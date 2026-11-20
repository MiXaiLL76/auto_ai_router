package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/require"
)

// cacheEligibilityProbe wires newOrgCacheControlProxy's fake native-Anthropic
// upstream together with a stubKafkaManager, sends req, and returns the
// CacheEligibilitySource/CacheControlOutcome the kafka spend event recorded
// for it -- the end-to-end path a human would actually query in ClickHouse.
func cacheEligibilityProbe(t *testing.T, req *http.Request) (source, outcome string) {
	t.Helper()
	upstream, captured := anthropicPassthroughUpstream(t)
	db := &orgCacheControlTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1"},
	}}
	prx := newOrgCacheControlProxy(t, upstream.URL, db, nil)
	stub := &stubKafkaManager{enabled: true}
	prx.kafkaLog = stub

	req.Header.Set("Authorization", "Bearer token")
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	select {
	case <-captured:
	default:
		t.Fatal("the upstream received no request")
	}
	require.Len(t, stub.events, 1, "exactly one kafka spend event must be published")
	return stub.events[0].CacheEligibilitySource, stub.events[0].CacheControlOutcome
}

// TestCacheEligibility_SessionHeaderInjectsAndLabelsSource covers the common
// /v1/messages case: a Session-Id header makes the request eligible, the
// router injects cache_control itself, and both new fields reflect that.
func TestCacheEligibility_SessionHeaderInjectsAndLabelsSource(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Session-Id", "session-1")

	source, outcome := cacheEligibilityProbe(t, req)
	require.Equal(t, "session_header", source)
	require.Equal(t, "router_injected", outcome)
}

// TestCacheEligibility_NoSignalAtAllReportsNone covers the "plain batch of
// requests" case directly raised in conversation: no Session-Id, no
// session_id/litellm_session_id/user/safety_identifier field, no
// previous_response_id. The router must not inject anything, and both
// fields must say so plainly instead of leaving an ambiguous blank.
func TestCacheEligibility_NoSignalAtAllReportsNone(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}]}`,
	))

	source, outcome := cacheEligibilityProbe(t, req)
	require.Equal(t, "", source, "no signal was sent, so no source should be attributed")
	require.Equal(t, "none", outcome, "not eligible and the client sent no cache_control of its own")
}

// TestCacheEligibility_ClientSuppliedCacheControlIsLabeledEvenWithoutSession
// is the case the earlier investigation had no way to see directly: a client
// manages its own cache_control breakpoints and never sends any session
// signal at all. The router must not touch the body, but must still report
// that the client attempted caching on its own -- this is the whole point of
// the new fields: telling "the client tried" apart from "the router did it".
func TestCacheEligibility_ClientSuppliedCacheControlIsLabeledEvenWithoutSession(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","system":[{"type":"text","text":"Be helpful.","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`,
	))

	source, outcome := cacheEligibilityProbe(t, req)
	require.Equal(t, "", source, "the client never sent a session signal")
	require.Equal(t, "client_supplied", outcome, "the client's own cache_control must be recognized even though the router was never eligible to act")
}

// TestCacheEligibility_ClientIDFallbackIsLabeled covers the third
// session-less eligibility path: no session_id/header, but safety_identifier
// or the user field it replaces makes the request sticky-eligible anyway.
// The two are reported under distinct source labels (not one generic
// "client_id") because user is routinely just an application-level
// end-user id with no session intent at all, unlike safety_identifier.
func TestCacheEligibility_ClientIDFallbackIsLabeled(t *testing.T) {
	t.Run("safety_identifier", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}],"safety_identifier":"end-user-42"}`,
		))
		source, outcome := cacheEligibilityProbe(t, req)
		require.Equal(t, "safety_identifier", source)
		require.Equal(t, "router_injected", outcome)
	})

	t.Run("user_field", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}],"user":"end-user-42"}`,
		))
		source, outcome := cacheEligibilityProbe(t, req)
		require.Equal(t, "user_field", source)
		require.Equal(t, "router_injected", outcome)
	})
}

// TestCacheEligibility_ResponsesAPINativePathIsLabeledAfterConversion guards
// the exact bug this observability was built to stop recurring: a
// /v1/responses request routed through the native Responses->Anthropic
// converter only becomes Messages-shaped in proxy.go, well after
// orchestrator.go's own InjectCacheControl call saw a shape it couldn't use.
// The final outcome must still be attributed correctly, not silently "none".
func TestCacheEligibility_ResponsesAPINativePathIsLabeledAfterConversion(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"claude","instructions":"Be helpful.","input":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Session-Id", "session-1")

	source, outcome := cacheEligibilityProbe(t, req)
	require.Equal(t, "session_header", source)
	require.Equal(t, "router_injected", outcome, "must reflect the post-conversion check in sendToCredential, not orchestrator.go's premature CacheControlNone on the still-Responses-shaped body")
}

// TestCacheEligibility_PreviousResponseIDAloneIsLabeled covers the one
// eligibility path that needs no client-sent session signal at all: a
// resolved previous_response_id makes stickyCacheEligible true on its own
// (preferredCredentialName), via the stored-entry lookup in
// orchestrateRequest. This is the path that explains "cache_control was
// present even though the client never sent a session" for genuine Responses
// API conversations.
func TestCacheEligibility_PreviousResponseIDAloneIsLabeled(t *testing.T) {
	upstream, captured := anthropicPassthroughUpstream(t)
	db := &orgCacheControlTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1"},
	}}
	prx := newOrgCacheControlProxy(t, upstream.URL, db, nil)
	prx.responseStore = &fakeResponseStore{
		entry: &responsestore.StoredEntry{
			ResponseID:     "resp-prev",
			APIKeyHash:     "hash",
			CredentialName: "anthropic-direct",
			ResponseJSON:   &responses.Response{ID: "resp-prev", Model: "claude"},
		},
	}
	stub := &stubKafkaManager{enabled: true}
	prx.kafkaLog = stub

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"claude","instructions":"Be helpful.","input":[{"role":"user","content":"hi"}],"previous_response_id":"resp-prev"}`,
	))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	select {
	case <-captured:
	default:
		t.Fatal("the upstream received no request")
	}
	require.Len(t, stub.events, 1)
	require.Equal(t, "previous_response_id", stub.events[0].CacheEligibilitySource)
	require.Equal(t, "router_injected", stub.events[0].CacheControlOutcome)
}

// TestCacheEligibility_SurvivesFallbackToNonAnthropicCredential guards a
// retry-specific bug: sendToCredential runs once per credential attempt and
// reuses the same logCtx, so a later attempt on a non-Anthropic-wire
// credential must not blank out an earlier attempt's "router_injected" --
// the final kafka spend event is built once, after the last attempt, from
// whatever logCtx holds at that point.
func TestCacheEligibility_SurvivesFallbackToNonAnthropicCredential(t *testing.T) {
	anthropicDown := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(anthropicDown.Close)
	openaiUp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	t.Cleanup(openaiUp.Close)

	anthropicCred := config.CredentialConfig{
		Name: "anthropic-flaky", Type: config.ProviderTypeAnthropic, BaseURL: anthropicDown.URL, APIKey: "k1",
		RPM: 100, TPM: 10000, FallbackPriority: 10,
	}
	openaiCred := config.CredentialConfig{
		Name: "openai-fallback", Type: config.ProviderTypeOpenAI, BaseURL: openaiUp.URL, APIKey: "k2",
		RPM: 100, TPM: 10000, FallbackPriority: 20,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(anthropicCred, openaiCred).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()
	prx.stickyAutoCacheCtrl = true
	setTestModelPrice(prx, "claude", &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerToken: 0.002})
	stub := &stubKafkaManager{enabled: true}
	prx.kafkaLog = stub

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"claude","messages":[{"role":"system","content":"Be helpful."},{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Session-Id", "session-1")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, stub.events, 1)
	require.Equal(t, "router_injected", stub.events[0].CacheControlOutcome,
		"the failed first attempt's cache_control injection must survive the fallback to a non-Anthropic credential")
}

// TestCacheEligibility_AppliesAcrossAnthropicWireProviderTypes checks that
// injection and outcome-labeling aren't accidentally Anthropic-type-only:
// the eligibility switches in orchestrator.go/proxy.go list all four
// Anthropic-wire provider types (Anthropic, CometAPI, ProMan, Bedrock)
// side by side, and a typo dropping one from either switch would otherwise
// go unnoticed, since every other test in this file only exercises
// ProviderTypeAnthropic.
func TestCacheEligibility_AppliesAcrossAnthropicWireProviderTypes(t *testing.T) {
	for _, providerType := range []config.ProviderType{
		config.ProviderTypeAnthropic,
		config.ProviderTypeCometAPI,
		config.ProviderTypeProMan,
		config.ProviderTypeBedrock,
	} {
		t.Run(string(providerType), func(t *testing.T) {
			upstream, captured := anthropicPassthroughUpstream(t)
			credential := config.CredentialConfig{
				Name: "cred", Type: providerType, BaseURL: upstream.URL, APIKey: "k", RPM: 100, TPM: 10000,
			}
			logger := testhelpers.NewTestLogger()
			manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
				{Name: "claude", Credential: credential.Name},
			})
			manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
			manager.SetCredentials([]config.CredentialConfig{credential})

			builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
			builder.config.ModelManager = manager
			prx := builder.Build()
			prx.stickyAutoCacheCtrl = true
			setTestModelPrice(prx, "claude", &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerToken: 0.002})
			stub := &stubKafkaManager{enabled: true}
			prx.kafkaLog = stub

			// /v1/messages passthrough, not the native-Responses path: Bedrock's
			// chat-shaped converter only speaks Anthropic for model IDs prefixed
			// "anthropic." (isAnthropicBedrockModel), which "claude" isn't, but
			// passthroughMessages forwards the body as-is regardless of model
			// name, so it's the one shape all four provider types handle alike.
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
				`{"model":"claude","max_tokens":100,"system":"Be helpful.","messages":[{"role":"user","content":"hi"}]}`,
			))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Session-Id", "session-1")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			select {
			case <-captured:
			default:
				t.Fatal("the upstream received no request")
			}
			require.Len(t, stub.events, 1)
			require.Equal(t, "session_header", stub.events[0].CacheEligibilitySource)
			require.Equal(t, "router_injected", stub.events[0].CacheControlOutcome)
		})
	}
}
