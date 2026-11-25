package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type orgForceStreamTestDB struct {
	litellmdb.NoopManager
	tokens map[string]*dbmodels.TokenInfo
}

func (d *orgForceStreamTestDB) IsEnabled() bool           { return true }
func (d *orgForceStreamTestDB) IsHealthy() bool           { return true }
func (d *orgForceStreamTestDB) SpendLoggingEnabled() bool { return true }

func (d *orgForceStreamTestDB) ValidateToken(_ context.Context, rawToken string) (*dbmodels.TokenInfo, error) {
	info := d.tokens[rawToken]
	if info == nil {
		return nil, litellmdb.ErrTokenNotFound
	}
	return info.Clone(), nil
}

func (d *orgForceStreamTestDB) LogSpend(*dbmodels.SpendLogEntry) error { return nil }

// forceStreamUpstream starts a fake Chat Completions upstream that answers a
// genuine "stream":true body with SSE (what handleSimulatedNonStream needs to
// reassemble) and a non-streamed body with a plain JSON chat completion, and
// hands every request body it receives to the returned channel.
func forceStreamUpstream(t *testing.T) (*httptest.Server, <-chan map[string]any) {
	t.Helper()
	captured := make(chan map[string]any, 4)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		captured <- body

		if streamed, _ := body["stream"].(bool); streamed {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: " + `{"id":"c","model":"gpt","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"id":"c","model":"gpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"c","object":"chat.completion","model":"gpt",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}
		}`))
	}))
	t.Cleanup(upstream.Close)
	return upstream, captured
}

// newOrgForceStreamProxy builds a proxy with a single direct OpenAI credential,
// an org-policy registry, and force_stream_for_ttft on globally -- the
// precondition an organization-level opt-out must override.
func newOrgForceStreamProxy(t *testing.T, upstreamURL string, db *orgForceStreamTestDB, policies []config.OrganizationPolicyConfig) *Proxy {
	t.Helper()
	credential := config.CredentialConfig{
		Name: "openai-direct", Type: config.ProviderTypeOpenAI, BaseURL: upstreamURL, APIKey: "provider-key", RPM: 100, TPM: 10000,
	}
	logger := testhelpers.NewTestLogger()
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "gpt", Credential: credential.Name},
	})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})

	registry, err := routermodels.LoadOrganizationPolicies(policies, manager, routermodels.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      true,
		LiteLLMDBRequired:     true,
		DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").WithForceStreamForTTFT(true)
	builder.config.ModelManager = manager
	builder.config.OrganizationPolicies = registry
	prx := builder.Build()
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "gpt", &routermodels.ModelPrice{InputCostPerToken: 0.001, OutputCostPerToken: 0.002})
	return prx
}

// sendNonStreamProbe sends a non-streaming chat completion request and returns
// the body the upstream received.
func sendNonStreamProbe(t *testing.T, prx *Proxy, captured <-chan map[string]any) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt","messages":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	select {
	case body := <-captured:
		return body
	default:
		t.Fatal("the upstream received no request")
		return nil
	}
}

func TestOrganizationPolicy_DisablesForceStreamForTTFT(t *testing.T) {
	upstream, captured := forceStreamUpstream(t)
	db := &orgForceStreamTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-no-ttft", OrganizationID: "org-no-ttft"},
	}}
	prx := newOrgForceStreamProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID:            "org-no-ttft",
		DisableForceStreamForTTFT: true,
	}})

	body := sendNonStreamProbe(t, prx, captured)

	streamed, _ := body["stream"].(bool)
	assert.False(t, streamed,
		"the organization opted out, so the non-streaming request must reach upstream unchanged")
}

func TestOrganizationPolicy_ForceStreamForTTFTAppliedWithoutOptOut(t *testing.T) {
	upstream, captured := forceStreamUpstream(t)
	db := &orgForceStreamTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1", DirectOrganizationID: "org-default", OrganizationID: "org-default"},
	}}
	prx := newOrgForceStreamProxy(t, upstream.URL, db, []config.OrganizationPolicyConfig{{
		OrganizationID: "org-default",
	}})

	body := sendNonStreamProbe(t, prx, captured)

	streamed, _ := body["stream"].(bool)
	assert.True(t, streamed,
		"no opt-out on this organization, so the global force_stream_for_ttft toggle must still apply")
}

func TestOrganizationPolicy_NoPolicyStillAppliesForceStreamForTTFT(t *testing.T) {
	// A token with no organization at all (effective policy is nil) must fall
	// back to the global default: force_stream_for_ttft stays on.
	upstream, captured := forceStreamUpstream(t)
	db := &orgForceStreamTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"token": {Token: "token-hash", UserID: "user-1"},
	}}
	prx := newOrgForceStreamProxy(t, upstream.URL, db, nil)

	body := sendNonStreamProbe(t, prx, captured)

	streamed, _ := body["stream"].(bool)
	assert.True(t, streamed,
		"force_stream_for_ttft must still apply when no organization policy exists")
}

// TestNextRetryCredential_HonoursForceStreamOptOut and
// TestNextCrossTypeRetryCredential_HonoursForceStreamOptOut pin that the
// opt-out survives the two retry entry points that rebuild the request via
// prepareRequestForCredential (nextRetryCredential, used by the simulated
// non-stream handlers' own internal retry; nextCrossTypeRetryCredential, used
// by the main direct/cross-type retry loop), not just the first attempt's.
// This is the exact class of bug the repo already had with
// autoCacheControlDisabled (a field threaded through the first attempt but
// forgotten on a retry path): a direct call, mirroring
// TestSimulatedStreamRetryDoesNotSelectAIRForDirectSend's style, so it
// exercises the rebuild itself rather than depending on a specific HTTP
// failure actually routing through one of these functions end to end.
func TestNextRetryCredential_HonoursForceStreamOptOut(t *testing.T) {
	first := config.CredentialConfig{Name: "first", Type: config.ProviderTypeOpenAI, BaseURL: "http://first.invalid", APIKey: "k1", RPM: 100, TPM: 10000}
	second := config.CredentialConfig{Name: "second", Type: config.ProviderTypeOpenAI, BaseURL: "http://second.invalid", APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000}
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)

	for _, c := range []struct {
		name     string
		disabled bool
		want     assert.BoolAssertionFunc
	}{
		{"opted out", true, assert.False},
		{"not opted out", false, assert.True},
	} {
		t.Run(c.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().WithCredentials(first, second).WithForceStreamForTTFT(true).Build()
			tried := map[string]bool{"first": true}

			_, nextReq, ready, _ := prx.nextRetryCredential(
				httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
				&RequestLogContext{Scope: scope.PublicContext()}, "gpt-4", &first,
				&orchestratedRequest{
					baseBody: body, baseProxyBody: body, basePath: "/v1/chat/completions",
					forceStreamForTTFTDisabled: c.disabled,
				},
				false, 1, tried, map[string]bool{"first": true},
			)

			require.True(t, ready)
			streamed := strings.Contains(string(nextReq.body), `"stream":true`)
			c.want(t, streamed, "nextRetryCredential's rebuilt body: stream:true present = %v, opted out = %v", streamed, c.disabled)
		})
	}
}

func TestNextCrossTypeRetryCredential_HonoursForceStreamOptOut(t *testing.T) {
	first := config.CredentialConfig{Name: "first", Type: config.ProviderTypeOpenAI, BaseURL: "http://first.invalid", APIKey: "k1", RPM: 100, TPM: 10000}
	second := config.CredentialConfig{Name: "second", Type: config.ProviderTypeOpenAI, BaseURL: "http://second.invalid", APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000}
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)

	for _, c := range []struct {
		name     string
		disabled bool
		want     assert.BoolAssertionFunc
	}{
		{"opted out", true, assert.False},
		{"not opted out", false, assert.True},
	} {
		t.Run(c.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().WithCredentials(first, second).WithForceStreamForTTFT(true).Build()
			tried := map[string]bool{"first": true}

			_, nextReq, ready, _ := prx.nextCrossTypeRetryCredential(
				httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
				&orchestratedRequest{
					modelID: "gpt-4", baseBody: body, baseProxyBody: body, basePath: "/v1/chat/completions",
					forceStreamForTTFTDisabled: c.disabled,
				},
				&RequestLogContext{Scope: scope.PublicContext()}, &first, tried, map[string]bool{"first": true}, 1, true,
			)

			require.True(t, ready)
			streamed := strings.Contains(string(nextReq.body), `"stream":true`)
			c.want(t, streamed, "nextCrossTypeRetryCredential's rebuilt body: stream:true present = %v, opted out = %v", streamed, c.disabled)
		})
	}
}
