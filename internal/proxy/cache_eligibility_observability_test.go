package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
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
// session-less eligibility path: no session_id/header, but a safety_identifier
// (or the user field it replaces) makes the request sticky-eligible anyway.
func TestCacheEligibility_ClientIDFallbackIsLabeled(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","system":"Be helpful.","messages":[{"role":"user","content":"hi"}],"user":"end-user-42"}`,
	))

	source, outcome := cacheEligibilityProbe(t, req)
	require.Equal(t, "client_id", source)
	require.Equal(t, "router_injected", outcome)
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
