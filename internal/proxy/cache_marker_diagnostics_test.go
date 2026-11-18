package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/require"
)

// cacheMarkerProbe mirrors cacheEligibilityProbe but returns the full kafka
// spend event, so tests can inspect CacheMarkersIn*/CacheMarkersOut*
// alongside CacheControlOutcome.
func cacheMarkerProbe(t *testing.T, req *http.Request) *spendEventStub {
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
	e := stub.events[0]
	return &spendEventStub{
		CacheControlOutcome:     e.CacheControlOutcome,
		CacheEligibilitySource:  e.CacheEligibilitySource,
		CacheMarkersInSystem:    e.CacheMarkersInSystem,
		CacheMarkersInMessages:  e.CacheMarkersInMessages,
		CacheMarkersInTools:     e.CacheMarkersInTools,
		CacheMarkersInTTL1h:     e.CacheMarkersInTTL1h,
		CacheMarkersOutSystem:   e.CacheMarkersOutSystem,
		CacheMarkersOutMessages: e.CacheMarkersOutMessages,
		CacheMarkersOutTools:    e.CacheMarkersOutTools,
		CacheMarkersOutTTL1h:    e.CacheMarkersOutTTL1h,
	}
}

type spendEventStub struct {
	CacheControlOutcome                                                  string
	CacheEligibilitySource                                               string
	CacheMarkersInSystem, CacheMarkersInMessages, CacheMarkersInTools    int
	CacheMarkersInTTL1h                                                  int
	CacheMarkersOutSystem, CacheMarkersOutMessages, CacheMarkersOutTools int
	CacheMarkersOutTTL1h                                                 int
}

// TestCacheMarkers_ToolOnlyMarkerSurvivesConversionButOutcomeSaysNone is the
// exact production gap reported: a client marks only a tool definition with
// cache_control (no messages/system marker, no session signal at all).
// CacheControlOutcome still says "none" -- it only looks at messages/system
// -- but the marker is real, forwarded verbatim by the OpenAI->Anthropic
// tool converter (tools.go), and CacheMarkersIn/Out must show it.
func TestCacheMarkers_ToolOnlyMarkerSurvivesConversionButOutcomeSaysNone(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "big_schema", "parameters": {"type": "object"}}, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
		]
	}`))

	e := cacheMarkerProbe(t, req)

	require.Equal(t, "", e.CacheEligibilitySource, "no session signal was sent")
	require.Equal(t, "none", e.CacheControlOutcome,
		"CacheControlOutcome's known blind spot: it never looks at tools")

	require.Equal(t, 0, e.CacheMarkersInSystem)
	require.Equal(t, 0, e.CacheMarkersInMessages)
	require.Equal(t, 1, e.CacheMarkersInTools, "the client's tool-level marker must be counted on the way in")
	require.Equal(t, 1, e.CacheMarkersInTTL1h)

	require.Equal(t, 1, e.CacheMarkersOutTools, "the marker must still be there in the body actually sent upstream")
	require.Equal(t, 1, e.CacheMarkersOutTTL1h)
}

// TestCacheMarkers_RouterInjectedMessagesStillCountedOnBothSides covers the
// ordinary case for comparison: no client marker anywhere, a session header
// makes the request eligible, the router injects its own two breakpoints
// (system + history). Both In (zero) and Out (two: system + the marked
// history message) must reflect that distinctly.
func TestCacheMarkers_RouterInjectedMessagesStillCountedOnBothSides(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model": "claude",
		"system": "Be helpful.",
		"messages": [
			{"role": "user", "content": "first"},
			{"role": "assistant", "content": "ok"},
			{"role": "user", "content": "second"}
		]
	}`))
	req.Header.Set("Session-Id", "session-1")

	e := cacheMarkerProbe(t, req)

	require.Equal(t, "session_header", e.CacheEligibilitySource)
	require.Equal(t, "router_injected", e.CacheControlOutcome)

	require.Equal(t, 0, e.CacheMarkersInSystem, "nothing from the client")
	require.Equal(t, 0, e.CacheMarkersInMessages)
	require.Equal(t, 0, e.CacheMarkersInTools)

	require.Equal(t, 1, e.CacheMarkersOutSystem, "the router's system breakpoint")
	require.Equal(t, 1, e.CacheMarkersOutMessages, "the router's history breakpoint, on the second-to-last user message")
	require.Equal(t, 0, e.CacheMarkersOutTools)
	require.Equal(t, 0, e.CacheMarkersOutTTL1h, "the router's own markers use the 5m default, never 1h")
}

// TestCacheMarkers_NoCredentialAttemptedLeavesOutAtZero covers a request
// that never reaches an Anthropic-wire credential at all (CacheControlOutcome
// stays ""): CacheMarkersOut must stay all-zero too, not carry over some
// stale value.
func TestCacheMarkers_NoMarkersAnywhereIsAllZero(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}]
	}`))

	e := cacheMarkerProbe(t, req)

	require.Equal(t, "", e.CacheEligibilitySource)
	require.Equal(t, "none", e.CacheControlOutcome, "no system field, single user message: nothing for the router to mark")
	require.Equal(t, 0, e.CacheMarkersInSystem+e.CacheMarkersInMessages+e.CacheMarkersInTools)
	require.Equal(t, 0, e.CacheMarkersOutSystem+e.CacheMarkersOutMessages+e.CacheMarkersOutTools)
}
