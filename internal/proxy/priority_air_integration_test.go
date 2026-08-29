package proxy

// These tests exercise the FULL real path end-to-end with actual HTTP round-trips
// between "nodes" (httptest.Server, not mocked interfaces):
//
//	main router (RoundRobin balancer under test)
//	 ├─ proxy credential "router2" (priority learned from router2's own /health)
//	 └─ proxy credential "router3" (priority learned from router3's own /health)
//
//	1. GET /health on router2/router3 (simulating one poll cycle of the background
//	   poller — see internal/proxy/remote.go UpdateStatsFromRemoteProxy)
//	2. → updateCredentialLimits/updateModelLimits aggregate limits+priority
//	3. → modelManager.ReplaceModelPrioritiesForCredential (T3)
//	4. → balancer.RoundRobin.effectivePriority resolves it via ModelChecker (T3)
//	5. → nextExcludingScoped/selectPriorityGroupCandidate picks a credential
//	6. POST completion request is actually proxied to the winning node's mock server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockHealthAndCompletionServer builds an httptest.Server that serves GET /health from
// the (mutable, atomically-swappable) healthFn and treats every other path as the
// completion endpoint, recording a call and returning a distinctive 200 JSON body.
func mockHealthAndCompletionServer(t *testing.T, completionCalls *int32, healthFn func() *httputil.ProxyHealthResponse, responseContent string) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(healthFn())
			return
		}
		atomic.AddInt32(completionCalls, 1)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-" + responseContent,
			"object": "chat.completion",
			"choices": []map[string]interface{}{
				{"message": map[string]string{"role": "assistant", "content": responseContent}},
			},
		})
	}))
}

func doPriorityTestRequest(t *testing.T, prx *Proxy, modelID string) *httptest.ResponseRecorder {
	t.Helper()
	body := `{"model": "` + modelID + `", "messages": [{"role": "user", "content": "hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// TestPriorityGroupCascade_ThreeRouterTopology is the requested three-router scenario:
// the main router has exactly two proxy/AIR credentials — "router2" (learns priority 1
// from its own /health) and "router3" (priority 2) — and nothing else. While router2 is
// healthy, every completion request must be routed to router2 and router3 must never be
// contacted for completions. Once router2 reports (via /health) that it has no RPM/TPM
// headroom left for the model, the next poll cycle must pull that in, and subsequent
// completion requests must cascade to router3 — router2 stops receiving completion
// traffic entirely.
func TestPriorityGroupCascade_ThreeRouterTopology(t *testing.T) {
	var router2CompletionCalls, router3CompletionCalls int32
	var router2Exhausted atomic.Bool

	router2Health := func() *httputil.ProxyHealthResponse {
		limitRPM, limitTPM, curRPM, curTPM := 1000, 1000000, 0, 0
		if router2Exhausted.Load() {
			// Upstream reports it has fully used its RPM/TPM budget for this model —
			// this is the "sync limits from upstream" path (remote.go:225-227 /
			// updateModelLimits), not a real 429/ban on the main router's side.
			limitRPM, limitTPM, curRPM, curTPM = 5, 5000, 5, 5000
		}
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"router2-upstream": {Type: "openai", Priority: 1, LimitRPM: limitRPM, LimitTPM: limitTPM, CurrentRPM: curRPM, CurrentTPM: curTPM},
			},
			Models: map[string]httputil.ModelHealthStats{
				"m": {Credential: "router2-upstream", Model: "shared-model", Priority: 1, LimitRPM: limitRPM, LimitTPM: limitTPM, CurrentRPM: curRPM, CurrentTPM: curTPM},
			},
		}
	}
	router3Health := func() *httputil.ProxyHealthResponse {
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"router3-upstream": {Type: "openai", Priority: 2, LimitRPM: 1000, LimitTPM: 1000000},
			},
			Models: map[string]httputil.ModelHealthStats{
				"m": {Credential: "router3-upstream", Model: "shared-model", Priority: 2, LimitRPM: 1000, LimitTPM: 1000000},
			},
		}
	}

	router2 := mockHealthAndCompletionServer(t, &router2CompletionCalls, router2Health, "response from router2")
	defer router2.Close()
	router3 := mockHealthAndCompletionServer(t, &router3CompletionCalls, router3Health, "response from router3")
	defer router3.Close()

	credRouter2 := config.CredentialConfig{Name: "router2", Type: config.ProviderTypeProxy, APIKey: "router2-key", BaseURL: router2.URL, RPM: 1000, TPM: 1000000}
	credRouter3 := config.CredentialConfig{Name: "router3", Type: config.ProviderTypeProxy, APIKey: "router3-key", BaseURL: router3.URL, RPM: 1000, TPM: 1000000}

	builder := NewTestProxyBuilder().WithCredentials(credRouter2, credRouter3)
	rl := ratelimit.New()
	builder.config.RateLimiter = rl
	mm := builder.config.ModelManager
	logger := builder.config.Logger
	prx := builder.Build()

	ctx := context.Background()
	pollAll := func() {
		c2, c3 := credRouter2, credRouter3
		UpdateStatsFromRemoteProxy(ctx, &c2, rl, logger, mm)
		UpdateStatsFromRemoteProxy(ctx, &c3, rl, logger, mm)
	}

	// Poll cycle 1: both nodes healthy. router2's learned priority (1) beats router3's (2).
	pollAll()
	require.Equal(t, 1, mm.GetModelPriorityForCredential("shared-model", "router2"))
	require.Equal(t, 2, mm.GetModelPriorityForCredential("shared-model", "router3"))

	for i := 0; i < 5; i++ {
		w := doPriorityTestRequest(t, prx, "shared-model")
		require.Equal(t, http.StatusOK, w.Code)
		require.Contains(t, w.Body.String(), "response from router2")
	}
	assert.EqualValues(t, 5, atomic.LoadInt32(&router2CompletionCalls), "all requests must go to router2 while it's the sole live member of the lower-priority group")
	assert.EqualValues(t, 0, atomic.LoadInt32(&router3CompletionCalls), "router3 must not be contacted for completions while router2's group is live")

	// router2's node reports (via /health) that it has no headroom left for this model.
	router2Exhausted.Store(true)
	pollAll()

	for i := 0; i < 3; i++ {
		w := doPriorityTestRequest(t, prx, "shared-model")
		require.Equal(t, http.StatusOK, w.Code)
		require.Contains(t, w.Body.String(), "response from router3")
	}
	assert.EqualValues(t, 5, atomic.LoadInt32(&router2CompletionCalls), "router2 must not receive any more completion traffic once its group is fully exhausted")
	assert.EqualValues(t, 3, atomic.LoadInt32(&router3CompletionCalls), "requests must cascade to router3's group once router2's is down")
}

// TestPriorityGroupCascade_UpstreamIsFallbackCredentialExcludedForPrimaryConnection:
// router2's own upstream node has TWO credentials — one regular (serves "model-a") and
// one is_fallback (serves "model-b" only). The main router's "router2" proxy credential
// is itself NOT is_fallback, so the single poller applies the same rule as the old
// model-snapshot path: an is_fallback upstream credential (reserved capacity for the
// upstream's own fallback traffic) is excluded from a primary connection's aggregation.
// model-b must NOT surface on router2, and a request for it must fail rather than be
// silently routed onto reserved fallback capacity.
func TestPriorityGroupCascade_UpstreamIsFallbackCredentialExcludedForPrimaryConnection(t *testing.T) {
	var router2CompletionCalls int32

	router2Health := func() *httputil.ProxyHealthResponse {
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"router2-regular":  {Type: "openai", IsFallback: false, Priority: 1, LimitRPM: 1000, LimitTPM: 1000000},
				"router2-fallback": {Type: "openai", IsFallback: true, Priority: 5, LimitRPM: 1000, LimitTPM: 1000000},
			},
			Models: map[string]httputil.ModelHealthStats{
				"a": {Credential: "router2-regular", Model: "model-a", Priority: 1, LimitRPM: 1000, LimitTPM: 1000000},
				"b": {Credential: "router2-fallback", Model: "model-b", Priority: 5, LimitRPM: 1000, LimitTPM: 1000000},
			},
		}
	}

	router2 := mockHealthAndCompletionServer(t, &router2CompletionCalls, router2Health, "response from router2")
	defer router2.Close()

	credRouter2 := config.CredentialConfig{Name: "router2", Type: config.ProviderTypeProxy, IsFallback: false, APIKey: "router2-key", BaseURL: router2.URL, RPM: 1000, TPM: 1000000}

	builder := NewTestProxyBuilder().WithCredentials(credRouter2)
	rl := ratelimit.New()
	builder.config.RateLimiter = rl
	mm := builder.config.ModelManager
	logger := builder.config.Logger
	prx := builder.Build()

	ctx := context.Background()
	c2 := credRouter2
	UpdateStatsFromRemoteProxy(ctx, &c2, rl, logger, mm)

	// model-a (regular upstream credential) surfaces; model-b (is_fallback-only) does not.
	require.True(t, mm.HasModel("router2", "model-a"))
	require.Equal(t, 1, mm.GetModelPriorityForCredential("model-a", "router2"))
	require.False(t, mm.HasModel("router2", "model-b"), "a model served only via an is_fallback upstream credential must not surface on a primary connection")
	require.Equal(t, 0, mm.GetModelPriorityForCredential("model-b", "router2"))

	// End-to-end: model-a routes; model-b has no live credential and must fail.
	w := doPriorityTestRequest(t, prx, "model-a")
	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "response from router2")
	assert.EqualValues(t, 1, atomic.LoadInt32(&router2CompletionCalls))

	wb := doPriorityTestRequest(t, prx, "model-b")
	require.NotEqual(t, http.StatusOK, wb.Code, "model-b must not be routable via a primary connection to an is_fallback-only upstream credential")
	assert.EqualValues(t, 1, atomic.LoadInt32(&router2CompletionCalls))
}
