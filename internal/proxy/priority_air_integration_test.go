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

// TestPriorityGroupCascade_MixedTierUpstreamLocalCascade is the review_158 #3 Design-B
// regression: one proxy credential ("router2") fronts an upstream whose /health lists the
// same model at TWO priority tiers — "r2-cheap" (tier 1, RPM budget 4) and "r2-expensive"
// (tier 5, large budget). The main router also has a direct mid-priced alternative
// ("routerMid", tier 3).
//
// router2 expands into two balancer candidates: router2@tier-1 (cumulative RPM cap 4) in
// primary group 1, and router2@tier-5 in group 5. routerMid is a single implicit tier in
// group 3. While the local aggregate usage for (router2, shared-model) is below 4, group 1
// serves. Once THIS router has itself sent 4 requests to router2 the tier-1 cumulative cap
// is hit locally — no /health poll needed — and selection cascades straight to group 3
// (routerMid), NOT to router2@tier-5, because tier 3 < tier 5.
func TestPriorityGroupCascade_MixedTierUpstreamLocalCascade(t *testing.T) {
	var router2Calls, routerMidCalls int32

	router2Health := func() *httputil.ProxyHealthResponse {
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"r2-cheap":     {Type: "openai", Priority: 1, LimitRPM: 4, LimitTPM: 1000000},
				"r2-expensive": {Type: "openai", Priority: 5, LimitRPM: 1000, LimitTPM: 1000000},
			},
			Models: map[string]httputil.ModelHealthStats{
				"c": {Credential: "r2-cheap", Model: "shared-model", Priority: 1, LimitRPM: 4, LimitTPM: 1000000},
				"e": {Credential: "r2-expensive", Model: "shared-model", Priority: 5, LimitRPM: 1000, LimitTPM: 1000000},
			},
		}
	}
	routerMidHealth := func() *httputil.ProxyHealthResponse {
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"mid-upstream": {Type: "openai", Priority: 3, LimitRPM: 1000, LimitTPM: 1000000},
			},
			Models: map[string]httputil.ModelHealthStats{
				"m": {Credential: "mid-upstream", Model: "shared-model", Priority: 3, LimitRPM: 1000, LimitTPM: 1000000},
			},
		}
	}

	router2 := mockHealthAndCompletionServer(t, &router2Calls, router2Health, "response from router2")
	defer router2.Close()
	routerMid := mockHealthAndCompletionServer(t, &routerMidCalls, routerMidHealth, "response from routerMid")
	defer routerMid.Close()

	credRouter2 := config.CredentialConfig{Name: "router2", Type: config.ProviderTypeProxy, APIKey: "k2", BaseURL: router2.URL, RPM: 10000, TPM: 100000000}
	credRouterMid := config.CredentialConfig{Name: "routerMid", Type: config.ProviderTypeProxy, APIKey: "km", BaseURL: routerMid.URL, RPM: 10000, TPM: 100000000}

	builder := NewTestProxyBuilder().WithCredentials(credRouter2, credRouterMid)
	rl := ratelimit.New()
	builder.config.RateLimiter = rl
	mm := builder.config.ModelManager
	logger := builder.config.Logger
	prx := builder.Build()

	ctx := context.Background()
	c2, cm := credRouter2, credRouterMid
	UpdateStatsFromRemoteProxy(ctx, &c2, rl, logger, mm)
	UpdateStatsFromRemoteProxy(ctx, &cm, rl, logger, mm)

	// router2 learned a 2-tier breakdown; routerMid is a single implicit tier.
	tiers := mm.GetModelPriorityTiersForCredential("shared-model", "router2")
	require.Len(t, tiers, 2)
	require.Equal(t, 1, tiers[0].Priority)
	require.Equal(t, 4, tiers[0].LimitRPM)
	require.Equal(t, 5, tiers[1].Priority)
	require.Nil(t, mm.GetModelPriorityTiersForCredential("shared-model", "routerMid"))
	require.Equal(t, 1, mm.GetModelPriorityForCredential("shared-model", "router2"))
	require.Equal(t, 3, mm.GetModelPriorityForCredential("shared-model", "routerMid"))

	// First 4 requests fill router2's tier-1 cumulative cap (4) — all served by router2.
	for i := 0; i < 4; i++ {
		w := doPriorityTestRequest(t, prx, "shared-model")
		require.Equal(t, http.StatusOK, w.Code, "request %d", i)
		require.Contains(t, w.Body.String(), "response from router2")
	}
	require.EqualValues(t, 4, atomic.LoadInt32(&router2Calls))
	require.EqualValues(t, 0, atomic.LoadInt32(&routerMidCalls))

	// No re-poll. tier-1 is now locally saturated, so the next requests cascade to
	// group 3 (routerMid) — never to router2's tier-5 group.
	for i := 0; i < 5; i++ {
		w := doPriorityTestRequest(t, prx, "shared-model")
		require.Equal(t, http.StatusOK, w.Code, "cascade request %d", i)
		require.Contains(t, w.Body.String(), "response from routerMid")
	}
	assert.EqualValues(t, 4, atomic.LoadInt32(&router2Calls), "router2 must stop once its tier-1 cumulative cap is hit locally — no poll lag")
	assert.EqualValues(t, 5, atomic.LoadInt32(&routerMidCalls), "traffic cascades to the tier-3 alternative, not router2's tier-5")
}

// TestPriorityGroupCascade_UpstreamFallbackCredentialSurfacesAsLastResortTier:
// router2's own upstream node has TWO credentials — one regular (serves "model-a") and
// one last-resort (serves "model-b" only, priority 999). Post-unification model-b is NOT
// hidden: it surfaces on router2 at priority 999 and is routable via the local
// last-resort tier — an all-last-resort upstream node no longer contributes zero
// routable models.
func TestPriorityGroupCascade_UpstreamFallbackCredentialSurfacesAsLastResortTier(t *testing.T) {
	var router2CompletionCalls int32

	router2Health := func() *httputil.ProxyHealthResponse {
		return &httputil.ProxyHealthResponse{
			Status: "healthy",
			Credentials: map[string]httputil.CredentialHealthStats{
				"router2-regular":  {Type: "openai", Priority: 1, LimitRPM: 1000, LimitTPM: 1000000},
				"router2-fallback": {Type: "openai", Priority: config.FallbackPriorityGroup, LastResort: true, LimitRPM: 1000, LimitTPM: 1000000},
			},
			Models: map[string]httputil.ModelHealthStats{
				"a": {Credential: "router2-regular", Model: "model-a", Priority: 1, LimitRPM: 1000, LimitTPM: 1000000},
				"b": {Credential: "router2-fallback", Model: "model-b", Priority: config.FallbackPriorityGroup, LimitRPM: 1000, LimitTPM: 1000000},
			},
		}
	}

	router2 := mockHealthAndCompletionServer(t, &router2CompletionCalls, router2Health, "response from router2")
	defer router2.Close()

	credRouter2 := config.CredentialConfig{Name: "router2", Type: config.ProviderTypeProxy, APIKey: "router2-key", BaseURL: router2.URL, RPM: 1000, TPM: 1000000}

	builder := NewTestProxyBuilder().WithCredentials(credRouter2)
	rl := ratelimit.New()
	builder.config.RateLimiter = rl
	mm := builder.config.ModelManager
	logger := builder.config.Logger
	prx := builder.Build()

	ctx := context.Background()
	c2 := credRouter2
	UpdateStatsFromRemoteProxy(ctx, &c2, rl, logger, mm)

	// model-a (regular upstream credential) surfaces at its tier; model-b (served only via
	// the last-resort upstream credential) now surfaces too, at the last-resort group.
	require.True(t, mm.HasModel("router2", "model-a"))
	require.Equal(t, 1, mm.GetModelPriorityForCredential("model-a", "router2"))
	require.True(t, mm.HasModel("router2", "model-b"), "a model served only via a last-resort upstream credential now surfaces, tiered locally")
	require.Equal(t, config.FallbackPriorityGroup, mm.GetModelPriorityForCredential("model-b", "router2"))

	// End-to-end: both route through router2. model-b via the local last-resort tier.
	w := doPriorityTestRequest(t, prx, "model-a")
	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "response from router2")
	assert.EqualValues(t, 1, atomic.LoadInt32(&router2CompletionCalls))

	wb := doPriorityTestRequest(t, prx, "model-b")
	require.Equal(t, http.StatusOK, wb.Code, "model-b routes via the local last-resort tier")
	require.Contains(t, wb.Body.String(), "response from router2")
	assert.EqualValues(t, 2, atomic.LoadInt32(&router2CompletionCalls))
}
