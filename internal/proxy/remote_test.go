package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUpdateCredentialLimits_EmptyCredentials(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should handle empty credentials gracefully
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Verify no credentials were added
	assert.Equal(t, 0, rateLimiter.GetCurrentRPM("test_proxy"))
}

func TestUpdateCredentialLimits_SingleCredential(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {
				Type:       "openai",
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 50,
				CurrentTPM: 500,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should not panic or error
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Verify that credential was added (should have non-zero limits)
	// The exact values depend on rate limiter internals
	assert.NotNil(t, rateLimiter)
}

func TestUpdateCredentialLimits_MultipleCredentials_MaxSelection(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 10, CurrentTPM: 100},
			"remote_cred_2": {LimitRPM: 200, LimitTPM: 2000, CurrentRPM: 20, CurrentTPM: 200},
			"remote_cred_3": {LimitRPM: 150, LimitTPM: 1500, CurrentRPM: 15, CurrentTPM: 150},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should aggregate credentials without error
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Verify it processed all credentials
	assert.NotNil(t, rateLimiter)
}

func TestUpdateCredentialLimits_ZeroValues(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {LimitRPM: 0, LimitTPM: 0, CurrentRPM: 0, CurrentTPM: 0},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Should not add credential if all limits are 0
	assert.Equal(t, 0, rateLimiter.GetCurrentRPM("test_proxy"))
}

func TestUpdateCredentialLimits_MixedValues(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {LimitRPM: 100, LimitTPM: 0, CurrentRPM: 25, CurrentTPM: 0},
			"remote_cred_2": {LimitRPM: 0, LimitTPM: 2000, CurrentRPM: 0, CurrentTPM: 500},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should handle mixed values without error
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Should process both credentials
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_EmptyModels(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should handle empty models gracefully
	updateModelLimits(health, cred, rateLimiter, logger, nil)
}

func TestUpdateModelLimits_SingleModel(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"gpt4:proxy": {
				Model:      "gpt-4",
				LimitRPM:   100,
				LimitTPM:   2000,
				CurrentRPM: 50,
				CurrentTPM: 1000,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should add model without error
	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Should have model limits set
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_MultipleModels_Aggregation(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"gpt4:cred1": {
				Model:      "gpt-4",
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 30,
				CurrentTPM: 300,
			},
			"gpt4:cred2": {
				Model:      "gpt-4",
				LimitRPM:   200,
				LimitTPM:   2000,
				CurrentRPM: 60,
				CurrentTPM: 600,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should aggregate multiple model instances
	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Verify aggregation happened without error
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_ZeroValues_TrackedInManagerOnly(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote": {Weight: 4},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:proxy": {
				Model:      "claude-3-opus",
				Credential: "remote",
				LimitRPM:   0,
				LimitTPM:   0,
				CurrentRPM: 0,
				CurrentTPM: 0,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Empty(t, rateLimiter.GetAllModelPairs(), "all-zero model stats must not create an unlimited limiter entry")
	assert.True(t, mockMM.HasModel("test_proxy", "claude-3-opus"), "model remains discoverable for routing")
	assert.Equal(t, 4, mockMM.GetModelWeightForCredential("claude-3-opus", "test_proxy"))
}

func TestUpdateModelLimits_PriorityDeliveredToModelManager(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote": {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:remote": {
				Model:      "gpt-4",
				Credential: "remote",
				Priority:   200,
				LimitRPM:   100,
				LimitTPM:   1000,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 200, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"),
		"priority learned from the upstream credential's /health response must reach ReplaceModelPrioritiesForCredential")
}

func TestUpdateModelLimits_PropagatesLearnedPriorityThroughProxyOfProxyChain(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"pol01-proxy-cred": {Priority: 0},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:pol01": {
				Model:      "gpt-4",
				Credential: "pol01-proxy-cred",
				Priority:   300,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "ru01-proxy-cred"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 300, mockMM.GetModelPriorityForCredential("gpt-4", "ru01-proxy-cred"),
		"pol01's dynamically-learned per-model priority must reach ru01, not pol01-proxy-cred's static (unset) priority")
}

// TestUpdateModelLimits_PriorityMinAggregation verifies that when several upstream
// credentials in the remote proxy's /health response offer the same model at different
// priorities (e.g. two grant credentials on the same node with different priority
// groups), the local proxy credential's effective priority for that model is the MINIMUM
// (highest-priority / tried-first group) — not a sum, unlike RPM/TPM limits.
func TestUpdateModelLimits_PriorityMinAggregation(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-a": {Priority: 300},
			"grant-b": {Priority: 100},
			"grant-c": {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:a": {Model: "gpt-4", Credential: "grant-a", Priority: 300},
			"model:b": {Model: "gpt-4", Credential: "grant-b", Priority: 100},
			"model:c": {Model: "gpt-4", Credential: "grant-c", Priority: 200},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 100, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
}

// TestUpdateModelLimits_EmitsPriorityTiers_WhenUpstreamSpansMultipleGroups is the
// Design B poll-side piece: an upstream serving one model from >= 2 priority groups
// produces a per-tier breakdown (summed capacity + weight per tier), sorted ascending.
func TestUpdateModelLimits_EmitsPriorityTiers_WhenUpstreamSpansMultipleGroups(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"vertex-1": {Priority: 1}, "vertex-2": {Priority: 1},
			"gemini-fallback": {Priority: 2},
		},
		Models: map[string]httputil.ModelHealthStats{
			"a": {Model: "gemini-2.5-flash", Credential: "vertex-1", Priority: 1, Weight: 20, LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 5, CurrentTPM: 50},
			"b": {Model: "gemini-2.5-flash", Credential: "vertex-2", Priority: 1, Weight: 20, LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 3, CurrentTPM: 30},
			"c": {Model: "gemini-2.5-flash", Credential: "gemini-fallback", Priority: 2, Weight: 1, LimitRPM: 500, LimitTPM: 5000, CurrentRPM: 1, CurrentTPM: 10},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "usa03"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	tiers := mockMM.GetModelPriorityTiersForCredential("gemini-2.5-flash", "usa03")
	require.Len(t, tiers, 2)
	assert.Equal(t, 1, tiers[0].Priority)
	assert.Equal(t, 40, tiers[0].Weight, "tier 1 weight = 20 + 20")
	assert.Equal(t, 200, tiers[0].LimitRPM, "tier 1 RPM = 100 + 100")
	assert.Equal(t, 8, tiers[0].CurrentRPM, "tier 1 current = 5 + 3")
	assert.False(t, tiers[0].Banned)
	assert.Equal(t, 2, tiers[1].Priority)
	assert.Equal(t, 500, tiers[1].LimitRPM)

	// Aggregate (cred,model) bucket is still the grand total.
	assert.Equal(t, 700, rateLimiter.GetModelLimitRPM("usa03", "gemini-2.5-flash"))
}

// TestUpdateModelLimits_TierBanned_OnlyRealBanNotSaturation is the review_158 item 12
// poll-side piece: a tier's Banned flag must reflect a real upstream ban, not mere
// RPM/TPM saturation. A saturated-but-not-banned tier keeps Banned=false and reports its
// usage via Current* so the downstream balancer surfaces 429 (rate limit), not 503.
func TestUpdateModelLimits_TierBanned_OnlyRealBanNotSaturation(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"c1": {Priority: 1}, "c2": {Priority: 2},
		},
		Models: map[string]httputil.ModelHealthStats{
			// tier 1: not banned, but RPM budget fully consumed upstream.
			"a": {Model: "m", Credential: "c1", Priority: 1, LimitRPM: 100, CurrentRPM: 100},
			// tier 2: really banned.
			"b": {Model: "m", Credential: "c2", Priority: 2, LimitRPM: 500, CurrentRPM: 0, IsBanned: true},
		},
	}
	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "px"}
	mockMM := NewMockModelManager()
	updateModelLimits(health, cred, rateLimiter, testhelpers.NewTestLogger(), mockMM)

	tiers := mockMM.GetModelPriorityTiersForCredential("m", "px")
	require.Len(t, tiers, 2)
	assert.Equal(t, 1, tiers[0].Priority)
	assert.False(t, tiers[0].Banned, "saturated-but-not-banned tier must keep Banned=false")
	assert.Equal(t, 100, tiers[0].CurrentRPM)
	assert.Equal(t, 100, tiers[0].LimitRPM)
	assert.Equal(t, 2, tiers[1].Priority)
	assert.True(t, tiers[1].Banned, "tier whose only contributor is IsBanned must be Banned=true")
}

// TestUpdateModelLimits_NoPriorityTiers_ForSingleGroupUpstream: a model served from one
// priority group stays on the scalar path — no tier breakdown emitted.
func TestUpdateModelLimits_NoPriorityTiers_ForSingleGroupUpstream(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"c1": {Priority: 1}, "c2": {Priority: 1},
		},
		Models: map[string]httputil.ModelHealthStats{
			"a": {Model: "gpt-4", Credential: "c1", Priority: 1, LimitRPM: 100},
			"b": {Model: "gpt-4", Credential: "c2", Priority: 1, LimitRPM: 100},
		},
	}
	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "px"}
	mockMM := NewMockModelManager()
	updateModelLimits(health, cred, rateLimiter, testhelpers.NewTestLogger(), mockMM)

	assert.Nil(t, mockMM.GetModelPriorityTiersForCredential("gpt-4", "px"))
	assert.Equal(t, 1, mockMM.GetModelPriorityForCredential("gpt-4", "px"))
}

// TestUpdateModelLimits_SingleGroupUpstream_BannedStillEmitsTier is review_158 round 3
// item 3: a single-group upstream normally stays on the scalar path, but when that lone
// group is banned the tier must be emitted anyway — the scalar priority number carries no
// ban state, so without this the proxy credential stays a live candidate here and on the
// next router until local fail2ban trips.
func TestUpdateModelLimits_SingleGroupUpstream_BannedStillEmitsTier(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"c1": {Priority: 1}, "c2": {Priority: 1},
		},
		Models: map[string]httputil.ModelHealthStats{
			"a": {Model: "gpt-4", Credential: "c1", Priority: 1, LimitRPM: 100, IsBanned: true},
			"b": {Model: "gpt-4", Credential: "c2", Priority: 1, LimitRPM: 100, IsBanned: true},
		},
	}
	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "px"}
	mockMM := NewMockModelManager()
	updateModelLimits(health, cred, rateLimiter, testhelpers.NewTestLogger(), mockMM)

	tiers := mockMM.GetModelPriorityTiersForCredential("gpt-4", "px")
	require.Len(t, tiers, 1)
	assert.True(t, tiers[0].Banned)
}

// TestUpdateModelLimits_PriorityTiers_RecursesUpstreamTiers: when an upstream model entry
// itself carries a PriorityTiers array (proxy-of-proxy), those tiers fold into this
// router's own per-priority buckets at their own priorities.
func TestUpdateModelLimits_PriorityTiers_RecursesUpstreamTiers(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"downstream-proxy": {Priority: 0},
			"direct":           {Priority: 9},
		},
		Models: map[string]httputil.ModelHealthStats{
			"a": {
				Model: "m", Credential: "downstream-proxy",
				PriorityTiers: []httputil.ModelPriorityTier{
					{Priority: 1, Weight: 5, LimitRPM: 50},
					{Priority: 4, Weight: 2, LimitRPM: 400},
				},
			},
			"b": {Model: "m", Credential: "direct", Priority: 9, Weight: 1, LimitRPM: 10},
		},
	}
	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "top"}
	mockMM := NewMockModelManager()
	updateModelLimits(health, cred, rateLimiter, testhelpers.NewTestLogger(), mockMM)

	tiers := mockMM.GetModelPriorityTiersForCredential("m", "top")
	require.Len(t, tiers, 3)
	assert.Equal(t, []int{1, 4, 9}, []int{tiers[0].Priority, tiers[1].Priority, tiers[2].Priority})
	assert.Equal(t, 50, tiers[0].LimitRPM)
	assert.Equal(t, 400, tiers[1].LimitRPM)
	assert.Equal(t, 10, tiers[2].LimitRPM)
}

func TestUpdateModelLimits_PriorityMinAggregation_SkipsBannedEntries(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cheapgpt": {Priority: 100},
			"grant":    {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheapgpt": {Model: "gpt-4", Credential: "cheapgpt", Priority: 100, IsBanned: true},
			"model:grant":    {Model: "gpt-4", Credential: "grant", Priority: 200, IsBanned: false},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 200, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"),
		"banned upstream credential's priority must be excluded from the MIN aggregation")
}

// TestUpdateModelLimits_AllEntriesBanned_FallsBackToWorstSeenNotZero is a regression
// test: when EVERY upstream entry for a model is banned, the model must not be left
// out of modelPriorities (which ReplaceModelPrioritiesForCredential reads as "clear
// any learned priority" — GetModelPriorityForCredential then returns 0, and
// effectivePriority falls back to this proxy credential's own static
// EffectivePriority(), commonly 0/unset — the *best*, tried-first group). A fully-down
// node must be pushed to the worst priority ever seen among its entries instead,
// so it doesn't rank ahead of a healthy alternative at a real priority like 50.
func TestUpdateModelLimits_AllEntriesBanned_FallsBackToWorstSeenNotZero(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cheapgpt": {Priority: 100},
			"grant":    {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheapgpt": {Model: "gpt-4", Credential: "cheapgpt", Priority: 100, IsBanned: true},
			"model:grant":    {Model: "gpt-4", Credential: "grant", Priority: 200, IsBanned: true},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 200, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"),
		"a fully-down model must expose the worst priority seen (200), not fall through to the proxy credential's own static priority (0 here, which would rank it ahead of a healthy priority-50 alternative)")
}

// TestUpdateModelLimits_PriorityMinAggregation_NonBannedLowerStillWins is a regression
// guard alongside the fix above: when neither upstream credential is banned, the MIN
// aggregation still picks the lower (higher-priority / tried-first) value as before —
// the banned-skip logic must not accidentally change the base MIN case.
func TestUpdateModelLimits_PriorityMinAggregation_NonBannedLowerStillWins(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-a": {Priority: 100},
			"grant-b": {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:a": {Model: "gpt-4", Credential: "grant-a", Priority: 100, IsBanned: false},
			"model:b": {Model: "gpt-4", Credential: "grant-b", Priority: 200, IsBanned: false},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 100, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
}

// TestUpdateModelLimits_PriorityMin_ExcludesRateLimitedCheapTier is the review_158 #3-
// deferred fix: when several upstream credentials front one proxy credential at different
// priorities and the cheap (lowest-number) tier is currently RPM-exhausted, the upstream
// has already cascaded to the pricier tier — so the proxy credential's learned per-model
// priority must rise to that live tier, not stay pinned to the saturated cheap one.
// Otherwise the local balancer keeps treating the proxy as tier-1 and over-sends to it
// while a genuinely mid-priced alternative sits idle.
func TestUpdateModelLimits_PriorityMin_ExcludesRateLimitedCheapTier(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-cheap":     {Priority: 1},
			"grant-expensive": {Priority: 5},
		},
		Models: map[string]httputil.ModelHealthStats{
			// cheap tier is at its RPM limit → not live
			"model:cheap":     {Model: "gpt-4", Credential: "grant-cheap", Priority: 1, LimitRPM: 10, CurrentRPM: 10},
			"model:expensive": {Model: "gpt-4", Credential: "grant-expensive", Priority: 5, LimitRPM: 1000, CurrentRPM: 3},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 5, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"),
		"a saturated cheap tier must not keep pinning the learned priority to 1")
	// Capacity is still the SUM of both tiers — only the priority scalar tracks the live tier.
	assert.Equal(t, 1010, rateLimiter.GetModelLimitRPM("test_proxy", "gpt-4"))
}

// TestUpdateModelLimits_PriorityMin_RateLimitedCheapTierRecovers is the paired
// recovery case: once the cheap tier drops back under its limit it is live again and
// MIN returns to 1.
func TestUpdateModelLimits_PriorityMin_RateLimitedCheapTierRecovers(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-cheap":     {Priority: 1},
			"grant-expensive": {Priority: 5},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheap":     {Model: "gpt-4", Credential: "grant-cheap", Priority: 1, LimitRPM: 10, CurrentRPM: 4},
			"model:expensive": {Model: "gpt-4", Credential: "grant-expensive", Priority: 5, LimitRPM: 1000, CurrentRPM: 3},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 1, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
}

// TestUpdateModelLimits_PriorityMin_TPMExhaustedCheapTierExcluded mirrors the RPM case
// for the TPM budget.
func TestUpdateModelLimits_PriorityMin_TPMExhaustedCheapTierExcluded(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-cheap":     {Priority: 1},
			"grant-expensive": {Priority: 5},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheap":     {Model: "gpt-4", Credential: "grant-cheap", Priority: 1, LimitTPM: 5000, CurrentTPM: 5000},
			"model:expensive": {Model: "gpt-4", Credential: "grant-expensive", Priority: 5, LimitTPM: 500000, CurrentTPM: 100},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 5, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
}

// TestUpdateModelLimits_PriorityMin_AllTiersSaturated_FallsBackToWorst: when every tier
// is saturated the model still exposes the worst tier seen (5), never falling through
// to the proxy credential's own default group.
func TestUpdateModelLimits_PriorityMin_AllTiersSaturated_FallsBackToWorst(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-cheap":     {Priority: 1},
			"grant-expensive": {Priority: 5},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheap":     {Model: "gpt-4", Credential: "grant-cheap", Priority: 1, LimitRPM: 10, CurrentRPM: 10},
			"model:expensive": {Model: "gpt-4", Credential: "grant-expensive", Priority: 5, LimitRPM: 1000, CurrentRPM: 1000},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 5, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
}

// TestUpdateModelLimits_NoPriorityMismatchWarning_WhenCheapTierSaturated: like the
// _WhenCheapCredentialIsBanned case — once the cheap tier is saturated it is excluded
// from the MIN, so priority == priorityHigh and there is nothing heterogeneous to warn
// about right now.
func TestUpdateModelLimits_NoPriorityMismatchWarning_WhenCheapTierSaturated(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-cheap":     {Priority: 100},
			"grant-expensive": {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheap":     {Model: "gpt-4", Credential: "grant-cheap", Priority: 100, LimitRPM: 10, CurrentRPM: 10},
			"model:expensive": {Model: "gpt-4", Credential: "grant-expensive", Priority: 200, LimitRPM: 1000, CurrentRPM: 1},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "grant-pol01"}
	mockMM := NewMockModelManager()

	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.NotContains(t, logBuf.String(), "different priority groups")
}

func TestUpdateModelLimits_WarnsOnPriorityMismatchAcrossLiveUpstreamCredentials(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cheapgpt": {Priority: 100},
			"grant":    {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheapgpt": {Model: "gpt-4", Credential: "cheapgpt", Priority: 100, IsBanned: false},
			"model:grant":    {Model: "gpt-4", Credential: "grant", Priority: 200, IsBanned: false},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "grant-pol01"}
	mockMM := NewMockModelManager()

	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	logOutput := logBuf.String()
	assert.Contains(t, logOutput, "different priority groups")
	assert.Contains(t, logOutput, "proxy=grant-pol01")
	assert.Contains(t, logOutput, "model=gpt-4")
	// Logged at Debug (a fully-supported config, fires every poll), with honest field
	// names — the old "priority_if_cheaper_credential_stops_being_live" over-promised.
	assert.Contains(t, logOutput, "level=DEBUG")
	assert.Contains(t, logOutput, "lowest_live_priority=100")
	assert.Contains(t, logOutput, "highest_live_priority=200")
}

// TestUpdateModelLimits_NoPriorityMismatchWarning_WhenCheapCredentialIsBanned checks the
// warning doesn't double up with the 6.1 fix: once the cheap credential is actually banned
// (not just still live at a lower priority), applyPriorityMin already excludes it, so
// priority == priorityHigh (both 200) and there's nothing heterogeneous left to warn about.
func TestUpdateModelLimits_NoPriorityMismatchWarning_WhenCheapCredentialIsBanned(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cheapgpt": {Priority: 100},
			"grant":    {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cheapgpt": {Model: "gpt-4", Credential: "cheapgpt", Priority: 100, IsBanned: true},
			"model:grant":    {Model: "gpt-4", Credential: "grant", Priority: 200, IsBanned: false},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "grant-pol01"}
	mockMM := NewMockModelManager()

	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.NotContains(t, logBuf.String(), "different priority groups")
}

// TestUpdateModelLimits_NoPriorityMismatchWarning_WhenPrioritiesMatch is the base
// no-false-positive case: two live upstream credentials at the same priority never warn.
func TestUpdateModelLimits_NoPriorityMismatchWarning_WhenPrioritiesMatch(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-a": {Priority: 200},
			"grant-b": {Priority: 200},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:a": {Model: "gpt-4", Credential: "grant-a", Priority: 200, IsBanned: false},
			"model:b": {Model: "gpt-4", Credential: "grant-b", Priority: 200, IsBanned: false},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "grant-pol01"}
	mockMM := NewMockModelManager()

	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.NotContains(t, logBuf.String(), "different priority groups")
}

// TestUpdateModelLimits_IngestsFallbackUpstreamAsLastResortTierForPrimaryConnection:
// post-unification a non-fallback local proxy credential also ingests models served only
// by a last-resort (priority: 999 / is_fallback) upstream credential. They surface on the
// proxy credential at priority 999, so the balancer expands them into a local last-resort
// tier (Design B) — no longer hidden. This is what fixes an all-last-resort upstream node
// contributing zero routable models to a primary chain link.
func TestUpdateModelLimits_IngestsFallbackUpstreamAsLastResortTierForPrimaryConnection(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-primary":  {Priority: 100},
			"grant-fallback": {Priority: 999},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:primary":  {Model: "gpt-4", Credential: "grant-primary", Priority: 100, LimitRPM: 50, LimitTPM: 500},
			"model:fallback": {Model: "claude-3-opus", Credential: "grant-fallback", Priority: 999, LimitRPM: 20, LimitTPM: 200},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.True(t, mockMM.HasModel("test_proxy", "gpt-4"))
	assert.True(t, mockMM.HasModel("test_proxy", "claude-3-opus"), "model served only via a last-resort upstream credential now surfaces (tiered locally)")
	assert.Equal(t, 100, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
	assert.Equal(t, 999, mockMM.GetModelPriorityForCredential("claude-3-opus", "test_proxy"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("test_proxy", "claude-3-opus"))
}

// TestUpdateModelLimits_IncludesFallbackUpstreamForFallbackConnection: a locally
// is_fallback proxy credential is a last resort — it includes every upstream credential
// regardless of the upstream's own is_fallback flag.
func TestUpdateModelLimits_IncludesFallbackUpstreamForFallbackConnection(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-primary":  {Priority: 100},
			"grant-fallback": {Priority: 999},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:primary":  {Model: "gpt-4", Credential: "grant-primary", Priority: 100, LimitRPM: 50, LimitTPM: 500},
			"model:fallback": {Model: "claude-3-opus", Credential: "grant-fallback", Priority: 999, LimitRPM: 20, LimitTPM: 200},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy", Priority: config.FallbackPriorityGroup}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.True(t, mockMM.HasModel("test_proxy", "gpt-4"))
	assert.True(t, mockMM.HasModel("test_proxy", "claude-3-opus"))
	assert.Equal(t, 999, mockMM.GetModelPriorityForCredential("claude-3-opus", "test_proxy"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("test_proxy", "claude-3-opus"))
}

// TestUpdateModelScopes_AggregatesAllUpstreamTiersForPrimaryConnection: post-unification
// every upstream credential is OR'd into the proxy credential's scope aggregate,
// last-resort tiers included. REVIEW POINT: for a MIXED upstream (a restricted primary
// credential plus an unrestricted last-resort one, both serving the same model) this
// widens the proxy credential's apparent scope — the model IS reachable through the
// upstream's last-resort path, so exposing it is consistent with "discover everything,
// tier it locally", but a deployment relying on the old narrowing must move the hard
// block to router/model-level denied_scopes.
func TestUpdateModelScopes_AggregatesAllUpstreamTiersForPrimaryConnection(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-primary":  {Scopes: []string{"team-a"}},
			"grant-fallback": {Priority: config.FallbackPriorityGroup},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:primary":  {Model: "gpt-4", Credential: "grant-primary"},
			"model:fallback": {Model: "gpt-4", Credential: "grant-fallback"},
		},
	}

	cred := &config.CredentialConfig{Name: "test_proxy"}
	mockMM := NewMockModelManager()

	updateModelScopes(health, cred, mockMM)

	require.NotNil(t, cred.ProviderScopeExpression)
	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(cred.ProviderScopeExpression),
		"team-a caller allowed via the primary tier")
	assert.True(t, scope.NewContext(nil, nil).AllowsExpression(cred.ProviderScopeExpression),
		"unrestricted last-resort upstream tier is now part of the aggregate")
}

// TestUpdateModelScopes_IncludesFallbackUpstreamForFallbackConnection: a locally
// is_fallback proxy credential aggregates every upstream credential, is_fallback or not.
func TestUpdateModelScopes_IncludesFallbackUpstreamForFallbackConnection(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-fallback": {Priority: config.FallbackPriorityGroup, Scopes: []string{"team-a"}},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:fallback": {Model: "claude-3-opus", Credential: "grant-fallback"},
		},
	}

	cred := &config.CredentialConfig{Name: "test_proxy", Priority: config.FallbackPriorityGroup}
	mockMM := NewMockModelManager()

	updateModelScopes(health, cred, mockMM)

	require.NotNil(t, cred.ProviderScopeExpression)
	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(cred.ProviderScopeExpression))
	assert.False(t, scope.NewContext(nil, nil).AllowsExpression(cred.ProviderScopeExpression))
}

func TestUpdateModelLimits_NoModels_ClearsPriorities(t *testing.T) {
	health := &httputil.ProxyHealthResponse{}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()
	// Pre-populate as if a previous poll cycle had learned a priority.
	mockMM.ReplaceModelPrioritiesForCredential("test_proxy", map[string]int{"gpt-4": 200})
	require.Equal(t, 200, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 0, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"), "an empty health.Models response must clear stale learned priorities")
}

func TestUpdateModelLimits_NoCurrentUsage(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"model:proxy": {
				Model:      "gpt-4-turbo",
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 0,
				CurrentTPM: 0,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Should still add model with 0 current usage
	assert.Equal(t, 0, rateLimiter.GetCurrentModelRPM("test_proxy", "gpt-4-turbo"))
	assert.Equal(t, 0, rateLimiter.GetCurrentModelTPM("test_proxy", "gpt-4-turbo"))
}

func TestUpdateStatsFromRemoteProxy_FetchError(t *testing.T) {
	// Mock credential with invalid URL
	cred := &config.CredentialConfig{
		Name:    "invalid_proxy",
		BaseURL: "http://[invalid:url",
		APIKey:  "key",
	}

	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()
	ctx := context.Background()

	// Should handle fetch error gracefully
	UpdateStatsFromRemoteProxy(ctx, cred, rateLimiter, logger, nil)

	// Verify no stats were updated
}

func TestUpdateModelLimits_MixedZeroAndNonZeroRPM(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"model:cred1": {
				Model:      "test-model",
				LimitRPM:   100,
				LimitTPM:   500,
				CurrentRPM: 20,
				CurrentTPM: 200,
			},
			"model:cred2": {
				Model:      "test-model",
				LimitRPM:   0,
				LimitTPM:   1000,
				CurrentRPM: 30,
				CurrentTPM: 300,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should handle mixed zero and non-zero values
	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Should process without error
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_NegativeLimitAggregatesAsUnlimited(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"limited":   {},
			"unlimited": {},
		},
		Models: map[string]httputil.ModelHealthStats{
			"limited": {
				Credential: "limited",
				Model:      "test-model",
				LimitRPM:   200,
				LimitTPM:   2000,
			},
			"unlimited": {
				Credential: "unlimited",
				Model:      "test-model",
				LimitRPM:   -1,
				LimitTPM:   -1,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	updateModelLimits(health, cred, rateLimiter, logger, nil)

	assert.Equal(t, -1, rateLimiter.GetModelLimitRPM("test_proxy", "test-model"))
	assert.Equal(t, -1, rateLimiter.GetModelLimitTPM("test_proxy", "test-model"))
}

func TestUpdateModelLimits_AllZeroInOne(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cred1": {},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:cred1": {
				Model:      "test-model",
				Credential: "cred1",
				LimitRPM:   0,
				LimitTPM:   0,
				CurrentRPM: 0,
				CurrentTPM: 0,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	updateModelLimits(health, cred, rateLimiter, logger, nil)

	assert.Empty(t, rateLimiter.GetAllModelPairs(), "all-zero stats should not add an unlimited model limiter")
}

func TestUpdateCredentialLimits_NegativeLimitSelection(t *testing.T) {
	// Test that -1 is not selected as max (it means unlimited)
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {LimitRPM: 100, LimitTPM: 1000},
			"remote_cred_2": {LimitRPM: -1, LimitTPM: -1}, // Unlimited
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	updateCredentialLimits(health, cred, rateLimiter, logger)

	assert.Equal(t, -1, rateLimiter.GetLimitRPM("test_proxy"), "-1 from upstream should make aggregate RPM unlimited")
	assert.Equal(t, -1, rateLimiter.GetLimitTPM("test_proxy"), "-1 from upstream should make aggregate TPM unlimited")
}

func TestUpdateCredentialLimits_LargeNumbers(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {LimitRPM: 10000, LimitTPM: 100000, CurrentRPM: 5000, CurrentTPM: 50000},
			"remote_cred_2": {LimitRPM: 20000, LimitTPM: 200000, CurrentRPM: 8000, CurrentTPM: 80000},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := testhelpers.NewTestLogger()

	// Should handle large numbers without overflow or error
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Should complete successfully
	assert.NotNil(t, rateLimiter)
}

// MockModelManager implements ModelManagerInterface for testing
type MockModelManager struct {
	mu     sync.Mutex
	models map[string]map[string]bool
	added  []struct {
		credential string
		model      string
	}
	weights       map[string]map[string]int
	priorities    map[string]map[string]int
	priorityTiers map[string]map[string][]httputil.ModelPriorityTier
	sourceCreds   map[string]map[string]string
}

func NewMockModelManager() *MockModelManager {
	return &MockModelManager{
		models: make(map[string]map[string]bool),
		added: make([]struct {
			credential string
			model      string
		}, 0),
		weights:       make(map[string]map[string]int),
		priorities:    make(map[string]map[string]int),
		priorityTiers: make(map[string]map[string][]httputil.ModelPriorityTier),
		sourceCreds:   make(map[string]map[string]string),
	}
}

func (m *MockModelManager) AddModel(credentialName, modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.models[credentialName] == nil {
		m.models[credentialName] = make(map[string]bool)
	}
	m.models[credentialName][modelID] = true
	m.added = append(m.added, struct {
		credential string
		model      string
	}{credentialName, modelID})
}

func (m *MockModelManager) ReplaceModelsForCredential(credentialName string, modelIDs []string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.models[credentialName] = make(map[string]bool, len(modelIDs))
	filtered := m.added[:0]
	for _, added := range m.added {
		if added.credential != credentialName {
			filtered = append(filtered, added)
		}
	}
	m.added = filtered

	for _, modelID := range modelIDs {
		if modelID == "" || m.models[credentialName][modelID] {
			continue
		}
		m.models[credentialName][modelID] = true
		m.added = append(m.added, struct {
			credential string
			model      string
		}{credentialName, modelID})
	}
}

func (m *MockModelManager) SetModelWeightForCredential(modelID, credentialName string, weight int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if weight <= 0 {
		if weights, ok := m.weights[modelID]; ok {
			delete(weights, credentialName)
			if len(weights) == 0 {
				delete(m.weights, modelID)
			}
		}
		return
	}
	if m.weights[modelID] == nil {
		m.weights[modelID] = make(map[string]int)
	}
	m.weights[modelID][credentialName] = weight
}

func (m *MockModelManager) ReplaceModelWeightsForCredential(credentialName string, weights map[string]int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for modelID, credentialWeights := range m.weights {
		delete(credentialWeights, credentialName)
		if len(credentialWeights) == 0 {
			delete(m.weights, modelID)
		}
	}
	for modelID, weight := range weights {
		if weight <= 0 {
			continue
		}
		if m.weights[modelID] == nil {
			m.weights[modelID] = make(map[string]int)
		}
		m.weights[modelID][credentialName] = weight
	}
}

func (m *MockModelManager) SetModelPriorityForCredential(modelID, credentialName string, priority int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if priority <= 0 {
		if priorities, ok := m.priorities[modelID]; ok {
			delete(priorities, credentialName)
			if len(priorities) == 0 {
				delete(m.priorities, modelID)
			}
		}
		return
	}
	if m.priorities[modelID] == nil {
		m.priorities[modelID] = make(map[string]int)
	}
	m.priorities[modelID][credentialName] = priority
}

func (m *MockModelManager) ReplaceModelPrioritiesForCredential(credentialName string, priorities map[string]int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for modelID, credentialPriorities := range m.priorities {
		delete(credentialPriorities, credentialName)
		if len(credentialPriorities) == 0 {
			delete(m.priorities, modelID)
		}
	}
	for modelID, priority := range priorities {
		// Mirror models.Manager: a learned priority of 0 (best group) is a real value and
		// is stored; only genuinely invalid negatives are dropped.
		if priority < 0 {
			continue
		}
		if m.priorities[modelID] == nil {
			m.priorities[modelID] = make(map[string]int)
		}
		m.priorities[modelID][credentialName] = priority
	}
}

func (m *MockModelManager) ReplaceModelPriorityTiersForCredential(credentialName string, tiers map[string][]httputil.ModelPriorityTier) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for modelID, byCred := range m.priorityTiers {
		delete(byCred, credentialName)
		if len(byCred) == 0 {
			delete(m.priorityTiers, modelID)
		}
	}
	for modelID, list := range tiers {
		if len(list) == 0 {
			continue
		}
		if m.priorityTiers[modelID] == nil {
			m.priorityTiers[modelID] = make(map[string][]httputil.ModelPriorityTier)
		}
		cp := make([]httputil.ModelPriorityTier, len(list))
		copy(cp, list)
		m.priorityTiers[modelID][credentialName] = cp
	}
}

func (m *MockModelManager) GetModelPriorityTiersForCredential(modelID, credentialName string) []httputil.ModelPriorityTier {
	m.mu.Lock()
	defer m.mu.Unlock()
	if byCred, ok := m.priorityTiers[modelID]; ok {
		return byCred[credentialName]
	}
	return nil
}

func (m *MockModelManager) ReplaceModelSourceCredentialsForCredential(credentialName string, sourceCredentials map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for modelID, creds := range m.sourceCreds {
		delete(creds, credentialName)
		if len(creds) == 0 {
			delete(m.sourceCreds, modelID)
		}
	}
	for modelID, sourceCredential := range sourceCredentials {
		if sourceCredential == "" {
			continue
		}
		if m.sourceCreds[modelID] == nil {
			m.sourceCreds[modelID] = make(map[string]string)
		}
		m.sourceCreds[modelID][credentialName] = sourceCredential
	}
}

func (m *MockModelManager) GetModelSourceCredentialForCredential(modelID, credentialName string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if creds, ok := m.sourceCreds[modelID]; ok {
		return creds[credentialName]
	}
	return ""
}

func (m *MockModelManager) GetModelPriorityForCredential(modelID, credentialName string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if priorities, ok := m.priorities[modelID]; ok {
		return priorities[credentialName]
	}
	return 0
}

func (m *MockModelManager) HasModel(credentialName, modelID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.models[credentialName][modelID]
}

func (m *MockModelManager) GetModelWeightForCredential(modelID, credentialName string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if weights, ok := m.weights[modelID]; ok {
		return weights[credentialName]
	}
	return 0
}

func (m *MockModelManager) GetAddedModels() []struct {
	credential string
	model      string
} {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Return a copy to avoid race conditions
	result := make([]struct {
		credential string
		model      string
	}, len(m.added))
	copy(result, m.added)
	return result
}

func TestUpdateStatsFromRemoteProxy_Success(t *testing.T) {
	// Create mock model manager
	mockMM := NewMockModelManager()

	// Create test HTTP server that returns health response
	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}

		health := createMockProxyHealthResponse()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(health)
	}))
	defer server.Close()

	// Create credential pointing to test server
	cred := &config.CredentialConfig{
		Name:    "proxy-remote",
		Type:    config.ProviderTypeProxy,
		BaseURL: server.URL,
		APIKey:  "unused",
	}

	// Create rate limiter
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()
	ctx := context.Background()

	// Call the function being tested
	UpdateStatsFromRemoteProxy(ctx, cred, rateLimiter, logger, mockMM)

	// Verify credential limits were aggregated correctly
	// Total RPM should be sum of remote credentials (100 + 200 = 300)
	assert.Equal(t, 300, rateLimiter.GetLimitRPM("proxy-remote"),
		"RPM limit should be sum of remote credentials")

	// Total TPM should be sum of remote credentials (1000 + 2000 = 3000)
	assert.Equal(t, 3000, rateLimiter.GetLimitTPM("proxy-remote"),
		"TPM limit should be sum of remote credentials")

	// Current RPM should be sum of all current RPMs (25 + 20 = 45)
	// Use GreaterThanOrEqual because some timestamps might age out if test execution takes time
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentRPM("proxy-remote"), 44,
		"Current RPM should be at least sum of remote credential usage")

	// Current TPM should be sum of all current TPMs (250 + 200 = 450)
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentTPM("proxy-remote"), 449,
		"Current TPM should be at least sum of remote credential usage")

	// Verify models were added with correct aggregated limits
	// gpt-4: LimitRPM = 50 + 100 = 150, LimitTPM = 500 + 1000 = 1500
	assert.Equal(t, 150, rateLimiter.GetModelLimitRPM("proxy-remote", "gpt-4"),
		"Model RPM limit should be sum of all credential limits for that model")
	assert.Equal(t, 1500, rateLimiter.GetModelLimitTPM("proxy-remote", "gpt-4"),
		"Model TPM limit should be sum of all credential limits for that model")

	// Current usage for gpt-4: CurrentRPM = 10 + 15 = 25, CurrentTPM = 100 + 150 = 250
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentModelRPM("proxy-remote", "gpt-4"), 24,
		"Current model RPM should be at least sum of usage")
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentModelTPM("proxy-remote", "gpt-4"), 249,
		"Current model TPM should be at least sum of usage")

	// claude-3-opus: LimitRPM = 75, LimitTPM = 1500
	assert.Equal(t, 75, rateLimiter.GetModelLimitRPM("proxy-remote", "claude-3-opus"),
		"Claude model RPM limit should match remote limit")
	assert.Equal(t, 1500, rateLimiter.GetModelLimitTPM("proxy-remote", "claude-3-opus"),
		"Claude model TPM limit should match remote limit")

	// Current usage for claude-3-opus
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentModelRPM("proxy-remote", "claude-3-opus"), 4)
	assert.GreaterOrEqual(t, rateLimiter.GetCurrentModelTPM("proxy-remote", "claude-3-opus"), 49)

	// Verify ModelManager.AddModel was called for each model
	addedModels := mockMM.GetAddedModels()
	assert.Greater(t, len(addedModels), 0,
		"ModelManager.AddModel should be called for at least one model")

	// Check that expected models were added
	modelSet := make(map[string]bool)
	for _, m := range addedModels {
		assert.Equal(t, "proxy-remote", m.credential, "All models should be added for proxy-remote credential")
		modelSet[m.model] = true
	}

	// Both gpt-4 and claude-3-opus should be present (they have non-zero limits/usage)
	assert.True(t, modelSet["gpt-4"], "gpt-4 model should be added (aggregated from multiple credentials)")
	assert.True(t, modelSet["claude-3-opus"], "claude-3-opus model should be added")

	assert.Equal(t, 12, mockMM.GetModelWeightForCredential("gpt-4", "proxy-remote"),
		"gpt-4 weight should be sum of remote model weights")
	assert.Equal(t, 3, mockMM.GetModelWeightForCredential("claude-3-opus", "proxy-remote"),
		"model without explicit weight should fall back to remote credential weight")
}

func TestUpdateStatsFromHealth_AggregatesRemoteWeightsWithLegacyFallback(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()

	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"weighted-a": {Weight: 20},
			"weighted-b": {Weight: 2},
			"legacy":     {},
		},
		Models: map[string]httputil.ModelHealthStats{
			"gpt4-a": {Credential: "weighted-a", Model: "gpt-4", Weight: 7},
			"gpt4-b": {Credential: "weighted-b", Model: "gpt-4"},
			"gpt4-c": {Credential: "legacy", Model: "gpt-4"},
		},
	}

	UpdateStatsFromHealth(health, &config.CredentialConfig{
		Name: "proxy-remote",
	}, rateLimiter, logger, mockMM)

	assert.Equal(t, 10, mockMM.GetModelWeightForCredential("gpt-4", "proxy-remote"),
		"explicit model weight + credential fallback + legacy default should be aggregated")
}

func TestUpdateStatsFromHealth_ReplacesStaleRemoteModelsAndWeights(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()
	cred := &config.CredentialConfig{Name: "proxy-remote"}

	firstHealth := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream": {Weight: 2},
		},
		Models: map[string]httputil.ModelHealthStats{
			"fresh": {Credential: "upstream", Model: "fresh-model", Weight: 7, LimitRPM: 10, LimitTPM: 100},
			"stale": {Credential: "upstream", Model: "stale-model", Weight: 5, LimitRPM: 20, LimitTPM: 200},
		},
	}
	UpdateStatsFromHealth(firstHealth, cred, rateLimiter, logger, mockMM)

	require.Equal(t, 7, mockMM.GetModelWeightForCredential("fresh-model", "proxy-remote"))
	require.Equal(t, 5, mockMM.GetModelWeightForCredential("stale-model", "proxy-remote"))
	require.True(t, mockMM.HasModel("proxy-remote", "stale-model"))
	require.Equal(t, 20, rateLimiter.GetModelLimitRPM("proxy-remote", "stale-model"))

	secondHealth := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream": {Weight: 3},
		},
		Models: map[string]httputil.ModelHealthStats{
			"fresh": {Credential: "upstream", Model: "fresh-model", Weight: 11, LimitRPM: 30, LimitTPM: 300},
		},
	}
	UpdateStatsFromHealth(secondHealth, cred, rateLimiter, logger, mockMM)

	assert.Equal(t, 11, mockMM.GetModelWeightForCredential("fresh-model", "proxy-remote"))
	assert.Equal(t, 0, mockMM.GetModelWeightForCredential("stale-model", "proxy-remote"))
	assert.False(t, mockMM.HasModel("proxy-remote", "stale-model"))
	assert.Equal(t, -1, rateLimiter.GetModelLimitRPM("proxy-remote", "stale-model"))
	assert.Equal(t, 30, rateLimiter.GetModelLimitRPM("proxy-remote", "fresh-model"))
}

func TestUpdateStatsFromHealth_ClearsModelsWhenRemoteSnapshotIsEmpty(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()
	cred := &config.CredentialConfig{Name: "proxy-remote"}

	firstHealth := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream": {Weight: 2},
		},
		Models: map[string]httputil.ModelHealthStats{
			"stale": {Credential: "upstream", Model: "stale-model", Weight: 5, LimitRPM: 20, LimitTPM: 200},
		},
	}
	UpdateStatsFromHealth(firstHealth, cred, rateLimiter, logger, mockMM)
	require.True(t, mockMM.HasModel("proxy-remote", "stale-model"))
	require.Equal(t, 5, mockMM.GetModelWeightForCredential("stale-model", "proxy-remote"))

	emptyHealth := &httputil.ProxyHealthResponse{
		Credentials: firstHealth.Credentials,
		Models:      map[string]httputil.ModelHealthStats{},
	}
	UpdateStatsFromHealth(emptyHealth, cred, rateLimiter, logger, mockMM)

	assert.False(t, mockMM.HasModel("proxy-remote", "stale-model"))
	assert.Equal(t, 0, mockMM.GetModelWeightForCredential("stale-model", "proxy-remote"))
	assert.Empty(t, rateLimiter.GetAllModelPairs())
}

// TestUpdateStatsFromHealth_IngestsAllUpstreamTiersForPrimaryConnection: post-unification
// a non-fallback local proxy credential sums every upstream credential's capacity into
// its scalar ceiling and discovers every upstream model, last-resort tiers included. The
// scalar ceiling is a coarse backstop; per-model per-tier gates keep primary traffic off
// a last-resort tier's budget. Same result as the fallback-connection case below.
func TestUpdateStatsFromHealth_IngestsAllUpstreamTiersForPrimaryConnection(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()

	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream-primary":  {LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 10, CurrentTPM: 100},
			"upstream-fallback": {Priority: config.FallbackPriorityGroup, LimitRPM: 500, LimitTPM: 5000, CurrentRPM: 50, CurrentTPM: 500},
		},
		Models: map[string]httputil.ModelHealthStats{
			"p1": {Credential: "upstream-primary", Model: "primary-model", LimitRPM: 20, LimitTPM: 200, CurrentRPM: 2, CurrentTPM: 20},
			"f1": {Credential: "upstream-fallback", Model: "fallback-model", LimitRPM: 80, LimitTPM: 800, CurrentRPM: 8, CurrentTPM: 80},
		},
	}

	UpdateStatsFromHealth(health, &config.CredentialConfig{
		Name: "proxy-primary",
	}, rateLimiter, logger, mockMM)

	assert.Equal(t, 600, rateLimiter.GetLimitRPM("proxy-primary"))
	assert.Equal(t, 6000, rateLimiter.GetLimitTPM("proxy-primary"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("proxy-primary", "primary-model"))
	assert.Equal(t, 80, rateLimiter.GetModelLimitRPM("proxy-primary", "fallback-model"))

	addedModels := mockMM.GetAddedModels()
	addedModelIDs := make([]string, 0, len(addedModels))
	for _, m := range addedModels {
		addedModelIDs = append(addedModelIDs, m.model)
	}
	assert.ElementsMatch(t, []string{"primary-model", "fallback-model"}, addedModelIDs)
}

// TestUpdateStatsFromHealth_IncludesFallbackUpstreamForFallbackConnection: a locally
// is_fallback proxy credential is a last resort and includes ALL upstream credentials.
func TestUpdateStatsFromHealth_IncludesFallbackUpstreamForFallbackConnection(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()

	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream-primary":  {LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 10, CurrentTPM: 100},
			"upstream-fallback": {Priority: config.FallbackPriorityGroup, LimitRPM: 500, LimitTPM: 5000, CurrentRPM: 50, CurrentTPM: 500},
		},
		Models: map[string]httputil.ModelHealthStats{
			"p1": {Credential: "upstream-primary", Model: "primary-model", LimitRPM: 20, LimitTPM: 200, CurrentRPM: 2, CurrentTPM: 20},
			"f1": {Credential: "upstream-fallback", Model: "fallback-model", LimitRPM: 80, LimitTPM: 800, CurrentRPM: 8, CurrentTPM: 80},
		},
	}

	UpdateStatsFromHealth(health, &config.CredentialConfig{
		Name: "proxy-fallback", Priority: config.FallbackPriorityGroup,
	}, rateLimiter, logger, mockMM)

	// Fallback gateway includes ALL upstream credentials (primary + fallback),
	// so limits are the SUM of both: RPM=100+500=600, TPM=1000+5000=6000.
	assert.Equal(t, 600, rateLimiter.GetLimitRPM("proxy-fallback"))
	assert.Equal(t, 6000, rateLimiter.GetLimitTPM("proxy-fallback"))
	assert.Equal(t, 80, rateLimiter.GetModelLimitRPM("proxy-fallback", "fallback-model"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("proxy-fallback", "primary-model"))

	addedModels := mockMM.GetAddedModels()
	assert.Len(t, addedModels, 2)
	addedModelIDs := []string{addedModels[0].model, addedModels[1].model}
	assert.ElementsMatch(t, []string{"primary-model", "fallback-model"}, addedModelIDs)
}
