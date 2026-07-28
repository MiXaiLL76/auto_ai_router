package balancer

import (
	"fmt"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// drawN selects model count times and returns the pick sequence by credential name.
func drawN(t *testing.T, bal *RoundRobin, model string, count int) []string {
	t.Helper()
	seq := make([]string, 0, count)
	for i := 0; i < count; i++ {
		cred, err := bal.NextForModel(model)
		require.NoError(t, err, "pick %d", i)
		seq = append(seq, cred.Name)
	}
	return seq
}

func tally(seq []string) map[string]int {
	counts := make(map[string]int)
	for _, name := range seq {
		counts[name]++
	}
	return counts
}

func maxConsecutive(seq []string, name string) int {
	best, run := 0, 0
	for _, s := range seq {
		if s == name {
			run++
			if run > best {
				best = run
			}
		} else {
			run = 0
		}
	}
	return best
}

// SWRR over a full cycle of length sum(weights) selects each credential exactly its
// weight number of times, so the distribution is exactly proportional and deterministic.
func TestWeighted_ExactProportionalDistribution(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "ours", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 100},
		{Name: "azure", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	counts := tally(drawN(t, bal, "gpt-4o", 101*50))

	assert.Equal(t, 100*50, counts["ours"])
	assert.Equal(t, 1*50, counts["azure"])
}

// Explicitly setting weight 1 everywhere must reproduce the historical round-robin
// sequence (the default even-rotation path must stay unchanged).
func TestWeighted_EqualWeightsMatchRoundRobin(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 1},
		{Name: "cred2", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
		{Name: "cred3", APIKey: "k3", BaseURL: "http://3", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	seq := drawN(t, bal, "", 6)
	assert.Equal(t, []string{"cred1", "cred2", "cred3", "cred1", "cred2", "cred3"}, seq)
}

// With model filtering disabled the candidate set is identical for every model name, so all
// models must share a single SWRR cycle. Otherwise an authorized client could grow the state
// map without bound by sending endless unique model names.
func TestWeighted_DisabledModelCheckerSharesOneCycle(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "k1", BaseURL: "http://1", RPM: -1},
		{Name: "cred2", APIKey: "k2", BaseURL: "http://2", RPM: -1},
	}
	bal := New(credentials, f2b, rl) // no model checker → filtering disabled

	for i := 0; i < 1000; i++ {
		_, err := bal.NextForModel(fmt.Sprintf("ghost-model-%d", i))
		require.NoError(t, err)
	}

	assert.Len(t, bal.swrr, 1, "disabled model filtering must collapse all model names to one cycle")
}

// With model filtering enabled, distinct models have distinct candidate sets and therefore
// their own independent SWRR cycles.
func TestWeighted_EnabledModelCheckerKeysPerModel(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "k1", BaseURL: "http://1", RPM: -1},
		{Name: "cred2", APIKey: "k2", BaseURL: "http://2", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "model-a")
	mc.AddModel("cred2", "model-a")
	mc.AddModel("cred1", "model-b")
	mc.AddModel("cred2", "model-b")
	bal.SetModelChecker(mc)

	_, err := bal.NextForModel("model-a")
	require.NoError(t, err)
	_, err = bal.NextForModel("model-b")
	require.NoError(t, err)

	assert.Len(t, bal.swrr, 2, "each model with its own candidate set gets its own cycle")
}

// Unset weight (0) must behave identically to weight 1.
func TestWeighted_ZeroWeightDefaultsToOne(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "k1", BaseURL: "http://1", RPM: -1},
		{Name: "cred2", APIKey: "k2", BaseURL: "http://2", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	counts := tally(drawN(t, bal, "", 100))
	assert.Equal(t, 50, counts["cred1"])
	assert.Equal(t, 50, counts["cred2"])
}

// Weights are spread over time (smooth), not delivered in one long burst.
func TestWeighted_SmoothNoBurst(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "heavy", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 5},
		{Name: "light", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	seq := drawN(t, bal, "", 60)

	// A naive "100 then 1" scheme would run heavy 5 in a row each cycle; smooth WRR keeps
	// the heavy run at or below its weight and interleaves the light pick.
	assert.LessOrEqual(t, maxConsecutive(seq, "heavy"), 5)
	assert.Equal(t, 10, tally(seq)["light"])
}

// A model-level weight overrides the credential default, mirroring per-model RPM, and is
// scoped to that model only.
func TestWeighted_ModelOverride(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "ours", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 1},
		{Name: "azure", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("ours", "gpt-4o")
	mc.AddModel("azure", "gpt-4o")
	mc.AddModel("ours", "gpt-5-mini")
	mc.AddModel("azure", "gpt-5-mini")
	// We support gpt-4o → heavy weight to us; gpt-5-mini stays even (default RR to Azure).
	mc.SetModelWeight("gpt-4o", "ours", 99)
	bal.SetModelChecker(mc)

	supported := tally(drawN(t, bal, "gpt-4o", 100))
	assert.Equal(t, 99, supported["ours"])
	assert.Equal(t, 1, supported["azure"])

	unsupported := tally(drawN(t, bal, "gpt-5-mini", 100))
	assert.Equal(t, 50, unsupported["ours"])
	assert.Equal(t, 50, unsupported["azure"])
}

// When the heavy provider hits its own RPM we skip it and serve the next live provider,
// keeping the existing fail2ban/rate-limit failover semantics under weighting.
func TestWeighted_RPMSkipFallsThrough(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "ours", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 100},
		{Name: "azure", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("ours", "gpt-4o")
	mc.AddModel("azure", "gpt-4o")
	bal.SetModelChecker(mc)
	// Our model-level RPM caps us at 3/min regardless of the high weight.
	rl.AddModel("ours", "gpt-4o", 3)
	rl.AddModel("azure", "gpt-4o", 1000)

	counts := tally(drawN(t, bal, "gpt-4o", 50))

	assert.Equal(t, 3, counts["ours"], "ours must be capped by its RPM despite weight 100")
	assert.Equal(t, 47, counts["azure"], "overflow falls through to the next live provider")
}

func TestWeighted_ExcludingDoesNotPrunePrimaryCycle(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "k1", BaseURL: "http://1", RPM: -1},
		{Name: "cred2", APIKey: "k2", BaseURL: "http://2", RPM: -1},
		{Name: "cred3", APIKey: "k3", BaseURL: "http://3", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	_, err := bal.NextForModel("")
	require.NoError(t, err)

	primary := bal.swrr[schedKey{scopeKey: "cred1\x00cred2\x00cred3"}]
	require.NotNil(t, primary)
	require.Contains(t, primary.nodes, "cred2")

	_, err = bal.NextForModelExcluding("", map[string]bool{"cred2": true})
	require.NoError(t, err)

	assert.Contains(t, primary.nodes, "cred2", "retry exclusion must not prune the primary SWRR cycle")
	assert.Len(t, primary.nodes, 3, "primary SWRR cycle should still track all primary credentials")
}

func TestWeighted_NoBurstAfterRPMRecovery(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "heavy", APIKey: "k1", BaseURL: "http://1", RPM: 100, Weight: 10},
		{Name: "light", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	// Set remote-synced usage well above the 100 RPM limit. The limiter spreads
	// synced usage across the trailing minute, so exact-at-limit values can shed
	// a bucket on slow CI runners before the assertions run.
	rl.SetCredentialCurrentUsage("heavy", 1000, 0)

	during := tally(drawN(t, bal, "", 30))
	assert.Equal(t, 30, during["light"])
	assert.Zero(t, during["heavy"])

	rl.SetCredentialCurrentUsage("heavy", 0, 0)

	after := drawN(t, bal, "", 11)
	assert.GreaterOrEqual(t, tally(after)["light"], 1, "heavy burst after RPM recovery: %v", after)
}

// A banned high-weight provider must not accumulate weight while down, otherwise it would
// burst (serve a long uninterrupted run) the moment it recovers.
func TestWeighted_NoBurstAfterUnban(t *testing.T) {
	f2b := fail2ban.New(3, 50*time.Millisecond, []int{401})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "heavy", APIKey: "k1", BaseURL: "http://1", RPM: -1, Weight: 10},
		{Name: "light", APIKey: "k2", BaseURL: "http://2", RPM: -1, Weight: 1},
	}
	bal := New(credentials, f2b, rl)

	// Warm up so both have realistic running counters.
	drawN(t, bal, "gpt-4o", 22)

	// Ban heavy (3 × 401 reaches the threshold).
	for i := 0; i < 3; i++ {
		bal.RecordResponse("heavy", "gpt-4o", 401)
	}
	require.True(t, bal.IsBanned("heavy", "gpt-4o"))

	// While banned all traffic goes to light; heavy must not be accruing weight.
	during := tally(drawN(t, bal, "gpt-4o", 30))
	assert.Equal(t, 30, during["light"])
	assert.Zero(t, during["heavy"])

	// Wait out the ban.
	time.Sleep(70 * time.Millisecond)
	require.False(t, bal.IsBanned("heavy", "gpt-4o"))

	// After recovery, within one 11-pick cycle the light provider must still appear — proof
	// that heavy did not bank ~300 weight while banned and burst on return.
	after := drawN(t, bal, "gpt-4o", 11)
	assert.GreaterOrEqual(t, tally(after)["light"], 1, "heavy burst after unban: %v", after)
}
