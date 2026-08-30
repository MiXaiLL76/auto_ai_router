package modelupdate

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestUpdateAllProxyCredentials_LegacyPathClearsFrozenSnapshot is the review_158 #22
// guard: once an upstream stops serving an AIR-shaped /health (a genuine non-AIR
// downgrade — 404 on /health, falling back to /v1/models), the poller must clear the
// proxy credential's frozen per-model tier / priority / weight snapshot instead of
// letting it (including Banned tiers and cumulative caps) live until process restart.
func TestUpdateAllProxyCredentials_LegacyPathClearsFrozenSnapshot(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	var airShaped atomic.Bool
	airShaped.Store(true)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			if !airShaped.Load() {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"upstream-primary": {Type: "openai"},
				},
				Models: map[string]httputil.ModelHealthStats{
					"m1": {Credential: "upstream-primary", Model: "gpt-4", Priority: 1},
				},
			})
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": "gpt-4"}},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	proxyCred := config.CredentialConfig{
		Name:    "proxy-a",
		Type:    config.ProviderTypeProxy,
		APIKey:  "sk-test",
		BaseURL: server.URL,
		RPM:     -1,
	}

	rl := ratelimit.New()
	f2b := fail2ban.New(3, 0, []int{500})
	bal := balancer.New([]config.CredentialConfig{proxyCred}, f2b, rl)
	mm := models.New(logger, 50, nil)
	mm.SetCredentials([]config.CredentialConfig{proxyCred})

	// syncFromHealth mirrors cmd/server's wiring: when an AIR-shaped /health was cached,
	// it owns this credential's dynamic snapshot. Here it stands in for
	// proxy.UpdateStatsFromHealth (which modelupdate cannot import) and writes exactly the
	// dynamic maps the real sync would.
	syncFromHealth := func(cred *config.CredentialConfig) bool {
		if mm.CachedRemoteHealth(cred.Name) == nil {
			return false
		}
		mm.ReplaceModelPriorityTiersForCredential(cred.Name, map[string][]httputil.ModelPriorityTier{
			"gpt-4": {
				{Priority: 1, Weight: 1, LimitRPM: 100, Banned: true},
				{Priority: 5, Weight: 1, LimitRPM: 500, Banned: true},
			},
		})
		mm.ReplaceModelPrioritiesForCredential(cred.Name, map[string]int{"gpt-4": 5})
		mm.ReplaceModelWeightsForCredential(cred.Name, map[string]int{"gpt-4": 7})
		return true
	}

	var mu sync.Mutex
	ctx := context.Background()

	UpdateAllProxyCredentials(ctx, bal, rl, logger, mm, &mu, syncFromHealth)
	require.Len(t, mm.GetModelPriorityTiersForCredential("gpt-4", "proxy-a"), 2, "AIR poll populates the tier snapshot")
	require.Equal(t, 7, mm.GetModelWeightForCredential("gpt-4", "proxy-a"))
	p, learned := mm.LearnedModelPriorityForCredential("gpt-4", "proxy-a")
	require.True(t, learned)
	require.Equal(t, 5, p)

	// Upstream downgrades to a non-AIR build.
	airShaped.Store(false)
	UpdateAllProxyCredentials(ctx, bal, rl, logger, mm, &mu, syncFromHealth)

	assert.Nil(t, mm.GetModelPriorityTiersForCredential("gpt-4", "proxy-a"), "legacy poll must clear the frozen tier snapshot")
	_, learned = mm.LearnedModelPriorityForCredential("gpt-4", "proxy-a")
	assert.False(t, learned, "legacy poll must clear the frozen scalar priority")
	assert.Equal(t, 0, mm.GetModelWeightForCredential("gpt-4", "proxy-a"), "legacy poll must clear the frozen weight")
}
