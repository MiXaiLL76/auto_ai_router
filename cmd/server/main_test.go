package main

import (
	"fmt"
	"log/slog"
	"net"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/modelupdate"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSplitCredentialModel(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "standard format",
			input:    "openai_main:gpt-4o",
			expected: []string{"openai_main", "gpt-4o"},
		},
		{
			name:     "with multiple colons in model name",
			input:    "openai_main:gpt-4o:turbo",
			expected: []string{"openai_main", "gpt-4o:turbo"},
		},
		{
			name:     "simple names",
			input:    "cred1:model1",
			expected: []string{"cred1", "model1"},
		},
		{
			name:     "with dashes and underscores",
			input:    "openai_backup:gpt-3.5-turbo",
			expected: []string{"openai_backup", "gpt-3.5-turbo"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := modelupdate.SplitCredentialModel(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestInitializeModelManagerKeepsBackendRateLimitsBehindClientSurface(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{DefaultModelsRPM: -1},
		Fail2Ban: config.Fail2BanConfig{
			MaxAttempts: 3,
			BanDuration: time.Minute,
			ErrorCodes:  []int{429},
		},
		Credentials: []config.CredentialConfig{{
			Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1, TPM: -1,
		}},
		Models: []config.ModelRPMConfig{{
			Name: "backend-chat", Credential: "provider", RPM: 7, TPM: 70,
		}},
		ModelAlias:     map[string]string{"public/chat": "backend-chat"},
		ClientModelIDs: []string{"public/chat"},
	}
	logger := slog.New(slog.DiscardHandler)
	_, limiter, bal, _ := initializeBalancer(cfg, logger, nil, nil)
	manager := initializeModelManager(logger, cfg, limiter, bal)

	assert.Equal(t, 7, limiter.GetModelLimitRPM("provider", "backend-chat"))
	assert.Equal(t, 70, limiter.GetModelLimitTPM("provider", "backend-chat"))
	assert.Equal(t, -1, limiter.GetModelLimitRPM("provider", "public/chat"))
	models := manager.GetClientModels()
	if assert.Len(t, models.Data, 1) {
		assert.Equal(t, "public/chat", models.Data[0].ID)
	}
}

func TestInitializeModelManagerRegistersAcceptedModelAliases(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{DefaultModelsRPM: -1},
		Credentials: []config.CredentialConfig{{
			Name: "provider", Type: config.ProviderTypeOpenAI, RPM: -1, TPM: -1,
		}},
		Models: []config.ModelRPMConfig{{
			Name: "backend-chat", Credential: "provider", RPM: 7, TPM: 70,
		}},
		ModelAlias:     map[string]string{"public/chat": "backend-chat"},
		ClientModelIDs: []string{"public/chat"},
		AcceptedModelAlias: map[string]config.ClientModelAliasConfig{
			"legacy-chat": {Target: "public/chat"},
		},
	}
	logger := slog.New(slog.DiscardHandler)
	_, limiter, bal, _ := initializeBalancer(cfg, logger, nil, nil)
	manager := initializeModelManager(logger, cfg, limiter, bal)

	assert.True(t, manager.IsClientModelIDRoutable("legacy-chat"))
	canonical, alias, err := manager.ResolvePublicModelAlias("legacy-chat")
	require.NoError(t, err)
	assert.True(t, alias)
	assert.Equal(t, "public/chat", canonical)

	models := manager.GetClientModels()
	require.Len(t, models.Data, 1)
	assert.Equal(t, "public/chat", models.Data[0].ID)
}

// TestInitializeBalancerReturnsHybridBackendForCallerToClose is a regression
// test: initializeBalancer must NOT close the HybridBackend itself (a defer
// inside the function would fire when it returns, within microseconds of
// construction, killing the background writeWorker/syncWorker before the
// server ever serves a request — see the bug this signature change fixes).
// Instead it must hand the HybridBackend back so main() owns its lifetime.
func TestInitializeBalancerReturnsHybridBackendForCallerToClose(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	baseCfg := &config.Config{
		Server: config.ServerConfig{DefaultModelsRPM: -1},
	}

	t.Run("hybrid enabled with a redis backend", func(t *testing.T) {
		cfg := *baseCfg
		cfg.Redis = config.RedisConfig{Hybrid: true, SyncInterval: time.Minute}
		redisBackend := ratelimit.NewRedisBackendFromClient(nil, "test:")

		_, _, _, hybridBackend := initializeBalancer(&cfg, logger, redisBackend, nil)
		require.NotNil(t, hybridBackend, "expected a non-nil HybridBackend when Redis.Hybrid is set with a redisBackend")
		hybridBackend.Close() // must not already be closed; a double-close would hang/panic otherwise
	})

	t.Run("hybrid disabled", func(t *testing.T) {
		cfg := *baseCfg
		cfg.Redis = config.RedisConfig{Hybrid: false}
		redisBackend := ratelimit.NewRedisBackendFromClient(nil, "test:")

		_, _, _, hybridBackend := initializeBalancer(&cfg, logger, redisBackend, nil)
		assert.Nil(t, hybridBackend)
	})

	t.Run("no redis backend", func(t *testing.T) {
		cfg := *baseCfg
		cfg.Redis = config.RedisConfig{Hybrid: true, SyncInterval: time.Minute}

		_, _, _, hybridBackend := initializeBalancer(&cfg, logger, nil, nil)
		assert.Nil(t, hybridBackend)
	})
}

// TestConnectRedisWithRetry_BoundedByOverallDeadline is a regression test for
// the ~75s worst-case startup block found in review: connectRedisWithRetry
// must give up and fall back to nil well before its full
// (attempts x (connect+ping timeout)) + backoff-sum arithmetic would allow,
// so the HTTP listener/readiness probe is never held up for over a minute on
// a genuine (not just cold-start-blip) Redis outage.
func TestConnectRedisWithRetry_BoundedByOverallDeadline(t *testing.T) {
	// Bind then immediately close a local port: connecting to it afterward
	// fails fast with "connection refused" (not a timeout), so this test
	// exercises the retry/backoff/deadline bookkeeping, not raw dial latency.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())

	cfg := config.RedisConfig{
		InitAddresses:     []string{addr},
		ForceSingleClient: true,
		ConnectTimeout:    time.Second,
	}
	logger := slog.New(slog.DiscardHandler)

	start := time.Now()
	rb := connectRedisWithRetry(cfg, logger, monitoring.New(false))
	elapsed := time.Since(start)

	assert.Nil(t, rb, "expected fallback to nil against an unreachable address")
	assert.Less(t, elapsed, 20*time.Second,
		fmt.Sprintf("connectRedisWithRetry took %s — should be bounded well under the old ~75s worst case", elapsed))
}
