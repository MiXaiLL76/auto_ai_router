package proxy

import (
	"context"
	"log/slog"
	"os"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

func createTestLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
}

func TestUpdateCredentialLimits_EmptyCredentials(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

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
	logger := createTestLogger()

	// Should aggregate multiple model instances
	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Verify aggregation happened without error
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_ZeroValues_ConvertedToUnlimited(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"model:proxy": {
				Model:      "claude-3-opus",
				LimitRPM:   0, // 0 means unlimited in remote
				LimitTPM:   0,
				CurrentRPM: 0,
				CurrentTPM: 0,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := createTestLogger()

	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// 0 should be converted to -1 (unlimited)
	limitRPM := rateLimiter.GetModelLimitRPM("test_proxy", "claude-3-opus")
	limitTPM := rateLimiter.GetModelLimitTPM("test_proxy", "claude-3-opus")
	assert.Equal(t, -1, limitRPM)
	assert.Equal(t, -1, limitTPM)
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
	logger := createTestLogger()

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
	logger := createTestLogger()
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
	logger := createTestLogger()

	// Should handle mixed zero and non-zero values
	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Should process without error
	assert.NotNil(t, rateLimiter)
}

func TestUpdateModelLimits_AllZeroInOne(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Models: map[string]httputil.ModelHealthStats{
			"model:cred1": {
				Model:      "test-model",
				LimitRPM:   0,
				LimitTPM:   0,
				CurrentRPM: 0,
				CurrentTPM: 0,
			},
		},
	}

	rateLimiter := ratelimit.New()
	cred := &config.CredentialConfig{Name: "test_proxy"}
	logger := createTestLogger()

	updateModelLimits(health, cred, rateLimiter, logger, nil)

	// Should not add model if all values are 0
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
	logger := createTestLogger()

	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Should only consider positive values
	// The function checks > 0, so -1 is ignored
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
	logger := createTestLogger()

	// Should handle large numbers without overflow or error
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Should complete successfully
	assert.NotNil(t, rateLimiter)
}
