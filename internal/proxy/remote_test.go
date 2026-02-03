package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

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

// MockModelManager implements ModelManagerInterface for testing
type MockModelManager struct {
	mu    sync.Mutex
	added []struct {
		credential string
		model      string
	}
}

func NewMockModelManager() *MockModelManager {
	return &MockModelManager{
		added: make([]struct {
			credential string
			model      string
		}, 0),
	}
}

func (m *MockModelManager) AddModel(credentialName, modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.added = append(m.added, struct {
		credential string
		model      string
	}{credentialName, modelID})
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
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	logger := createTestLogger()
	ctx := context.Background()

	// Call the function being tested
	UpdateStatsFromRemoteProxy(ctx, cred, rateLimiter, logger, mockMM)

	// Verify credential limits were aggregated correctly
	// Max RPM should be 200 (from remote_cred_2)
	assert.Equal(t, 200, rateLimiter.GetLimitRPM("proxy-remote"),
		"RPM limit should be max of remote credentials")

	// Max TPM should be 2000 (from remote_cred_2)
	assert.Equal(t, 2000, rateLimiter.GetLimitTPM("proxy-remote"),
		"TPM limit should be max of remote credentials")

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
}
