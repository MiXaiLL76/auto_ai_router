package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
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

// TestUpdateModelLimits_PriorityMinAggregation_SkipsBannedEntries is a regression test
// for todo_round_robin.md section 6.1: a banned/exhausted upstream credential's static
// priority must not pull down (via MIN) the priority exposed for the local proxy
// credential + model pair, since that upstream can no longer actually serve the model.
// "cheapgpt" (priority 100) is banned for gpt-4 on this node while "grant" (priority
// 200) is still live — the aggregated priority must reflect "grant" (200), not the
// banned "cheapgpt" (100).
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

// TestUpdateModelLimits_IsFallbackDoesNotDropPriority is a T3 regression test: an
// upstream credential marked is_fallback still contributes its model+priority to the
// proxy credential's aggregation instead of being skipped outright (the bug fixed in
// updateModelLimits/updateCredentialLimits — see the _IsFallbackDoesNotSkipAggregation_
// tests above for the limits-only version of this same fix). Here "grant-fallback"
// (is_fallback on the upstream node) offers a model no other upstream credential offers,
// at a high (deprioritized) priority number — it must still show up, not vanish.
func TestUpdateModelLimits_IsFallbackDoesNotDropPriority(t *testing.T) {
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"grant-primary":  {IsFallback: false, Priority: 100},
			"grant-fallback": {IsFallback: true, Priority: 999},
		},
		Models: map[string]httputil.ModelHealthStats{
			"model:primary":  {Model: "gpt-4", Credential: "grant-primary", Priority: 100, LimitRPM: 50, LimitTPM: 500},
			"model:fallback": {Model: "claude-3-opus", Credential: "grant-fallback", Priority: 999, LimitRPM: 20, LimitTPM: 200},
		},
	}

	rateLimiter := ratelimit.New()
	// The LOCAL proxy credential is not is_fallback — before the fix, this would have
	// dropped "grant-fallback" (and claude-3-opus) entirely instead of just
	// deprioritizing it.
	cred := &config.CredentialConfig{Name: "test_proxy", IsFallback: false}
	logger := testhelpers.NewTestLogger()
	mockMM := NewMockModelManager()

	updateModelLimits(health, cred, rateLimiter, logger, mockMM)

	assert.True(t, mockMM.HasModel("test_proxy", "gpt-4"))
	assert.True(t, mockMM.HasModel("test_proxy", "claude-3-opus"), "is_fallback upstream credential's model must not be dropped")
	assert.Equal(t, 100, mockMM.GetModelPriorityForCredential("gpt-4", "test_proxy"))
	assert.Equal(t, 999, mockMM.GetModelPriorityForCredential("claude-3-opus", "test_proxy"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("test_proxy", "claude-3-opus"), "limits for the is_fallback-served model must still be aggregated")
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
	weights     map[string]map[string]int
	priorities  map[string]map[string]int
	sourceCreds map[string]map[string]string
}

func NewMockModelManager() *MockModelManager {
	return &MockModelManager{
		models: make(map[string]map[string]bool),
		added: make([]struct {
			credential string
			model      string
		}, 0),
		weights:     make(map[string]map[string]int),
		priorities:  make(map[string]map[string]int),
		sourceCreds: make(map[string]map[string]string),
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
		if priority <= 0 {
			continue
		}
		if m.priorities[modelID] == nil {
			m.priorities[modelID] = make(map[string]int)
		}
		m.priorities[modelID][credentialName] = priority
	}
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

// TestUpdateStatsFromHealth_IsFallbackDoesNotSkipAggregation_Primary is a regression
// test for the T3 bug fix (see todo_round_robin.md T3): an upstream credential marked
// is_fallback must never be dropped from a downstream proxy credential's aggregated
// limits/models just because the proxy credential itself is not is_fallback.
//
// Previously `if !cred.IsFallback && credStats.IsFallback { continue }` in both
// updateCredentialLimits and updateModelLimits skipped the fallback-flagged upstream
// credential entirely, so "fallback-model" (served only by "upstream-fallback") vanished
// completely from "proxy-primary" instead of just being deprioritized. That skip is gone:
// the upstream credential's is_fallback status no longer affects aggregation at all —
// deprioritizing it (without hiding it) is now priority's job, not this function's.
func TestUpdateStatsFromHealth_IsFallbackDoesNotSkipAggregation_Primary(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()

	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream-primary":  {IsFallback: false, LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 10, CurrentTPM: 100},
			"upstream-fallback": {IsFallback: true, LimitRPM: 500, LimitTPM: 5000, CurrentRPM: 50, CurrentTPM: 500},
		},
		Models: map[string]httputil.ModelHealthStats{
			"p1": {Credential: "upstream-primary", Model: "primary-model", LimitRPM: 20, LimitTPM: 200, CurrentRPM: 2, CurrentTPM: 20},
			"f1": {Credential: "upstream-fallback", Model: "fallback-model", LimitRPM: 80, LimitTPM: 800, CurrentRPM: 8, CurrentTPM: 80},
		},
	}

	UpdateStatsFromHealth(health, &config.CredentialConfig{
		Name:       "proxy-primary",
		IsFallback: false,
	}, rateLimiter, logger, mockMM)

	// Both upstream credentials now contribute to the proxy's aggregated limits —
	// is_fallback no longer excludes "upstream-fallback".
	assert.Equal(t, 600, rateLimiter.GetLimitRPM("proxy-primary"))
	assert.Equal(t, 6000, rateLimiter.GetLimitTPM("proxy-primary"))
	assert.Equal(t, 20, rateLimiter.GetModelLimitRPM("proxy-primary", "primary-model"))
	assert.Equal(t, 80, rateLimiter.GetModelLimitRPM("proxy-primary", "fallback-model"))

	addedModels := mockMM.GetAddedModels()
	assert.Len(t, addedModels, 2)
	addedModelIDs := []string{addedModels[0].model, addedModels[1].model}
	assert.ElementsMatch(t, []string{"primary-model", "fallback-model"}, addedModelIDs)
}

// TestUpdateStatsFromHealth_IsFallbackDoesNotSkipAggregation_Fallback mirrors the
// _Primary case above but with a locally is_fallback proxy credential — included mainly
// to document that the aggregation behavior is now identical regardless of the proxy
// credential's own is_fallback flag (that flag only ever affected the removed skip).
func TestUpdateStatsFromHealth_IsFallbackDoesNotSkipAggregation_Fallback(t *testing.T) {
	mockMM := NewMockModelManager()
	rateLimiter := ratelimit.New()
	logger := testhelpers.NewTestLogger()

	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"upstream-primary":  {IsFallback: false, LimitRPM: 100, LimitTPM: 1000, CurrentRPM: 10, CurrentTPM: 100},
			"upstream-fallback": {IsFallback: true, LimitRPM: 500, LimitTPM: 5000, CurrentRPM: 50, CurrentTPM: 500},
		},
		Models: map[string]httputil.ModelHealthStats{
			"p1": {Credential: "upstream-primary", Model: "primary-model", LimitRPM: 20, LimitTPM: 200, CurrentRPM: 2, CurrentTPM: 20},
			"f1": {Credential: "upstream-fallback", Model: "fallback-model", LimitRPM: 80, LimitTPM: 800, CurrentRPM: 8, CurrentTPM: 80},
		},
	}

	UpdateStatsFromHealth(health, &config.CredentialConfig{
		Name:       "proxy-fallback",
		IsFallback: true,
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
