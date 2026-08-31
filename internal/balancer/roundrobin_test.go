package balancer

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockModelChecker implements ModelChecker interface for testing
type MockModelChecker struct {
	enabled            bool
	credentialModels   map[string][]string       // credential -> models
	modelToCredentials map[string][]string       // model -> credentials
	modelWeights       map[string]map[string]int // model -> credential -> weight
	modelPriorities    map[string]map[string]int // model -> credential -> priority
	modelTiers         map[string]map[string][]httputil.ModelPriorityTier
}

func NewMockModelChecker(enabled bool) *MockModelChecker {
	return &MockModelChecker{
		enabled:            enabled,
		credentialModels:   make(map[string][]string),
		modelToCredentials: make(map[string][]string),
		modelWeights:       make(map[string]map[string]int),
		modelPriorities:    make(map[string]map[string]int),
		modelTiers:         make(map[string]map[string][]httputil.ModelPriorityTier),
	}
}

func (m *MockModelChecker) HasModel(credentialName, modelID string) bool {
	if !m.enabled {
		return true
	}
	models, ok := m.credentialModels[credentialName]
	if !ok {
		// If credentialModels are configured, unknown credentials don't have models
		// If credentialModels are empty, allow all (backward compatibility)
		return len(m.credentialModels) == 0
	}
	return slices.Contains(models, modelID)
}

func (m *MockModelChecker) GetCredentialsForModel(modelID string) []string {
	if !m.enabled {
		return nil
	}
	return m.modelToCredentials[modelID]
}

func (m *MockModelChecker) GetModelWeightForCredential(modelID, credentialName string) int {
	if creds, ok := m.modelWeights[modelID]; ok {
		return creds[credentialName]
	}
	return 0
}

func (m *MockModelChecker) SetModelWeight(modelID, credentialName string, weight int) {
	if m.modelWeights[modelID] == nil {
		m.modelWeights[modelID] = make(map[string]int)
	}
	m.modelWeights[modelID][credentialName] = weight
}

func (m *MockModelChecker) LearnedModelPriorityForCredential(modelID, credentialName string) (int, bool) {
	if creds, ok := m.modelPriorities[modelID]; ok {
		if p, ok := creds[credentialName]; ok {
			return p, true
		}
	}
	return 0, false
}

func (m *MockModelChecker) SetModelPriority(modelID, credentialName string, priority int) {
	if m.modelPriorities[modelID] == nil {
		m.modelPriorities[modelID] = make(map[string]int)
	}
	m.modelPriorities[modelID][credentialName] = priority
}

func (m *MockModelChecker) GetModelPriorityTiersForCredential(modelID, credentialName string) []httputil.ModelPriorityTier {
	if creds, ok := m.modelTiers[modelID]; ok {
		return creds[credentialName]
	}
	return nil
}

func (m *MockModelChecker) SetModelTiers(modelID, credentialName string, tiers []httputil.ModelPriorityTier) {
	if m.modelTiers[modelID] == nil {
		m.modelTiers[modelID] = make(map[string][]httputil.ModelPriorityTier)
	}
	m.modelTiers[modelID][credentialName] = tiers
}

func (m *MockModelChecker) IsEnabled() bool {
	return m.enabled
}

func (m *MockModelChecker) AddModel(credentialName, modelID string) {
	m.credentialModels[credentialName] = append(m.credentialModels[credentialName], modelID)
	m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], credentialName)
}

func TestNew(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 200},
	}

	bal := New(credentials, f2b, rl)

	assert.NotNil(t, bal)
	assert.Len(t, bal.credentials, 2)
	assert.Empty(t, bal.swrr)
}

func TestNextForModel_WithModelFilter(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Setup mock model checker
	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	mc.AddModel("cred1", "gpt-4o-mini")
	mc.AddModel("cred2", "gpt-4o-mini")
	mc.AddModel("cred3", "gpt-3.5-turbo")

	bal.SetModelChecker(mc)

	// Request for gpt-4o should only return cred1
	cred, err := bal.NextForModel("gpt-4o")
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred.Name)

	// Request for gpt-4o-mini can return cred1 or cred2 (second call should return cred2)
	cred, err = bal.NextForModel("gpt-4o-mini")
	require.NoError(t, err)
	assert.Contains(t, []string{"cred1", "cred2"}, cred.Name)
}

func TestNextForModel_NoModelSupport(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Setup mock model checker - no credentials have the requested model
	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	mc.AddModel("cred2", "gpt-4o-mini")

	bal.SetModelChecker(mc)

	// Request for unsupported model
	_, err := bal.NextForModel("unsupported-model")
	assert.Error(t, err)
	assert.Equal(t, ErrNoCredentialsAvailable, err)
}

func TestNextForModel_ModelRPMExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Set very low model RPM limit for each credential
	rl.AddModel("cred1", "gpt-4o", 1)
	rl.AddModel("cred2", "gpt-4o", 1)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	mc.AddModel("cred2", "gpt-4o")

	bal.SetModelChecker(mc)

	// First request should succeed (uses cred1)
	cred, err := bal.NextForModel("gpt-4o")
	require.NoError(t, err)
	assert.NotNil(t, cred)

	// Second request should succeed (uses cred2)
	cred, err = bal.NextForModel("gpt-4o")
	require.NoError(t, err)
	assert.NotNil(t, cred)

	// Third request should fail (both credentials exhausted their model RPM)
	_, err = bal.NextForModel("gpt-4o")
	assert.Error(t, err)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestSetModelChecker(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	assert.Nil(t, bal.modelChecker)

	mc := NewMockModelChecker(true)
	bal.SetModelChecker(mc)

	assert.NotNil(t, bal.modelChecker)
	assert.Equal(t, mc, bal.modelChecker)
}

func TestRecordResponse(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Record error responses
	bal.RecordResponse("cred1", "gpt-4", 401)
	bal.RecordResponse("cred1", "gpt-4", 401)
	bal.RecordResponse("cred1", "gpt-4", 401)

	// Credential should be banned
	assert.True(t, f2b.IsBanned("cred1", "gpt-4"))
}

func TestGetCredentialsSnapshot(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 200},
	}

	bal := New(credentials, f2b, rl)

	creds := bal.GetCredentialsSnapshot()
	assert.Len(t, creds, 2)
	assert.Equal(t, "cred1", creds[0].Name)
	assert.Equal(t, "cred2", creds[1].Name)
}

func TestGetAvailableCount(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Initially all available
	assert.Equal(t, 3, bal.GetAvailableCount())

	// Ban one credential
	f2b.RecordResponse("cred2", "gpt-4", 401)
	f2b.RecordResponse("cred2", "gpt-4", 401)
	f2b.RecordResponse("cred2", "gpt-4", 401)

	// Should have 2 available
	assert.Equal(t, 2, bal.GetAvailableCount())
}

func TestGetAvailableCount_ExcludesCredentialsWithoutProviderRoute(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	bal := New([]config.CredentialConfig{
		{Name: "available", RPM: 100},
		{Name: "no-route", RPM: 100, ProviderScopeExpression: scope.FalseExpression()},
	}, f2b, rl)

	assert.Equal(t, 1, bal.GetAvailableCount())
}

func TestGetBannedCount(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Initially 0 banned
	assert.Equal(t, 0, bal.GetBannedCount())

	// Ban one credential
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)
	f2b.RecordResponse("cred1", "gpt-4", 401)

	// Should have 1 banned
	assert.Equal(t, 1, bal.GetBannedCount())

	// Ban another
	f2b.RecordResponse("cred2", "gpt-4", 500)
	f2b.RecordResponse("cred2", "gpt-4", 500)
	f2b.RecordResponse("cred2", "gpt-4", 500)

	// Should have 2 banned
	assert.Equal(t, 2, bal.GetBannedCount())
}

func TestNext_ModelCheckerDisabled(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Setup mock model checker (disabled)
	mc := NewMockModelChecker(false)
	bal.SetModelChecker(mc)

	// Even with model specified, should work when disabled
	cred, err := bal.NextForModel("any-model")
	require.NoError(t, err)
	assert.NotNil(t, cred)
}

func TestRoundRobinCycling(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1000},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1000},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000},
	}

	bal := New(credentials, f2b, rl)

	// Request 6 times and verify round-robin order
	expectedOrder := []string{"cred1", "cred2", "cred3", "cred1", "cred2", "cred3"}
	for i, expectedName := range expectedOrder {
		cred, err := bal.NextForModel("")
		require.NoError(t, err, "Request %d failed", i+1)
		assert.Equal(t, expectedName, cred.Name, "Request %d: expected %s, got %s", i+1, expectedName, cred.Name)
	}
}

func TestNextForModel_SkipsFallback(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "primary1", Type: config.ProviderTypeOpenAI, IsFallback: false, RPM: 100, TPM: 10000},
		{Name: "fallback1", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "primary2", Type: config.ProviderTypeAnthropic, IsFallback: false, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// Should only return non-fallback credentials
	seen := make(map[string]bool)
	for range 4 {
		cred, err := bal.NextForModel("gpt-4o")
		require.NoError(t, err)
		assert.False(t, cred.IsFallback)
		seen[cred.Name] = true
	}

	assert.True(t, seen["primary1"])
	assert.True(t, seen["primary2"])
	assert.False(t, seen["fallback1"])
}

func TestNextForModel_BannedCredential(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Ban cred1
	bal.RecordResponse("cred1", "", 401)
	bal.RecordResponse("cred1", "", 401)
	bal.RecordResponse("cred1", "", 401)

	// Next should skip cred1 and return cred2
	cred, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)
}

func TestNextForModel_AllBanned(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Ban all credentials
	for range 3 {
		bal.RecordResponse("cred1", "", 401)
		bal.RecordResponse("cred2", "", 401)
	}

	// Next should return error
	_, err := bal.NextForModel("")
	assert.Error(t, err)
	assert.Equal(t, ErrNoCredentialsAvailable, err)
}

func TestNextForModel_CredentialRPMExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1},
	}

	bal := New(credentials, f2b, rl)

	// First request should succeed (uses cred1)
	cred, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred.Name)

	// Second request should succeed (uses cred2)
	cred, err = bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)

	// Third request should fail (both credentials exhausted their RPM)
	_, err = bal.NextForModel("")
	assert.Error(t, err)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestNextForModel_CredentialTPMExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100, TPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100, TPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Consume tokens to exceed TPM limit
	rl.ConsumeTokens("cred1", 100)
	rl.ConsumeTokens("cred2", 100)

	// Next request should fail (both credentials exhausted their TPM)
	_, err := bal.NextForModel("")
	assert.Error(t, err)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestNextForModel_ModelTPMExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Set model TPM limits and consume them
	rl.AddModelWithTPM("cred1", "gpt-4o", 100, 100)
	rl.AddModelWithTPM("cred2", "gpt-4o", 100, 100)
	rl.ConsumeModelTokens("cred1", "gpt-4o", 100)
	rl.ConsumeModelTokens("cred2", "gpt-4o", 100)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	mc.AddModel("cred2", "gpt-4o")
	bal.SetModelChecker(mc)

	// Next request should fail (both credentials exhausted their model TPM)
	_, err := bal.NextForModel("gpt-4o")
	assert.Error(t, err)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestNextForModel_EmptyModelID(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Setup mock model checker but request with empty modelID
	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	bal.SetModelChecker(mc)

	// Should work without model filtering
	cred, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.NotNil(t, cred)
}

func TestNextFallbackForModel_Success(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.NotNil(t, cred)
	assert.Equal(t, "proxy1", cred.Name)
	assert.True(t, cred.IsFallback)
}

func TestNextFallbackForModel_SkipsNonFallback(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: false, RPM: 100, TPM: 10000},
		{Name: "proxy2", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "proxy2", cred.Name)
	assert.True(t, cred.IsFallback)
}

func TestNextFallbackForModel_AllowsNonProxyTypes(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "openai1", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "openai1", cred.Name)
	assert.Equal(t, config.ProviderTypeOpenAI, cred.Type)
}

func TestNextFallbackForModel_NoFallbacksAvailable(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.Error(t, err)
	assert.Nil(t, cred)
	assert.Equal(t, ErrNoCredentialsAvailable, err)
}

func TestNextFallbackProxyForModel_SkipsNonProxyTypes(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "openai1", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackProxyForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)
	assert.Equal(t, config.ProviderTypeProxy, cred.Type)
}

func TestNextFallbackForModel_SkipsBannedFallback(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy2", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// Ban first proxy
	bal.RecordResponse("proxy1", "gpt-4o", 500)
	bal.RecordResponse("proxy1", "gpt-4o", 500)
	bal.RecordResponse("proxy1", "gpt-4o", 500)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "proxy2", cred.Name)
}

func TestNextFallbackForModel_RoundRobinFallbacks(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy2", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "openai1", Type: config.ProviderTypeOpenAI, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// First call should return proxy1
	cred, err := bal.NextFallbackForModel("gpt-4o")
	require.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)

	// Second call should return proxy2
	cred, err = bal.NextFallbackForModel("gpt-4o")
	require.NoError(t, err)
	assert.Equal(t, "proxy2", cred.Name)

	// Third call should return proxy1 again (round robin)
	cred, err = bal.NextFallbackForModel("gpt-4o")
	require.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)
}

func TestNextFallbackForModel_RPMLimitExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 1, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// First call succeeds
	cred, err := bal.NextFallbackForModel("gpt-4o")
	require.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)

	// Second call should fail with rate limit exceeded
	cred, err = bal.NextFallbackForModel("gpt-4o")
	assert.Error(t, err)
	assert.Nil(t, cred)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestNextFallbackForModel_TPMLimitExceeded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 100},
		{Name: "proxy2", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Consume TPM tokens to exceed the limit
	rl.ConsumeTokens("proxy1", 100)
	rl.ConsumeTokens("proxy2", 100)

	// Both proxies should be exhausted
	cred, err := bal.NextFallbackForModel("gpt-4o")
	assert.Error(t, err)
	assert.Nil(t, cred)
	assert.Equal(t, ErrRateLimitExceeded, err)
}

// TestNextForModel_MixedPrimaryFallback_RateLimit verifies that when primary credentials
// are TPM-exhausted and fallback credentials also are TPM-exhausted, ErrRateLimitExceeded
// is returned (not ErrNoCredentialsAvailable). This was a bug where filtering out fallback
// credentials set otherReasonsHit=true, masking the actual rate limit error.
func TestNextForModel_MixedPrimaryFallback_RateLimit(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "primary1", Type: config.ProviderTypeOpenAI, RPM: 100, TPM: 100},
		{Name: "fallback1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 100},
	}

	bal := New(credentials, f2b, rl)
	rl.ConsumeTokens("primary1", 100)

	// Primary is TPM-exhausted, but fallback exists → should get ErrRateLimitExceeded (not ErrNoCredentialsAvailable)
	_, err := bal.NextForModel("gpt-4o")
	assert.Equal(t, ErrRateLimitExceeded, err)

	// Fallback path: fallback TPM also exhausted → should still get ErrRateLimitExceeded
	rl.ConsumeTokens("fallback1", 100)
	_, err = bal.NextFallbackForModel("gpt-4o")
	assert.Equal(t, ErrRateLimitExceeded, err)
}

func TestNextFallbackForModel_WithModelChecker(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// Set model checker - should still return proxy (proxies ignore model checker)
	mc := NewMockModelChecker(true)
	mc.AddModel("proxy1", "gpt-4o")
	bal.SetModelChecker(mc)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)
}

func TestRoundRobin_GetCredentialsSnapshot_NoRace(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 200},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 300},
	}

	bal := New(credentials, f2b, rl)

	// Run concurrent reads and writes
	numReaders := 10
	numWriteOps := 100
	done := make(chan bool, numReaders)

	// Start multiple concurrent readers
	for range numReaders {
		go func() {
			for range 1000 {
				snap := bal.GetCredentialsSnapshot()
				assert.Len(t, snap, 3)
				// Verify snapshot is a copy (modifying it shouldn't affect balancer)
				if len(snap) > 0 {
					snap[0].APIKey = "modified"
				}
			}
			done <- true
		}()
	}

	// Start a writer that performs operations that acquire the lock
	go func() {
		for j := range numWriteOps {
			// These operations acquire locks internally
			bal.GetAvailableCount()
			bal.GetBannedCount()
			if j%3 == 0 {
				f2b.RecordResponse("cred1", "gpt-4", 401)
			}
		}
		done <- true
	}()

	// Wait for all goroutines
	for i := 0; i <= numReaders; i++ {
		<-done
	}

	// Verify the snapshot still returns unmodified data
	finalSnapshot := bal.GetCredentialsSnapshot()
	assert.Len(t, finalSnapshot, 3)
	assert.Equal(t, "key1", finalSnapshot[0].APIKey)
	assert.Equal(t, "key2", finalSnapshot[1].APIKey)
	assert.Equal(t, "key3", finalSnapshot[2].APIKey)
}

// Fallback Configuration Validation Tests

func TestValidateFallbackConfiguration_WithFallbacks(t *testing.T) {
	// Test that balancer correctly counts fallback credentials
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "proxy-a",
			Type:       config.ProviderTypeProxy,
			BaseURL:    "http://a.com",
			RPM:        100,
			IsFallback: false,
		},
		{
			Name:       "proxy-b",
			Type:       config.ProviderTypeProxy,
			BaseURL:    "http://b.com",
			RPM:        100,
			IsFallback: true,
		},
		{
			Name:       "proxy-c",
			Type:       config.ProviderTypeProxy,
			BaseURL:    "http://c.com",
			RPM:        100,
			IsFallback: true,
		},
	}

	// Should initialize successfully with fallback credentials
	bal := New(credentials, f2b, rl)
	require.NotNil(t, bal)
	assert.Equal(t, 3, len(bal.credentials))

	// Verify fallback count
	fallbackCount := 0
	for _, cred := range bal.credentials {
		if cred.IsFallback {
			fallbackCount++
		}
	}
	assert.Equal(t, 2, fallbackCount, "Should have 2 fallback credentials")
}

func TestValidateFallbackConfiguration_NoFallbacks(t *testing.T) {
	// Test configuration with no fallbacks (normal case)
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:    "proxy-a",
			Type:    config.ProviderTypeProxy,
			BaseURL: "http://a.com",
			RPM:     100,
		},
		{
			Name:    "proxy-b",
			Type:    config.ProviderTypeProxy,
			BaseURL: "http://b.com",
			RPM:     100,
		},
	}

	// Initialize balancer - no fallbacks to validate
	bal := New(credentials, f2b, rl)
	require.NotNil(t, bal)

	// No credentials should have IsFallback set
	for _, cred := range bal.credentials {
		assert.False(t, cred.IsFallback, "No credentials should be marked as fallback")
	}
}

func TestNextForModelExcluding_Basic(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
		{Name: "cred4", APIKey: "key4", BaseURL: "http://test4.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Exclude cred1 and cred3
	exclude := map[string]bool{"cred1": true, "cred3": true}

	// Should return cred2 first, then cred4
	cred, err := bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)

	cred, err = bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred4", cred.Name)

	// Should cycle back to cred2
	cred, err = bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)
}

func TestNextForModelExcluding_AllExcluded(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Exclude all credentials
	exclude := map[string]bool{"cred1": true, "cred2": true}

	_, err := bal.NextForModelExcluding("", exclude)
	assert.Error(t, err)
	assert.Equal(t, ErrNoCredentialsAvailable, err)
}

func TestNextForModelExcluding_RoundRobin(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://test1.com", RPM: 1000},
		{Name: "cred2", Type: config.ProviderTypeVertexAI, APIKey: "key2", RPM: 1000, ProjectID: "p", Location: "l"},
		{Name: "cred3", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000},
		{Name: "cred4", Type: config.ProviderTypeVertexAI, APIKey: "key4", RPM: 1000, ProjectID: "p", Location: "l"},
	}

	bal := New(credentials, f2b, rl)

	// First call without exclude gets cred1
	cred, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred.Name)

	// Now exclude cred1 (already tried), should get cred2
	exclude := map[string]bool{"cred1": true}
	cred, err = bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)

	// Exclude cred1 and cred2, should get cred3
	exclude["cred2"] = true
	cred, err = bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred3", cred.Name)
}

func TestNextForModelExcluding_SkipsFallback(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "fallback1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Exclude cred1, should skip fallback1 and return cred2
	exclude := map[string]bool{"cred1": true}
	cred, err := bal.NextForModelExcluding("", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)
}

func TestNextRetryForModelExcluding_FallbackPriorityOrder(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cheapgpt", Type: config.ProviderTypeAnthropic, APIKey: "key1", BaseURL: "http://cheapgpt.com", RPM: 100, FallbackPriority: 10},
		{Name: "grant", Type: config.ProviderTypeBedrock, APIKey: "key2", BaseURL: "http://grant.com", RPM: 100, FallbackPriority: 30, Weight: 100},
		{Name: "cometapi", Type: config.ProviderTypeAnthropic, APIKey: "key3", BaseURL: "http://cometapi.com", RPM: 100, FallbackPriority: 20},
	}

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("cheapgpt", "claude")
	mc.AddModel("cometapi", "claude")
	mc.AddModel("grant", "claude")
	bal.SetModelChecker(mc)

	tried := map[string]bool{"cheapgpt": true}
	cred, err := bal.NextRetryForModelExcluding("claude", &credentials[0], tried)
	require.NoError(t, err)
	assert.Equal(t, "cometapi", cred.Name)

	tried["cometapi"] = true
	cred, err = bal.NextRetryForModelExcluding("claude", cred, tried)
	require.NoError(t, err)
	assert.Equal(t, "grant", cred.Name)
}

func TestNextRetryForModelExcluding_FallbackPriorityFallsBackToUnprioritized(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cheapgpt", Type: config.ProviderTypeAnthropic, APIKey: "key1", BaseURL: "http://cheapgpt.com", RPM: 100, FallbackPriority: 10},
		{Name: "cometapi", Type: config.ProviderTypeCometAPI, APIKey: "key2", BaseURL: "http://cometapi.com", RPM: 100, FallbackPriority: 20},
		{Name: "cloudru", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://cloudru.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	tried := map[string]bool{"cheapgpt": true}
	cred, err := bal.NextRetryForModelExcluding("", &credentials[0], tried)
	require.NoError(t, err)
	assert.Equal(t, "cometapi", cred.Name)

	tried["cometapi"] = true
	cred, err = bal.NextRetryForModelExcluding("", cred, tried)
	require.NoError(t, err)
	assert.Equal(t, "cloudru", cred.Name)
}

func TestNextRetryForModelExcluding_UnprioritizedTailContinuesAcrossTypes(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cheapgpt", Type: config.ProviderTypeAnthropic, APIKey: "key1", BaseURL: "http://cheapgpt.com", RPM: 100, FallbackPriority: 10},
		{Name: "cometapi", Type: config.ProviderTypeCometAPI, APIKey: "key2", BaseURL: "http://cometapi.com", RPM: 100, FallbackPriority: 20},
		{Name: "cloudru", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://cloudru.com", RPM: 100},
		{Name: "yandex", Type: config.ProviderTypeAnthropic, APIKey: "key4", BaseURL: "http://yandex.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	tried := map[string]bool{"cheapgpt": true, "cometapi": true}
	cred, err := bal.NextRetryForModelExcluding("", &credentials[1], tried)
	require.NoError(t, err)
	assert.Equal(t, "cloudru", cred.Name)

	tried["cloudru"] = true
	cred, err = bal.NextRetryForModelExcluding("", cred, tried)
	require.NoError(t, err)
	assert.Equal(t, "yandex", cred.Name)
}

func TestNextRetryForModelExcluding_DefaultsToSameType(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "openai-a", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://a.com", RPM: 100},
		{Name: "anthropic-a", Type: config.ProviderTypeAnthropic, APIKey: "key2", BaseURL: "http://b.com", RPM: 100},
		{Name: "openai-b", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://c.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextRetryForModelExcluding("", &credentials[0], map[string]bool{"openai-a": true})
	require.NoError(t, err)
	assert.Equal(t, "openai-b", cred.Name)
}

func TestNextRetryForModelExcluding_FallbackPrioritySkipsFallbackCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "primary-a", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://a.com", RPM: 100, FallbackPriority: 10},
		{Name: "reserve", Type: config.ProviderTypeAnthropic, APIKey: "key2", BaseURL: "http://b.com", RPM: 100, IsFallback: true, FallbackPriority: 20},
		{Name: "primary-b", Type: config.ProviderTypeBedrock, APIKey: "key3", BaseURL: "http://c.com", RPM: 100, FallbackPriority: 30},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextRetryForModelExcluding("", &credentials[0], map[string]bool{"primary-a": true})
	require.NoError(t, err)
	assert.Equal(t, "primary-b", cred.Name)
}

func TestNextRetryForModelExcluding_UsesSeparateSWRRStatePerPriority(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "primary-a", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://a.com", RPM: 100, FallbackPriority: 10},
		{Name: "tier-20-a", Type: config.ProviderTypeOpenAI, APIKey: "key2", BaseURL: "http://b.com", RPM: 100, FallbackPriority: 20, Weight: 2},
		{Name: "tier-20-b", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://c.com", RPM: 100, FallbackPriority: 20},
		{Name: "tier-30-a", Type: config.ProviderTypeBedrock, APIKey: "key4", BaseURL: "http://d.com", RPM: 100, FallbackPriority: 30, Weight: 3},
		{Name: "tier-30-b", Type: config.ProviderTypeBedrock, APIKey: "key5", BaseURL: "http://e.com", RPM: 100, FallbackPriority: 30},
	}

	bal := New(credentials, f2b, rl)

	_, err := bal.NextRetryForModelExcluding("", &credentials[0], map[string]bool{"primary-a": true})
	require.NoError(t, err)
	_, err = bal.NextRetryForModelExcluding("", &credentials[0], map[string]bool{
		"primary-a": true,
		"tier-20-a": true,
		"tier-20-b": true,
	})
	require.NoError(t, err)

	priorities := make(map[int]bool)
	for key := range bal.swrr {
		if key.priority > 0 {
			priorities[key.priority] = true
		}
	}
	assert.True(t, priorities[20])
	assert.True(t, priorities[30])
}

func TestNextForModelExcluding_WithModelChecker(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeVertexAI, APIKey: "key1", RPM: 1000, ProjectID: "p", Location: "l"},
		{Name: "cred2", Type: config.ProviderTypeVertexAI, APIKey: "key2", RPM: 1000, ProjectID: "p", Location: "l"},
		{Name: "cred3", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000},
	}

	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gemini-2.5-flash")
	mc.AddModel("cred2", "gemini-2.5-flash")
	// cred3 does NOT support gemini-2.5-flash
	bal.SetModelChecker(mc)

	// Exclude cred1 for gemini-2.5-flash model
	exclude := map[string]bool{"cred1": true}
	cred, err := bal.NextForModelExcluding("gemini-2.5-flash", exclude)
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name, "Should return cred2 since cred1 is excluded and cred3 doesn't support the model")

	// Exclude both vertex creds
	exclude["cred2"] = true
	_, err = bal.NextForModelExcluding("gemini-2.5-flash", exclude)
	assert.Error(t, err, "Should error when all model-supporting creds are excluded")
}

func TestRoundRobin_MultipleCredentialsSameModel(t *testing.T) {
	// Test case: 4 credentials with same model should be cycled properly
	// This reproduces the issue where storied-port-482316-u0 was being used 90% of the time
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "matvey-01", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://test1.com", RPM: -1},
		{Name: "matvey-02", Type: config.ProviderTypeOpenAI, APIKey: "key2", BaseURL: "http://test2.com", RPM: -1},
		{Name: "matvey-03", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://test3.com", RPM: -1},
		{Name: "storied-port-482316-u0", Type: config.ProviderTypeVertexAI, APIKey: "key4", BaseURL: "http://vertex1.com", RPM: -1},
		{Name: "sunlit-flag-482317-i9", Type: config.ProviderTypeVertexAI, APIKey: "key5", BaseURL: "http://vertex2.com", RPM: -1},
		{Name: "aqueous-heading-482215-q8", Type: config.ProviderTypeVertexAI, APIKey: "key6", BaseURL: "http://vertex3.com", RPM: -1},
		{Name: "spatial-shore-482315-p6", Type: config.ProviderTypeVertexAI, APIKey: "key7", BaseURL: "http://vertex4.com", RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	// Setup model checker: only the 4 vertex credentials support gemini-2.5-flash
	mc := NewMockModelChecker(true)
	mc.AddModel("storied-port-482316-u0", "gemini-2.5-flash")
	mc.AddModel("sunlit-flag-482317-i9", "gemini-2.5-flash")
	mc.AddModel("aqueous-heading-482215-q8", "gemini-2.5-flash")
	mc.AddModel("spatial-shore-482315-p6", "gemini-2.5-flash")

	bal.SetModelChecker(mc)

	// Request for gemini-2.5-flash multiple times
	// Should cycle through all 4 vertex credentials
	requests := make([]string, 8)
	expectedCycle := []string{
		"storied-port-482316-u0",
		"sunlit-flag-482317-i9",
		"aqueous-heading-482215-q8",
		"spatial-shore-482315-p6",
		"storied-port-482316-u0", // cycle repeats
		"sunlit-flag-482317-i9",
		"aqueous-heading-482215-q8",
		"spatial-shore-482315-p6",
	}

	for i := range 8 {
		cred, err := bal.NextForModel("gemini-2.5-flash")
		require.NoError(t, err)
		requests[i] = cred.Name
	}

	// Verify the cycle
	for i, expected := range expectedCycle {
		assert.Equal(t, expected, requests[i], "Request %d should get %s, got %s", i+1, expected, requests[i])
	}

	// Verify distribution: each vertex credential should appear at least once in first 4 requests
	firstFourCreds := make(map[string]int)
	for i := range 4 {
		firstFourCreds[requests[i]]++
	}
	assert.Equal(t, 4, len(firstFourCreds), "First 4 requests should use 4 different credentials")
	for cred, count := range firstFourCreds {
		assert.Equal(t, 1, count, "Each credential in first 4 requests should appear exactly once, %s appeared %d times", cred, count)
	}
}

// TestNextSameTypeForModelExcluding_ProxyDoesNotReturnVertexAI reproduces the bug where
// same-type proxy retry could return a vertex-ai credential with empty BaseURL, causing
// "unsupported protocol scheme" errors.
func TestNextSameTypeForModelExcluding_ProxyDoesNotReturnVertexAI(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, BaseURL: "http://proxy1.com", RPM: -1},
		{Name: "proxy2", Type: config.ProviderTypeProxy, BaseURL: "http://proxy2.com", RPM: -1},
		{Name: "vertex1", Type: config.ProviderTypeVertexAI, RPM: -1},
		{Name: "vertex2", Type: config.ProviderTypeVertexAI, RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	// Simulate: proxy1 tried, retry should return proxy2 (not vertex-ai)
	exclude := map[string]bool{"proxy1": true}
	cred, err := bal.NextSameTypeForModelExcluding("gemini-model", config.ProviderTypeProxy, exclude)
	require.NoError(t, err)
	assert.Equal(t, config.ProviderTypeProxy, cred.Type, "Same-type retry must return proxy type, not vertex-ai")
	assert.Equal(t, "proxy2", cred.Name)
}

func TestNextSameTypeForModelExcluding_AIRDoesNotReturnGenericProxy(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "air1", Type: config.ProviderTypeAIR, BaseURL: "http://air1.com", RPM: -1},
		{Name: "air2", Type: config.ProviderTypeAIR, BaseURL: "http://air2.com", RPM: -1},
		{Name: "proxy1", Type: config.ProviderTypeProxy, BaseURL: "http://proxy1.com", RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	exclude := map[string]bool{"air1": true}
	cred, err := bal.NextSameTypeForModelExcluding("gpt-cache", config.ProviderTypeAIR, exclude)
	require.NoError(t, err)
	assert.Equal(t, "air2", cred.Name)

	_, err = bal.NextSameTypeForModelExcluding("gpt-cache", config.ProviderTypeAIR, map[string]bool{
		"air1": true,
		"air2": true,
	})
	require.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

func TestNextRetryForModelExcluding_ProxyLikeRetriesStayExactType(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "air1", Type: config.ProviderTypeAIR, BaseURL: "http://air1.com", RPM: -1},
		{Name: "proxy1", Type: config.ProviderTypeProxy, BaseURL: "http://proxy1.com", RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	_, err := bal.NextRetryForModelExcluding("gpt-cache", &credentials[0], map[string]bool{"air1": true})
	require.ErrorIs(t, err, ErrNoCredentialsAvailable)

	_, err = bal.NextRetryForModelExcluding("gpt-cache", &credentials[1], map[string]bool{"proxy1": true})
	require.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

// TestNextSameTypeForModelExcluding_VertexAIDoesNotReturnProxy verifies that same-type
// retry for vertex-ai credentials does not return proxy credentials.
func TestNextSameTypeForModelExcluding_VertexAIDoesNotReturnProxy(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, BaseURL: "http://proxy1.com", RPM: -1},
		{Name: "vertex1", Type: config.ProviderTypeVertexAI, RPM: -1},
		{Name: "vertex2", Type: config.ProviderTypeVertexAI, RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	// Simulate: vertex1 tried, retry should return vertex2 (not proxy)
	exclude := map[string]bool{"vertex1": true}
	cred, err := bal.NextSameTypeForModelExcluding("gemini-model", config.ProviderTypeVertexAI, exclude)
	require.NoError(t, err)
	assert.Equal(t, config.ProviderTypeVertexAI, cred.Type, "Same-type retry must return vertex-ai type, not proxy")
	assert.Equal(t, "vertex2", cred.Name)
}

// TestNextSameTypeForModelExcluding_NoSameTypeAvailable verifies proper error
// when no same-type credentials are available.
func TestNextSameTypeForModelExcluding_NoSameTypeAvailable(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, BaseURL: "http://proxy1.com", RPM: -1},
		{Name: "vertex1", Type: config.ProviderTypeVertexAI, RPM: -1},
	}

	bal := New(credentials, f2b, rl)

	// All proxy credentials excluded
	exclude := map[string]bool{"proxy1": true}
	_, err := bal.NextSameTypeForModelExcluding("", config.ProviderTypeProxy, exclude)
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

func TestNextSpecific_ReturnsNamedCredentialWithoutAdvancingRoundRobin(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	credentials := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, RPM: 100},
		{Name: "cred2", Type: config.ProviderTypeOpenAI, RPM: 100},
		{Name: "cred3", Type: config.ProviderTypeOpenAI, RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextSpecific("cred2", "")
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred.Name)

	next, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "cred1", next.Name, "NextSpecific must not advance round-robin state")
}

func TestNextSpecific_NotFound(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	bal := New([]config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, RPM: 100},
	}, f2b, rl)

	_, err := bal.NextSpecific("missing", "")
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

func TestNextSpecific_RespectsModelCheckerBanAndRateLimit(t *testing.T) {
	f2b := fail2ban.New(1, time.Minute, []int{429})
	rl := ratelimit.New()
	credentials := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, RPM: 1},
	}
	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "model-a")
	bal.SetModelChecker(mc)

	_, err := bal.NextSpecific("cred1", "model-b")
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)

	cred, err := bal.NextSpecific("cred1", "model-a")
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred.Name)

	_, err = bal.NextSpecific("cred1", "model-a")
	assert.ErrorIs(t, err, ErrRateLimitExceeded)

	f2b.RecordResponse("cred1", "model-a", 429)
	_, err = bal.NextSpecific("cred1", "model-a")
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

func TestNextForModelScoped_FiltersCredentialScopes(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()
	credentials := []config.CredentialConfig{
		{Name: "shared", APIKey: "key1", BaseURL: "http://shared.com", RPM: -1},
		{Name: "team-a", APIKey: "key2", BaseURL: "http://team-a.example", RPM: -1, Scopes: []string{"team-a"}},
		{Name: "team-b", APIKey: "key3", BaseURL: "http://team-b.example", RPM: -1, Scopes: []string{"team-b"}},
		{Name: "blocked", APIKey: "key4", BaseURL: "http://blocked.com", RPM: -1, Scopes: []string{"team-a"}, DeniedScopes: []string{"team-a"}},
	}
	bal := New(credentials, f2b, rl)
	teamAScope := scope.NewContext([]string{"team-a"}, nil)

	for range 8 {
		cred, err := bal.NextForModelScoped("", teamAScope)
		require.NoError(t, err)
		assert.Contains(t, []string{"shared", "team-a"}, cred.Name)
	}

	_, err := bal.NextSpecificScoped("team-b", "", teamAScope)
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
	_, err = bal.NextSpecificScoped("blocked", "", teamAScope)
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

func TestNextForModelScoped_SharesCycleForSameVisiblePool(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401})
	rl := ratelimit.New()
	credentials := []config.CredentialConfig{
		{Name: "shared-a", APIKey: "key1", BaseURL: "http://a.com", RPM: -1},
		{Name: "shared-b", APIKey: "key2", BaseURL: "http://b.com", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	first, err := bal.NextForModelScoped("", scope.NewContext([]string{"team-a"}, nil))
	require.NoError(t, err)
	second, err := bal.NextForModelScoped("", scope.NewContext([]string{"team-b"}, nil))
	require.NoError(t, err)

	assert.Equal(t, "shared-a", first.Name)
	assert.Equal(t, "shared-b", second.Name)
}

func TestSetLogger(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()
	creds := []config.CredentialConfig{
		{Name: "cred1", Type: config.ProviderTypeOpenAI, RPM: 100},
	}

	rr := New(creds, f2b, rl)

	logger := rr.logger
	assert.NotNil(t, logger, "logger should not be nil after creation")

	// Set a new logger
	newLogger := rr.logger // just reuse for simplicity
	rr.SetLogger(newLogger)
	assert.NotNil(t, rr.logger, "logger should not be nil after SetLogger")
}

func TestGetCredentialByName(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()
	creds := []config.CredentialConfig{
		{Name: "alpha", Type: config.ProviderTypeOpenAI, RPM: 100},
		{Name: "beta", Type: config.ProviderTypeVertexAI, RPM: 200},
	}

	rr := New(creds, f2b, rl)

	t.Run("existing", func(t *testing.T) {
		// getCredentialByName is unexported, test via IsProxyCredential
		// or test indirectly through exported methods
		rr.mu.RLock()
		cred := rr.getCredentialByName("alpha")
		rr.mu.RUnlock()
		require.NotNil(t, cred)
		assert.Equal(t, "alpha", cred.Name)
		assert.Equal(t, config.ProviderTypeOpenAI, cred.Type)
	})

	t.Run("not_existing", func(t *testing.T) {
		rr.mu.RLock()
		cred := rr.getCredentialByName("nonexistent")
		rr.mu.RUnlock()
		assert.Nil(t, cred)
	})
}

func TestIsProxyCredential_And_GetProxyCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()
	creds := []config.CredentialConfig{
		{Name: "openai-1", Type: config.ProviderTypeOpenAI, RPM: 100},
		{Name: "proxy-1", Type: config.ProviderTypeProxy, RPM: 100, BaseURL: "http://localhost:8080"},
		{Name: "air-1", Type: config.ProviderTypeAIR, RPM: 100, BaseURL: "http://localhost:8082"},
		{Name: "vertex-1", Type: config.ProviderTypeVertexAI, RPM: 100},
		{Name: "proxy-2", Type: config.ProviderTypeProxy, RPM: 200, BaseURL: "http://localhost:8081"},
	}

	rr := New(creds, f2b, rl)

	// IsProxyCredential
	assert.False(t, rr.IsProxyCredential("openai-1"))
	assert.True(t, rr.IsProxyCredential("proxy-1"))
	assert.True(t, rr.IsProxyCredential("air-1"))
	assert.False(t, rr.IsProxyCredential("vertex-1"))
	assert.True(t, rr.IsProxyCredential("proxy-2"))
	assert.False(t, rr.IsProxyCredential("nonexistent"))

	// GetProxyCredentials
	proxies := rr.GetProxyCredentials()
	assert.Len(t, proxies, 3)

	proxyNames := make(map[string]bool)
	for _, p := range proxies {
		proxyNames[p.Name] = true
	}
	assert.True(t, proxyNames["proxy-1"])
	assert.True(t, proxyNames["air-1"])
	assert.True(t, proxyNames["proxy-2"])
}

// TestRoundRobin_MixedTypeTrafficIndependence verifies that high-frequency OpenAI
// traffic does not interfere with Vertex AI credential cycling.
// This is the real-world bug: with a shared r.current counter, 500 RPM of OpenAI
// requests reset the counter to positions 0-1, causing every Vertex request to
// start from the same position and always pick the first available Vertex credential.
func TestRoundRobin_MixedTypeTrafficIndependence(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "openai-1", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://openai1.com", RPM: -1},
		{Name: "openai-2", Type: config.ProviderTypeOpenAI, APIKey: "key2", BaseURL: "http://openai2.com", RPM: -1},
		{Name: "vertex-1", Type: config.ProviderTypeVertexAI, APIKey: "key3", RPM: -1, ProjectID: "p1", Location: "us-central1"},
		{Name: "vertex-2", Type: config.ProviderTypeVertexAI, APIKey: "key4", RPM: -1, ProjectID: "p2", Location: "us-central1"},
		{Name: "vertex-3", Type: config.ProviderTypeVertexAI, APIKey: "key5", RPM: -1, ProjectID: "p3", Location: "us-central1"},
		{Name: "vertex-4", Type: config.ProviderTypeVertexAI, APIKey: "key6", RPM: -1, ProjectID: "p4", Location: "us-central1"},
	}

	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("openai-1", "gpt-4o")
	mc.AddModel("openai-2", "gpt-4o")
	mc.AddModel("vertex-1", "gemini-2.5-flash")
	mc.AddModel("vertex-2", "gemini-2.5-flash")
	mc.AddModel("vertex-3", "gemini-2.5-flash")
	mc.AddModel("vertex-4", "gemini-2.5-flash")
	bal.SetModelChecker(mc)

	// Simulate heavy OpenAI traffic interleaved with Vertex requests.
	// Without per-type counters, OpenAI requests reset r.current to 0-1,
	// causing every Vertex request to always pick vertex-1.
	vertexResults := make([]string, 0, 8)
	for range 8 {
		// 10 OpenAI requests between each Vertex request (simulates ~500 RPM)
		for range 10 {
			_, err := bal.NextForModel("gpt-4o")
			require.NoError(t, err)
		}
		cred, err := bal.NextForModel("gemini-2.5-flash")
		require.NoError(t, err)
		vertexResults = append(vertexResults, cred.Name)
	}

	// Vertex requests must cycle through all 4 credentials in strict round-robin order.
	expectedOrder := []string{
		"vertex-1", "vertex-2", "vertex-3", "vertex-4",
		"vertex-1", "vertex-2", "vertex-3", "vertex-4",
	}
	assert.Equal(t, expectedOrder, vertexResults,
		"Vertex creds should cycle evenly regardless of interleaved OpenAI traffic")
}

func TestUpdateDBCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	staticCreds := []config.CredentialConfig{
		{Name: "yaml-1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}
	bal := New(staticCreds, f2b, rl)

	dbCreds := []config.CredentialConfig{
		{Name: "yaml-1", APIKey: "dup", RPM: 10}, // should be ignored (static wins)
		{Name: "db-1", APIKey: "dbkey1", RPM: 10, TPM: 0},
		{Name: "db-2", APIKey: "dbkey2", RPM: 20, TPM: 50},
	}

	bal.UpdateDBCredentials(dbCreds)

	// Static credential must remain, duplicate DB name should be filtered out.
	assert.Len(t, bal.credentials, 3)
	assert.Equal(t, []string{"yaml-1", "db-1", "db-2"}, []string{
		bal.credentials[0].Name,
		bal.credentials[1].Name,
		bal.credentials[2].Name,
	})

	// Credential index must include new DB creds.
	assert.Equal(t, 0, bal.credentialIndex["yaml-1"])
	assert.Equal(t, 1, bal.credentialIndex["db-1"])
	assert.Equal(t, 2, bal.credentialIndex["db-2"])

	// DB credentials should be registered in the rate limiter with TPM defaulting to -1 when 0.
	assert.True(t, bal.rateLimiter.AllowTokens("db-1"), "TPM=0 should be treated as unlimited")
	assert.True(t, bal.rateLimiter.AllowTokens("db-2"))
}

// TestUpdateDBCredentials_FallbackPinnedToGroup999 verifies that a DB-sourced credential
// with IsFallback set is pinned to FallbackPriorityGroup even though DB credentials never
// pass through config.Validate (review_158 item 24).
func TestUpdateDBCredentials_FallbackPinnedToGroup999(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()
	bal := New([]config.CredentialConfig{
		{Name: "yaml-1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}, f2b, rl)

	bal.UpdateDBCredentials([]config.CredentialConfig{
		{Name: "db-fallback", APIKey: "k", RPM: 10, IsFallback: true},
		{Name: "db-primary", APIKey: "k", RPM: 10},
	})

	idx := bal.credentialIndex["db-fallback"]
	assert.Equal(t, config.FallbackPriorityGroup, bal.credentials[idx].Priority)
	assert.Equal(t, 0, bal.credentials[bal.credentialIndex["db-primary"]].Priority)
}

func TestUpdateDBCredentials_PreservesProviderScopeMetadata(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	staticProxy := config.CredentialConfig{
		Name: "yaml-proxy", Type: config.ProviderTypeProxy, BaseURL: "http://yaml.example", APIKey: "key", RPM: -1,
	}
	bal := New([]config.CredentialConfig{staticProxy}, f2b, rl)
	teamA := scope.FromScopes([]string{"team-a"}, nil)
	require.True(t, bal.UpdateProviderScopes(staticProxy, []string{"team-a"}, nil, teamA, true))

	bal.UpdateDBCredentials(nil)

	staticSnapshot := bal.GetCredentialsSnapshot()[0]
	assert.True(t, staticSnapshot.ProviderScopeKnown)
	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(staticSnapshot.ProviderScopeExpression))
	assert.False(t, scope.NewContext([]string{"team-b"}, nil).AllowsExpression(staticSnapshot.ProviderScopeExpression))

	dbProxy := config.CredentialConfig{
		Name: "db-proxy", Type: config.ProviderTypeProxy, BaseURL: "http://db.example", APIKey: "key", RPM: -1,
	}
	bal.UpdateDBCredentials([]config.CredentialConfig{dbProxy})
	require.True(t, bal.UpdateProviderScopes(dbProxy, []string{"team-a"}, nil, teamA, true))
	bal.UpdateDBCredentials([]config.CredentialConfig{dbProxy})

	dbSnapshot := bal.GetCredentialsSnapshot()[1]
	assert.True(t, dbSnapshot.ProviderScopeKnown)
	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(dbSnapshot.ProviderScopeExpression))

	dbProxy.IsFallback = true
	bal.UpdateDBCredentials([]config.CredentialConfig{dbProxy})
	dbSnapshot = bal.GetCredentialsSnapshot()[1]
	assert.False(t, dbSnapshot.ProviderScopeKnown)
	assert.False(t, scope.AdminContext().AllowsExpression(dbSnapshot.ProviderScopeExpression))
}

func TestUpdateDBCredentials_PreservesSWRRState(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	staticCreds := []config.CredentialConfig{
		{Name: "yaml-1", APIKey: "key1", BaseURL: "http://test1.com", RPM: -1, Weight: 2},
		{Name: "yaml-2", APIKey: "key2", BaseURL: "http://test2.com", RPM: -1, Weight: 1},
	}
	bal := New(staticCreds, f2b, rl)

	_, err := bal.NextForModel("")
	require.NoError(t, err)

	key := schedKey{scopeKey: "yaml-1\x00yaml-2"}
	state := bal.swrr[key]
	require.NotNil(t, state)
	beforeYAML1 := state.currentOf("yaml-1")
	beforeYAML2 := state.currentOf("yaml-2")

	bal.UpdateDBCredentials([]config.CredentialConfig{
		{Name: "db-1", APIKey: "dbkey1", BaseURL: "http://db1.com", RPM: -1},
	})

	assert.Same(t, state, bal.swrr[key], "DB sync should not reset existing SWRR cycles")
	assert.Equal(t, beforeYAML1, state.currentOf("yaml-1"))
	assert.Equal(t, beforeYAML2, state.currentOf("yaml-2"))
}

// --- T2: priority-group primary selection tests ---

// TestNextForModel_PriorityGroup_PartialAvailability verifies that when a priority group
// still has at least one live member, the selector keeps serving that group instead of
// cascading to the next one, even though most of the group is down.
func TestNextForModel_PriorityGroup_PartialAvailability(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "g100-a", Type: config.ProviderTypeOpenAI, APIKey: "k1", BaseURL: "http://a.com", RPM: 100, Priority: 100},
		{Name: "g100-b", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 1, Priority: 100},
		{Name: "g100-c", Type: config.ProviderTypeOpenAI, APIKey: "k3", BaseURL: "http://c.com", RPM: 100, Priority: 100},
		{Name: "g200", Type: config.ProviderTypeOpenAI, APIKey: "k4", BaseURL: "http://d.com", RPM: 100, Priority: 200},
	}

	bal := New(credentials, f2b, rl)

	// Ban g100-a (3x 500 trips fail2ban).
	for range 3 {
		bal.RecordResponse("g100-a", "", 500)
	}
	// Exhaust g100-b's RPM (limit is 1) directly against the shared rate limiter.
	require.True(t, rl.Allow("g100-b"))

	// The remaining live member of group 100 (g100-c) must keep being selected;
	// group 200 must never be reached while group 100 has a live member.
	for range 5 {
		cred, err := bal.NextForModel("")
		require.NoError(t, err)
		assert.Equal(t, "g100-c", cred.Name)
	}
}

// TestNextForModel_PriorityGroup_CascadesWhenGroupFullyDown verifies that once every
// member of the lowest priority group is down, selection cascades to the next group.
func TestNextForModel_PriorityGroup_CascadesWhenGroupFullyDown(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "g100-a", Type: config.ProviderTypeOpenAI, APIKey: "k1", BaseURL: "http://a.com", RPM: 100, Priority: 100},
		{Name: "g100-b", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 100, Priority: 100},
		{Name: "g200", Type: config.ProviderTypeOpenAI, APIKey: "k3", BaseURL: "http://c.com", RPM: 100, Priority: 200},
	}

	bal := New(credentials, f2b, rl)

	// Before banning anything, group 100 serves the request.
	cred, err := bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "g100-a", cred.Name)

	// Ban both members of group 100.
	for range 3 {
		bal.RecordResponse("g100-a", "", 500)
		bal.RecordResponse("g100-b", "", 500)
	}

	// Group 100 is fully down now — selection must cascade to group 200.
	cred, err = bal.NextForModel("")
	require.NoError(t, err)
	assert.Equal(t, "g200", cred.Name)
}

func TestSelectPriorityGroupCandidate_ConcurrentRateLimitRaceCascadesToNextGroup(t *testing.T) {
	const trials = 300

	for i := range trials {
		f2b := fail2ban.New(3, 0, []int{500})
		rl := ratelimit.New()

		credentials := []config.CredentialConfig{
			{Name: "g100", Type: config.ProviderTypeOpenAI, APIKey: "k1", BaseURL: "http://a.com", RPM: 1, Priority: 100},
			{Name: "g200", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 10, Priority: 200},
		}
		r := New(credentials, f2b, rl)

		candidates := []candidateEntry{
			{absIdx: 0, cred: &r.credentials[0]},
			{absIdx: 1, cred: &r.credentials[1]},
		}
		keyBase := r.schedKeyFor("", false, false, "", false, "")

		var victimCred *config.CredentialConfig
		var victimFound bool

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			victimCred, victimFound, _ = r.selectPriorityGroupCandidate("", candidates, keyBase, r.primaryPriority)
		}()
		go func() {
			defer wg.Done()
			rl.TryAllowAll("g100", "") // the "other process" racing for g100's only unit
		}()
		wg.Wait()

		require.True(t, victimFound, "trial %d: g200 has spare capacity, must never surface ErrRateLimitExceeded", i)
		require.NotNil(t, victimCred)
		assert.Contains(t, []string{"g100", "g200"}, victimCred.Name)
	}
}

// TestNextForModel_PriorityGroup_DefaultGroupSelectedFirst verifies that credentials
// without an explicit priority (group 0) are always tried before credentials with an
// explicit, positive priority.
func TestNextForModel_PriorityGroup_DefaultGroupSelectedFirst(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "explicit-100", Type: config.ProviderTypeOpenAI, APIKey: "k1", BaseURL: "http://a.com", RPM: 100, Priority: 100},
		{Name: "unprioritized", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	for range 4 {
		cred, err := bal.NextForModel("")
		require.NoError(t, err)
		assert.Equal(t, "unprioritized", cred.Name)
	}
}

// TestNextForModel_PriorityGroup_WeightedWithinGroup verifies SWRR by weight continues to
// work as before, scoped to credentials inside a single priority group.
func TestNextForModel_PriorityGroup_WeightedWithinGroup(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "heavy", Type: config.ProviderTypeOpenAI, APIKey: "k1", BaseURL: "http://a.com", RPM: 1000, Priority: 100, Weight: 3},
		{Name: "light", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 1000, Priority: 100, Weight: 1},
	}

	bal := New(credentials, f2b, rl)

	counts := map[string]int{}
	const n = 400
	for range n {
		cred, err := bal.NextForModel("")
		require.NoError(t, err)
		counts[cred.Name]++
	}

	// With weights 3:1 the smooth weighted round-robin should distribute selections
	// roughly proportionally; allow generous slack since this is a scheduling algorithm,
	// not an exact statistical draw.
	assert.InDelta(t, 300, counts["heavy"], 20)
	assert.InDelta(t, 100, counts["light"], 20)
}

// TestNextForModel_PriorityGroup_ProxyCredentialUsesPerModelPriority verifies the T3
// balancer-side piece: a proxy/AIR credential's priority group for a given model can be
// overridden per-model via ModelChecker.LearnedModelPriorityForCredential — the mechanism
// remote.go's updateModelLimits feeds via modelManager.ReplaceModelPrioritiesForCredential
// after polling an upstream node's /health. Without this, every proxy credential would
// collapse into one flat group (its own static EffectivePriority(), typically 0)
// regardless of what priority is actually configured on the node it proxies to.
func TestNextForModel_PriorityGroup_ProxyCredentialUsesPerModelPriority(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy-a", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: 100},
		{Name: "proxy-b", Type: config.ProviderTypeProxy, APIKey: "k2", BaseURL: "http://b.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("proxy-a", "shared-model")
	mc.AddModel("proxy-b", "shared-model")
	// Both credentials have no static Priority (EffectivePriority() == 0 for both), but
	// the upstream nodes they proxy to report different priorities for this model.
	mc.SetModelPriority("shared-model", "proxy-a", 200)
	mc.SetModelPriority("shared-model", "proxy-b", 50)
	bal.SetModelChecker(mc)

	// proxy-b's learned priority (50) is lower than proxy-a's (200), so proxy-b's group
	// must be selected exclusively while it's live — proxy-a is never reached.
	for range 5 {
		cred, err := bal.NextForModel("shared-model")
		require.NoError(t, err)
		assert.Equal(t, "proxy-b", cred.Name)
	}

	// A request for a model with no learned per-model priority falls back to each
	// credential's static EffectivePriority() (0 for both here) — flat group, SWRR
	// alternates.
	mc.AddModel("proxy-a", "other-model")
	mc.AddModel("proxy-b", "other-model")
	seen := map[string]bool{}
	for range 4 {
		cred, err := bal.NextForModel("other-model")
		require.NoError(t, err)
		seen[cred.Name] = true
	}
	assert.True(t, seen["proxy-a"], "without a per-model override, proxy-a must be reachable")
	assert.True(t, seen["proxy-b"], "without a per-model override, proxy-b must be reachable")
}

// TestNextForModel_PriorityGroup_LearnedZeroIsAuthoritative is the review_158 #14
// balancer piece: a proxy credential with a static priority: field but whose upstream
// serves a model in its best group (learned priority 0) must route in group 0, not fall
// back to the static field. Before the fix a learned 0 was indistinguishable from
// "nothing learned" and the static field won — so a better upstream tier produced a
// worse local group.
func TestNextForModel_PriorityGroup_LearnedZeroIsAuthoritative(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy-slow", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: 100, Priority: 50},
		{Name: "direct-mid", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: 100, Priority: 10},
	}
	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("proxy-slow", "shared-model")
	mc.AddModel("direct-mid", "shared-model")
	// Upstream reports the model in its best group.
	mc.SetModelPriority("shared-model", "proxy-slow", 0)
	bal.SetModelChecker(mc)

	// proxy-slow's learned 0 beats direct-mid's static 10 — it is served exclusively.
	for range 5 {
		cred, err := bal.NextForModel("shared-model")
		require.NoError(t, err)
		assert.Equal(t, "proxy-slow", cred.Name, "learned priority 0 must place the proxy in group 0, ahead of the static-10 direct credential")
	}
}

// TestNextForModel_TierCandidateExpansion_LocalCumulativeCascade is the Design B
// balancer piece (review_158 item 3): a proxy credential with a learned multi-tier
// breakdown for a model expands into one candidate per tier, each in its own primary
// priority group with its own cumulative local RPM cap. When this router's own recorded
// usage crosses the tier-1 cap, selection cascades to the next group immediately — with
// no /health poll — and lands on a genuinely lower-priority alternative rather than the
// same proxy's tier-5 group.
func TestNextForModel_TierCandidateExpansion_LocalCumulativeCascade(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
		{Name: "alt", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: -1, Priority: 3},
	}
	rl.AddCredential("px", -1)
	rl.AddCredential("alt", -1)
	// Aggregate (cred,model) bucket = grand total of all tiers, so TryAllowAll records
	// into the counter that tierHasHeadroom reads.
	rl.AddModelWithTPM("px", "m", 3+100, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.AddModel("alt", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: 3},
		{Priority: 5, Weight: 1, LimitRPM: 100},
	})
	bal.SetModelChecker(mc)

	// First 3 requests: px@tier-1 (group 1) has cumulative headroom (cap 3).
	for i := 0; i < 3; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "request %d", i)
		assert.Equal(t, "px", cred.Name, "request %d", i)
	}
	// tier-1 cumulative cap now reached locally — cascade to group 3 (alt), never px@tier-5.
	for i := 0; i < 4; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "cascade request %d", i)
		assert.Equal(t, "alt", cred.Name, "cascade request %d must land on the tier-3 alternative, not px@tier-5", i)
	}
}

// TestNextForModel_TierCandidate_SWRRWeightWithinGroup: a tier candidate and a plain
// credential sharing a priority group split traffic by the tier's learned weight.
func TestNextForModel_TierCandidate_SWRRWeightWithinGroup(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
		{Name: "alt", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: -1, Priority: 1},
	}
	rl.AddCredential("px", -1)
	rl.AddCredential("alt", -1)
	rl.AddModelWithTPM("px", "m", -1, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.AddModel("alt", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 3, LimitRPM: -1},
		{Priority: 9, Weight: 1, LimitRPM: -1},
	})
	bal.SetModelChecker(mc)

	counts := map[string]int{}
	for i := 0; i < 8; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err)
		counts[cred.Name]++
	}
	// group 1 = {px@tier-1 weight 3, alt weight 1} → 3:1 over 8 picks.
	assert.Equal(t, 6, counts["px"])
	assert.Equal(t, 2, counts["alt"])
}

// TestNextForModel_AllTiersSaturated_Returns429NotNoCredentials is review_158 item 12:
// when the only candidate is a proxy whose every learned tier is RPM-saturated upstream
// (Banned=false, CurrentRPM>=LimitRPM), selection must fail with ErrRateLimitExceeded so
// the caller does rate-limit backoff — not ErrNoCredentialsAvailable (503).
func TestNextForModel_AllTiersSaturated_Returns429NotNoCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
	}
	rl.AddCredential("px", -1)
	rl.AddModelWithTPM("px", "m", -1, -1) // aggregate bucket unlimited — per-tier gate is what trips

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: 10, CurrentRPM: 10},
		{Priority: 2, Weight: 1, LimitRPM: 20, CurrentRPM: 20},
	})
	bal.SetModelChecker(mc)

	_, err := bal.NextForModel("m")
	assert.ErrorIs(t, err, ErrRateLimitExceeded)
}

// TestNextForModel_AllTiersReallyBanned_ReturnsNoCredentials is the item 12 counterpart:
// a real upstream ban on every tier is not a rate-limit condition — it must still surface
// as ErrNoCredentialsAvailable.
func TestNextForModel_AllTiersReallyBanned_ReturnsNoCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
	}
	rl.AddCredential("px", -1)
	rl.AddModelWithTPM("px", "m", -1, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: -1, Banned: true},
		{Priority: 2, Weight: 1, LimitRPM: -1, Banned: true},
	})
	bal.SetModelChecker(mc)

	_, err := bal.NextForModel("m")
	assert.ErrorIs(t, err, ErrNoCredentialsAvailable)
}

// TestNextForModel_TierHeadroom_GatesOnUpstreamPerTierUsage is review_158 item 13: the
// cumulative-cap check reads a single aggregate (cred,model) counter that cannot see
// which tier the upstream actually consumed. When the upstream's own per-tier view shows
// the cheap tier saturated while the local aggregate counter is still low, the per-tier
// gate must drop that tier candidate and let selection cascade to the next group.
func TestNextForModel_TierHeadroom_GatesOnUpstreamPerTierUsage(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
		// A direct alternative sharing priority group 1 with px's cheap tier.
		{Name: "alt", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: -1, Priority: 1},
	}
	rl.AddCredential("px", -1)
	rl.AddCredential("alt", -1)
	rl.AddModelWithTPM("px", "m", -1, -1) // local aggregate counter stays 0 in this test
	rl.AddModelWithTPM("alt", "m", -1, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.AddModel("alt", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		// tier 1: upstream reports it fully consumed (100/100) — not banned.
		{Priority: 1, Weight: 1, LimitRPM: 100, CurrentRPM: 100},
		// tier 2: free.
		{Priority: 2, Weight: 1, LimitRPM: 900, CurrentRPM: 0},
	})
	bal.SetModelChecker(mc)

	// The local aggregate counter is 0 (< tier-1 cumulative cap 100), so the cumulative
	// gate alone would keep px@tier-1 in group 1 and SWRR would split traffic with alt.
	// The per-tier gate sees the upstream's 100/100 and drops px@tier-1: group 1 now has
	// only alt, and px@tier-2 (group 2) is never reached.
	for i := 0; i < 6; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "request %d", i)
		assert.Equal(t, "alt", cred.Name, "request %d must route to alt, not px's saturated tier-1 or its tier-2", i)
	}
}

// TestNextForModel_TierHeadroom_PricierTierUsageDoesNotCloseRecoveredCheapTier is
// review_158 round 3 item 1: the cumulative-cap gate reads a single aggregate
// (cred, model) counter with no tier attribution. Load the upstream serves from a pricier
// fallback tier must not count against a cheaper tier's small cumulative cap and evict it
// once it has recovered.
func TestNextForModel_TierHeadroom_PricierTierUsageDoesNotCloseRecoveredCheapTier(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
		{Name: "alt", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: -1, Priority: 3},
	}
	rl.AddCredential("px", -1)
	rl.AddCredential("alt", -1)
	rl.AddModelWithTPM("px", "m", -1, -1)
	rl.AddModelWithTPM("alt", "m", -1, -1)

	// 50 requests this router already sent to px for m — the upstream served them all from
	// its pricier tier 5; its cheap tier 1 window has since reset to 0/10.
	for i := 0; i < 50; i++ {
		require.True(t, rl.TryAllowAll("px", "m"))
	}

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.AddModel("alt", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: 10, CurrentRPM: 0},
		{Priority: 5, Weight: 1, LimitRPM: 100, CurrentRPM: 50},
	})
	bal.SetModelChecker(mc)

	// px@tier-1 (group 1) is free upstream; its cumulative cap (10), net of the 50
	// requests the upstream attributes to tier 5, still has headroom — px wins over the
	// group-3 alternative instead of being wrongly cascaded past.
	for i := 0; i < 5; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "request %d", i)
		assert.Equal(t, "px", cred.Name, "request %d must stay on px@tier-1, not cascade to alt", i)
	}
}

// TestNextForModel_BannedTierCapacityDoesNotInflateLiveTierCap is review_158 round 3
// item 2: a really-banned tier's recoverable capacity must not fold into a live tier's
// cumulative cap, or the live tier over-admits past the capacity actually available now.
func TestNextForModel_BannedTierCapacityDoesNotInflateLiveTierCap(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
	}
	rl.AddCredential("px", -1)
	rl.AddModelWithTPM("px", "m", -1, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: 100, Banned: true},
		{Priority: 2, Weight: 1, LimitRPM: 10, CurrentRPM: 0},
	})
	bal.SetModelChecker(mc)

	// tier 2's cumulative cap is its own 10, not 110 (the banned tier 1 contributes none).
	for i := 0; i < 10; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "request %d", i)
		assert.Equal(t, "px", cred.Name)
	}
	_, err := bal.NextForModel("m")
	assert.ErrorIs(t, err, ErrRateLimitExceeded)
}

// TestNextForModel_SingleBannedTier_CredentialDropped is review_158 round 3 item 3: a
// proxy whose sole learned upstream tier is banned must not stay a live candidate.
func TestNextForModel_SingleBannedTier_CredentialDropped(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "px", Type: config.ProviderTypeProxy, APIKey: "k1", BaseURL: "http://a.com", RPM: -1},
		{Name: "alt", Type: config.ProviderTypeOpenAI, APIKey: "k2", BaseURL: "http://b.com", RPM: -1, Priority: 5},
	}
	rl.AddCredential("px", -1)
	rl.AddCredential("alt", -1)
	rl.AddModelWithTPM("px", "m", -1, -1)
	rl.AddModelWithTPM("alt", "m", -1, -1)

	bal := New(credentials, f2b, rl)
	mc := NewMockModelChecker(true)
	mc.AddModel("px", "m")
	mc.AddModel("alt", "m")
	mc.SetModelTiers("m", "px", []httputil.ModelPriorityTier{
		{Priority: 1, Weight: 1, LimitRPM: -1, Banned: true},
	})
	bal.SetModelChecker(mc)

	for i := 0; i < 5; i++ {
		cred, err := bal.NextForModel("m")
		require.NoError(t, err, "request %d", i)
		assert.Equal(t, "alt", cred.Name, "request %d must skip the single-banned-tier proxy", i)
	}
}

// TestNextForModel_NoPriorityFields_BehavesAsFlatPool is a backward-compatibility check:
// a config without any priority/fallback_priority fields set must behave exactly like the
// historical flat weighted round-robin pool (single implicit group 0).
func TestNextForModel_NoPriorityFields_BehavesAsFlatPool(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1000},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1000},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000},
	}

	bal := New(credentials, f2b, rl)

	expectedOrder := []string{"cred1", "cred2", "cred3", "cred1", "cred2", "cred3"}
	for i, expectedName := range expectedOrder {
		cred, err := bal.NextForModel("")
		require.NoError(t, err, "Request %d failed", i+1)
		assert.Equal(t, expectedName, cred.Name, "Request %d: expected %s, got %s", i+1, expectedName, cred.Name)
	}
}

// TestNextForModel_FallbackPriorityOnly_StaysFlatPrimaryPool is the #3 regression guard:
// fallback_priority is a retry-only ordering knob. A config that sets fallback_priority
// but never the explicit priority: field must keep serving primary traffic as one flat
// weighted pool (matching pre-priority-groups behavior), NOT as hard exclusive tiers.
func TestNextForModel_FallbackPriorityOnly_StaysFlatPrimaryPool(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1000, FallbackPriority: 10},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1000, FallbackPriority: 20},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000, FallbackPriority: 30},
	}

	bal := New(credentials, f2b, rl)

	counts := map[string]int{}
	for i := range 30 {
		cred, err := bal.NextForModel("")
		require.NoError(t, err, "request %d", i)
		counts[cred.Name]++
	}

	assert.Equal(t, map[string]int{"cred1": 10, "cred2": 10, "cred3": 10}, counts,
		"fallback_priority must not create primary-pool priority steps — expected an even flat split")
}

func TestMinRemainingBanForModel_IgnoresExcludedCredential(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{429})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred-excluded", APIKey: "key1", BaseURL: "http://test1.com", RPM: -1},
		{Name: "cred-relevant", APIKey: "key2", BaseURL: "http://test2.com", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	// cred-excluded is banned much longer, but has already been tried and
	// rejected earlier in this same request (e.g. a 500) — it must not
	// influence the Retry-After hint at all.
	bal.BanUntil("cred-excluded", "shared-model", 429, time.Now().Add(time.Hour), "test")
	bal.BanUntil("cred-relevant", "shared-model", 429, time.Now().Add(2*time.Second), "test")

	exclude := map[string]bool{"cred-excluded": true}
	remaining, ok := bal.MinRemainingBanForModel("shared-model", exclude, scope.AdminContext())

	require.True(t, ok)
	assert.Greater(t, remaining, time.Duration(0))
	assert.LessOrEqual(t, remaining, 2*time.Second, "must report the excluded candidate's ban, not the 1h one from the excluded credential")
}

func TestMinRemainingBanForModel_IgnoresCredentialOutOfScope(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{429})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred-out-of-scope", APIKey: "key1", BaseURL: "http://test1.com", RPM: -1, ProviderScopeExpression: scope.FalseExpression()},
		{Name: "cred-in-scope", APIKey: "key2", BaseURL: "http://test2.com", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	// The out-of-scope credential has the shortest ban of the two — if scope
	// filtering were missing, it would incorrectly win the "shortest" pick
	// even though this caller can never route to it.
	bal.BanUntil("cred-out-of-scope", "shared-model", 429, time.Now().Add(1*time.Second), "test")
	bal.BanUntil("cred-in-scope", "shared-model", 429, time.Now().Add(5*time.Second), "test")

	remaining, ok := bal.MinRemainingBanForModel("shared-model", nil, scope.AdminContext())

	require.True(t, ok)
	assert.Greater(t, remaining, 2*time.Second, "must report the in-scope candidate's ban, not the shorter out-of-scope one")
	assert.LessOrEqual(t, remaining, 5*time.Second)
}

func TestMinRemainingBanForModel_NoRelevantCandidateBanned_ReturnsFalse(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{429})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred-excluded", APIKey: "key1", BaseURL: "http://test1.com", RPM: -1},
	}
	bal := New(credentials, f2b, rl)

	bal.BanUntil("cred-excluded", "shared-model", 429, time.Now().Add(time.Hour), "test")

	exclude := map[string]bool{"cred-excluded": true}
	_, ok := bal.MinRemainingBanForModel("shared-model", exclude, scope.AdminContext())

	assert.False(t, ok, "no relevant candidate is banned once the only one is excluded — must not fall back to guessing")
}
