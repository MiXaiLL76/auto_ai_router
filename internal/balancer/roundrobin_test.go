package balancer

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockModelChecker implements ModelChecker interface for testing
type MockModelChecker struct {
	enabled            bool
	credentialModels   map[string][]string // credential -> models
	modelToCredentials map[string][]string // model -> credentials
}

func NewMockModelChecker(enabled bool) *MockModelChecker {
	return &MockModelChecker{
		enabled:            enabled,
		credentialModels:   make(map[string][]string),
		modelToCredentials: make(map[string][]string),
	}
}

func (m *MockModelChecker) HasModel(credentialName, modelID string) bool {
	if !m.enabled {
		return true
	}
	models, ok := m.credentialModels[credentialName]
	if !ok {
		return true
	}
	for _, model := range models {
		if model == modelID {
			return true
		}
	}
	return false
}

func (m *MockModelChecker) GetCredentialsForModel(modelID string) []string {
	if !m.enabled {
		return nil
	}
	return m.modelToCredentials[modelID]
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
	assert.Equal(t, 0, bal.current)
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
	bal.RecordResponse("cred1", 401)
	bal.RecordResponse("cred1", 401)
	bal.RecordResponse("cred1", 401)

	// Credential should be banned
	assert.True(t, f2b.IsBanned("cred1"))
}

func TestGetCredentials(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 200},
	}

	bal := New(credentials, f2b, rl)

	creds := bal.GetCredentials()
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
	f2b.RecordResponse("cred2", 401)
	f2b.RecordResponse("cred2", 401)
	f2b.RecordResponse("cred2", 401)

	// Should have 2 available
	assert.Equal(t, 2, bal.GetAvailableCount())
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
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)
	f2b.RecordResponse("cred1", 401)

	// Should have 1 banned
	assert.Equal(t, 1, bal.GetBannedCount())

	// Ban another
	f2b.RecordResponse("cred2", 500)
	f2b.RecordResponse("cred2", 500)
	f2b.RecordResponse("cred2", 500)

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
	bal.RecordResponse("cred1", 401)
	bal.RecordResponse("cred1", 401)
	bal.RecordResponse("cred1", 401)

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
	for i := 0; i < 3; i++ {
		bal.RecordResponse("cred1", 401)
		bal.RecordResponse("cred2", 401)
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

func TestNextFallbackForModel_SkipsNonProxyTypes(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "openai1", Type: config.ProviderTypeOpenAI, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	cred, err := bal.NextFallbackForModel("gpt-4o")

	assert.NoError(t, err)
	assert.Equal(t, "proxy1", cred.Name)
	assert.Equal(t, config.ProviderTypeProxy, cred.Type)
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

func TestNextFallbackForModel_SkipsBannedFallback(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "proxy1", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
		{Name: "proxy2", Type: config.ProviderTypeProxy, IsFallback: true, RPM: 100, TPM: 10000},
	}

	bal := New(credentials, f2b, rl)

	// Ban first proxy
	bal.RecordResponse("proxy1", 500)
	bal.RecordResponse("proxy1", 500)
	bal.RecordResponse("proxy1", 500)

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
	for i := 0; i < numReaders; i++ {
		go func() {
			for j := 0; j < 1000; j++ {
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
		for j := 0; j < numWriteOps; j++ {
			// These operations acquire locks internally
			bal.GetAvailableCount()
			bal.GetBannedCount()
			if j%3 == 0 {
				f2b.RecordResponse("cred1", 401)
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
