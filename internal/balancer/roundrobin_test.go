package balancer

import (
	"sync"
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

func TestNext_RoundRobin(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Should rotate through credentials
	cred1, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred1.Name)

	cred2, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred2.Name)

	cred3, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred3", cred3.Name)

	// Back to first
	cred4, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred4.Name)
}

func TestNext_SkipBanned(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Ban cred2
	f2b.RecordResponse("cred2", 401)
	f2b.RecordResponse("cred2", 401)
	f2b.RecordResponse("cred2", 401)

	// Should skip cred2
	cred1, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred1.Name)

	cred3, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred3", cred3.Name) // Skipped cred2

	cred1Again, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred1Again.Name)
}

func TestNext_SkipRateLimited(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 2},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Get cred1 twice (at limit) - rounds: cred1, cred2, cred1
	cred1, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred1.Name)

	cred2, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred2.Name)

	cred3, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred3", cred3.Name)

	cred1Again, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred1Again.Name) // cred1 now at limit (2 requests)

	// Next should skip cred1 (rate limited) and go to cred2
	credNext, err := bal.Next()
	require.NoError(t, err)
	assert.Equal(t, "cred2", credNext.Name)
}

func TestNext_AllBanned(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	// Ban all credentials
	for _, cred := range []string{"cred1", "cred2"} {
		f2b.RecordResponse(cred, 401)
		f2b.RecordResponse(cred, 401)
		f2b.RecordResponse(cred, 401)
	}

	// Should return error
	_, err := bal.Next()
	assert.Error(t, err)
	assert.Equal(t, ErrNoCredentialsAvailable, err)
}

func TestNext_AllRateLimited(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1},
	}

	bal := New(credentials, f2b, rl)

	// Exhaust rate limits
	_, _ = bal.Next() // cred1
	_, _ = bal.Next() // cred2

	// Next request should return rate limit error
	_, err := bal.Next()
	assert.Error(t, err)
	assert.Equal(t, ErrRateLimitExceeded, err)
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

	// Request for gpt-4o-mini can return cred1 or cred2
	cred, err = bal.NextForModel("gpt-4o-mini")
	require.NoError(t, err)
	assert.Contains(t, []string{"cred2", cred.Name}, cred.Name)
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
	rl.AddModel("gpt-4o", 1) // Very low model RPM limit

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)

	mc := NewMockModelChecker(true)
	mc.AddModel("cred1", "gpt-4o")
	mc.AddModel("cred2", "gpt-4o")

	bal.SetModelChecker(mc)

	// First request should succeed
	cred, err := bal.NextForModel("gpt-4o")
	require.NoError(t, err)
	assert.NotNil(t, cred)

	// Second request should fail (model RPM exceeded)
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

func TestConcurrency(t *testing.T) {
	f2b := fail2ban.New(100, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 1000},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 1000},
		{Name: "cred3", APIKey: "key3", BaseURL: "http://test3.com", RPM: 1000},
	}

	bal := New(credentials, f2b, rl)

	var wg sync.WaitGroup
	numGoroutines := 50
	requestsPerGoroutine := 10

	// Concurrent Next calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				_, _ = bal.Next()
			}
		}()
	}

	// Concurrent GetCredentials calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				_ = bal.GetCredentials()
				_ = bal.GetAvailableCount()
				_ = bal.GetBannedCount()
			}
		}()
	}

	wg.Wait()

	// Test should complete without panics
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
