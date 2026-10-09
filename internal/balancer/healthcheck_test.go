// Tests for the health-check integration in the balancer: dead accounts are
// skipped during selection, and the exact-pick path refuses them.
package balancer

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockHealthChecker marks configured accounts dead for the balancer tests.
type MockHealthChecker struct {
	dead map[string]bool
}

func NewMockHealthChecker(dead map[string]bool) *MockHealthChecker {
	return &MockHealthChecker{dead: dead}
}

func (m *MockHealthChecker) IsDead(name string) bool {
	v, ok := m.dead[name]
	return ok && v
}

func newHealthTestBalancer(t *testing.T, hc *MockHealthChecker) *RoundRobin {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "cred2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	bal := New(credentials, f2b, rl)
	bal.SetHealthChecker(hc)
	return bal
}

func publicScope() scope.Context {
	return scope.NewContext([]string{}, nil)
}

func TestNextSkipsDeadAccount(t *testing.T) {
	bal := newHealthTestBalancer(t, NewMockHealthChecker(map[string]bool{"cred1": true}))

	// cred1 is dead per the health-check service: selection must land on
	// cred2 every time instead of failing or looping on the dead one.
	for i := 0; i < 6; i++ {
		cred, err := bal.NextForModelScoped("model1", publicScope())
		require.NoError(t, err)
		assert.Equal(t, "cred2", cred.Name, "dead account is skipped")
	}
}

func TestNextAllDeadReturnsNoCredentials(t *testing.T) {
	bal := newHealthTestBalancer(t, NewMockHealthChecker(map[string]bool{"cred1": true, "cred2": true}))

	cred, err := bal.NextForModelScoped("model1", publicScope())
	require.Error(t, err)
	_ = cred
}

func TestNextSpecificRefusesDeadAccount(t *testing.T) {
	bal := newHealthTestBalancer(t, NewMockHealthChecker(map[string]bool{"cred1": true}))

	cred, err := bal.NextSpecificScoped("cred1", "model1", publicScope())
	require.Error(t, err, "exact pick of a dead account is refused")
	_ = cred

	// The alive account is still pickable exactly.
	cred2, err := bal.NextSpecificScoped("cred2", "model1", publicScope())
	require.NoError(t, err)
	assert.Equal(t, "cred2", cred2.Name)
}

func TestDefaultCheckerKeepsAllAlive(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	credentials := []config.CredentialConfig{
		{Name: "cred1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}
	// No SetHealthChecker call: the no-op checker must keep behavior unchanged.
	bal := New(credentials, f2b, rl)

	cred, err := bal.NextForModelScoped("model1", publicScope())
	require.NoError(t, err)
	assert.Equal(t, "cred1", cred.Name, "default checker never reports dead")
}