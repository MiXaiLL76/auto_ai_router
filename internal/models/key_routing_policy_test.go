package models

import (
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func keyRoutingCatalog(t *testing.T) (*Manager, []config.CredentialConfig, config.KeyRoutingPolicyConfig) {
	t.Helper()
	t.Setenv("TEST_POLICY_KEY", strings.Repeat("a", 64))
	credentials := []config.CredentialConfig{{Name: "provider-a", Type: config.ProviderTypeOpenAI}, {Name: "provider-b", Type: config.ProviderTypeOpenAI}}
	var routes []config.ModelRPMConfig
	for _, credential := range credentials {
		for _, name := range []string{"model-a", "model-b"} {
			routes = append(routes, config.ModelRPMConfig{Name: name, Credential: credential.Name})
		}
	}
	manager := New(testhelpers.NewTestLogger(), 100, routes)
	manager.LoadModelsFromConfig(credentials)
	manager.SetModelAliases(map[string]string{"vendor/model-a": "model-a"})
	manager.SetAcceptedModelAliases(map[string]string{"friendly-name": "vendor/model-a"})
	policy := config.KeyRoutingPolicyConfig{ID: "restricted", KeyHash: "os.environ/TEST_POLICY_KEY", Rules: []config.KeyRoutingRuleConfig{{Models: []string{"vendor/model-a"}, CredentialAllowlist: []string{"provider-a"}}}}
	return manager, credentials, policy
}

func TestKeyRoutingPolicyMatching(t *testing.T) {
	manager, credentials, policy := keyRoutingCatalog(t)
	registry, err := LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, manager)
	require.NoError(t, err)
	for _, public := range []string{"model-a", "vendor/model-a", "friendly-name"} {
		visibility, id := registry.Restrict(scope.PublicContext(), strings.Repeat("a", 64), public, "model-a", "model-a")
		assert.Equal(t, "restricted", id)
		assert.True(t, visibility.AllowsCredential("provider-a"))
		assert.False(t, visibility.AllowsCredential("provider-b"))
	}
	for _, test := range []struct{ hash, model string }{{strings.Repeat("b", 64), "model-a"}, {strings.Repeat("a", 64), "model-b"}} {
		visibility, id := registry.Restrict(scope.PublicContext(), test.hash, test.model, test.model, test.model)
		assert.Empty(t, id)
		assert.False(t, visibility.HasCredentialRestriction())
	}
	// An organization mapping can change the route without removing the
	// restriction attached to the admitted public model.
	visibility, id := registry.Restrict(scope.PublicContext(), strings.Repeat("a", 64), "vendor/model-a", "model-b", "model-b")
	assert.Equal(t, "restricted", id)
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestKeyRoutingPolicyOverlapsIntersect(t *testing.T) {
	manager, credentials, policy := keyRoutingCatalog(t)
	policy.Rules = append(policy.Rules, config.KeyRoutingRuleConfig{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-b"}})
	registry, err := LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, manager)
	require.NoError(t, err)
	visibility, id := registry.Restrict(scope.AdminContext(), strings.Repeat("a", 64), "model-a", "model-a", "model-a")
	assert.NotEmpty(t, id)
	assert.True(t, visibility.HasCredentialRestriction())
	assert.False(t, visibility.AllowsCredential("provider-a"))
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestKeyRoutingPolicyCatalogValidation(t *testing.T) {
	manager, credentials, policy := keyRoutingCatalog(t)
	_, err := LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, nil)
	require.Error(t, err)
	policy.Rules[0].Models = []string{"missing-model"}
	_, err = LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, manager)
	require.ErrorContains(t, err, "no permitted direct route")
	policy.Rules[0].Models = []string{"friendly-name"}
	manager.SetAcceptedModelAliases(map[string]string{"friendly-name": "missing-model"})
	_, err = LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, manager)
	require.ErrorContains(t, err, "not uniquely routable")
	assert.NotContains(t, err.Error(), strings.Repeat("a", 64))
}

func TestKeyRoutingPolicyFollowsAliasRefreshAndKeepsOriginalIdentity(t *testing.T) {
	manager, credentials, policy := keyRoutingCatalog(t)
	registry, err := LoadKeyRoutingPolicies([]config.KeyRoutingPolicyConfig{policy}, credentials, manager)
	require.NoError(t, err)
	manager.SetModelAliases(map[string]string{"vendor/model-a": "model-b"})
	visibility, _ := registry.Restrict(scope.PublicContext(), strings.Repeat("a", 64), "model-b", "model-b", "model-b")
	assert.False(t, visibility.AllowsCredential("provider-b"))
	visibility, _ = registry.Restrict(scope.PublicContext(), strings.Repeat("a", 64), "model-a", "model-a", "model-a")
	assert.False(t, visibility.AllowsCredential("provider-b"))
}
