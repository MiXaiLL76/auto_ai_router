package models

import (
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func routingProfileCatalog(t *testing.T) (*Manager, []config.CredentialConfig, config.RoutingProfileConfig) {
	t.Helper()

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
	policy := config.RoutingProfileConfig{ID: "restricted", Rules: []config.RoutingRuleConfig{{Models: []string{"vendor/model-a"}, CredentialAllowlist: []string{"provider-a"}}}}
	return manager, credentials, policy
}

func TestRoutingProfileMatching(t *testing.T) {
	manager, credentials, policy := routingProfileCatalog(t)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.NoError(t, err)
	for _, public := range []string{"model-a", "vendor/model-a", "friendly-name"} {
		visibility, err := registry.Restrict(scope.PublicContext(), []string{policy.ID}, public, "model-a")
		require.NoError(t, err)
		assert.True(t, visibility.AllowsCredential("provider-a"))
		assert.False(t, visibility.AllowsCredential("provider-b"))
	}
	for _, test := range []struct {
		profiles []string
		model    string
	}{{nil, "model-a"}, {[]string{policy.ID}, "model-b"}} {
		visibility, err := registry.Restrict(scope.PublicContext(), test.profiles, test.model, test.model)
		require.NoError(t, err)
		assert.False(t, visibility.HasCredentialRestriction())
	}
	// An organization mapping can change the route without removing the
	// restriction attached to the admitted public model.
	visibility, err := registry.Restrict(scope.PublicContext(), []string{policy.ID}, "vendor/model-a", "model-b")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestRoutingProfileOverlapsIntersect(t *testing.T) {
	manager, credentials, policy := routingProfileCatalog(t)
	policy.Rules = append(policy.Rules, config.RoutingRuleConfig{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-b"}})
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.NoError(t, err)
	visibility, err := registry.Restrict(scope.AdminContext(), []string{policy.ID}, "model-a", "model-a")
	require.NoError(t, err)
	assert.True(t, visibility.HasCredentialRestriction())
	assert.False(t, visibility.AllowsCredential("provider-a"))
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestRoutingProfileCatalogValidation(t *testing.T) {
	manager, credentials, policy := routingProfileCatalog(t)
	_, err := LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, nil)
	require.Error(t, err)
	policy.Rules[0].Models = []string{"missing-model"}
	_, err = LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.ErrorContains(t, err, "no configured route")
	policy.Rules[0].Models = []string{"friendly-name"}
	manager.SetAcceptedModelAliases(map[string]string{"friendly-name": "missing-model"})
	_, err = LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.ErrorContains(t, err, "not uniquely routable")
	assert.NotContains(t, err.Error(), "missing-model")
}

func TestRoutingProfileFollowsAliasRefreshAndKeepsOriginalIdentity(t *testing.T) {
	manager, credentials, policy := routingProfileCatalog(t)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.NoError(t, err)
	manager.SetModelAliases(map[string]string{"vendor/model-a": "model-b"})
	visibility, _ := registry.Restrict(scope.PublicContext(), []string{policy.ID}, "model-b", "model-b")
	assert.False(t, visibility.AllowsCredential("provider-b"))
	visibility, _ = registry.Restrict(scope.PublicContext(), []string{policy.ID}, "model-a", "model-a")
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestRoutingProfileDeploymentAliasIsolation(t *testing.T) {

	credentials := []config.CredentialConfig{{Name: "provider-a", Type: config.ProviderTypeOpenAI}, {Name: "provider-b", Type: config.ProviderTypeOpenAI}}
	manager := New(testhelpers.NewTestLogger(), 100, nil)
	routes := []config.ModelRPMConfig{
		{Name: "model-a", Model: "deployment-a", Credential: "provider-a"},
		{Name: "bridge", Model: "deployment-a", Credential: "provider-a"},
		{Name: "bridge", Model: "deployment-b", Credential: "provider-b"},
		{Name: "alternate", Model: "deployment-b", Credential: "provider-b"},
		// Identical deployment names on different accounts are not model identities.
		{Name: "unrelated", Model: "deployment-a", Credential: "provider-b"},
	}
	manager.UpdateDBModels(routes, nil, credentials)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "pool", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a"}}}}}, credentials, manager)
	require.NoError(t, err)
	for _, name := range []string{"model-a", "bridge", "alternate"} {
		visibility, err := registry.Restrict(scope.PublicContext(), []string{"pool"}, name, name)
		require.NoError(t, err, name)
		assert.False(t, visibility.AllowsCredential("provider-b"), name)
	}
	visibility, err := registry.Restrict(scope.PublicContext(), []string{"pool"}, "unrelated", "unrelated")
	require.NoError(t, err)
	assert.False(t, visibility.HasCredentialRestriction())
	// An organization mapping must not erase the original public identity.
	visibility, err = registry.Restrict(scope.PublicContext(), []string{"pool"}, "alternate", "unrelated")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-b"))
	// Catalog replacement cannot reopen a startup-equivalent name by removing
	// the bridge that originally connected it to the configured policy model.
	manager.UpdateDBModels(routes[3:], nil, credentials)
	visibility, err = registry.Restrict(scope.PublicContext(), []string{"pool"}, "alternate", "alternate")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestRoutingProfileDeploymentAliasAddedAfterStartup(t *testing.T) {
	manager, credentials, policy := routingProfileCatalog(t)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{policy}, credentials, manager)
	require.NoError(t, err)
	manager.UpdateDBModels([]config.ModelRPMConfig{{Name: "new-alias", Model: "model-a", Credential: "provider-a"}}, credentials, credentials)
	visibility, err := registry.Restrict(scope.PublicContext(), []string{policy.ID}, "new-alias", "model-b")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func BenchmarkRoutingProfileCatalog(b *testing.B) {

	var credentials []config.CredentialConfig
	var routes []config.ModelRPMConfig
	for c := 0; c < 50; c++ {
		name := fmt.Sprintf("provider-%d", c)
		credentials = append(credentials, config.CredentialConfig{Name: name, Type: config.ProviderTypeOpenAI})
		for m := 0; m < 200; m++ {
			routes = append(routes, config.ModelRPMConfig{Name: fmt.Sprintf("model-%d", m), Model: fmt.Sprintf("deployment-%d", m/2), Credential: name})
		}
	}
	manager := New(testhelpers.NewTestLogger(), 100, routes)
	manager.LoadModelsFromConfig(credentials)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{{ID: "pool", Rules: []config.RoutingRuleConfig{{Models: []string{"model-0"}, CredentialAllowlist: []string{"provider-0"}}}}}, credentials, manager)
	if err != nil {
		b.Fatal(err)
	}
	for _, test := range []struct {
		name string
		ids  []string
	}{{"matched-key", []string{"pool"}}, {"unmatched-key", nil}} {
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = registry.Restrict(scope.PublicContext(), test.ids, "model-1", "model-1")
			}
		})
	}
}

func TestRoutingProfileDenylistAndIntersection(t *testing.T) {
	manager, credentials, profile := routingProfileCatalog(t)
	profile.Rules[0].CredentialAllowlist = nil
	profile.Rules[0].CredentialDenylist = []string{"provider-b"}
	profiles := []config.RoutingProfileConfig{profile, {ID: "narrower", Rules: []config.RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-b"}}}}}
	registry, err := LoadRoutingProfiles(profiles, credentials, manager)
	require.NoError(t, err)
	visibility, err := registry.Restrict(scope.PublicContext(), []string{profile.ID}, "model-a", "model-a")
	require.NoError(t, err)
	assert.True(t, visibility.AllowsCredential("provider-a"))
	assert.False(t, visibility.AllowsCredential("provider-b"))
	visibility, err = registry.Restrict(scope.AdminContext(), []string{profile.ID, "narrower"}, "model-a", "model-a")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-a"))
	assert.False(t, visibility.AllowsCredential("provider-b"))
	// An allowlist never overrides a matching denylist, even in the same rule.
	profiles[0].Rules[0].CredentialAllowlist = []string{"provider-b"}
	registry, err = LoadRoutingProfiles(profiles, credentials, manager)
	require.NoError(t, err)
	visibility, err = registry.Restrict(scope.PublicContext(), []string{profile.ID}, "model-a", "model-a")
	require.NoError(t, err)
	assert.False(t, visibility.AllowsCredential("provider-b"))
}

func TestRoutingProfileMissingBindingFailsClosed(t *testing.T) {
	manager, credentials, profile := routingProfileCatalog(t)
	registry, err := LoadRoutingProfiles([]config.RoutingProfileConfig{profile}, credentials, manager)
	require.NoError(t, err)
	empty, err := LoadRoutingProfiles(nil, nil, nil)
	require.NoError(t, err)
	for _, r := range []*RoutingProfileRegistry{registry, empty, nil} {
		visibility, err := r.Restrict(scope.AdminContext(), []string{"missing"}, "model-b", "model-b")
		require.ErrorIs(t, err, ErrInvalidRoutingProfile)
		assert.True(t, visibility.HasCredentialRestriction())
		assert.False(t, visibility.AllowsCredential("provider-a"))
		visibility, err = r.Restrict(scope.PublicContext(), nil, "model-a", "model-a")
		require.NoError(t, err)
		assert.False(t, visibility.HasCredentialRestriction())
	}
}
