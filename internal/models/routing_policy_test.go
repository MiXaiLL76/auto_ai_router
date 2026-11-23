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

func TestKeyRoutingPolicySchemaAndLegacyCompatibility(t *testing.T) {
	for _, legacy := range []interface{}{nil, "pool-a", 123, []string{}, []string{"pool-a"}} {
		policy, err := ParseKeyRoutingPolicy(map[string]interface{}{RoutingProfilesField: legacy}, nil, nil)
		require.NoError(t, err)
		assert.Nil(t, policy, "profiles alone keep legacy behavior")
	}
	for name, value := range map[string]interface{}{
		"null": nil, "string": "model-a", "empty": []string{}, "mixed": []interface{}{"model-a", 1},
		"blank": []string{""}, "space": []string{" model-a"}, "wildcard": []string{"*"}, "oversize": make([]string, 65),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ParseKeyRoutingPolicy(map[string]interface{}{RoutingModelsField: value, RoutingProfilesField: []string{"pool-a"}}, nil, nil)
			require.ErrorIs(t, err, ErrRoutingPolicy)
		})
	}
	// A selector without a profile list names no pool. Failing the key here would
	// take down models it never mentioned, so it degrades to ordinary routing.
	for _, absent := range []interface{}{nil, []string{}} {
		policy, err := ParseKeyRoutingPolicy(map[string]interface{}{RoutingModelsField: []string{"model-a"}, RoutingProfilesField: absent}, nil, nil)
		require.NoError(t, err)
		assert.Nil(t, policy, "a selector without profiles selects nothing")
	}
	// A present profile list of the wrong shape is still an operator error.
	for _, pools := range []interface{}{"pool-a", []interface{}{"pool-a", 1}, []string{" invalid"}, make([]string, 17)} {
		_, err := ParseKeyRoutingPolicy(map[string]interface{}{RoutingModelsField: []string{"model-a"}, RoutingProfilesField: pools}, nil, nil)
		require.Error(t, err)
	}
	valid := map[string]interface{}{RoutingModelsField: []interface{}{"model-a"}, RoutingProfilesField: []interface{}{"pool-a"}}
	policy, err := ParseKeyRoutingPolicy(valid, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, &RoutingPolicy{Models: []string{"model-a"}, Pools: []string{"pool-a"}}, policy)
	_, err = ParseKeyRoutingPolicy(nil, valid, nil)
	require.Error(t, err)
	_, err = ParseKeyRoutingPolicy(nil, nil, valid)
	require.Error(t, err)
}

func TestRoutingPoolAliasRefreshAndRestart(t *testing.T) {
	creds := []config.CredentialConfig{{Name: "a"}, {Name: "b"}}
	manager := New(testhelpers.NewTestLogger(), 100, nil)
	entries := []config.ModelRPMConfig{
		{Name: "model-a", Model: "wire-a", Credential: "a", Pools: []string{"pool-a"}},
		{Name: "bridge", Model: "wire-a", Credential: "a"},
		{Name: "bridge", Model: "wire-b", Credential: "b"},
		{Name: "alternate", Model: "wire-b", Credential: "b"},
		{Name: "other", Model: "wire-a", Credential: "b", Pools: []string{"pool-b"}},
	}
	manager.UpdateDBModels(entries, nil, creds)
	policy := &RoutingPolicy{Models: []string{"model-a"}, Pools: []string{"pool-a"}}
	for _, name := range []string{"model-a", "bridge", "alternate"} {
		v, err := manager.RestrictToModelPools(scope.PublicContext(), policy, name, name)
		require.NoError(t, err)
		assert.True(t, v.AllowsCredential("a"))
		assert.False(t, v.AllowsCredential("b"))
	}
	v, err := manager.RestrictToModelPools(scope.PublicContext(), policy, "other", "other")
	require.NoError(t, err)
	assert.False(t, v.HasCredentialRestriction(), "same wire name on a different account alone is not an alias")
	// Removing the alias bridge must not detach a known alias on hot refresh.
	manager.UpdateDBModels([]config.ModelRPMConfig{entries[0], entries[3], entries[4]}, nil, creds)
	v, err = manager.RestrictToModelPools(scope.PublicContext(), policy, "alternate", "alternate")
	require.NoError(t, err)
	assert.False(t, v.AllowsCredential("b"))
	// Explicit model scope survives loss of every pool member AND a restart.
	changed := []config.ModelRPMConfig{{Name: "model-a", Credential: "a"}, {Name: "other", Credential: "b", Pools: []string{"pool-a"}}}
	manager.UpdateDBModels(changed, nil, creds)
	for _, m := range []*Manager{manager, New(testhelpers.NewTestLogger(), 100, changed)} {
		v, err = m.RestrictToModelPools(scope.AdminContext(), policy, "model-a", "model-a")
		require.NoError(t, err)
		assert.True(t, v.HasCredentialRestriction())
		assert.False(t, v.AllowsCredential("a"))
		assert.False(t, v.AllowsCredential("b"))
		v, err = m.RestrictToModelPools(scope.PublicContext(), policy, "other", "other")
		require.NoError(t, err)
		assert.False(t, v.HasCredentialRestriction())
	}
}

func TestRoutingPoolMultiplePoolsNarrowExistingScope(t *testing.T) {
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "model-a", Credential: "a", Pools: []string{"one"}},
		{Name: "model-a", Credential: "b", Pools: []string{"two"}},
		{Name: "model-a", Pools: []string{"global"}},
	})
	p := &RoutingPolicy{Models: []string{"model-a"}, Pools: []string{"one", "two"}}
	v, err := m.RestrictToModelPools(scope.PublicContext().RestrictCredentials([]string{"b", "c"}), p, "model-a", "model-a")
	require.NoError(t, err)
	assert.False(t, v.AllowsCredential("a"))
	assert.True(t, v.AllowsCredential("b"))
	assert.False(t, v.AllowsCredential("c"))
	p.Pools = []string{"global"}
	v, err = m.RestrictToModelPools(scope.PublicContext(), p, "model-a", "model-a")
	require.NoError(t, err)
	assert.False(t, v.AllowsCredential("a"), "global tags cannot authorize arbitrary accounts in strict mode")
}

func BenchmarkRoutingPoolLookup(b *testing.B) {
	var entries []config.ModelRPMConfig
	for c := 0; c < 50; c++ {
		for i := 0; i < 200; i++ {
			entries = append(entries, config.ModelRPMConfig{Name: fmt.Sprintf("model-%d", i), Credential: fmt.Sprintf("account-%d", c), Pools: []string{"pool-a"}})
		}
	}
	m := New(testhelpers.NewTestLogger(), 100, entries)
	p := &RoutingPolicy{Models: []string{"model-0"}, Pools: []string{"pool-a"}}
	for _, tc := range []struct {
		name   string
		policy *RoutingPolicy
	}{{"unbound", nil}, {"restricted", p}} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = m.RestrictToModelPools(scope.PublicContext(), tc.policy, "model-0", "model-0")
			}
		})
	}
}

func TestRoutingPoolOrganizationMappingCannotReuseOriginalMembership(t *testing.T) {
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "public-model", Credential: "a", Pools: []string{"one"}},
		{Name: "mapped-model", Credential: "b", Pools: []string{"two"}},
	})
	p := &RoutingPolicy{Models: []string{"public-model"}, Pools: []string{"one"}}
	v, err := m.RestrictToModelPools(scope.PublicContext(), p, "public-model", "mapped-model")
	require.NoError(t, err)
	assert.True(t, v.HasCredentialRestriction(), "organization mapping retains the public model restriction")
	assert.False(t, v.AllowsCredential("a"), "original model membership does not authorize the mapped model")
	assert.False(t, v.AllowsCredential("b"), "the mapped deployment is outside the permitted pool")
}

// A re-pointed alias must not merge two live models: otherwise membership of the
// new target authorises the old one, which widens the restriction this feature
// exists to enforce. Removing a bridge, by contrast, must stay closed.
func TestRoutingPoolAliasRepointDoesNotWiden(t *testing.T) {
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "model-a", Credential: "a", Pools: []string{"one"}},
		{Name: "model-a", Credential: "b", Pools: []string{"one"}},
		{Name: "model-b", Credential: "b", Pools: []string{"two"}},
	})
	m.SetModelAliases(map[string]string{"foo": "model-a"})
	m.SetModelAliases(map[string]string{"foo": "model-b"})

	policy := &RoutingPolicy{Models: []string{"model-a"}, Pools: []string{"two"}}
	vis, err := m.RestrictToModelPools(scope.PublicContext(), policy, "model-a", "model-a")
	require.NoError(t, err)
	assert.False(t, vis.AllowsCredential("a"), "model-a is not in pool two")
	assert.False(t, vis.AllowsCredential("b"), "model-b's pool must not authorise model-a")
}

func TestRoutingPoolDroppedBridgeStaysClosed(t *testing.T) {
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "model-a", Credential: "a", Pools: []string{"one"}},
		{Name: "alias", Credential: "a", Pools: []string{"one"}},
		{Name: "model-b", Credential: "b"},
	})
	m.SetModelAliases(map[string]string{"alias": "model-a"})
	m.SetModelAliases(nil)

	policy := &RoutingPolicy{Models: []string{"alias"}, Pools: []string{"one"}}
	vis, err := m.RestrictToModelPools(scope.PublicContext(), policy, "alias", "alias")
	require.NoError(t, err)
	assert.True(t, vis.AllowsCredential("a"), "the alias still resolves to its pooled deployment")
	assert.False(t, vis.AllowsCredential("b"), "dropping the bridge must not reopen routing")
}

// The soft lookup must not scan the whole catalog: a large catalog is the point
// of the reverse index. Kept as a benchmark so a future change that reintroduces
// a full scan shows up as a thousandfold jump, not a subtle slowdown.
func BenchmarkCredentialsInPoolsLargeCatalog(b *testing.B) {
	m := New(testhelpers.NewTestLogger(), 100, nil)
	credentials := make([]config.CredentialConfig, 0, 10000)
	entries := make([]config.ModelRPMConfig, 0, 10000)
	for i := 0; i < 10000; i++ {
		name := fmt.Sprintf("c%d", i)
		credentials = append(credentials, config.CredentialConfig{Name: name})
		entries = append(entries, config.ModelRPMConfig{Name: fmt.Sprintf("model-%d", i), Credential: name, Pools: []string{"pool-01"}})
	}
	m.UpdateDBModels(entries, nil, credentials)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.CredentialsInPools("model-5000", []string{"pool-01"})
	}
}
