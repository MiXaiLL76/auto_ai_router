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
	for _, pools := range []interface{}{nil, "pool-a", []string{}, []interface{}{"pool-a", 1}, []string{" invalid"}, make([]string, 17)} {
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
