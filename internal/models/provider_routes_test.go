package models

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderRoutesPreserveAlternativesAndIgnoreTransientAvailability(t *testing.T) {
	cred := config.CredentialConfig{Name: "air-child", Type: config.ProviderTypeAIR, Scopes: []string{"local"}}
	health := &httputil.ProxyHealthResponse{
		Credentials: map[string]httputil.CredentialHealthStats{
			"cometapi01": {Type: "openai", IsBanned: true},
			"middle":     {Type: "air", IsProxyLike: true},
			"fallback":   {Type: "openai", IsFallback: true},
		},
		Models: map[string]httputil.ModelHealthStats{
			"direct": {Model: "claude-sonnet-4.6", Credential: "cometapi01", IsBanned: true, LimitRPM: 1, CurrentRPM: 1},
			"relay": {Model: "claude-sonnet-4.6", Credential: "middle", ProviderRoutes: map[string]*scope.Expression{
				"requesty-claude": scope.FromScopes([]string{"team-a"}, nil),
			}},
			"other-path": {Model: "claude-sonnet-4.6", Credential: "middle", ProviderRoutes: map[string]*scope.Expression{
				"requesty-claude": scope.FromScopes([]string{"team-b"}, nil),
			}},
			"fallback": {Model: "claude-sonnet-4.6", Credential: "fallback"},
			"different-model": {Model: "gpt-image-2", Credential: "cometapi01", ProviderRoutes: map[string]*scope.Expression{
				"image-provider": nil,
			}},
		},
	}
	m := New(testhelpers.NewTestLogger(), 100, nil)
	m.remoteModelsCache[cred.Name] = remoteModelCache{credential: cred, health: health}
	routes := m.ProviderRoutesForModel(&cred, "claude-sonnet-4.6")
	require.Len(t, routes, 2)
	assert.Contains(t, routes, "cometapi01", "bans and exhausted counters do not change stable eligibility")
	require.Contains(t, routes, "requesty-claude")
	assert.True(t, scope.NewContext([]string{"local", "team-a"}, nil).AllowsExpression(routes["requesty-claude"]))
	assert.True(t, scope.NewContext([]string{"local", "team-b"}, nil).AllowsExpression(routes["requesty-claude"]))
	assert.False(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(routes["requesty-claude"]), "local scope remains required")
	assert.False(t, scope.NewContext([]string{"local"}, nil).AllowsExpression(routes["requesty-claude"]), "leaf scope remains required")

	delete(routes, "cometapi01")
	routes["requesty-claude"].Alternatives = nil
	fresh := m.ProviderRoutesForModel(&cred, "claude-sonnet-4.6")
	assert.Len(t, fresh, 2, "callers must not mutate the health snapshot")
	assert.True(t, scope.NewContext([]string{"local", "team-a"}, nil).AllowsExpression(fresh["requesty-claude"]))

	cred.IsFallback = true
	m.remoteModelsCache[cred.Name] = remoteModelCache{credential: cred, health: health}
	assert.Contains(t, m.ProviderRoutesForModel(&cred, "claude-sonnet-4.6"), "fallback")

	cred.APIKey = "changed-key"
	assert.Contains(t, m.ProviderRoutesForModel(&cred, "claude-sonnet-4.6"), "", "do not reuse another provider identity's metadata")
}

func TestProviderRoutesLegacyAndEmptyMetadata(t *testing.T) {
	for _, tc := range []struct {
		name, credentialType string
		providerRoutes       map[string]*scope.Expression
		wantName             string
		empty                bool
	}{
		{name: "legacy_direct_provider", credentialType: "openai", wantName: "provider"},
		{name: "legacy_relay_unknown", credentialType: "air", wantName: ""},
		{name: "legacy_proxy_unknown", credentialType: "proxy", wantName: ""},
		{name: "known_empty", credentialType: "air", providerRoutes: map[string]*scope.Expression{}, empty: true},
		{name: "partial_legacy", credentialType: "air", providerRoutes: map[string]*scope.Expression{"": nil, "cometapi01": nil}, wantName: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cred := config.CredentialConfig{Name: "air-child", Type: config.ProviderTypeAIR}
			health := httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{"provider": {Type: tc.credentialType}},
				Models: map[string]httputil.ModelHealthStats{"provider:model": {
					Model: "model", Credential: "provider", RealCredential: "display-only",
					ProviderRoutes: tc.providerRoutes,
				}},
			}
			// Empty and absent metadata must remain distinct across a chain hop.
			encoded, err := json.Marshal(health)
			require.NoError(t, err)
			var decoded httputil.ProxyHealthResponse
			require.NoError(t, json.Unmarshal(encoded, &decoded))
			m := New(testhelpers.NewTestLogger(), 100, nil)
			m.remoteModelsCache[cred.Name] = remoteModelCache{credential: cred, health: &decoded}
			routes := m.ProviderRoutesForModel(&cred, "model")
			if tc.empty {
				assert.Empty(t, routes)
			} else {
				assert.Contains(t, routes, tc.wantName)
			}
			assert.NotContains(t, routes, "display-only")
			assert.Empty(t, m.ProviderRoutesForModel(&cred, "unknown-model"))
			m.remoteModelsCache[cred.Name] = remoteModelCache{credential: cred}
			assert.Contains(t, m.ProviderRoutesForModel(&cred, "model"), "", "missing/legacy discovery retains compatibility")
		})
	}
}

func TestProviderRoutesComposeLocalRestrictionsOnce(t *testing.T) {
	leafScope := scope.FromScopes([]string{"team-a"}, []string{"leaf-blocked"})
	for _, tc := range []struct {
		name, upstreamType, leafName string
		leaves                       map[string]*scope.Expression
		modelScope                   *scope.Expression
	}{
		{
			name: "complete_leaf_path", upstreamType: "air", leafName: "provider",
			leaves:     map[string]*scope.Expression{"provider": leafScope},
			modelScope: scope.Or(leafScope, scope.FromScopes([]string{"team-b"}, nil)),
		},
		{name: "legacy_direct_model_scope", upstreamType: "openai", leafName: "provider", modelScope: leafScope},
		{name: "legacy_relay_model_scope", upstreamType: "air", leafName: "", modelScope: leafScope},
		{name: "legacy_credential_scope", upstreamType: "openai", leafName: "provider"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cred := config.CredentialConfig{
				Name: "air-child", Type: config.ProviderTypeAIR,
				Scopes: []string{"local"}, DeniedScopes: []string{"local-blocked"},
			}
			health := &httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"provider": {Type: tc.upstreamType, ScopeExpression: leafScope},
				},
				Models: map[string]httputil.ModelHealthStats{"provider:model": {
					Model: "model", Credential: "provider", ScopeExpression: tc.modelScope, ProviderRoutes: tc.leaves,
				}},
			}
			m := New(testhelpers.NewTestLogger(), 100, nil)
			m.remoteModelsCache[cred.Name] = remoteModelCache{credential: cred, health: health}
			routes := m.ProviderRoutesForModel(&cred, "model")
			require.Contains(t, routes, tc.leafName)
			path := routes[tc.leafName]
			require.NotNil(t, path)
			assert.Len(t, path.Alternatives, 1, "unrelated providers must not multiply a leaf's alternatives")
			assert.True(t, scope.NewContext([]string{"local", "team-a"}, nil).AllowsExpression(path))
			for _, scopes := range [][]string{
				{"team-a"}, {"local"}, {"local", "team-b"},
				{"local", "team-a", "local-blocked"}, {"local", "team-a", "leaf-blocked"},
			} {
				assert.False(t, scope.NewContext(scopes, nil).AllowsExpression(path), "scopes=%v", scopes)
			}
		})
	}
}
