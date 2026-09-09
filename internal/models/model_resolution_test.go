package models

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveModelSharedPrecedence(t *testing.T) {
	cred := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI}
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "openai/gpt-image-2", Model: "gpt-image-2"},
		{Name: "gpt-image-2-vsellm", Model: "gpt-image-2"},
		{Name: "accepted-and-direct", Model: "direct-real"},
	})
	m.LoadModelsFromConfig([]config.CredentialConfig{cred})
	m.SetModelAliases(map[string]string{
		"openai/gpt-image-2": "gpt-image-2-vsellm",
		"public-image":       "gpt-image-2-vsellm",
		"orphan":             "missing",
		"cycle-a":            "cycle-b",
		"cycle-b":            "cycle-a",
	})
	m.SetAcceptedModelAliases(map[string]string{
		"image":               "openai/gpt-image-2",
		"accepted-and-direct": "openai/gpt-image-2",
		"accepted-orphan":     "missing",
	})
	m.SetClientModelIDs([]string{"openai/gpt-image-2", "public-image", "orphan", "cycle-a"})
	price := &ModelPrice{InputCostPerToken: 0.25}
	org := &OrganizationPolicy{prices: map[string]*ModelPrice{"openai/gpt-image-2": price}}
	mappedOrg := &OrganizationPolicy{
		mappings: map[string]string{"org-image": "public-image", "bad-target": "image"},
		prices:   map[string]*ModelPrice{"org-image": price},
	}
	for _, tc := range []struct {
		name, requested, canonical, route, real string
		master                                  bool
		policy                                  *OrganizationPolicy
		unavailable                             bool
	}{
		{name: "direct_beats_model_alias", requested: "openai/gpt-image-2", canonical: "openai/gpt-image-2", route: "openai/gpt-image-2", real: "gpt-image-2"},
		{name: "master_direct_beats_model_alias", requested: "openai/gpt-image-2", canonical: "openai/gpt-image-2", route: "openai/gpt-image-2", real: "gpt-image-2", master: true},
		{name: "organization_uses_same_precedence", requested: "openai/gpt-image-2", canonical: "openai/gpt-image-2", route: "openai/gpt-image-2", real: "gpt-image-2", policy: org},
		{name: "accepted_alias", requested: "image", canonical: "openai/gpt-image-2", route: "openai/gpt-image-2", real: "gpt-image-2"},
		{name: "client_accepted_beats_direct", requested: "accepted-and-direct", canonical: "openai/gpt-image-2", route: "openai/gpt-image-2", real: "gpt-image-2"},
		{name: "master_exact_bypasses_accepted", requested: "accepted-and-direct", canonical: "accepted-and-direct", route: "accepted-and-direct", real: "direct-real", master: true},
		{name: "ordinary_model_alias", requested: "public-image", canonical: "public-image", route: "gpt-image-2-vsellm", real: "gpt-image-2"},
		{name: "organization_mapping", requested: "org-image", canonical: "public-image", route: "gpt-image-2-vsellm", real: "gpt-image-2", policy: mappedOrg},
		{name: "internal_route_hidden", requested: "gpt-image-2-vsellm", unavailable: true},
		{name: "internal_route_master", requested: "gpt-image-2-vsellm", canonical: "gpt-image-2-vsellm", route: "gpt-image-2-vsellm", real: "gpt-image-2", master: true},
		{name: "orphan", requested: "orphan", unavailable: true},
		{name: "master_orphan", requested: "orphan", master: true, unavailable: true},
		{name: "cycle", requested: "cycle-a", unavailable: true},
		{name: "accepted_orphan", requested: "accepted-orphan", unavailable: true},
		{name: "organization_mapping_cannot_target_accepted", requested: "bad-target", policy: mappedOrg, unavailable: true},
		{name: "organization_allowlist", requested: "openai/gpt-image-2", policy: &OrganizationPolicy{AllowlistSet: true}, unavailable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolved, ok := m.ResolveModel(tc.requested, tc.policy, tc.master)
			require.Equal(t, !tc.unavailable, ok)
			if !ok {
				return
			}
			assert.Equal(t, tc.requested, resolved.PublicModelID)
			assert.Equal(t, tc.canonical, resolved.CanonicalModelID)
			assert.Equal(t, tc.route, resolved.ModelID)
			assert.Equal(t, tc.real, resolved.RealModelID)
			if tc.policy != nil {
				assert.Equal(t, tc.requested, resolved.PriceModelID)
				assert.Same(t, price, resolved.ModelPrice)
				organizationResolved, err := m.ResolveOrganizationModel(tc.policy, tc.requested)
				require.NoError(t, err)
				assert.Equal(t, resolved, organizationResolved)
			} else if !tc.master {
				assert.True(t, m.IsClientModelIDRoutable(tc.requested))
			}
		})
	}
}

func TestModelScopeMatchesResolvedAliasPrecedence(t *testing.T) {
	cred := config.CredentialConfig{Name: "provider", Type: config.ProviderTypeOpenAI}
	m := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "A"}, {Name: "B"}, {Name: "C"}, {Name: "openai/shadowed"},
	})
	m.LoadModelsFromConfig([]config.CredentialConfig{cred})
	m.SetModelAliases(map[string]string{
		"A": "B", "alias": "B", "accepted": "B",
		"openai/shadowed": "C", "openai/live": "B",
		"anthropic/shadowed": "C",
	})
	m.SetAcceptedModelAliases(map[string]string{"accepted": "C", "anthropic/shadowed": "B"})
	for _, tc := range []struct {
		name, requested, allowed, route string
		wantAllowed                     bool
	}{
		{name: "direct_own_permission", requested: "A", allowed: "A", route: "A", wantAllowed: true},
		{name: "shadowed_target_cannot_authorize_direct", requested: "A", allowed: "B", route: "A"},
		{name: "shadowed_target_cannot_supply_wildcard", requested: "A", allowed: "openai/*", route: "A"},
		{name: "active_alias_target_permission", requested: "alias", allowed: "B", route: "B", wantAllowed: true},
		{name: "no_reverse_alias_permission", requested: "B", allowed: "alias", route: "B"},
		{name: "accepted_alias_canonical_permission", requested: "accepted", allowed: "C", route: "C", wantAllowed: true},
		{name: "accepted_alias_shadows_global_alias", requested: "accepted", allowed: "B", route: "C"},
		{name: "accepted_alias_shadows_global_wildcard", requested: "accepted", allowed: "openai/*", route: "C"},
		{name: "shadowed_alias_cannot_infer_provider", requested: "C", allowed: "openai/*", route: "C"},
		{name: "accepted_shadowed_alias_cannot_infer_provider", requested: "C", allowed: "anthropic/*", route: "C"},
		{name: "active_alias_can_infer_provider", requested: "B", allowed: "openai/*", route: "B", wantAllowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolved, ok := m.ResolveModel(tc.requested, nil, false)
			require.True(t, ok)
			assert.Equal(t, tc.route, resolved.ModelID)
			assert.Equal(t, tc.wantAllowed, m.IsModelIDAllowedByScope(tc.requested, []string{tc.allowed}))
		})
	}
}
