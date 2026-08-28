package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestOrganizationPolicyConfig_UnmarshalStrictFieldsAndPresence(t *testing.T) {
	var cfg Config
	err := yaml.Unmarshal([]byte(`
organization_policies:
  - organization_id: " org-1 "
    price_profile_id: profile-1
    model_prices_link: /tmp/prices.json
    model_allowlist: []
    model_mappings:
      public/model: route-model
`), &cfg)

	require.NoError(t, err)
	require.Len(t, cfg.OrganizationPolicies, 1)
	policy := cfg.OrganizationPolicies[0]
	assert.Equal(t, "org-1", policy.OrganizationID)
	assert.True(t, policy.AllowlistSet)
	assert.Empty(t, policy.ModelAllowlist)
	assert.Equal(t, map[string]string{"public/model": "route-model"}, policy.ModelMappings)
}

func TestOrganizationPolicyConfig_RejectsUnknownField(t *testing.T) {
	var cfg Config
	err := yaml.Unmarshal([]byte(`
organization_policies:
  - organization_id: org-1
    price_profile_id: profile-1
    model_prices_link: /tmp/prices.json
    unsupported: true
`), &cfg)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown field")
}

func TestOrganizationPolicyConfig_RejectsDuplicateMappingKey(t *testing.T) {
	var cfg Config
	err := yaml.Unmarshal([]byte(`
organization_policies:
  - organization_id: org-1
    price_profile_id: profile-1
    model_prices_link: /tmp/prices.json
    model_mappings:
      public/model: target-a
      public/model: target-b
`), &cfg)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate model mapping key")
}

func TestValidateOrganizationPolicies(t *testing.T) {
	err := ValidateOrganizationPolicies([]OrganizationPolicyConfig{
		{
			OrganizationID:  "org-1",
			PriceProfileID:  "profile-1",
			ModelPricesLink: "/tmp/prices.json",
		},
		{
			OrganizationID:  "org-1",
			PriceProfileID:  "profile-2",
			ModelPricesLink: "/tmp/prices-2.json",
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate organization_id")

	err = ValidateOrganizationPolicies([]OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: "/tmp/prices.json",
		AllowlistSet:    true,
		ModelAllowlist:  []string{"public/allowed"},
		ModelMappings:   map[string]string{"public/other": "target"},
	}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "outside model_allowlist")
}
