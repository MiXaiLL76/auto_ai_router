package config

import (
	"fmt"
	"strings"
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
    credential_denylist:
      - " provider-a "
      - provider-b
`), &cfg)

	require.NoError(t, err)
	require.Len(t, cfg.OrganizationPolicies, 1)
	policy := cfg.OrganizationPolicies[0]
	assert.Equal(t, "org-1", policy.OrganizationID)
	assert.True(t, policy.AllowlistSet)
	assert.Empty(t, policy.ModelAllowlist)
	assert.Equal(t, map[string]string{"public/model": "route-model"}, policy.ModelMappings)
	assert.Equal(t, []string{"provider-a", "provider-b"}, policy.CredentialDenylist)
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

func TestOrganizationPolicyConfig_RejectsDuplicateCredentialDenylistEntry(t *testing.T) {
	var cfg Config
	err := yaml.Unmarshal([]byte(`
organization_policies:
  - organization_id: org-1
    price_profile_id: profile-1
    model_prices_link: /tmp/prices.json
    credential_denylist:
      - provider-a
      - " provider-a "
`), &cfg)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate credential_denylist entry")
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

	err = ValidateOrganizationPolicies([]OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    "/tmp/prices.json",
		CredentialDenylist: []string{""},
	}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty credential name")

	validDenylist := make([]string, maxOrganizationCredentialDenylistEntries)
	for index := range validDenylist {
		validDenylist[index] = fmt.Sprintf("credential-%04d", index)
	}
	err = ValidateOrganizationPolicies([]OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    "/tmp/prices.json",
		CredentialDenylist: validDenylist,
	}})
	require.NoError(t, err)

	err = ValidateOrganizationPolicies([]OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    "/tmp/prices.json",
		CredentialDenylist: append(validDenylist, "credential-overflow"),
	}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "credential_denylist exceeds 1024 entries")

	largeDenylist := make([]string, 300)
	for index := range largeDenylist {
		prefix := fmt.Sprintf("%03d-", index)
		largeDenylist[index] = prefix + strings.Repeat("x", maxOrganizationCredentialNameBytes-len(prefix))
	}
	err = ValidateOrganizationPolicies([]OrganizationPolicyConfig{{
		OrganizationID:     "org-1",
		PriceProfileID:     "profile-1",
		ModelPricesLink:    "/tmp/prices.json",
		CredentialDenylist: largeDenylist,
	}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "credential_denylist exceeds 65536 bytes")
}

func TestOrganizationPolicyConfig_DefaultPricing(t *testing.T) {
	for _, tt := range []struct {
		name      string
		fields    string
		wantError bool
	}{
		{name: "denylist", fields: "credential_denylist: [provider-a]"},
		{name: "profile only", fields: "price_profile_id: custom", wantError: true},
		{name: "source only", fields: "model_prices_link: /tmp/prices.json", wantError: true},
		{name: "empty allowlist", fields: "model_allowlist: []", wantError: true},
		{name: "allowlist", fields: "model_allowlist: [public/model]", wantError: true},
		{name: "mapping", fields: "model_mappings: {public/model: route-a}", wantError: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var policy OrganizationPolicyConfig
			require.NoError(t, yaml.Unmarshal([]byte("organization_id: org-1\n"+tt.fields+"\n"), &policy))
			err := ValidateOrganizationPolicies([]OrganizationPolicyConfig{policy})
			if tt.wantError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, []string{"provider-a"}, policy.CredentialDenylist)
			}
		})
	}
}

func TestOrganizationPolicyConfig_ResponseCostMode(t *testing.T) {
	cases := []struct {
		doc      string
		wantMode ResponseCostMode
		wantBool bool
	}{
		{"organization_id: o\nresponse_cost_mode: off\n", ResponseCostModeOff, false},
		{"organization_id: o\nresponse_cost_mode: total\n", ResponseCostModeTotal, true},
		{"organization_id: o\nresponse_cost_mode: breakdown\n", ResponseCostModeBreakdown, true},
		{"organization_id: o\n", ResponseCostModeOff, false},
		// legacy boolean
		{"organization_id: o\nreturn_response_cost: true\n", ResponseCostModeTotal, true},
		{"organization_id: o\nreturn_response_cost: false\n", ResponseCostModeOff, false},
		// mode wins over the legacy boolean
		{"organization_id: o\nreturn_response_cost: true\nresponse_cost_mode: breakdown\n", ResponseCostModeBreakdown, true},
	}
	for _, c := range cases {
		var p OrganizationPolicyConfig
		if err := yaml.Unmarshal([]byte(c.doc), &p); err != nil {
			t.Fatalf("unmarshal %q: %v", c.doc, err)
		}
		if p.ResponseCostMode != c.wantMode {
			t.Fatalf("%q: mode=%q want %q", c.doc, p.ResponseCostMode, c.wantMode)
		}
		if p.ReturnResponseCost != c.wantBool {
			t.Fatalf("%q: bool=%v want %v", c.doc, p.ReturnResponseCost, c.wantBool)
		}
	}

	// an unknown mode is rejected
	var p OrganizationPolicyConfig
	if err := yaml.Unmarshal([]byte("organization_id: o\nresponse_cost_mode: everything\n"), &p); err == nil {
		t.Fatalf("unknown response_cost_mode must be rejected")
	}
}

func TestOrganizationPolicyConfig_ResponseCostInBody(t *testing.T) {
	cases := []struct {
		doc  string
		want bool
	}{
		{"organization_id: o\nresponse_cost_in_body: true\n", true},
		{"organization_id: o\nresponse_cost_in_body: false\n", false},
		{"organization_id: o\n", false},
		// independent of response_cost_mode: both can be set together, or
		// the body form alone with no header mode at all.
		{"organization_id: o\nresponse_cost_mode: total\nresponse_cost_in_body: true\n", true},
	}
	for _, c := range cases {
		var p OrganizationPolicyConfig
		if err := yaml.Unmarshal([]byte(c.doc), &p); err != nil {
			t.Fatalf("unmarshal %q: %v", c.doc, err)
		}
		if p.ReturnResponseCostInBody != c.want {
			t.Fatalf("%q: ReturnResponseCostInBody=%v want %v", c.doc, p.ReturnResponseCostInBody, c.want)
		}
	}
}
