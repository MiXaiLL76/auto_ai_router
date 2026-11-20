package config

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
	"testing"
)

func validTestRoutingProfile() RoutingProfileConfig {
	return RoutingProfileConfig{ID: "pool-a", Rules: []RoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a"}}}}
}

func TestRoutingProfileConfigRoundTrip(t *testing.T) {
	for _, rule := range []RoutingRuleConfig{
		{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a"}},
		{Models: []string{"model-a"}, CredentialDenylist: []string{"provider-a"}},
	} {
		profile := RoutingProfileConfig{ID: "pool-a", Rules: []RoutingRuleConfig{rule}}
		encoded, err := yaml.Marshal(Config{RoutingProfiles: []RoutingProfileConfig{profile}})
		require.NoError(t, err)
		var cfg Config
		require.NoError(t, yaml.Unmarshal(encoded, &cfg))
		require.Equal(t, []RoutingProfileConfig{profile}, cfg.RoutingProfiles)
		require.NoError(t, ValidateRoutingProfiles(cfg.RoutingProfiles, []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}}))
	}
}

func TestRoutingProfileConfigRejectsInvalidRules(t *testing.T) {
	cases := map[string]func(*RoutingProfileConfig){
		"empty id":                   func(p *RoutingProfileConfig) { p.ID = "" },
		"invalid id":                 func(p *RoutingProfileConfig) { p.ID = "pool/private" },
		"no rules":                   func(p *RoutingProfileConfig) { p.Rules = nil },
		"no models":                  func(p *RoutingProfileConfig) { p.Rules[0].Models = nil },
		"wildcard model":             func(p *RoutingProfileConfig) { p.Rules[0].Models = []string{"model-*"} },
		"whitespace model":           func(p *RoutingProfileConfig) { p.Rules[0].Models = []string{" model-a"} },
		"no filter":                  func(p *RoutingProfileConfig) { p.Rules[0].CredentialAllowlist = nil },
		"empty filter":               func(p *RoutingProfileConfig) { p.Rules[0].CredentialAllowlist = []string{} },
		"unknown allowed credential": func(p *RoutingProfileConfig) { p.Rules[0].CredentialAllowlist = []string{"private-missing"} },
		"unknown denied credential":  func(p *RoutingProfileConfig) { p.Rules[0].CredentialDenylist = []string{"private-missing"} },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			profile := validTestRoutingProfile()
			change(&profile)
			err := ValidateRoutingProfiles([]RoutingProfileConfig{profile}, []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}})
			require.Error(t, err)
			assert.NotContains(t, err.Error(), "private-missing")
		})
	}
}

func TestRoutingProfileRejectsProxyAndDuplicateIDs(t *testing.T) {
	profile := validTestRoutingProfile()
	for _, kind := range []ProviderType{ProviderTypeAIR, ProviderTypeProxy} {
		assert.Error(t, ValidateRoutingProfiles([]RoutingProfileConfig{profile}, []CredentialConfig{{Name: "provider-a", Type: kind}}))
	}
	assert.Error(t, ValidateRoutingProfiles([]RoutingProfileConfig{profile, profile}, []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}}))
}

func TestRoutingProfileRequiresDatabaseAuthentication(t *testing.T) {
	for _, db := range []LiteLLMDBConfig{{}, {Enabled: true}, {IsRequired: true}} {
		cfg := Config{RoutingProfiles: []RoutingProfileConfig{validTestRoutingProfile()}, LiteLLMDB: db}
		require.ErrorContains(t, cfg.Validate(), "routing_profiles requires")
	}
}
