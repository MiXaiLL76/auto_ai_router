package config

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func validTestKeyRoutingPolicy(t *testing.T) KeyRoutingPolicyConfig {
	t.Helper()
	t.Setenv("TEST_ROUTING_HASH", strings.Repeat("a", 64))
	return KeyRoutingPolicyConfig{ID: "restricted-pool", KeyHash: "os.environ/TEST_ROUTING_HASH", Rules: []KeyRoutingRuleConfig{{Models: []string{"model-a"}, CredentialAllowlist: []string{"provider-a"}}}}
}

func TestKeyRoutingPolicyConfigRoundTrip(t *testing.T) {
	policy := validTestKeyRoutingPolicy(t)
	encoded, err := yaml.Marshal(Config{KeyRoutingPolicies: []KeyRoutingPolicyConfig{policy}})
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), strings.Repeat("a", 64))
	var cfg Config
	require.NoError(t, yaml.Unmarshal(encoded, &cfg))
	require.Equal(t, []KeyRoutingPolicyConfig{policy}, cfg.KeyRoutingPolicies)
	assert.NoError(t, ValidateKeyRoutingPolicies(cfg.KeyRoutingPolicies, []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}}))
}

func TestKeyRoutingPolicyConfigRejectsInvalidRules(t *testing.T) {
	cases := map[string]func(*KeyRoutingPolicyConfig){
		"empty id":                  func(p *KeyRoutingPolicyConfig) { p.ID = "" },
		"literal fingerprint":       func(p *KeyRoutingPolicyConfig) { p.KeyHash = strings.Repeat("a", 64) },
		"empty environment name":    func(p *KeyRoutingPolicyConfig) { p.KeyHash = "os.environ/" },
		"missing environment value": func(p *KeyRoutingPolicyConfig) { p.KeyHash = "os.environ/TEST_MISSING_ROUTING_HASH" },
		"no rules":                  func(p *KeyRoutingPolicyConfig) { p.Rules = nil },
		"no models":                 func(p *KeyRoutingPolicyConfig) { p.Rules[0].Models = nil },
		"wildcard model":            func(p *KeyRoutingPolicyConfig) { p.Rules[0].Models = []string{"model-*"} },
		"whitespace model":          func(p *KeyRoutingPolicyConfig) { p.Rules[0].Models = []string{" model-a"} },
		"no credentials":            func(p *KeyRoutingPolicyConfig) { p.Rules[0].CredentialAllowlist = nil },
		"unknown credential":        func(p *KeyRoutingPolicyConfig) { p.Rules[0].CredentialAllowlist = []string{"missing"} },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			policy := validTestKeyRoutingPolicy(t)
			t.Setenv("TEST_MISSING_ROUTING_HASH", "")
			change(&policy)
			err := ValidateKeyRoutingPolicies([]KeyRoutingPolicyConfig{policy}, []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}})
			require.Error(t, err)
			assert.NotContains(t, err.Error(), strings.Repeat("a", 64))
		})
	}
}

func TestKeyRoutingPolicyRejectsProxyAndDuplicateBindings(t *testing.T) {
	policy := validTestKeyRoutingPolicy(t)
	for _, kind := range []ProviderType{ProviderTypeAIR, ProviderTypeProxy} {
		assert.Error(t, ValidateKeyRoutingPolicies([]KeyRoutingPolicyConfig{policy}, []CredentialConfig{{Name: "provider-a", Type: kind}}))
	}
	credentials := []CredentialConfig{{Name: "provider-a", Type: ProviderTypeOpenAI}}
	assert.Error(t, ValidateKeyRoutingPolicies([]KeyRoutingPolicyConfig{policy, policy}, credentials))
	other := policy
	other.ID = "other-id"
	assert.Error(t, ValidateKeyRoutingPolicies([]KeyRoutingPolicyConfig{policy, other}, credentials))
	for _, invalid := range []string{"", strings.Repeat("g", 64), "private-test-value"} {
		t.Setenv("TEST_ROUTING_HASH", invalid)
		_, err := policy.ResolveKeyHash()
		require.Error(t, err)
		if invalid != "" {
			assert.NotContains(t, err.Error(), invalid)
		}
	}
}

func TestKeyRoutingPolicyRequiresDatabaseAuthentication(t *testing.T) {
	policy := validTestKeyRoutingPolicy(t)
	for _, db := range []LiteLLMDBConfig{{}, {Enabled: true}, {IsRequired: true}} {
		cfg := Config{KeyRoutingPolicies: []KeyRoutingPolicyConfig{policy}, LiteLLMDB: db}
		require.ErrorContains(t, cfg.Validate(), "key_routing_policies requires")
	}
}
