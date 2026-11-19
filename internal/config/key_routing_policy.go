package config

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// KeyRoutingPolicyConfig restricts selected models of one authenticated virtual
// key to a fixed set of existing, direct provider credentials. KeyHash is an
// environment reference, never a literal key or fingerprint in the YAML.
type KeyRoutingPolicyConfig struct {
	ID      string                 `yaml:"id"`
	KeyHash string                 `yaml:"key_hash"`
	Rules   []KeyRoutingRuleConfig `yaml:"rules"`
}

type KeyRoutingRuleConfig struct {
	Models              []string `yaml:"models"`
	CredentialAllowlist []string `yaml:"credential_allowlist"`
}

// ResolveKeyHash deliberately never includes the reference or its value in an
// error. The caller may log the error during startup.
func (p KeyRoutingPolicyConfig) ResolveKeyHash() (string, error) {
	const prefix = "os.environ/"
	if !strings.HasPrefix(p.KeyHash, prefix) || len(p.KeyHash) == len(prefix) {
		return "", fmt.Errorf("key_hash must be an environment reference")
	}
	value, ok := os.LookupEnv(strings.TrimPrefix(p.KeyHash, prefix))
	if !ok || len(value) != 64 {
		return "", fmt.Errorf("key_hash environment value must be a SHA-256 fingerprint")
	}
	if _, err := hex.DecodeString(value); err != nil {
		return "", fmt.Errorf("key_hash environment value must be a SHA-256 fingerprint")
	}
	return strings.ToLower(value), nil
}

func validRoutingName(s string) bool {
	if s == "" || len(s) > 256 || strings.TrimSpace(s) != s || strings.ContainsAny(s, "*?\r\n") {
		return false
	}
	for _, r := range s {
		if unicode.IsControl(r) {
			return false
		}
	}
	return true
}

// ValidateKeyRoutingPolicies checks static references. Model/alias validation
// happens after the model catalog has loaded. Overlapping rules intersect.
func ValidateKeyRoutingPolicies(policies []KeyRoutingPolicyConfig, credentials []CredentialConfig) error {
	if len(policies) > 1024 {
		return fmt.Errorf("key_routing_policies exceeds 1024 entries")
	}
	known := make(map[string]CredentialConfig, len(credentials))
	for _, credential := range credentials {
		known[credential.Name] = credential
	}
	ids, hashes := make(map[string]bool), make(map[string]bool)
	for i, policy := range policies {
		if !validRoutingName(policy.ID) || ids[policy.ID] {
			return fmt.Errorf("key_routing_policies[%d]: id must be nonempty and unique", i)
		}
		ids[policy.ID] = true
		hash, err := policy.ResolveKeyHash()
		if err != nil {
			return fmt.Errorf("key_routing_policies[%d]: %w", i, err)
		}
		if hashes[hash] {
			return fmt.Errorf("key_routing_policies[%d]: duplicate key binding", i)
		}
		hashes[hash] = true
		if len(policy.Rules) == 0 || len(policy.Rules) > 128 {
			return fmt.Errorf("key_routing_policies[%d]: rules must contain 1 to 128 entries", i)
		}
		for j, rule := range policy.Rules {
			if len(rule.Models) == 0 || len(rule.Models) > 128 || len(rule.CredentialAllowlist) == 0 || len(rule.CredentialAllowlist) > 128 {
				return fmt.Errorf("key_routing_policies[%d].rules[%d]: models and credential_allowlist must contain 1 to 128 entries", i, j)
			}
			for _, model := range rule.Models {
				if !validRoutingName(model) {
					return fmt.Errorf("key_routing_policies[%d].rules[%d]: models must be exact nonempty names", i, j)
				}
			}
			for _, name := range rule.CredentialAllowlist {
				credential, exists := known[name]
				if !validRoutingName(name) || !exists || credential.IsProxyLike() {
					return fmt.Errorf("key_routing_policies[%d].rules[%d]: credential_allowlist must reference existing direct provider credentials", i, j)
				}
			}
		}
	}
	return nil
}
