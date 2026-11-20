package config

import (
	"fmt"
	"strings"
	"unicode"
)

// RoutingProfileConfig defines an operator-owned provider pool. Database
// metadata binds its opaque ID; account names remain in server configuration.
type RoutingProfileConfig struct {
	ID    string              `yaml:"id"`
	Rules []RoutingRuleConfig `yaml:"rules"`
}

type RoutingRuleConfig struct {
	Models []string `yaml:"models"`
	// Empty means unspecified. Combine allow/deny to exclude individual accounts.
	CredentialAllowlist []string `yaml:"credential_allowlist,omitempty"`
	CredentialDenylist  []string `yaml:"credential_denylist,omitempty"`
}

func ValidRoutingProfileID(s string) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.') {
			return false
		}
	}
	return true
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

// ValidateRoutingProfiles checks static references. Model/alias validation
// happens after the model catalog has loaded. Matching profiles intersect.
func ValidateRoutingProfiles(profiles []RoutingProfileConfig, credentials []CredentialConfig) error {
	if len(profiles) > 1024 {
		return fmt.Errorf("routing_profiles exceeds 1024 entries")
	}
	known := make(map[string]CredentialConfig, len(credentials))
	for _, credential := range credentials {
		known[credential.Name] = credential
	}
	ids := make(map[string]bool)
	for i, profile := range profiles {
		if !ValidRoutingProfileID(profile.ID) || ids[profile.ID] {
			return fmt.Errorf("routing_profiles[%d]: id must be unique and contain 1 to 64 letters, digits, dots, hyphens or underscores", i)
		}
		ids[profile.ID] = true
		if len(profile.Rules) == 0 || len(profile.Rules) > 128 {
			return fmt.Errorf("routing_profiles[%d]: rules must contain 1 to 128 entries", i)
		}
		for j, rule := range profile.Rules {
			if len(rule.Models) == 0 || len(rule.Models) > 128 || len(rule.CredentialAllowlist) > 128 || len(rule.CredentialDenylist) > 128 || (len(rule.CredentialAllowlist) == 0 && len(rule.CredentialDenylist) == 0) {
				return fmt.Errorf("routing_profiles[%d].rules[%d]: expected 1 to 128 models and an allowlist or nonempty denylist (at most 128 entries each)", i, j)
			}
			for _, model := range rule.Models {
				if !validRoutingName(model) {
					return fmt.Errorf("routing_profiles[%d].rules[%d]: models must be exact nonempty names", i, j)
				}
			}
			for _, name := range rule.CredentialAllowlist {
				credential, exists := known[name]
				if !validRoutingName(name) || !exists || credential.IsProxyLike() {
					return fmt.Errorf("routing_profiles[%d].rules[%d]: credential_allowlist must reference existing direct provider credentials", i, j)
				}
			}
			for _, name := range rule.CredentialDenylist {
				if _, exists := known[name]; !validRoutingName(name) || !exists {
					return fmt.Errorf("routing_profiles[%d].rules[%d]: credential_denylist must reference existing credentials", i, j)
				}
			}
		}
	}
	return nil
}
