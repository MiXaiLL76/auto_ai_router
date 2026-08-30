package config

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

const (
	maxOrganizationCredentialDenylistEntries = 128
	maxOrganizationCredentialNameBytes       = 256
	maxOrganizationCredentialDenylistBytes   = 4096
)

type OrganizationPolicyConfig struct {
	OrganizationID     string            `yaml:"organization_id"`
	PriceProfileID     string            `yaml:"price_profile_id"`
	ModelPricesLink    string            `yaml:"model_prices_link"`
	ModelAllowlist     []string          `yaml:"model_allowlist,omitempty"`
	AllowlistSet       bool              `yaml:"-"`
	ModelMappings      map[string]string `yaml:"model_mappings,omitempty"`
	CredentialDenylist []string          `yaml:"credential_denylist,omitempty"`
}

func (p *OrganizationPolicyConfig) UnmarshalYAML(value *yaml.Node) error {
	if value == nil || value.Kind != yaml.MappingNode {
		return fmt.Errorf("organization policy must be a mapping")
	}

	allowedFields := map[string]struct{}{
		"organization_id":     {},
		"price_profile_id":    {},
		"model_prices_link":   {},
		"model_allowlist":     {},
		"model_mappings":      {},
		"credential_denylist": {},
	}

	var raw struct {
		OrganizationID     string   `yaml:"organization_id"`
		PriceProfileID     string   `yaml:"price_profile_id"`
		ModelPricesLink    string   `yaml:"model_prices_link"`
		ModelAllowlist     []string `yaml:"model_allowlist,omitempty"`
		CredentialDenylist []string `yaml:"credential_denylist,omitempty"`
	}
	mappings := make(map[string]string)

	for i := 0; i+1 < len(value.Content); i += 2 {
		key := value.Content[i]
		field := key.Value
		if _, ok := allowedFields[field]; !ok {
			return fmt.Errorf("organization policy: unknown field %q", field)
		}
		if field == "model_allowlist" {
			p.AllowlistSet = true
		}
		if field != "model_mappings" {
			continue
		}
		mapNode := value.Content[i+1]
		if mapNode.Kind != yaml.MappingNode {
			return fmt.Errorf("organization policy: model_mappings must be a mapping")
		}
		for j := 0; j+1 < len(mapNode.Content); j += 2 {
			rawKey := strings.TrimSpace(resolveEnvString(mapNode.Content[j].Value))
			if _, exists := mappings[rawKey]; exists {
				return fmt.Errorf("organization policy: duplicate model mapping key %q", rawKey)
			}
			var target string
			if err := mapNode.Content[j+1].Decode(&target); err != nil {
				return err
			}
			mappings[rawKey] = strings.TrimSpace(resolveEnvString(target))
		}
	}

	if err := value.Decode(&raw); err != nil {
		return err
	}
	p.OrganizationID = strings.TrimSpace(resolveEnvString(raw.OrganizationID))
	p.PriceProfileID = strings.TrimSpace(resolveEnvString(raw.PriceProfileID))
	p.ModelPricesLink = strings.TrimSpace(resolveEnvString(raw.ModelPricesLink))
	p.ModelAllowlist = make([]string, 0, len(raw.ModelAllowlist))
	for _, modelID := range raw.ModelAllowlist {
		p.ModelAllowlist = append(p.ModelAllowlist, strings.TrimSpace(resolveEnvString(modelID)))
	}
	seenCredentials := make(map[string]struct{}, len(raw.CredentialDenylist))
	p.CredentialDenylist = make([]string, 0, len(raw.CredentialDenylist))
	for _, credentialName := range raw.CredentialDenylist {
		credentialName = strings.TrimSpace(resolveEnvString(credentialName))
		if _, exists := seenCredentials[credentialName]; exists {
			return fmt.Errorf("organization policy: duplicate credential_denylist entry %q", credentialName)
		}
		seenCredentials[credentialName] = struct{}{}
		p.CredentialDenylist = append(p.CredentialDenylist, credentialName)
	}
	p.ModelMappings = mappings
	return nil
}

func ValidateOrganizationPolicies(policies []OrganizationPolicyConfig) error {
	seen := make(map[string]struct{}, len(policies))
	for _, policy := range policies {
		if policy.OrganizationID == "" {
			return fmt.Errorf("organization policy: organization_id is required")
		}
		if _, exists := seen[policy.OrganizationID]; exists {
			return fmt.Errorf("organization policy: duplicate organization_id %q", policy.OrganizationID)
		}
		seen[policy.OrganizationID] = struct{}{}
		if policy.PriceProfileID == "" {
			return fmt.Errorf("organization policy %q: price_profile_id is required", policy.OrganizationID)
		}
		if policy.ModelPricesLink == "" {
			return fmt.Errorf("organization policy %q: model_prices_link is required", policy.OrganizationID)
		}
		for _, modelID := range policy.ModelAllowlist {
			if modelID == "" {
				return fmt.Errorf("organization policy %q: model_allowlist contains an empty model ID", policy.OrganizationID)
			}
		}
		for _, credentialName := range policy.CredentialDenylist {
			if credentialName == "" {
				return fmt.Errorf("organization policy %q: credential_denylist contains an empty credential name", policy.OrganizationID)
			}
			if !utf8.ValidString(credentialName) || len(credentialName) > maxOrganizationCredentialNameBytes {
				return fmt.Errorf("organization policy %q: credential_denylist contains an invalid credential name", policy.OrganizationID)
			}
			for _, character := range credentialName {
				if unicode.IsControl(character) {
					return fmt.Errorf("organization policy %q: credential_denylist contains an invalid credential name", policy.OrganizationID)
				}
			}
		}
		if len(policy.CredentialDenylist) > maxOrganizationCredentialDenylistEntries {
			return fmt.Errorf("organization policy %q: credential_denylist exceeds %d entries", policy.OrganizationID, maxOrganizationCredentialDenylistEntries)
		}
		encodedDenylist, err := json.Marshal(policy.CredentialDenylist)
		if err != nil || len(encodedDenylist) > maxOrganizationCredentialDenylistBytes {
			return fmt.Errorf("organization policy %q: credential_denylist exceeds %d bytes", policy.OrganizationID, maxOrganizationCredentialDenylistBytes)
		}
		allowlisted := make(map[string]struct{}, len(policy.ModelAllowlist))
		for _, modelID := range policy.ModelAllowlist {
			allowlisted[modelID] = struct{}{}
		}
		for source, target := range policy.ModelMappings {
			if source == "" || target == "" {
				return fmt.Errorf("organization policy %q: model_mappings contains an empty key or target", policy.OrganizationID)
			}
			if policy.AllowlistSet {
				if _, ok := allowlisted[source]; !ok {
					return fmt.Errorf("organization policy %q: mapped model %q is outside model_allowlist", policy.OrganizationID, source)
				}
			}
		}
	}
	return nil
}
