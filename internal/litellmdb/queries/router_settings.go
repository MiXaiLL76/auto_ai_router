package queries

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ParseRouterSettings extracts model_group_alias and fallbacks from the JSON value of
// the LiteLLM_Config router_settings row. Empty or null input yields empty settings.
//
// LiteLLM allows a model_group_alias value to be a plain group name or an object
// {"model": "<group>", "hidden": bool}, and a fallbacks entry to be a list of group
// names; anything else in those positions is skipped rather than failing the sync.
func ParseRouterSettings(raw []byte) (RouterSettings, error) {
	settings := RouterSettings{
		ModelGroupAlias: map[string]string{},
		Fallbacks:       map[string][]string{},
	}
	if len(raw) == 0 || string(raw) == "null" {
		return settings, nil
	}

	var doc struct {
		ModelGroupAlias map[string]json.RawMessage `json:"model_group_alias"`
		Fallbacks       []map[string][]any         `json:"fallbacks"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return settings, fmt.Errorf("parse router_settings: %w", err)
	}

	for alias, value := range doc.ModelGroupAlias {
		alias = strings.TrimSpace(alias)
		if alias == "" {
			continue
		}
		var target string
		if err := json.Unmarshal(value, &target); err != nil {
			var obj struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(value, &obj); err != nil {
				continue
			}
			target = obj.Model
		}
		if target = strings.TrimSpace(target); target != "" {
			settings.ModelGroupAlias[alias] = target
		}
	}

	for _, entry := range doc.Fallbacks {
		for group, targets := range entry {
			for _, target := range targets {
				if name, ok := target.(string); ok && strings.TrimSpace(name) != "" {
					settings.Fallbacks[group] = append(settings.Fallbacks[group], strings.TrimSpace(name))
				}
			}
		}
	}

	return settings, nil
}
