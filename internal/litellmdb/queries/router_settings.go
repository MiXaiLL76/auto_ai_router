package queries

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ParseRouterSettings extracts model_group_alias and fallbacks from the JSON value of
// the LiteLLM_Config router_settings row. Empty or null input yields empty settings.
//
// LiteLLM allows a model_group_alias value to be a plain group name or an object
// {"model": "<group>", "hidden": bool}, and a fallbacks entry to be a list of group
// names.
//
// Parsing is best-effort: the returned settings always hold everything that could be
// read, even when the error is non-nil. The error lists each part that had an
// unexpected shape and was ignored (a malformed document, a field of the wrong type, or
// a single alias/fallback entry of the wrong type). Callers should log it and carry on
// rather than fail: a router_settings row this parser does not understand must not
// stop the model sync.
func ParseRouterSettings(raw []byte) (RouterSettings, error) {
	settings := RouterSettings{
		ModelGroupAlias: map[string]string{},
		Fallbacks:       map[string][]string{},
	}
	if len(raw) == 0 || string(raw) == "null" {
		return settings, nil
	}

	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		return settings, fmt.Errorf("parse router_settings: %w", err)
	}

	var errs []error

	if rawAliases, ok := doc["model_group_alias"]; ok {
		var aliases map[string]json.RawMessage
		if err := json.Unmarshal(rawAliases, &aliases); err != nil {
			errs = append(errs, fmt.Errorf("ignoring router_settings.model_group_alias: %w", err))
		}
		for alias, value := range aliases {
			alias = strings.TrimSpace(alias)
			if alias == "" {
				continue
			}
			target, err := parseAliasTarget(value)
			if err != nil {
				errs = append(errs, fmt.Errorf("ignoring router_settings.model_group_alias[%q]: %w", alias, err))
				continue
			}
			if target != "" {
				settings.ModelGroupAlias[alias] = target
			}
		}
	}

	if rawFallbacks, ok := doc["fallbacks"]; ok {
		var entries []json.RawMessage
		if err := json.Unmarshal(rawFallbacks, &entries); err != nil {
			errs = append(errs, fmt.Errorf("ignoring router_settings.fallbacks: %w", err))
		}
		for i, rawEntry := range entries {
			var entry map[string][]any
			if err := json.Unmarshal(rawEntry, &entry); err != nil {
				errs = append(errs, fmt.Errorf("ignoring router_settings.fallbacks[%d]: %w", i, err))
				continue
			}
			for group, targets := range entry {
				for _, target := range targets {
					if name, ok := target.(string); ok && strings.TrimSpace(name) != "" {
						settings.Fallbacks[group] = append(settings.Fallbacks[group], strings.TrimSpace(name))
					}
				}
			}
		}
	}

	return settings, errors.Join(errs...)
}

// parseAliasTarget reads a model_group_alias value: a group name or {"model": "<group>"}.
// An object without a model yields "" and no error (LiteLLM allows a hidden-only entry).
func parseAliasTarget(value json.RawMessage) (string, error) {
	var target string
	if err := json.Unmarshal(value, &target); err != nil {
		var obj struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(value, &obj); err != nil {
			return "", err
		}
		target = obj.Model
	}
	return strings.TrimSpace(target), nil
}
