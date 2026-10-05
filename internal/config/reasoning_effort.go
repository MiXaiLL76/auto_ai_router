package config

import (
	"encoding/json"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

// ReasoningEffortDefaultKey is the reasoning_effort_map key whose value replaces
// any client value the map neither maps nor accepts.
const ReasoningEffortDefaultKey = "default"

// ReasoningEffortMap rewrites the reasoning effort a client asked for into one the
// model accepts, for models whose chat template rejects (or silently misreads)
// values outside its own set — e.g. a template that raises on anything but
// xhigh/medium/low.
//
// Resolution of a client value (case-insensitive):
//  1. a key of Mapping   -> its mapped value;
//  2. a value of Mapping -> kept as is (the model accepts it);
//  3. anything else      -> Default, or kept as is when Default is empty.
//
// In YAML it is one mapping (or the same object as a JSON string) where the key
// "default" sets Default:
//
//	reasoning_effort_map: {minimal: low, high: medium, max: xhigh, default: low}
type ReasoningEffortMap struct {
	Mapping map[string]string
	Default string
}

// Resolve returns the effort to send upstream for the client value v and whether
// it differs from v.
func (m *ReasoningEffortMap) Resolve(v string) (string, bool) {
	if m == nil {
		return v, false
	}
	key := strings.ToLower(strings.TrimSpace(v))
	if mapped, ok := m.Mapping[key]; ok {
		return mapped, mapped != v
	}
	for _, accepted := range m.Mapping {
		if strings.EqualFold(accepted, key) {
			return accepted, accepted != v
		}
	}
	if m.Default != "" {
		return m.Default, m.Default != v
	}
	return v, false
}

// UnmarshalYAML accepts a YAML mapping or a JSON object string.
func (m *ReasoningEffortMap) UnmarshalYAML(value *yaml.Node) error {
	raw := make(map[string]string)
	if value.Kind == yaml.ScalarNode {
		s := strings.TrimSpace(resolveEnvString(value.Value))
		if s == "" {
			*m = ReasoningEffortMap{}
			return nil
		}
		if err := json.Unmarshal([]byte(s), &raw); err != nil {
			return fmt.Errorf("reasoning_effort_map: want a mapping or a JSON object of strings: %w", err)
		}
	} else if err := value.Decode(&raw); err != nil {
		return fmt.Errorf("reasoning_effort_map: want a mapping of strings: %w", err)
	}
	parsed, err := NewReasoningEffortMap(raw)
	if err != nil {
		return err
	}
	*m = parsed
	return nil
}

// NewReasoningEffortMap builds a map from its config form (keys lower-cased, the
// "default" key split out). Values must be non-empty.
func NewReasoningEffortMap(raw map[string]string) (ReasoningEffortMap, error) {
	out := ReasoningEffortMap{Mapping: make(map[string]string, len(raw))}
	for k, v := range raw {
		key := strings.ToLower(strings.TrimSpace(k))
		value := strings.TrimSpace(v)
		if key == "" || value == "" {
			return ReasoningEffortMap{}, fmt.Errorf("reasoning_effort_map: empty key or value (%q: %q)", k, v)
		}
		if key == ReasoningEffortDefaultKey {
			out.Default = value
			continue
		}
		out.Mapping[key] = value
	}
	return out, nil
}

// IsEmpty reports whether the map would never change a value.
func (m *ReasoningEffortMap) IsEmpty() bool {
	return m == nil || (len(m.Mapping) == 0 && m.Default == "")
}
