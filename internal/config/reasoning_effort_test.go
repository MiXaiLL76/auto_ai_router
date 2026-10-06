package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestReasoningEffortMap_Resolve(t *testing.T) {
	m, err := NewReasoningEffortMap(map[string]string{
		"Minimal": "low", "high": "medium", "max": "xhigh", "default": "low",
	})
	require.NoError(t, err)

	for in, want := range map[string]string{
		"minimal": "low",    // mapped (keys are case-insensitive)
		"MINIMAL": "low",    // mapped
		"high":    "medium", // mapped
		"max":     "xhigh",  // mapped
		"medium":  "medium", // a mapping target is accepted as is
		"xhigh":   "xhigh",  // a mapping target is accepted as is
		"low":     "low",    // a mapping target is accepted as is
		"none":    "low",    // unknown -> default
		"":        "low",    // unknown -> default
	} {
		got, changed := m.Resolve(in)
		assert.Equal(t, want, got, in)
		assert.Equal(t, want != in, changed, in)
	}

	noDefault, err := NewReasoningEffortMap(map[string]string{"minimal": "low"})
	require.NoError(t, err)
	got, changed := noDefault.Resolve("none")
	assert.Equal(t, "none", got, "without a default an unknown value is kept")
	assert.False(t, changed)

	var nilMap *ReasoningEffortMap
	got, changed = nilMap.Resolve("high")
	assert.Equal(t, "high", got)
	assert.False(t, changed)
}

func TestReasoningEffortMap_UnmarshalYAML(t *testing.T) {
	var fromMap, fromJSON ReasoningEffortMap
	require.NoError(t, yaml.Unmarshal([]byte(`{minimal: low, default: medium}`), &fromMap))
	require.NoError(t, yaml.Unmarshal([]byte(`'{"minimal": "low", "default": "medium"}'`), &fromJSON))
	assert.Equal(t, fromMap, fromJSON)
	assert.Equal(t, "medium", fromMap.Default)

	var bad ReasoningEffortMap
	assert.Error(t, yaml.Unmarshal([]byte(`{minimal: ""}`), &bad))
	assert.Error(t, yaml.Unmarshal([]byte(`'not json'`), &bad))
	assert.Error(t, yaml.Unmarshal([]byte(`[low, medium]`), &bad))
}
