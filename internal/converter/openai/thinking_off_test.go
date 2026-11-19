package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSetThinkingDisabled(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		wantThinking any
		wantNoEffort bool
		wantSame     bool
	}{
		{
			name:         "reasoning_effort none becomes an explicit disable",
			body:         `{"model":"m","reasoning_effort":"none","messages":[]}`,
			wantThinking: map[string]any{"type": "disabled"},
			wantNoEffort: true,
		},
		{
			name:         "disable and off are the same request",
			body:         `{"model":"m","reasoning_effort":"disable","messages":[]}`,
			wantThinking: map[string]any{"type": "disabled"},
			wantNoEffort: true,
		},
		{
			name:         "an effort that asks for reasoning is left alone",
			body:         `{"model":"m","reasoning_effort":"high","messages":[]}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
		{
			name:         "an explicit thinking config is the client's own business",
			body:         `{"model":"m","reasoning_effort":"none","thinking":{"type":"enabled"},"messages":[]}`,
			wantThinking: map[string]any{"type": "enabled"},
			wantNoEffort: false,
		},
		{
			name:         "a request without an effort is untouched",
			body:         `{"model":"m","messages":[]}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
		{
			name:         "the responses api carries the effort nested",
			body:         `{"model":"m","reasoning":{"effort":"none"},"input":"hi"}`,
			wantThinking: map[string]any{"type": "disabled"},
			wantNoEffort: true,
		},
		{
			name:     "a malformed body is returned as it came",
			body:     `not json`,
			wantSame: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SetThinkingDisabled([]byte(tt.body))
			if tt.wantSame {
				assert.Equal(t, tt.body, string(got))
				return
			}
			var data map[string]any
			require.NoError(t, json.Unmarshal(got, &data))
			if tt.wantThinking == nil {
				assert.NotContains(t, data, "thinking")
			} else {
				assert.Equal(t, tt.wantThinking, data["thinking"])
			}
			if tt.wantNoEffort {
				assert.NotContains(t, data, "reasoning_effort",
					"the effort that contradicted the disable must go with it")
				assert.NotContains(t, data, "reasoning",
					"a nested reasoning.effort that contradicted the disable must go with it")
			}
		})
	}
}

// TestSetThinkingDisabledKeepsOtherFields guards the body rewrite: only the
// effort and the thinking config are the converter's business.
func TestSetThinkingDisabledKeepsOtherFields(t *testing.T) {
	body := `{"model":"m","temperature":0.5,"max_tokens":100,"stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`
	got := SetThinkingDisabled([]byte(body))

	var data map[string]any
	require.NoError(t, json.Unmarshal(got, &data))
	assert.Equal(t, 0.5, data["temperature"])
	assert.Equal(t, float64(100), data["max_tokens"])
	assert.Equal(t, true, data["stream"])
	assert.Equal(t, "m", data["model"])
	require.Len(t, data["messages"], 1)
}
