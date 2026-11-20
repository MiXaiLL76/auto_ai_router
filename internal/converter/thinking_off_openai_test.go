package converter

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider that speaks the OpenAI wire protocol but fronts a Claude model
// does not read "reasoning_effort": "none" as "no reasoning". Measured
// against Requesty (router.requesty.ai) with anthropic/claude-haiku-5-5, a
// model that thinks by default, the reasoning RAN.
//
// Anthropic's own words are needed instead, and the effort that
// contradicted them has to go with them -- with the disable AND
// reasoning_effort "none" both present, Requesty turns reasoning back ON.
func TestRequestFrom_OpenAIWire_ClaudeThinkingOff(t *testing.T) {
	// The model name a provider-credential carries is the provider's own
	// (here Requesty's routing prefix), which is what the converter sees.
	tests := []struct {
		name         string
		model        string
		body         string
		wantThinking any
		wantNoEffort bool
	}{
		{
			name:         "haiku 5.5 through an openai-wire provider",
			model:        "policy/claude-haiku-5-5",
			body:         `{"model":"policy/claude-haiku-5-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			wantThinking: map[string]any{"type": "disabled"},
			wantNoEffort: true,
		},
		{
			name:         "opus 5 through an openai-wire provider",
			model:        "policy/claude-opus-5",
			body:         `{"model":"policy/claude-opus-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			wantThinking: map[string]any{"type": "disabled"},
			wantNoEffort: true,
		},
		{
			name:         "a model that thinks only when asked needs nothing",
			model:        "policy/claude-haiku-4-5",
			body:         `{"model":"policy/claude-haiku-4-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
		{
			name:         "a non-claude model is untouched",
			model:        "policy/gemini-3.1-flash-lite",
			body:         `{"model":"policy/gemini-3.1-flash-lite","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
		{
			name:         "an openai model is untouched",
			model:        "gpt-4o-mini",
			body:         `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
		{
			name:         "an effort that asks for reasoning stays",
			model:        "policy/claude-haiku-5-5",
			body:         `{"model":"policy/claude-haiku-5-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`,
			wantThinking: nil,
			wantNoEffort: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := New(config.ProviderTypeOpenAI, RequestMode{ModelID: tt.model}).RequestFrom([]byte(tt.body))
			require.NoError(t, err)
			m := mustUnmarshal[map[string]any](t, got)
			if tt.wantThinking == nil {
				assert.NotContains(t, m, "thinking")
			} else {
				assert.Equal(t, tt.wantThinking, m["thinking"])
			}
			if tt.wantNoEffort {
				assert.NotContains(t, m, "reasoning_effort")
			}
		})
	}
}

// The Anthropic-wire path already sets the disable itself (see the Anthropic
// converter), so the OpenAI-wire rewrite must not fight it: a body that
// already carries a thinking config is left as the client sent it.
func TestRequestFrom_OpenAIWire_ExplicitThinkingWins(t *testing.T) {
	body := []byte(`{"model":"policy/claude-haiku-5-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none","thinking":{"type":"enabled"}}`)
	got, err := New(config.ProviderTypeOpenAI, RequestMode{ModelID: "policy/claude-haiku-5-5"}).RequestFrom(body)
	require.NoError(t, err)
	m := mustUnmarshal[map[string]any](t, got)
	assert.Equal(t, map[string]any{"type": "enabled"}, m["thinking"])
	assert.Contains(t, m, "reasoning_effort")
}
