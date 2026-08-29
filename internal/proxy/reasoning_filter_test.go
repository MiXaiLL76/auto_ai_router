package proxy

import (
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestUsesReasoning(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		want       bool
		wantSource string
	}{
		{name: "absent", body: `{"model":"claude","messages":[]}`},
		{name: "effort", body: `{"reasoning_effort":"high"}`, want: true, wantSource: "reasoning_effort"},
		{name: "disabled effort", body: `{"reasoning_effort":"none"}`},
		{name: "responses reasoning", body: `{"reasoning":{"effort":"medium"}}`, want: true, wantSource: "reasoning"},
		{name: "responses default effort", body: `{"reasoning":{"effort":null,"summary":"auto"}}`, want: true, wantSource: "reasoning"},
		{name: "anthropic thinking", body: `{"thinking":{"type":"enabled","budget_tokens":1024}}`, want: true, wantSource: "thinking"},
		{name: "adaptive thinking", body: `{"thinking":{"type":"adaptive"}}`, want: true, wantSource: "thinking"},
		{name: "disabled thinking", body: `{"thinking":{"type":"disabled"}}`},
		{name: "gemini budget", body: `{"thinking_budget":-1}`, want: true, wantSource: "thinking_budget"},
		{name: "gemini disabled budget", body: `{"thinking_budget":0}`},
		{name: "nested extra body", body: `{"extra_body":{"reasoning_effort":"low"}}`, want: true, wantSource: "extra_body.reasoning_effort"},
		{name: "nested thinking config", body: `{"thinking_config":{"thinking_level":"high"}}`, want: true, wantSource: "thinking_config.thinking_level"},
		{name: "include thoughts only", body: `{"thinking_config":{"include_thoughts":true}}`},
		{name: "invalid json", body: `{`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requested, source := requestReasoning([]byte(tt.body))
			assert.Equal(t, tt.want, requested)
			assert.Equal(t, tt.wantSource, source)
			assert.Equal(t, tt.want, requestUsesReasoning([]byte(tt.body)))
		})
	}
}

func TestReasoningOnlyExclusions(t *testing.T) {
	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "reasoning", Type: config.ProviderTypeAnthropic, ReasoningOnly: true, RPM: -1},
		config.CredentialConfig{Name: "general", Type: config.ProviderTypeAnthropic, RPM: -1},
	).Build()

	assert.Equal(t, map[string]bool{"reasoning": true}, prx.reasoningOnlyExclusions(false))
	assert.Empty(t, prx.reasoningOnlyExclusions(true))
}

func TestSelectCredentialForModelSkipsReasoningOnly(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "reasoning", Type: config.ProviderTypeAnthropic, ReasoningOnly: true, RPM: -1},
		{Name: "general", Type: config.ProviderTypeAnthropic, RPM: -1},
	}
	prx := NewTestProxyBuilder().WithCredentials(credentials...).Build()
	logCtx := testLogCtx(t)
	exclude := prx.reasoningOnlyExclusions(false)

	credential, ok := prx.selectCredentialForModel(httptest.NewRecorder(), "claude", "", "", exclude, logCtx)

	require.True(t, ok)
	assert.Equal(t, "general", credential.Name)
}
