package proxy

import (
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestIsCometAPICredential(t *testing.T) {
	tests := []struct {
		name string
		cred *config.CredentialConfig
		want bool
	}{
		{
			name: "dedicated provider type",
			cred: &config.CredentialConfig{Type: config.ProviderTypeCometAPI},
			want: true,
		},
		{
			name: "dedicated provider type, openai protocol",
			cred: &config.CredentialConfig{Type: config.ProviderTypeCometAPI, OpenAIProtocol: true},
			want: true,
		},
		{
			name: "comet host fallback",
			cred: &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.cometapi.com/v1"},
			want: true,
		},
		{
			name: "comet name fallback",
			cred: &config.CredentialConfig{Type: config.ProviderTypeAnthropic, Name: "comet-api-anthropic"},
			want: true,
		},
		{
			name: "regular anthropic",
			cred: &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.anthropic.com"},
			want: false,
		},
		{
			name: "nil credential",
			cred: nil,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isCometAPICredential(tt.cred))
			assert.Equal(t, tt.want, shouldMaskUpstreamErrors(tt.cred))
		})
	}
}

func TestIsDeepSeekModel(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		want    bool
	}{
		{
			name:    "plain client-facing alias",
			modelID: "deepseek-v4-flash-0731",
			want:    true,
		},
		{
			name:    "vendor-prefixed on openrouter",
			modelID: "deepseek/deepseek-v4-flash-0731",
			want:    true,
		},
		{
			name:    "policy-aliased on requesty",
			modelID: "policy/deepseek-v4-flash-0731-ateam",
			want:    true,
		},
		{
			name:    "differently-cased on deepinfra",
			modelID: "deepseek-ai/DeepSeek-V4-Flash-0731",
			want:    true,
		},
		{
			name:    "unrelated model on the same OpenAI-compatible credential",
			modelID: "glm-5.3",
			want:    false,
		},
		{
			name:    "real openai model",
			modelID: "gpt-5.6-sol",
			want:    false,
		},
		{
			name:    "empty",
			modelID: "",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isDeepSeekModel(tt.modelID))
		})
	}
}

func TestAppendResponseBodyForLogs_CometKeepsMaskedFlagAndLogsBody(t *testing.T) {
	cred := &config.CredentialConfig{Type: config.ProviderTypeCometAPI}
	body := `{"error":{"code":"permission_denied","message":"` + strings.Repeat("model access denied ", 50) + `","type":"comet_api_error"}}`

	args := appendResponseBodyForLogs([]any{}, cred, body)

	assert.Contains(t, args, "response_body_masked")
	assert.Contains(t, args, true)
	assert.Contains(t, args, "response_body")
	assert.Contains(t, args, body)
}
