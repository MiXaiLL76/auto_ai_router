package proxy

import (
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestIsCometAPICredential(t *testing.T) {
	tests := []struct {
		name      string
		cred      *config.CredentialConfig
		wantComet bool
		wantMask  bool
	}{
		{
			name:      "dedicated provider type",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeCometAPI},
			wantComet: true,
			wantMask:  true,
		},
		{
			name:      "comet host fallback",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.cometapi.com/v1"},
			wantComet: true,
			wantMask:  true,
		},
		{
			name:      "comet name fallback",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, Name: "comet-api-anthropic"},
			wantComet: true,
			wantMask:  true,
		},
		{
			name:      "regular anthropic",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.anthropic.com"},
			wantComet: false,
			wantMask:  false,
		},
		{
			name:      "nil credential",
			cred:      nil,
			wantComet: false,
			wantMask:  false,
		},
		{
			name:      "explicit mask flag",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeOpenAI, MaskUpstreamErrors: true},
			wantComet: false,
			wantMask:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantComet, isCometAPICredential(tt.cred))
			assert.Equal(t, tt.wantMask, shouldMaskUpstreamErrors(tt.cred))
		})
	}
}

func TestIsSosanaCredential(t *testing.T) {
	tests := []struct {
		name       string
		cred       *config.CredentialConfig
		wantSosana bool
		wantMask   bool
	}{
		{
			name:       "dedicated provider type",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeSosana},
			wantSosana: true,
			wantMask:   true,
		},
		{
			name:       "sosana host fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://sosana.art"},
			wantSosana: true,
			wantMask:   true,
		},
		{
			name:       "sosana name fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, Name: "sosana-art-images"},
			wantSosana: true,
			wantMask:   true,
		},
		{
			name:       "sasana host fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://api.sasana.example/v1"},
			wantSosana: true,
			wantMask:   true,
		},
		{
			name:       "regular openai",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com"},
			wantSosana: false,
			wantMask:   false,
		},
		{
			name:       "nil credential",
			cred:       nil,
			wantSosana: false,
			wantMask:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantSosana, isSosanaCredential(tt.cred))
			assert.Equal(t, tt.wantMask, shouldMaskUpstreamErrors(tt.cred))
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
