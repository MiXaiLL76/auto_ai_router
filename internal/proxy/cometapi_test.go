package proxy

import (
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsCometAPICredential(t *testing.T) {
	tests := []struct {
		name      string
		cred      *config.CredentialConfig
		wantComet bool
	}{
		{
			name:      "dedicated provider type",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeCometAPI},
			wantComet: true,
		},
		{
			name:      "comet host fallback",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.cometapi.com/v1"},
			wantComet: true,
		},
		{
			name:      "comet name fallback",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, Name: "comet-api-anthropic"},
			wantComet: true,
		},
		{
			name:      "regular anthropic",
			cred:      &config.CredentialConfig{Type: config.ProviderTypeAnthropic, BaseURL: "https://api.anthropic.com"},
			wantComet: false,
		},
		{
			name:      "nil credential",
			cred:      nil,
			wantComet: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantComet, isCometAPICredential(tt.cred))
		})
	}
}

func TestAppendResponseBodyForLogs_MaskedProviderKeepsMaskedFlagAndLogsBody(t *testing.T) {
	cred := &config.CredentialConfig{Type: config.ProviderTypeCometAPI}
	body := `{"error":{"code":"permission_denied","message":"` + strings.Repeat("model access denied ", 50) + `","type":"comet_api_error"}}`

	args := appendResponseBodyForLogs([]any{}, cred, body)

	assert.Contains(t, args, "response_body_masked")
	assert.Contains(t, args, true)
	loggedBody := responseBodyArg(t, args)
	assert.Contains(t, loggedBody, "permission_denied")
	assert.Contains(t, loggedBody, "model access denied")
	assert.NotEqual(t, body, loggedBody)
	assert.Less(t, len(loggedBody), len(body))
}

func TestIsSosanaCredential(t *testing.T) {
	tests := []struct {
		name       string
		cred       *config.CredentialConfig
		wantSosana bool
	}{
		{
			name:       "dedicated provider type",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeSosana},
			wantSosana: true,
		},
		{
			name:       "sosana host fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://sosana.art"},
			wantSosana: true,
		},
		{
			name:       "sosana name fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, Name: "sosana-art-images"},
			wantSosana: true,
		},
		{
			name:       "sasana host fallback",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://api.sasana.example/v1"},
			wantSosana: true,
		},
		{
			name:       "regular openai",
			cred:       &config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com"},
			wantSosana: false,
		},
		{
			name:       "nil credential",
			cred:       nil,
			wantSosana: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantSosana, isSosanaCredential(tt.cred))
		})
	}
}

func TestShouldMaskUpstreamErrorsForKnownResellers(t *testing.T) {
	assert.True(t, shouldMaskUpstreamErrors(&config.CredentialConfig{Type: config.ProviderTypeCometAPI}))
	assert.True(t, shouldMaskUpstreamErrors(&config.CredentialConfig{Type: config.ProviderTypeSosana}))
	assert.False(t, shouldMaskUpstreamErrors(&config.CredentialConfig{Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com"}))
	assert.False(t, shouldMaskUpstreamErrors(nil))
}

func TestAppendResponseBodyForLogs_CometKeepsMaskedFlagAndLogsBody(t *testing.T) {
	cred := &config.CredentialConfig{Type: config.ProviderTypeCometAPI}
	body := `{"error":{"code":"permission_denied","message":"` + strings.Repeat("model access denied ", 50) + `","type":"comet_api_error"}}`

	args := appendResponseBodyForLogs([]any{}, cred, body)

	assert.Contains(t, args, "response_body_masked")
	assert.Contains(t, args, true)
	loggedBody := responseBodyArg(t, args)
	assert.Contains(t, loggedBody, "permission_denied")
	assert.Contains(t, loggedBody, "model access denied")
	assert.NotEqual(t, body, loggedBody)
	assert.Less(t, len(loggedBody), len(body))
}

func responseBodyArg(t *testing.T, args []any) string {
	t.Helper()

	for i := 0; i < len(args)-1; i++ {
		if args[i] == "response_body" {
			body, ok := args[i+1].(string)
			require.True(t, ok)
			return body
		}
	}
	require.FailNow(t, "response_body arg not found")
	return ""
}
