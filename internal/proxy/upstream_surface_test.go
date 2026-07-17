package proxy

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsProManCredential(t *testing.T) {
	tests := []struct {
		name string
		cred *config.CredentialConfig
		want bool
	}{
		{
			name: "dedicated provider type",
			cred: &config.CredentialConfig{Type: config.ProviderTypeProMan},
			want: true,
		},
		{
			name: "known credential name fallback",
			cred: &config.CredentialConfig{Name: "anthropic-promanYT-01", Type: config.ProviderTypeAnthropic},
			want: true,
		},
		{
			name: "hyphenated provider name",
			cred: &config.CredentialConfig{Name: "pro-man-claude", Type: config.ProviderTypeProxy},
			want: true,
		},
		{
			name: "proman host",
			cred: &config.CredentialConfig{Name: "claude", BaseURL: "https://api.proman.ai/v1"},
			want: true,
		},
		{
			name: "aws elb host without explicit name is not enough",
			cred: &config.CredentialConfig{Name: "anthropic-regular", BaseURL: "http://intern-loadb-qo31w2z8vdyb-411810922.eu-central-1.elb.amazonaws.com/v1"},
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
			assert.Equal(t, tt.want, isProManCredential(tt.cred))
		})
	}
}

func TestSanitizeUpstreamJSONBodyRemovesLiteLLMSurface(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-1",
		"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a",
		"provider_specific_fields":{"trace":"hidden"},
		"caller":"litellm",
		"choices":[{"message":{"content":"ok"},"provider_specific_fields":{"router":"hidden"}}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)

	sanitized, changed := sanitizeUpstreamJSONBody(raw, "claude-haiku-4.5")

	require.True(t, changed)
	body := string(sanitized)
	assert.NotContains(t, body, "provider_specific_fields")
	assert.NotContains(t, body, "caller")
	assert.NotContains(t, body, "anthropic-direct-client")
	assert.NotContains(t, body, "litellm")

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(sanitized, &decoded))
	assert.Equal(t, "claude-haiku-4.5", decoded["model"])
	assert.Contains(t, body, `"usage"`)
}

func TestClientResponseBodyForProManMasksErrors(t *testing.T) {
	cred := &config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan}
	raw := []byte(`{"error":{"message":"litellm.BadRequestError: Received Model Group=anthropic/claude/anthropic-direct-client-0dce8b1a Available Model Group Fallbacks=None"}}`)

	body, changed := clientResponseBodyForCredential(400, raw, cred, "claude-haiku-4.5")

	require.True(t, changed)
	assert.NotContains(t, string(body), "litellm")
	assert.NotContains(t, string(body), "anthropic-direct-client")
	assert.NotContains(t, string(body), "Model Group")
	assert.Contains(t, string(body), "Upstream provider error")
}

func TestSanitizingSSEReadCloserRemovesInternalFields(t *testing.T) {
	stream := strings.Join([]string{
		"event: response.output_text.delta",
		`data: {"type":"response.completed","response":{"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a","provider_specific_fields":{"trace":"hidden"},"output":[]}}`,
		`data: {"type":"error","error":{"type":"invalid_request_error","message":"litellm.BadRequestError: Received Model Group=anthropic/claude/anthropic-direct-client-0dce8b1a Available Model Group Fallbacks=None"}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	rc := newSanitizingSSEReadCloser(io.NopCloser(strings.NewReader(stream)), "claude-haiku-4.5")

	out, err := io.ReadAll(rc)
	require.NoError(t, err)

	assert.Contains(t, string(out), "event: response.output_text.delta")
	assert.Contains(t, string(out), `"model":"claude-haiku-4.5"`)
	assert.Contains(t, string(out), "Upstream provider error")
	assert.NotContains(t, string(out), "provider_specific_fields")
	assert.NotContains(t, string(out), "anthropic-direct-client")
	assert.NotContains(t, string(out), "Model Group")
	assert.Contains(t, string(out), "data: [DONE]")
}
