package utils

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsCredential(t *testing.T) {
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
			name: "provider-looking credential name is not enough",
			cred: &config.CredentialConfig{Name: "anthropic-promanYT-01", Type: config.ProviderTypeAnthropic},
			want: false,
		},
		{
			name: "hyphenated provider name is not enough",
			cred: &config.CredentialConfig{Name: "pro-man-claude", Type: config.ProviderTypeProxy},
			want: false,
		},
		{
			name: "proman host is not enough",
			cred: &config.CredentialConfig{Name: "claude", BaseURL: "https://api.proman.ai/v1"},
			want: false,
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
			assert.Equal(t, tt.want, IsCredential(tt.cred))
			assert.Equal(t, tt.want, ShouldSanitizeUpstreamSurface(tt.cred))
		})
	}
}

func TestIsProviderInternalResponseHeader(t *testing.T) {
	for _, key := range []string{
		"Server",
		"X-Litellm-Version",
		"Llm_provider-Base",
		"X-Amzn-Requestid",
		"Anthropic-Ratelimit-Tokens-Remaining",
	} {
		t.Run(key, func(t *testing.T) {
			assert.True(t, IsProviderInternalResponseHeader(key))
		})
	}

	assert.False(t, IsProviderInternalResponseHeader("Content-Type"))
	assert.False(t, IsProviderInternalResponseHeader("Cache-Control"))
}

func TestUnsupportedRequest(t *testing.T) {
	tests := []struct {
		name  string
		model string
		body  string
		want  string
	}{
		{
			name:  "server tool use history rejected",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`,
			want:  "server_tool_use",
		},
		{
			name:  "server tool use in responses input rejected",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","input":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`,
			want:  "server_tool_use",
		},
		{
			name:  "server tool use inside user payload ignored",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":[{"type":"tool_result","content":[{"type":"server_tool_use","value":"customer json"}]}]}]}`,
			want:  "",
		},
		{
			name:  "thinking not provider-blocked",
			model: "claude-sonnet-5",
			body:  `{"model":"claude-sonnet-5","messages":[],"thinking":{"type":"adaptive","effort":"high"}}`,
			want:  "",
		},
		{
			name:  "reasoning effort not provider-blocked",
			model: "claude-sonnet-5",
			body:  `{"model":"claude-sonnet-5","messages":[],"reasoning_effort":"high"}`,
			want:  "",
		},
		{
			name:  "context management not provider-blocked",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"context_management":{"edits":[{"type":"compact_20260112"}]}}`,
			want:  "",
		},
		{
			name:  "tool choice none disable parallel not provider-blocked",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"tool_choice":{"type":"none","disable_parallel_tool_use":true}}`,
			want:  "",
		},
		{
			name:  "tool choice none without disable allowed",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"tool_choice":{"type":"none"}}`,
			want:  "",
		},
		{
			name:  "text plain document allowed",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"text/plain","data":"SGk="}}]}]}`,
			want:  "",
		},
		{
			name:  "assistant prefill not provider-blocked",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"pre"}]}`,
			want:  "",
		},
		{
			name:  "temperature top p allowed",
			model: "claude-haiku-4.5",
			body:  `{"model":"claude-haiku-4.5","messages":[],"temperature":0.2,"top_p":0.9}`,
			want:  "",
		},
		{
			name:  "top p allowed",
			model: "claude-sonnet-5",
			body:  `{"model":"claude-sonnet-5","messages":[],"top_p":0.9}`,
			want:  "",
		},
		{
			name:  "top k allowed",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"top_k":10}`,
			want:  "",
		},
		{
			name:  "temperature one allowed for deprecated sampling model",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"temperature":1}`,
			want:  "",
		},
		{
			name:  "low temperature allowed",
			model: "claude-opus-4.8",
			body:  `{"model":"claude-opus-4.8","messages":[],"temperature":0.2}`,
			want:  "",
		},
		{
			name:  "basic request allowed",
			model: "claude-sonnet-4.6",
			body:  `{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"hi"}]}`,
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, UnsupportedRequest([]byte(tt.body), tt.model))
		})
	}
}

func TestSanitizeUpstreamJSONBodyOnlyTouchesResponseEnvelope(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-1",
		"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a",
		"provider_specific_fields":{"trace":"hidden"},
		"caller":"litellm",
		"choices":[{
			"message":{
				"content":[{
					"type":"tool_use",
					"input":{"caller":"customer-app","model":"anthropic/customer-choice","provider_specific_fields":{"keep":"user-payload"}}
				}]
			},
			"provider_specific_fields":{"router":"hidden"}
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)

	sanitized, changed := SanitizeUpstreamJSONBody(raw, "claude-haiku-4.5")

	require.True(t, changed)
	body := string(sanitized)
	assert.NotContains(t, body, `"caller":"litellm"`)
	assert.NotContains(t, body, `"provider_specific_fields":{"trace":"hidden"}`)
	assert.NotContains(t, body, `"provider_specific_fields":{"router":"hidden"}`)
	assert.NotContains(t, body, "anthropic-direct-client")
	assert.NotContains(t, body, "litellm")
	assert.Contains(t, body, `"caller":"customer-app"`)
	assert.Contains(t, body, `"model":"anthropic/customer-choice"`)
	assert.Contains(t, body, `"provider_specific_fields":{"keep":"user-payload"}`)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(sanitized, &decoded))
	assert.Equal(t, "claude-haiku-4.5", decoded["model"])
	assert.Contains(t, body, `"usage"`)
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
	rc := NewSanitizingSSEReadCloser(io.NopCloser(strings.NewReader(stream)), "claude-haiku-4.5")

	out, err := io.ReadAll(rc)
	require.NoError(t, err)

	assert.Contains(t, string(out), "event: response.output_text.delta")
	assert.Contains(t, string(out), `"model":"claude-haiku-4.5"`)
	assert.Contains(t, string(out), "Request failed")
	assert.NotContains(t, string(out), "provider_specific_fields")
	assert.NotContains(t, string(out), "anthropic-direct-client")
	assert.NotContains(t, string(out), "Model Group")
	assert.Contains(t, string(out), "data: [DONE]")
}

func TestSanitizingSSEReadCloserPassesThroughOversizedLine(t *testing.T) {
	oversizedPayload := strings.Repeat("x", MaxSanitizingSSELineBytes+8192)
	oversizedLine := `data: {"blob":"` + oversizedPayload + `"}` + "\n"
	sanitizableLine := `data: {"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a","provider_specific_fields":{"trace":"hidden"}}` + "\n"
	stream := oversizedLine + sanitizableLine + "data: [DONE]\n"
	rc := NewSanitizingSSEReadCloser(io.NopCloser(strings.NewReader(stream)), "claude-haiku-4.5")

	out, err := io.ReadAll(rc)
	require.NoError(t, err)

	body := string(out)
	assert.Contains(t, body, oversizedLine)
	assert.Contains(t, body, `"model":"claude-haiku-4.5"`)
	assert.NotContains(t, body, "provider_specific_fields")
	assert.NotContains(t, body, "anthropic-direct-client")
	assert.Contains(t, body, "data: [DONE]")
}
