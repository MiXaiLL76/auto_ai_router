package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

func TestTokenUsageExtractionOptionsForCredential(t *testing.T) {
	tests := []struct {
		name     string
		cred     *config.CredentialConfig
		includes bool
	}{
		{
			name:     "generic proxy defaults to OpenAI semantics",
			cred:     &config.CredentialConfig{Type: config.ProviderTypeProxy},
			includes: true,
		},
		{
			name: "explicit OpenAI proxy",
			cred: &config.CredentialConfig{
				Type:             config.ProviderTypeProxy,
				ProxyUsageFormat: config.ProxyUsageFormatOpenAI,
			},
			includes: true,
		},
		{
			name: "explicit normalized proxy",
			cred: &config.CredentialConfig{
				Type:             config.ProviderTypeProxy,
				ProxyUsageFormat: config.ProxyUsageFormatNormalized,
			},
			includes: false,
		},
		{
			name:     "direct OpenAI provider",
			cred:     &config.CredentialConfig{Type: config.ProviderTypeOpenAI},
			includes: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenUsageExtractionOptionsForCredential(tt.cred)
			if got.AudioInputIncludesCachedAudio != tt.includes {
				t.Fatalf("unexpected usage semantics: got %v, want %v", got.AudioInputIncludesCachedAudio, tt.includes)
			}
		})
	}
}

func TestProxyUsageContractCachedAudioMatrix(t *testing.T) {
	tests := []struct {
		name      string
		cred      config.CredentialConfig
		audio     int
		wantAudio int
	}{
		{
			name:      "raw OpenAI-compatible proxy subtracts cached audio",
			cred:      config.CredentialConfig{Type: config.ProviderTypeProxy},
			audio:     100,
			wantAudio: 60,
		},
		{
			name: "normalized proxy preserves non-cached audio",
			cred: config.CredentialConfig{
				Type:             config.ProviderTypeProxy,
				ProxyUsageFormat: config.ProxyUsageFormatNormalized,
			},
			audio:     60,
			wantAudio: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"usage":{"prompt_tokens":200,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":` +
				fmt.Sprintf("%d", tt.audio) + `}}}`)
			usage := converter.ExtractTokenUsageWithOptions(
				body,
				tokenUsageExtractionOptionsForCredential(&tt.cred),
			)
			if usage == nil || usage.AudioInputTokens != tt.wantAudio {
				t.Fatalf("unexpected usage: %+v; want audio=%d", usage, tt.wantAudio)
			}
		})
	}
}

func TestInjectStreamOptions_AddsIncludeUsage(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
}

func TestInjectStreamOptions_UpdatesExisting(t *testing.T) {
	body := []byte(`{"stream_options":{"include_usage":false,"foo":1}}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
	if streamOptions["foo"] != float64(1) {
		t.Fatalf("expected foo to be preserved, got %v", streamOptions["foo"])
	}
}

func TestInjectStreamOptions_ReplacesNonMap(t *testing.T) {
	body := []byte(`{"stream_options":"bad"}`)
	modified := injectStreamOptions(body)

	var raw map[string]interface{}
	if err := json.Unmarshal(modified, &raw); err != nil {
		t.Fatalf("failed to unmarshal modified body: %v", err)
	}

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected stream_options map, got %T", raw["stream_options"])
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected include_usage=true, got %v", streamOptions["include_usage"])
	}
}

func TestInjectStreamOptions_InvalidJSON(t *testing.T) {
	body := []byte(`{"stream_options":`)
	modified := injectStreamOptions(body)
	if !bytes.Equal(modified, body) {
		t.Fatalf("expected invalid json to be returned as-is")
	}
}
