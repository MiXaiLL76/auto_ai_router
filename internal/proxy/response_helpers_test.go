package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

func TestTokenUsageExtractionOptionsForResponse(t *testing.T) {
	tests := []struct {
		name     string
		cred     *config.CredentialConfig
		headers  http.Header
		includes bool
	}{
		{
			name:     "generic proxy defaults to OpenAI semantics",
			cred:     &config.CredentialConfig{Type: config.ProviderTypeProxy},
			includes: true,
		},
		{
			name: "AIR response header marks normalized usage without credential config",
			cred: &config.CredentialConfig{Type: config.ProviderTypeProxy},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
			},
			includes: false,
		},
		{
			name: "AIR response header marks raw OpenAI-compatible usage",
			cred: &config.CredentialConfig{Type: config.ProviderTypeProxy},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensIncludeCached},
			},
			includes: true,
		},
		{
			name:     "AIR provider without header keeps legacy AIR normalized semantics",
			cred:     &config.CredentialConfig{Type: config.ProviderTypeAIR},
			includes: false,
		},
		{
			name: "AIR provider normalized response header",
			cred: &config.CredentialConfig{Type: config.ProviderTypeAIR},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
			},
			includes: false,
		},
		{
			name: "AIR provider raw response header",
			cred: &config.CredentialConfig{Type: config.ProviderTypeAIR},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensIncludeCached},
			},
			includes: true,
		},
		{
			name: "AIR response header tolerates comma joined duplicates",
			cred: &config.CredentialConfig{Type: config.ProviderTypeProxy},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{"exclude-cached, exclude-cached"},
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
			got := tokenUsageExtractionOptionsForResponse(tt.cred, tt.headers)
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
		headers   http.Header
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
			name: "AIR-normalized proxy response header preserves non-cached audio",
			cred: config.CredentialConfig{Type: config.ProviderTypeProxy},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
			},
			audio:     60,
			wantAudio: 60,
		},
		{
			name: "AIR provider raw response header subtracts cached audio",
			cred: config.CredentialConfig{Type: config.ProviderTypeAIR},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensIncludeCached},
			},
			audio:     100,
			wantAudio: 60,
		},
		{
			name: "AIR provider normalized response header preserves non-cached audio",
			cred: config.CredentialConfig{Type: config.ProviderTypeAIR},
			headers: http.Header{
				HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
			},
			audio:     60,
			wantAudio: 60,
		},
		{
			name:      "legacy AIR without usage header preserves normalized audio",
			cred:      config.CredentialConfig{Type: config.ProviderTypeAIR},
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
				tokenUsageExtractionOptionsForResponse(&tt.cred, tt.headers),
			)
			if usage == nil || usage.AudioInputTokens != tt.wantAudio {
				t.Fatalf("unexpected usage: %+v; want audio=%d", usage, tt.wantAudio)
			}
		})
	}
}

func TestExtractWebSearchRequestUsage(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantOn   bool
		wantSize string
	}{
		{
			name:     "web_search_options high",
			body:     `{"model":"gpt-4o-search-preview","web_search_options":{"search_context_size":"high"}}`,
			wantOn:   true,
			wantSize: "high",
		},
		{
			name:     "web_search tool defaults medium",
			body:     `{"model":"gpt-4o","tools":[{"type":"web_search"}]}`,
			wantOn:   true,
			wantSize: "medium",
		},
		{
			name:     "versioned web_search tool low",
			body:     `{"model":"gpt-4o","tools":[{"type":"web_search_preview_2025_03_11","search_context_size":"low"}]}`,
			wantOn:   true,
			wantSize: "low",
		},
		{
			name:   "no web search",
			body:   `{"model":"gpt-4o","tools":[{"type":"function","function":{"name":"f"}}]}`,
			wantOn: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOn, gotSize := extractWebSearchRequestUsage([]byte(tt.body), "application/json")
			if gotOn != tt.wantOn || gotSize != tt.wantSize {
				t.Fatalf("unexpected web search usage: got (%v,%q), want (%v,%q)", gotOn, gotSize, tt.wantOn, tt.wantSize)
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

// TestExtractTokenUsageFromPayloads_BatchedSSEMergesUsage reproduces a
// real-world batching case: upstream flushes a web-search-only annotation
// frame and the final usage frame close enough together that a single
// downstream Read (and therefore a single splitSSEPayloads call) returns
// both SSE frames as one payload batch. extractTokenUsageFromPayloads must
// merge usage across every payload in the batch — returning on the first
// non-nil hit would report only WebSearchRequests and silently drop the
// real prompt/completion tokens carried by the later frame.
func TestExtractTokenUsageFromPayloads_BatchedSSEMergesUsage(t *testing.T) {
	chunk := []byte("data: {\"usage\":{\"web_search_requests\":1}}\n\n" +
		"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")

	payloads := splitSSEPayloads(chunk, nil)
	if len(payloads) != 2 {
		t.Fatalf("expected 2 SSE payloads in the batch, got %d", len(payloads))
	}

	usage := extractTokenUsageFromPayloads(payloads, converter.TokenUsageExtractionOptions{})
	if usage == nil {
		t.Fatal("expected non-nil merged usage")
		return
	}
	if usage.PromptTokens != 10 || usage.CompletionTokens != 5 {
		t.Fatalf("expected PromptTokens=10 CompletionTokens=5, got PromptTokens=%d CompletionTokens=%d",
			usage.PromptTokens, usage.CompletionTokens)
	}
	if usage.WebSearchRequests != 1 {
		t.Fatalf("expected WebSearchRequests=1 preserved from the earlier frame, got %d", usage.WebSearchRequests)
	}
}

// TestExtractTokenUsageFromPayloads_SingleFrameStillWorks is a sanity check
// that the merge loop doesn't change single-payload behavior.
func TestExtractTokenUsageFromPayloads_SingleFrameStillWorks(t *testing.T) {
	payloads := [][]byte{[]byte(`{"usage":{"prompt_tokens":3,"completion_tokens":7}}`)}

	usage := extractTokenUsageFromPayloads(payloads, converter.TokenUsageExtractionOptions{})
	if usage == nil {
		t.Fatal("expected non-nil usage")
		return
	}
	if usage.PromptTokens != 3 || usage.CompletionTokens != 7 {
		t.Fatalf("expected PromptTokens=3 CompletionTokens=7, got PromptTokens=%d CompletionTokens=%d",
			usage.PromptTokens, usage.CompletionTokens)
	}
}

// TestExtractTokenUsageFromPayloads_NoUsageReturnsNil ensures the loop still
// returns nil (not an empty non-nil TokenUsage) when nothing in the batch
// carries usage.
func TestExtractTokenUsageFromPayloads_NoUsageReturnsNil(t *testing.T) {
	payloads := [][]byte{[]byte(`{"choices":[{"delta":{"content":"hi"}}]}`)}

	usage := extractTokenUsageFromPayloads(payloads, converter.TokenUsageExtractionOptions{})
	if usage != nil {
		t.Fatalf("expected nil usage, got %+v", usage)
	}
}
