package anthropic

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

func TestTransformAnthropicStreamToOpenAIUsesStableClientCompatibleCorrelatedID(t *testing.T) {
	providerID := "msg_fixture_stream"
	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"` + providerID + `","usage":{"input_tokens":2}}}`,
		`data: {"type":"content_block_start","content_block":{"type":"text"}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}`,
		`data: {"type":"content_block_stop"}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n")

	var output bytes.Buffer
	if err := TransformAnthropicStreamToOpenAI(strings.NewReader(stream), "anthropic/claude-sonnet-4.5", &output); err != nil {
		t.Fatalf("transform Anthropic stream: %v", err)
	}

	wantID := "chatcmpl-" + providerID
	idPattern := regexp.MustCompile(`^chatcmpl-[A-Za-z0-9_-]+$`)
	chunkCount := 0
	for _, line := range strings.Split(output.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("unmarshal OpenAI stream chunk: %v", err)
		}
		chunkCount++
		if !idPattern.MatchString(chunk.ID) {
			t.Fatalf("stream chunk ID is not OpenAI-compatible: %q", chunk.ID)
		}
		if chunk.ID != wantID {
			t.Fatalf("stream chunk ID is not stable or correlated: got %q, want %q", chunk.ID, wantID)
		}
	}
	if chunkCount == 0 {
		t.Fatal("expected at least one OpenAI stream chunk")
	}
}

func TestTransformAnthropicStreamToOpenAIEmitsOpenAITerminalSequence(t *testing.T) {
	tests := []struct {
		name             string
		contentEvents    []string
		stopReason       string
		withUsage        bool
		wantFinishReason string
		wantToolID       string
		wantChunkCount   int
	}{
		{
			name: "text with usage",
			contentEvents: []string{
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
				`data: {"type":"content_block_stop","index":0}`,
			},
			stopReason:       "end_turn",
			withUsage:        true,
			wantFinishReason: "stop",
			wantChunkCount:   4,
		},
		{
			name: "text without usage",
			contentEvents: []string{
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
				`data: {"type":"content_block_stop","index":0}`,
			},
			stopReason:       "end_turn",
			wantFinishReason: "stop",
			wantChunkCount:   3,
		},
		{
			name: "tool with usage",
			contentEvents: []string{
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_opaque:/+==_-.","name":"weather","input":{}}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Moscow\"}"}}`,
				`data: {"type":"content_block_stop","index":0}`,
			},
			stopReason:       "tool_use",
			withUsage:        true,
			wantFinishReason: "tool_calls",
			wantToolID:       "toolu_opaque:/+==_-.",
			wantChunkCount:   5,
		},
		{
			name: "tool without usage",
			contentEvents: []string{
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_opaque:/+==_-.","name":"weather","input":{}}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Moscow\"}"}}`,
				`data: {"type":"content_block_stop","index":0}`,
			},
			stopReason:       "tool_use",
			wantFinishReason: "tool_calls",
			wantToolID:       "toolu_opaque:/+==_-.",
			wantChunkCount:   4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messageStart := `data: {"type":"message_start","message":{"id":"msg_fixture_terminal"}}`
			messageDelta := `data: {"type":"message_delta","delta":{"stop_reason":"` + tt.stopReason + `"}}`
			if tt.withUsage {
				messageStart = `data: {"type":"message_start","message":{"id":"msg_fixture_terminal","usage":{"input_tokens":12,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}`
				messageDelta = `data: {"type":"message_delta","delta":{"stop_reason":"` + tt.stopReason + `"},"usage":{"output_tokens":7}}`
			}
			lines := []string{messageStart}
			lines = append(lines, tt.contentEvents...)
			lines = append(lines, messageDelta, `data: {"type":"message_stop"}`)

			var output bytes.Buffer
			if err := TransformAnthropicStreamToOpenAI(
				strings.NewReader(strings.Join(lines, "\n\n")),
				"anthropic/claude-sonnet-4.5",
				&output,
			); err != nil {
				t.Fatalf("transform Anthropic stream: %v", err)
			}

			chunks, terminal := parseOpenAIChatStream(t, output.String())
			if terminal != "[DONE]" {
				t.Fatalf("unexpected terminal marker: %q", terminal)
			}
			if len(chunks) != tt.wantChunkCount {
				t.Fatalf("unexpected chunk count: got %d, want %d; output=%s", len(chunks), tt.wantChunkCount, output.String())
			}

			finishIndex := -1
			usageIndex := -1
			for index, chunk := range chunks {
				if chunk.Usage != nil {
					if usageIndex != -1 {
						t.Fatalf("multiple usage chunks: %s", output.String())
					}
					usageIndex = index
					if len(chunk.Choices) != 0 {
						t.Fatalf("usage chunk choices must be empty, got %+v", chunk.Choices)
					}
				}
				for _, choice := range chunk.Choices {
					if choice.FinishReason != nil {
						if finishIndex != -1 {
							t.Fatalf("multiple finish chunks: %s", output.String())
						}
						finishIndex = index
						if got := *choice.FinishReason; got != tt.wantFinishReason {
							t.Fatalf("unexpected finish reason: got %q, want %q", got, tt.wantFinishReason)
						}
					}
					for _, toolCall := range choice.Delta.ToolCalls {
						if toolCall.ID != "" && toolCall.ID != tt.wantToolID {
							t.Fatalf("opaque tool ID changed: got %q, want %q", toolCall.ID, tt.wantToolID)
						}
					}
				}
			}

			if finishIndex == -1 {
				t.Fatal("missing finish chunk")
			}
			if tt.withUsage {
				if usageIndex != finishIndex+1 {
					t.Fatalf("usage chunk must immediately follow finish: finish=%d usage=%d", finishIndex, usageIndex)
				}
				usage := chunks[usageIndex].Usage
				if usage.PromptTokens != 17 || usage.CompletionTokens != 7 || usage.TotalTokens != 24 {
					t.Fatalf("unexpected usage: %+v", usage)
				}
				if usage.PromptTokensDetails == nil ||
					usage.PromptTokensDetails.CachedTokens != 3 ||
					usage.PromptTokensDetails.CacheCreationTokens != 2 {
					t.Fatalf("unexpected prompt token details: %+v", usage.PromptTokensDetails)
				}
				assertUsageChoicesIsEmptyArray(t, output.String())
			} else if usageIndex != -1 {
				t.Fatalf("usage must remain absent, got chunk %d", usageIndex)
			}
		})
	}
}

func TestTransformAnthropicStreamToOpenAIPreservesUsagePresenceAndExplicitZeroes(t *testing.T) {
	tests := []struct {
		name        string
		usageEvents []string
		wantUsage   openai.OpenAIUsage
		wantDetails *openai.TokenDetails
	}{
		{
			name: "explicitly zero usage remains present",
			usageEvents: []string{
				`data: {"type":"message_start","message":{"id":"msg_zero","usage":{}}}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}`,
			},
		},
		{
			name: "omitted terminal cache fields preserve start values",
			usageEvents: []string{
				`data: {"type":"message_start","message":{"id":"msg_cache","usage":{"input_tokens":12,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
			},
			wantUsage: openai.OpenAIUsage{PromptTokens: 17, CompletionTokens: 7, TotalTokens: 24},
			wantDetails: &openai.TokenDetails{
				CachedTokens:        3,
				CacheCreationTokens: 2,
			},
		},
		{
			name: "explicit terminal cache zero overrides start values",
			usageEvents: []string{
				`data: {"type":"message_start","message":{"id":"msg_cache_zero","usage":{"input_tokens":12,"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`,
			},
			wantUsage: openai.OpenAIUsage{PromptTokens: 12, CompletionTokens: 7, TotalTokens: 19},
		},
		{
			name: "usage event can precede finish event",
			usageEvents: []string{
				`data: {"type":"message_start","message":{"id":"msg_split_usage","usage":{"input_tokens":12}}}`,
				`data: {"type":"message_delta","usage":{"output_tokens":7}}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			},
			wantUsage: openai.OpenAIUsage{PromptTokens: 12, CompletionTokens: 7, TotalTokens: 19},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lines := append([]string{}, tt.usageEvents...)
			lines = append(lines, `data: {"type":"message_stop"}`)

			var output bytes.Buffer
			if err := TransformAnthropicStreamToOpenAI(
				strings.NewReader(strings.Join(lines, "\n\n")),
				"anthropic/claude-sonnet-4.5",
				&output,
			); err != nil {
				t.Fatalf("transform Anthropic stream: %v", err)
			}

			chunks, terminal := parseOpenAIChatStream(t, output.String())
			if terminal != "[DONE]" {
				t.Fatalf("unexpected terminal marker: %q", terminal)
			}
			if len(chunks) < 3 {
				t.Fatalf("expected role, finish, and usage chunks, got %d: %s", len(chunks), output.String())
			}
			finishChunk := chunks[len(chunks)-2]
			usageChunk := chunks[len(chunks)-1]
			if len(finishChunk.Choices) != 1 || finishChunk.Choices[0].FinishReason == nil {
				t.Fatalf("penultimate chunk must be finish chunk: %+v", finishChunk)
			}
			if len(usageChunk.Choices) != 0 || usageChunk.Usage == nil {
				t.Fatalf("last chunk before [DONE] must be usage-only: %+v", usageChunk)
			}
			if got := usageChunk.Usage; got.PromptTokens != tt.wantUsage.PromptTokens ||
				got.CompletionTokens != tt.wantUsage.CompletionTokens ||
				got.TotalTokens != tt.wantUsage.TotalTokens {
				t.Fatalf("unexpected usage: got %+v, want %+v", got, tt.wantUsage)
			}
			if tt.wantDetails == nil {
				if usageChunk.Usage.PromptTokensDetails != nil {
					t.Fatalf("prompt token details must be absent: %+v", usageChunk.Usage.PromptTokensDetails)
				}
			} else if got := usageChunk.Usage.PromptTokensDetails; got == nil ||
				got.CachedTokens != tt.wantDetails.CachedTokens ||
				got.CacheCreationTokens != tt.wantDetails.CacheCreationTokens {
				t.Fatalf("unexpected prompt token details: got %+v, want %+v", got, tt.wantDetails)
			}
			assertUsageChoicesIsEmptyArray(t, output.String())
		})
	}
}

func assertUsageChoicesIsEmptyArray(t *testing.T, body string) {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &raw); err != nil {
			t.Fatalf("unmarshal raw stream chunk: %v", err)
		}
		if _, hasUsage := raw["usage"]; !hasUsage {
			continue
		}
		if got := strings.TrimSpace(string(raw["choices"])); got != "[]" {
			t.Fatalf("usage chunk choices must serialize as [], got %s", got)
		}
		return
	}
	t.Fatal("missing usage chunk")
}

func parseOpenAIChatStream(t *testing.T, body string) ([]openai.OpenAIStreamingChunk, string) {
	t.Helper()
	var chunks []openai.OpenAIStreamingChunk
	terminal := ""
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			terminal = data
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("unmarshal OpenAI stream chunk: %v; data=%s", err, data)
		}
		chunks = append(chunks, chunk)
	}
	return chunks, terminal
}
