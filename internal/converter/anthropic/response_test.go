package anthropic

import (
	"encoding/json"
	"regexp"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

func TestAnthropicToOpenAIUsesClientCompatibleCorrelatedID(t *testing.T) {
	providerID := "msg_fixture_multiblock"
	body, err := json.Marshal(AnthropicResponse{
		ID:         providerID,
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-sonnet-4-5",
		StopReason: "end_turn",
		Content:    []ContentBlock{{Type: "text", Text: "hello"}},
	})
	if err != nil {
		t.Fatalf("marshal Anthropic response: %v", err)
	}

	converted, err := AnthropicToOpenAI(body, "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	if !regexp.MustCompile(`^chatcmpl-[A-Za-z0-9_-]+$`).MatchString(response.ID) {
		t.Fatalf("response ID is not OpenAI-compatible: %q", response.ID)
	}
	if response.ID != "chatcmpl-"+providerID {
		t.Fatalf("response ID does not preserve provider correlation: %q", response.ID)
	}
}

func TestAnthropicToOpenAIPreservesUsagePresence(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantUsage *openai.OpenAIUsage
	}{
		{
			name: "provider usage absent",
			body: `{
				"id":"msg_without_usage",
				"type":"message",
				"role":"assistant",
				"model":"claude-sonnet-4-5",
				"content":[{"type":"text","text":"hello"}],
				"stop_reason":"end_turn"
			}`,
		},
		{
			name: "provider usage present",
			body: `{
				"id":"msg_with_usage",
				"type":"message",
				"role":"assistant",
				"model":"claude-sonnet-4-5",
				"content":[{"type":"text","text":"hello"}],
				"stop_reason":"end_turn",
				"usage":{
					"input_tokens":12,
					"cache_creation_input_tokens":2,
					"cache_read_input_tokens":3,
					"output_tokens":7
				}
			}`,
			wantUsage: &openai.OpenAIUsage{
				PromptTokens:     17,
				CompletionTokens: 7,
				TotalTokens:      24,
				PromptTokensDetails: &openai.TokenDetails{
					CachedTokens:        3,
					CacheCreationTokens: 2,
				},
			},
		},
		{
			name: "provider usage explicitly zero",
			body: `{
				"id":"msg_with_zero_usage",
				"type":"message",
				"role":"assistant",
				"model":"claude-sonnet-4-5",
				"content":[{"type":"text","text":"hello"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":0,"output_tokens":0}
			}`,
			wantUsage: &openai.OpenAIUsage{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converted, err := AnthropicToOpenAI([]byte(tt.body), "anthropic/claude-sonnet-4.5")
			if err != nil {
				t.Fatalf("convert Anthropic response: %v", err)
			}

			var response openai.OpenAIResponse
			if err := json.Unmarshal(converted, &response); err != nil {
				t.Fatalf("unmarshal OpenAI response: %v", err)
			}
			if tt.wantUsage == nil {
				if response.Usage != nil {
					t.Fatalf("usage must remain absent, got %+v", response.Usage)
				}
				var raw map[string]json.RawMessage
				if err := json.Unmarshal(converted, &raw); err != nil {
					t.Fatalf("unmarshal raw OpenAI response: %v", err)
				}
				if _, exists := raw["usage"]; exists {
					t.Fatalf("usage field must be omitted, got %s", converted)
				}
				return
			}

			if response.Usage == nil {
				t.Fatal("usage must be present")
			}
			if response.Usage.PromptTokens != tt.wantUsage.PromptTokens ||
				response.Usage.CompletionTokens != tt.wantUsage.CompletionTokens ||
				response.Usage.TotalTokens != tt.wantUsage.TotalTokens {
				t.Fatalf("unexpected usage: got %+v, want %+v", response.Usage, tt.wantUsage)
			}
			if tt.wantUsage.PromptTokensDetails == nil {
				if response.Usage.PromptTokensDetails != nil {
					t.Fatalf("prompt token details must be absent, got %+v", response.Usage.PromptTokensDetails)
				}
			} else if response.Usage.PromptTokensDetails == nil ||
				response.Usage.PromptTokensDetails.CachedTokens != tt.wantUsage.PromptTokensDetails.CachedTokens ||
				response.Usage.PromptTokensDetails.CacheCreationTokens != tt.wantUsage.PromptTokensDetails.CacheCreationTokens {
				t.Fatalf("unexpected prompt token details: got %+v, want %+v", response.Usage.PromptTokensDetails, tt.wantUsage.PromptTokensDetails)
			}
		})
	}
}

func TestAnthropicToOpenAIPreservesOpaqueToolID(t *testing.T) {
	const toolID = "toolu_opaque:/+==_-."
	body := []byte(`{
		"id":"msg_tool",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-5",
		"content":[{"type":"tool_use","id":"` + toolID + `","name":"weather","input":{"city":"Moscow"}}],
		"stop_reason":"tool_use",
		"usage":{"input_tokens":4,"output_tokens":2}
	}`)

	converted, err := AnthropicToOpenAI(body, "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("convert Anthropic response: %v", err)
	}

	var response openai.OpenAIResponse
	if err := json.Unmarshal(converted, &response); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v", err)
	}
	if len(response.Choices) != 1 || len(response.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %+v", response.Choices)
	}
	if got := response.Choices[0].Message.ToolCalls[0].ID; got != toolID {
		t.Fatalf("opaque tool ID changed: got %q, want %q", got, toolID)
	}
}
