package responses

import (
	"encoding/json"
	"fmt"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// ResponseToChat converts a Responses API response body into a Chat
// Completions response body -- the mirror of ChatToResponse, used for a
// model marked responses_only (see ChatRequestToResponses/config.go's
// ResponsesOnly): AIR called the provider's native /v1/responses on the
// client's behalf, and must hand back exactly the shape a
// /v1/chat/completions caller expects.
func ResponseToChat(body []byte) ([]byte, error) {
	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse responses API response: %w", err)
	}

	var textParts, reasoningParts []string
	var refusal string
	var toolCalls []openai.OpenAIToolCall
	var images []openai.ImageData
	hasFunctionCall := false

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					if c.Text != "" {
						textParts = append(textParts, c.Text)
					}
				case "output_refusal", "refusal":
					if c.Refusal != "" {
						refusal = c.Refusal
					}
				}
			}
		case "reasoning":
			for _, s := range item.Summary {
				if s.Text != "" {
					reasoningParts = append(reasoningParts, s.Text)
				}
			}
		case "function_call":
			hasFunctionCall = true
			// item.CallID (not item.ID) becomes the Chat Completions tool_call
			// "id": that's the value the client will echo back as
			// tool_call_id on its next "tool" role message, and
			// chatMessagesToInput reads it back out under the same key
			// (call_id) to build the matching function_call_output item --
			// keeping a multi-turn tool-use conversation correlated across
			// the round trip.
			toolCalls = append(toolCalls, openai.OpenAIToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: openai.OpenAIToolFunction{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		case "image_generation_call":
			if item.Result != "" {
				images = append(images, openai.ImageData{
					Type:    "image_url",
					B64JSON: item.Result,
				})
			}
			// web_search_call, file_search_call, code_interpreter_call, etc.:
			// hosted tool calls with no Chat Completions message-visible
			// equivalent. Their token cost is already reflected in
			// resp.Usage.ServerToolUse (see responsesUsageToChat) if the
			// provider reports one; nothing further to surface here.
		}
	}

	message := openai.OpenAIResponseMessage{
		Role:    "assistant",
		Content: strings.Join(textParts, "\n\n"),
	}
	if refusal != "" {
		message.Refusal = refusal
	}
	if len(reasoningParts) > 0 {
		message.ReasoningContent = strings.Join(reasoningParts, "\n\n")
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}
	if len(images) > 0 {
		message.Images = images
	}

	// A response that failed outright (status="failed" inside an otherwise
	// 2xx HTTP response -- the Responses API's async-style error shape) has
	// no output items at all; surface the embedded error text as content
	// instead of handing back an empty message with no explanation.
	if resp.Status == "failed" && message.Content == "" && len(toolCalls) == 0 {
		if errMap, ok := resp.Error.(map[string]interface{}); ok {
			if msg, ok := errMap["message"].(string); ok && msg != "" {
				message.Content = msg
			}
		}
	}

	openAIResp := openai.OpenAIResponse{
		ID:      chatCompletionIDFromResponses(resp.ID),
		Object:  "chat.completion",
		Created: resp.CreatedAt,
		Model:   resp.Model,
		Choices: []openai.OpenAIChoice{
			{
				Index:        0,
				Message:      message,
				FinishReason: responsesFinishReason(resp.Status, resp.IncompleteDetails, hasFunctionCall),
			},
		},
		Usage: responsesUsageToChat(resp.Usage),
	}

	result, err := json.Marshal(openAIResp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat completions response: %w", err)
	}
	return result, nil
}

// chatCompletionIDFromResponses exposes a Chat-Completions-shaped response
// ID while keeping the upstream Responses API response ID in the value, for
// request correlation -- same convention as
// anthropic.openAIChatCompletionID/vertex's ID handling.
func chatCompletionIDFromResponses(responseID string) string {
	if responseID == "" {
		return converterutil.GenerateID()
	}
	if strings.HasPrefix(responseID, "chatcmpl-") {
		return responseID
	}
	return "chatcmpl-" + responseID
}

// responsesFinishReason maps a Responses API response's status/
// incomplete_details (and whether it produced any function_call output
// item) to a Chat Completions finish_reason. Mirrors ChatToResponse's own
// status derivation (response.go) in reverse, plus the same "tool_calls"
// override every other provider converter in this codebase already applies
// on a tool call (see anthropic.AnthropicToOpenAI / vertex.VertexToOpenAI).
func responsesFinishReason(status string, incomplete *IncompleteDetails, hasFunctionCall bool) string {
	if hasFunctionCall {
		return "tool_calls"
	}
	if status == "incomplete" && incomplete != nil {
		switch incomplete.Reason {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		}
	}
	return "stop"
}

// responsesUsageToChat converts a Responses API Usage into the Chat
// Completions OpenAIUsage shape. Mirrors ChatToResponse's usage conversion
// (response.go) in reverse.
func responsesUsageToChat(usage *Usage) *openai.OpenAIUsage {
	if usage == nil {
		return nil
	}
	result := &openai.OpenAIUsage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.TotalTokens,
	}

	d := usage.InputTokensDetails
	if d.CachedTokens > 0 || d.CachedAudioTokens > 0 || d.CacheCreationTokens > 0 || d.CacheWriteTokens > 0 || d.AudioTokens > 0 {
		result.PromptTokensDetails = &openai.TokenDetails{
			CachedTokens:        d.CachedTokens,
			CachedAudioTokens:   d.CachedAudioTokens,
			CacheCreationTokens: d.CacheCreationTokens,
			CacheWriteTokens:    d.CacheWriteTokens,
			AudioTokens:         d.AudioTokens,
		}
		if cc := d.CacheCreationTokenDetails; cc != nil {
			result.PromptTokensDetails.CacheCreationTokenDetails = &openai.CacheCreationTokenDetails{
				Ephemeral5mInputTokens: cc.Ephemeral5mInputTokens,
				Ephemeral1hInputTokens: cc.Ephemeral1hInputTokens,
			}
		}
	}

	od := usage.OutputTokensDetails
	if od.ReasoningTokens > 0 || od.AudioTokens > 0 || od.ImageTokens > 0 {
		result.CompletionTokensDetails = &openai.CompletionTokenDetails{
			ReasoningTokens: od.ReasoningTokens,
			AudioTokens:     od.AudioTokens,
			ImageTokens:     od.ImageTokens,
		}
	}

	if usage.ServerToolUse != nil && usage.ServerToolUse.WebSearchRequests > 0 {
		result.ServerToolUse = &openai.ServerToolUseDetails{
			WebSearchRequests: usage.ServerToolUse.WebSearchRequests,
		}
	}

	return result
}
