package anthropic

import (
	"encoding/json"
	"fmt"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// AnthropicToOpenAI converts an Anthropic Messages API response body to OpenAI
// Chat Completions response format.
func AnthropicToOpenAI(anthropicBody []byte, model string) ([]byte, error) {
	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(anthropicBody, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	openAIResp := openai.OpenAIResponse{
		ID:      anthropicResp.ID,
		Object:  "chat.completion",
		Created: converterutil.GetCurrentTimestamp(),
		Model:   model,
		Choices: make([]openai.OpenAIChoice, 0),
	}

	// Translate content blocks to OpenAI message fields.
	var textContent string
	var reasoningContent string
	var toolCalls []openai.OpenAIToolCall

	for _, block := range anthropicResp.Content {
		switch block.Type {
		case "text":
			textContent += block.Text
		case "thinking":
			reasoningContent += block.Thinking
		case "tool_use":
			argsJSON := "{}"
			if block.Input != nil {
				if data, err := json.Marshal(block.Input); err == nil {
					argsJSON = string(data)
				}
			}
			toolCalls = append(toolCalls, openai.OpenAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.OpenAIToolFunction{
					Name:      block.Name,
					Arguments: argsJSON,
				},
			})
		}
	}

	finishReason := mapAnthropicStopReason(anthropicResp.StopReason)

	message := openai.OpenAIResponseMessage{
		Role:    "assistant",
		Content: textContent,
	}
	if reasoningContent != "" {
		message.ReasoningContent = reasoningContent
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}

	choice := openai.OpenAIChoice{
		Index:        0,
		Message:      message,
		FinishReason: finishReason,
	}
	openAIResp.Choices = append(openAIResp.Choices, choice)

	// Usage
	openAIResp.Usage = convertAnthropicUsageToOpenAI(&anthropicResp.Usage)

	return json.Marshal(openAIResp)
}

// mapAnthropicStopReason maps an Anthropic stop_reason value to the OpenAI finish_reason.
func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	default:
		return "stop"
	}
}

// convertAnthropicUsageToOpenAI converts Anthropic usage to the OpenAI usage struct.
func convertAnthropicUsageToOpenAI(usage *AnthropicUsage) *openai.OpenAIUsage {
	if usage == nil {
		return nil
	}
	result := &openai.OpenAIUsage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.InputTokens + usage.OutputTokens,
	}
	if usage.CacheReadInputTokens > 0 {
		result.PromptTokensDetails = &openai.TokenDetails{
			CachedTokens: usage.CacheReadInputTokens,
		}
	}
	return result
}
