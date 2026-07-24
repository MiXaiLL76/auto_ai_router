package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

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
		ID:      openAIChatCompletionID(anthropicResp.ID),
		Object:  "chat.completion",
		Created: converterutil.GetCurrentTimestamp(),
		Model:   model,
		Choices: make([]openai.OpenAIChoice, 0),
	}

	// Translate content blocks to OpenAI message fields.
	var textParts []string // collect text parts separately, join with separator
	var reasoningContent string
	var toolCalls []openai.OpenAIToolCall

	for _, block := range anthropicResp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
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

	// join multiple text blocks with double newline separator
	textContent := ""
	if len(textParts) > 0 {
		textContent = strings.Join(textParts, "\n\n")
	}

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
	openAIResp.Usage = convertAnthropicUsageToOpenAI(anthropicResp.Usage)

	return json.Marshal(openAIResp)
}

// openAIChatCompletionID exposes an OpenAI-compatible response ID while keeping
// the upstream Anthropic message ID in the value for request correlation.
func openAIChatCompletionID(providerID string) string {
	if providerID == "" {
		return converterutil.GenerateID()
	}
	if strings.HasPrefix(providerID, "chatcmpl-") {
		return providerID
	}
	return "chatcmpl-" + providerID
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
	case "content_filter": // M2 — Anthropic content filtering stop reason
		return "content_filter"
	default:
		return "stop"
	}
}

// convertAnthropicUsageToOpenAI converts Anthropic usage to the OpenAI usage struct.
//
// Anthropic's input_tokens is exclusive of cache tokens:
//
//	total_input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
func convertAnthropicUsageToOpenAI(usage *AnthropicUsage) *openai.OpenAIUsage {
	if usage == nil {
		return nil
	}
	totalInputTokens := usage.InputTokens + usage.CacheCreationInputTokens + usage.CacheReadInputTokens
	result := &openai.OpenAIUsage{
		PromptTokens:     totalInputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      totalInputTokens + usage.OutputTokens,
	}
	// map cache_read + cache_creation tokens to prompt_tokens_details
	if usage.CacheReadInputTokens > 0 || usage.CacheCreationInputTokens > 0 {
		result.PromptTokensDetails = &openai.TokenDetails{
			CachedTokens:        usage.CacheReadInputTokens,
			CacheCreationTokens: usage.CacheCreationInputTokens,
		}
	}
	return result
}
