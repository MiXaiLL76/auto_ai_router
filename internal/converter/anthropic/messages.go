package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// OpenAIToAnthropic converts an OpenAI Chat Completions request body to Anthropic Messages API
// format.  The model parameter overrides the model field in the request body when non-empty.
//
// Unsupported OpenAI parameters (silently ignored):
//   - n: Anthropic does not support multiple candidates per request
//   - frequency_penalty / presence_penalty / logit_bias: no Anthropic equivalent
//   - seed: no Anthropic equivalent
//   - response_format: instruct via system prompt instead
//   - logprobs: not supported
//   - modalities: Anthropic is text-only
//   - service_tier / store: not supported
//   - parallel_tool_calls: Anthropic always allows parallel tool calls
//   - prediction / verbosity / prompt_cache_key: not supported
func OpenAIToAnthropic(openAIBody []byte, model string) ([]byte, error) {
	var req openai.OpenAIRequest
	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	// Resolve model: caller-supplied parameter takes precedence.
	if model == "" {
		model = req.Model
	}

	// max_tokens is mandatory in Anthropic; default to 4096.
	maxTokens := 4096
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	if req.MaxCompletionTokens != nil {
		maxTokens = *req.MaxCompletionTokens
	}

	anthropicReq := AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Stream:    req.Stream,
	}

	// Temperature
	if req.Temperature != nil {
		anthropicReq.Temperature = req.Temperature
	}

	// TopP
	if req.TopP != nil {
		anthropicReq.TopP = req.TopP
	}

	// TopK (Anthropic extension, passed via extra_body)
	if req.ExtraBody != nil {
		if topK, ok := req.ExtraBody["top_k"].(float64); ok {
			v := int(topK)
			anthropicReq.TopK = &v
		}
	}

	// Stop sequences
	if req.Stop != nil {
		switch stop := req.Stop.(type) {
		case string:
			anthropicReq.StopSequences = []string{stop}
		case []interface{}:
			for _, s := range stop {
				if str, ok := s.(string); ok {
					anthropicReq.StopSequences = append(anthropicReq.StopSequences, str)
				}
			}
		}
	}

	// Thinking / reasoning config.
	// Prefer req.Thinking (direct Anthropic-style param), then extra_body["thinking"],
	// then fall back to reasoning_effort mapping.
	var thinkingParam interface{}
	if req.ExtraBody != nil {
		thinkingParam = req.ExtraBody["thinking"]
	}
	if req.Thinking != nil {
		thinkingParam = req.Thinking
	}
	if tc := mapThinkingConfig(thinkingParam, req.ReasoningEffort); tc != nil {
		anthropicReq.Thinking = tc
		// Anthropic requires temperature=1.0 when thinking is enabled.
		temp := 1.0
		anthropicReq.Temperature = &temp
	}

	// user → metadata.user_id
	if req.User != "" {
		anthropicReq.Metadata = &AnthropicMetadata{UserID: req.User}
	}

	// Tools
	if len(req.Tools) > 0 {
		anthropicReq.Tools = convertOpenAIToolsToAnthropic(req.Tools)
	}

	// Tool choice
	if req.ToolChoice != nil {
		anthropicReq.ToolChoice = mapToolChoice(req.ToolChoice)
	}

	// Messages (system messages are extracted to the top-level system field)
	systemContent, messages := convertOpenAIMessagesToAnthropic(req.Messages)
	anthropicReq.Messages = messages
	if systemContent != nil {
		anthropicReq.System = systemContent
	}

	return json.Marshal(anthropicReq)
}

// convertOpenAIMessagesToAnthropic converts the OpenAI messages array to Anthropic format.
// System / developer messages are aggregated into the top-level system field.
// Tool result messages become user-role messages containing tool_result blocks.
// Returns (systemContent, messages).
func convertOpenAIMessagesToAnthropic(openAIMessages []openai.OpenAIMessage) (interface{}, []AnthropicMessage) {
	var systemTexts []string
	var messages []AnthropicMessage

	for _, msg := range openAIMessages {
		switch msg.Role {
		case "system", "developer":
			texts := extractSystemText(msg.Content)
			systemTexts = append(systemTexts, texts...)

		case "user":
			blocks := convertOpenAIContentToAnthropic(msg.Content)
			if len(blocks) > 0 {
				messages = append(messages, AnthropicMessage{
					Role:    "user",
					Content: blocks,
				})
			}

		case "assistant":
			var blocks []ContentBlock
			if textBlocks := convertOpenAIContentToAnthropic(msg.Content); len(textBlocks) > 0 {
				blocks = append(blocks, textBlocks...)
			}
			if len(msg.ToolCalls) > 0 {
				toolBlocks := convertToolCallsToAnthropicContent(msg.ToolCalls)
				blocks = append(blocks, toolBlocks...)
			}
			if len(blocks) > 0 {
				messages = append(messages, AnthropicMessage{
					Role:    "assistant",
					Content: blocks,
				})
			}

		case "tool":
			// OpenAI: {role:"tool", tool_call_id:"...", content:"..."}
			// Anthropic: user message with a tool_result block.
			toolUseID := msg.ToolCallID
			if toolUseID == "" {
				toolUseID = msg.Name
			}
			resultContent := convertOpenAIContentToAnthropic(msg.Content)
			if len(resultContent) == 0 {
				resultContent = []ContentBlock{{Type: "text", Text: ""}}
			}
			toolResult := ContentBlock{
				Type:      "tool_result",
				ToolUseID: toolUseID,
				Content:   resultContent,
			}
			messages = append(messages, AnthropicMessage{
				Role:    "user",
				Content: []ContentBlock{toolResult},
			})
		}
	}

	var systemContent interface{}
	if len(systemTexts) > 0 {
		systemContent = strings.Join(systemTexts, "\n")
	}

	return systemContent, messages
}

// extractSystemText extracts all text strings from a system/developer message content.
func extractSystemText(content interface{}) []string {
	switch c := content.(type) {
	case string:
		if c != "" {
			return []string{c}
		}
	case []interface{}:
		var texts []string
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockMap["type"] == "text" {
					if text, ok := blockMap["text"].(string); ok && text != "" {
						texts = append(texts, text)
					}
				}
			}
		}
		return texts
	}
	return nil
}
