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
	// then fall back to OpenAI reasoning effort mapping.
	var thinkingParam interface{}
	if req.ExtraBody != nil {
		thinkingParam = req.ExtraBody["thinking"]
	}
	if req.Thinking != nil {
		thinkingParam = req.Thinking
	}
	reasoningEffort := openAIReasoningEffort(req)
	if tc, oc := mapThinkingConfig(thinkingParam, reasoningEffort, anthropicReq.Model); tc != nil {
		anthropicReq.Thinking = tc
		anthropicReq.OutputConfig = oc
		anthropicReq.MaxTokens = EnsureMaxTokensForThinking(anthropicReq.MaxTokens, tc)
		if oc != nil {
			// effort-based adaptive thinking requires the effort beta header
			anthropicReq.AnthropicBeta = appendBetaUnique(anthropicReq.AnthropicBeta, "effort-2025-11-24")
		}
		// Anthropic requires temperature=1.0 when thinking is enabled.
		temp := 1.0
		anthropicReq.Temperature = &temp
	}

	// Anthropic has no native response_format; we inject a JSON instruction
	// into the system prompt when the caller requests JSON output.
	if req.ResponseFormat != nil {
		if rfMap, ok := req.ResponseFormat.(map[string]interface{}); ok {
			if rfType, _ := rfMap["type"].(string); rfType == "json_object" || rfType == "json_schema" {
				jsonInstruction := "\n\nYou must respond with valid JSON only. Do not include any text outside of the JSON object."
				if systemContent, ok := anthropicReq.System.(string); ok {
					anthropicReq.System = systemContent + jsonInstruction
				} else if anthropicReq.System == nil {
					anthropicReq.System = strings.TrimPrefix(jsonInstruction, "\n\n")
				}
			}
		}
	}

	// user → metadata.user_id
	if req.User != "" {
		anthropicReq.Metadata = &AnthropicMetadata{UserID: req.User}
	}

	// Tools
	if len(req.Tools) > 0 {
		anthropicReq.Tools = convertOpenAIToolsToAnthropic(req.Tools)
	}

	// Tool choice: map standard OpenAI format first, then let extra_body override
	// with Anthropic-native format (e.g. {"type":"allowed_tools",...}).
	if req.ToolChoice != nil {
		anthropicReq.ToolChoice = mapToolChoice(req.ToolChoice)
	}
	if req.ExtraBody != nil {
		if tc, ok := req.ExtraBody["tool_choice"]; ok && tc != nil {
			anthropicReq.ToolChoice = tc
		}
	}

	// allowed_tools is not supported by Bedrock/Anthropic API.
	// Convert it by filtering the tools array to only the allowed subset,
	// then replacing tool_choice with {"type": mode} (auto or any).
	if tc, ok := anthropicReq.ToolChoice.(map[string]interface{}); ok {
		if tc["type"] == "allowed_tools" {
			anthropicReq.ToolChoice, anthropicReq.Tools = expandAllowedTools(tc, anthropicReq.Tools)
		}
	}

	// anthropic_beta from extra_body (e.g. ["prompt-caching-2024-07-31"])
	if req.ExtraBody != nil {
		if beta, ok := req.ExtraBody["anthropic_beta"].([]interface{}); ok {
			for _, b := range beta {
				if s, ok := b.(string); ok {
					anthropicReq.AnthropicBeta = append(anthropicReq.AnthropicBeta, s)
				}
			}
		}
	}

	// Messages (system messages are extracted to the top-level system field)
	systemContent, messages := convertOpenAIMessagesToAnthropic(req.Messages)
	anthropicReq.Messages = messages
	if systemContent != nil {
		anthropicReq.System = systemContent
	}

	return json.Marshal(anthropicReq)
}

func openAIReasoningEffort(req openai.OpenAIRequest) string {
	effort := req.ReasoningEffort
	if req.ExtraBody != nil {
		if nested := reasoningObjectEffort(req.ExtraBody["reasoning"]); nested != "" {
			effort = nested
		}
	}
	if nested := reasoningObjectEffort(req.Reasoning); nested != "" {
		effort = nested
	}
	return effort
}

func reasoningObjectEffort(value interface{}) string {
	reasoning, ok := value.(map[string]interface{})
	if !ok {
		return ""
	}
	effort, _ := reasoning["effort"].(string)
	return strings.TrimSpace(effort)
}

// convertOpenAIMessagesToAnthropic converts the OpenAI messages array to Anthropic format.
// System / developer messages are aggregated into the top-level system field.
// Tool result messages become user-role messages containing tool_result blocks.
// Returns (systemContent, messages).
func convertOpenAIMessagesToAnthropic(openAIMessages []openai.OpenAIMessage) (interface{}, []AnthropicMessage) {
	var allSystemBlocks []ContentBlock
	var messages []AnthropicMessage

	for _, msg := range openAIMessages {
		switch msg.Role {
		case "system", "developer":
			sysBlocks := extractSystemBlocks(msg.Content)
			allSystemBlocks = append(allSystemBlocks, sysBlocks...)

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
			for _, thinking := range msg.ThinkingBlocks {
				if thinking.Type != "thinking" && thinking.Type != "redacted_thinking" {
					continue
				}
				blocks = append(blocks, ContentBlock{
					Type:         thinking.Type,
					Thinking:     thinking.Thinking,
					Signature:    thinking.Signature,
					Data:         thinking.Data,
					CacheControl: thinking.CacheControl,
				})
			}
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
	if len(allSystemBlocks) > 0 {
		hasCacheControl := false
		for _, b := range allSystemBlocks {
			if b.CacheControl != nil {
				hasCacheControl = true
				break
			}
		}
		if hasCacheControl {
			systemContent = allSystemBlocks
		} else {
			texts := make([]string, 0, len(allSystemBlocks))
			for _, b := range allSystemBlocks {
				if b.Text != "" {
					texts = append(texts, b.Text)
				}
			}
			if len(texts) > 0 {
				systemContent = strings.Join(texts, "\n")
			}
		}
	}

	// Merge consecutive same-role messages into a single message.
	messages = mergeConsecutiveSameRole(messages)

	return systemContent, messages
}

// mergeConsecutiveSameRole merges consecutive messages with the same role
// into a single message. Anthropic rejects requests with consecutive
// same-role messages (e.g. two user messages in a row).
func mergeConsecutiveSameRole(messages []AnthropicMessage) []AnthropicMessage {
	if len(messages) <= 1 {
		return messages
	}
	merged := make([]AnthropicMessage, 0, len(messages))
	for _, msg := range messages {
		blocks := toContentBlocks(msg.Content)
		if len(merged) > 0 && merged[len(merged)-1].Role == msg.Role {
			// Append blocks to previous message
			prevBlocks := toContentBlocks(merged[len(merged)-1].Content)
			prevBlocks = append(prevBlocks, blocks...)
			merged[len(merged)-1].Content = prevBlocks
		} else {
			merged = append(merged, AnthropicMessage{
				Role:    msg.Role,
				Content: blocks,
			})
		}
	}
	return merged
}

// toContentBlocks normalises a message Content value (string or []ContentBlock)
// into a []ContentBlock slice.
func toContentBlocks(content interface{}) []ContentBlock {
	switch c := content.(type) {
	case []ContentBlock:
		return c
	case string:
		if c == "" {
			return nil
		}
		return []ContentBlock{{Type: "text", Text: c}}
	}
	return nil
}

// OpenAIToBedrock converts an OpenAI Chat Completions request body to AWS Bedrock Runtime
// format. It reuses the Anthropic converter and then:
//   - Removes the "model" field (Bedrock gets model from the URL path)
//   - Adds "anthropic_version": "bedrock-2023-05-31"
func OpenAIToBedrock(openAIBody []byte, model string) ([]byte, error) {
	anthropicBody, err := OpenAIToAnthropic(openAIBody, model)
	if err != nil {
		return nil, err
	}

	var body map[string]interface{}
	if err := json.Unmarshal(anthropicBody, &body); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic request for Bedrock conversion: %w", err)
	}

	delete(body, "model")
	delete(body, "stream")
	body["anthropic_version"] = "bedrock-2023-05-31"

	return json.Marshal(body)
}

// extractSystemBlocks extracts system/developer message content into ContentBlocks,
// preserving cache_control markers for Anthropic prompt caching.
func extractSystemBlocks(content interface{}) []ContentBlock {
	switch c := content.(type) {
	case string:
		if c != "" {
			return []ContentBlock{{Type: "text", Text: c}}
		}
	case []interface{}:
		var blocks []ContentBlock
		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] == "text" {
				text, _ := blockMap["text"].(string)
				cb := ContentBlock{Type: "text", Text: text}
				cb.CacheControl = blockMap["cache_control"]
				blocks = append(blocks, cb)
			}
		}
		return blocks
	}
	return nil
}
