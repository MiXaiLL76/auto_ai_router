package anthropicresponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// ResponsesRequestToAnthropic converts a Responses API request body to the
// Anthropic Messages API format (as JSON).
func ResponsesRequestToAnthropic(body []byte, model string) ([]byte, error) {
	var req responses.Request
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("ResponsesRequestToAnthropic: parse: %w", err)
	}

	anthropicReq, err := buildAnthropicRequest(&req, model)
	if err != nil {
		return nil, fmt.Errorf("ResponsesRequestToAnthropic: build: %w", err)
	}

	result, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("ResponsesRequestToAnthropic: marshal: %w", err)
	}
	return result, nil
}

func buildAnthropicRequest(req *responses.Request, model string) (*anthropic.AnthropicRequest, error) {
	if model == "" {
		model = req.Model
	}

	// max_tokens is mandatory; default to 4096.
	maxTokens := 4096
	if req.MaxOutputTokens != nil {
		maxTokens = *req.MaxOutputTokens
	}

	ar := &anthropic.AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Stream:    req.Stream,
	}

	// Scalar params. Drop temperature/top_p for models that no longer accept
	// sampling params (Claude Opus 4.7+ — see anthropic.SamplingRemoved).
	dropSampling := anthropic.SamplingRemoved(model)
	if req.Temperature != nil && !dropSampling {
		ar.Temperature = req.Temperature
	}
	if req.TopP != nil && !dropSampling {
		ar.TopP = req.TopP
	}

	// Stop sequences
	if req.Stop != nil {
		switch s := req.Stop.(type) {
		case string:
			ar.StopSequences = []string{s}
		case []interface{}:
			for _, sv := range s {
				if str, ok := sv.(string); ok {
					ar.StopSequences = append(ar.StopSequences, str)
				}
			}
		}
	}

	// System instruction
	if inst := responses.ExtractInstructionsText(req.Instructions); inst != "" {
		ar.System = inst
	}

	// Reasoning → extended thinking
	if req.Reasoning != nil && req.Reasoning.Effort != "" && req.Reasoning.Effort != "none" {
		tc, oc := anthropic.MapThinkingConfigFromEffort(req.Reasoning.Effort, ar.Model)
		if tc != nil {
			ar.Thinking = tc
			ar.OutputConfig = oc
			ar.MaxTokens = anthropic.EnsureMaxTokensForThinking(ar.MaxTokens, tc)
			if oc != nil {
				ar.AnthropicBeta = appendUnique(ar.AnthropicBeta, "effort-2025-11-24")
			} else {
				ar.AnthropicBeta = appendUnique(ar.AnthropicBeta, "interleaved-thinking-2025-05-14")
			}
		}
	}

	// Tools
	if len(req.Tools) > 0 {
		tools, betas := responsesToolsToAnthropic(req.Tools)
		if len(tools) > 0 {
			ar.Tools = tools
		}
		for _, b := range betas {
			ar.AnthropicBeta = appendUnique(ar.AnthropicBeta, b)
		}
	}

	// Tool choice
	if req.ToolChoice != nil {
		ar.ToolChoice = responsesToolChoiceToAnthropic(req.ToolChoice)
	}

	// Messages from input
	messages, err := inputToAnthropicMessages(req.Input)
	if err != nil {
		return nil, err
	}
	ar.Messages = messages

	return ar, nil
}

// inputToAnthropicMessages converts the Responses API input to Anthropic messages.
func inputToAnthropicMessages(input interface{}) ([]anthropic.AnthropicMessage, error) {
	if input == nil {
		return nil, nil
	}

	if s, ok := input.(string); ok {
		return []anthropic.AnthropicMessage{
			{Role: "user", Content: s},
		}, nil
	}

	raw, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	var items []interface{}
	if err := json.Unmarshal(raw, &items); err != nil {
		var single interface{}
		if err2 := json.Unmarshal(raw, &single); err2 != nil {
			return nil, fmt.Errorf("inputToAnthropicMessages: parse input: %w", err)
		}
		items = []interface{}{single}
	}

	var messages []anthropic.AnthropicMessage
	var pendingToolUse []anthropic.ContentBlock // assistant tool calls, flushed before tool results

	flushPendingToolUse := func() {
		if len(pendingToolUse) == 0 {
			return
		}
		messages = append(messages, anthropic.AnthropicMessage{
			Role:    "assistant",
			Content: pendingToolUse,
		})
		pendingToolUse = nil
	}

	for _, item := range items {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := itemMap["type"].(string)
		role, _ := itemMap["role"].(string)
		if itemType == "" && role != "" {
			itemType = "message"
		}

		switch itemType {
		case "message", "":
			flushPendingToolUse()
			msg, err := messageItemToAnthropic(itemMap, role)
			if err != nil {
				return nil, err
			}
			if msg != nil {
				messages = append(messages, *msg)
			}

		case "function_call":
			// Accumulate as a tool_use block in an assistant message.
			callID, _ := itemMap["call_id"].(string)
			name, _ := itemMap["name"].(string)
			arguments, _ := itemMap["arguments"].(string)
			var input interface{}
			_ = json.Unmarshal([]byte(arguments), &input)
			if input == nil {
				input = map[string]interface{}{}
			}
			pendingToolUse = append(pendingToolUse, anthropic.ContentBlock{
				Type:  "tool_use",
				ID:    callID,
				Name:  name,
				Input: input,
			})

		case "function_call_output":
			// Flush pending tool_use before emitting tool_result.
			flushPendingToolUse()
			callID, _ := itemMap["call_id"].(string)
			var content interface{}
			switch o := itemMap["output"].(type) {
			case string:
				content = o
			default:
				if b, err := json.Marshal(itemMap["output"]); err == nil {
					content = string(b)
				}
			}
			messages = append(messages, anthropic.AnthropicMessage{
				Role: "user",
				Content: []anthropic.ContentBlock{{
					Type:      "tool_result",
					ToolUseID: callID,
					Content:   content,
				}},
			})

		case "computer_call":
			// computer_call → tool_use block in an assistant message (action is input).
			callID, _ := itemMap["call_id"].(string)
			name, _ := itemMap["name"].(string)
			action := itemMap["action"]
			if action == nil {
				action = map[string]interface{}{}
			}
			pendingToolUse = append(pendingToolUse, anthropic.ContentBlock{
				Type:  "tool_use",
				ID:    callID,
				Name:  name,
				Input: action,
			})

		case "computer_call_output":
			// computer_call_output → tool_result user message with optional screenshot.
			flushPendingToolUse()
			callID, _ := itemMap["call_id"].(string)
			var resultContent interface{}
			if output, ok := itemMap["output"].(map[string]interface{}); ok {
				outputType, _ := output["type"].(string)
				if outputType == "computer_screenshot" || outputType == "input_image" {
					if imageURL, ok := output["image_url"].(string); ok && imageURL != "" {
						resultContent = []anthropic.ContentBlock{{
							Type:   "image",
							Source: &anthropic.MediaSource{Type: "url", URL: imageURL},
						}}
					} else if b64, ok := output["image_base64"].(string); ok && b64 != "" {
						mediaType, _ := output["media_type"].(string)
						if mediaType == "" {
							mediaType = "image/png"
						}
						resultContent = []anthropic.ContentBlock{{
							Type:   "image",
							Source: &anthropic.MediaSource{Type: "base64", MediaType: mediaType, Data: b64},
						}}
					}
				}
			}
			if resultContent == nil {
				resultContent = ""
			}
			messages = append(messages, anthropic.AnthropicMessage{
				Role: "user",
				Content: []anthropic.ContentBlock{{
					Type:      "tool_result",
					ToolUseID: callID,
					Content:   resultContent,
				}},
			})

		case "reasoning":
			// Include reasoning as a thinking block in an assistant message.
			flushPendingToolUse()
			var thinkingText strings.Builder
			if summary, ok := itemMap["summary"].([]interface{}); ok {
				for _, s := range summary {
					if sm, ok := s.(map[string]interface{}); ok {
						if t, ok := sm["text"].(string); ok {
							thinkingText.WriteString(t)
						}
					}
				}
			}
			encContent, _ := itemMap["encrypted_content"].(string)
			if thinkingText.Len() > 0 || encContent != "" {
				block := anthropic.ContentBlock{Type: "thinking"}
				if encContent != "" {
					// Preserve encrypted thinking for round-trip to same provider.
					block.Signature = encContent
				} else {
					block.Thinking = thinkingText.String()
				}
				messages = append(messages, anthropic.AnthropicMessage{
					Role:    "assistant",
					Content: []anthropic.ContentBlock{block},
				})
			}

		default:
			flushPendingToolUse()
		}
	}

	flushPendingToolUse()
	return messages, nil
}

// messageItemToAnthropic converts an input message item to an AnthropicMessage.
func messageItemToAnthropic(itemMap map[string]interface{}, role string) (*anthropic.AnthropicMessage, error) {
	anthropicRole := "user"
	if role == "assistant" || role == "model" {
		anthropicRole = "assistant"
	}

	rawContent := itemMap["content"]

	var content interface{}
	switch c := rawContent.(type) {
	case string:
		content = c

	case []interface{}:
		var blocks []anthropic.ContentBlock
		for _, part := range c {
			pm, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			block, err := contentPartToAnthropic(pm)
			if err != nil {
				return nil, err
			}
			if block != nil {
				blocks = append(blocks, *block)
			}
		}
		content = blocks

	default:
		content = ""
	}

	return &anthropic.AnthropicMessage{
		Role:    anthropicRole,
		Content: content,
	}, nil
}
