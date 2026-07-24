package anthropicresponses

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// AnthropicToResponsesResponse converts an Anthropic Messages API response body to a
// responses.Response, using displayModelID as the echoed model name.
func AnthropicToResponsesResponse(body []byte, displayModelID, responseID string, createdAt int64) (*responses.Response, error) {
	var ar anthropic.AnthropicResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, fmt.Errorf("AnthropicToResponsesResponse: parse: %w", err)
	}
	if responseID == "" {
		responseID = responses.GenerateResponseID()
	}
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	if displayModelID == "" {
		displayModelID = ar.Model
	}
	return buildAnthropicResponse(&ar, displayModelID, responseID, createdAt), nil
}

func buildAnthropicResponse(
	ar *anthropic.AnthropicResponse,
	model, responseID string,
	createdAt int64,
) *responses.Response {
	status, incompleteDetails := anthropicStopReasonToStatus(ar.StopReason)
	output := anthropicContentToOutputItems(ar.Content)
	usage := anthropicUsageToUsage(ar.Usage)
	completedAt := createdAt
	return responses.BuildCompletedResponse(responses.CompletedResponseParams{
		ID:                responseID,
		Model:             model,
		CreatedAt:         createdAt,
		CompletedAt:       &completedAt,
		Status:            status,
		IncompleteDetails: incompleteDetails,
		Output:            output,
		Usage:             usage,
	})
}

// anthropicContentToOutputItems converts Anthropic content blocks to Responses API output items.
func anthropicContentToOutputItems(blocks []anthropic.ContentBlock) []responses.OutputItem {
	var output []responses.OutputItem
	var msgContent []responses.OutputContent

	flushMessage := func() {
		if len(msgContent) == 0 {
			return
		}
		output = append(output, responses.OutputItem{
			Type:    "message",
			ID:      responses.GenerateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: msgContent,
		})
		msgContent = nil
	}

	for _, block := range blocks {
		switch block.Type {
		case "text":
			msgContent = append(msgContent, responses.OutputContent{
				Type:        "output_text",
				Text:        block.Text,
				Annotations: []responses.Annotation{},
			})

		case "thinking":
			// Flush any accumulated text first.
			flushMessage()
			reasoningItem := responses.OutputItem{
				Type:   "reasoning",
				ID:     responses.GenerateItemID("rs_"),
				Status: "completed",
			}
			if block.Thinking != "" {
				reasoningItem.Summary = []responses.OutputContent{
					{Type: "summary_text", Text: block.Thinking},
				}
			}
			if block.Signature != "" {
				reasoningItem.EncryptedContent = block.Signature
			}
			output = append(output, reasoningItem)

		case "tool_use":
			// Flush accumulated text before a tool call.
			flushMessage()
			// Detect computer_use tool call by tool name.
			// The name "computer" is the canonical discriminator; the action-key
			// heuristic is kept as a fallback for non-standard names.
			isComputerCall := block.Name == "computer"
			if !isComputerCall {
				if inputMap, ok := block.Input.(map[string]interface{}); ok {
					_, isComputerCall = inputMap["action"]
				}
			}
			if isComputerCall {
				output = append(output, responses.OutputItem{
					Type:   "computer_call",
					ID:     responses.GenerateItemID("cc_"),
					Status: "completed",
					CallID: block.ID,
					Name:   block.Name,
					Action: block.Input,
				})
				continue
			}
			argsJSON := "{}"
			if block.Input != nil {
				if b, err := json.Marshal(block.Input); err == nil {
					argsJSON = string(b)
				}
			}
			output = append(output, responses.OutputItem{
				Type:      "function_call",
				ID:        responses.GenerateItemID("fc_"),
				Status:    "completed",
				CallID:    block.ID,
				Name:      block.Name,
				Arguments: argsJSON,
			})
		}
	}

	flushMessage()

	if len(output) == 0 {
		output = []responses.OutputItem{{
			Type:    "message",
			ID:      responses.GenerateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: []responses.OutputContent{{Type: "output_text", Text: "", Annotations: []responses.Annotation{}}},
		}}
	}

	return output
}

// anthropicStopReasonToStatus maps Anthropic stop_reason to Responses API status.
func anthropicStopReasonToStatus(stopReason string) (string, *responses.IncompleteDetails) {
	switch stopReason {
	case "end_turn", "tool_use", "":
		return "completed", nil
	case "max_tokens":
		return "incomplete", &responses.IncompleteDetails{Reason: "max_output_tokens"}
	case "stop_sequence":
		return "completed", nil
	default:
		return "completed", nil
	}
}

// anthropicUsageToUsage converts Anthropic usage to responses.Usage.
func anthropicUsageToUsage(au *anthropic.AnthropicUsage) *responses.Usage {
	if au == nil {
		return nil
	}
	return &responses.Usage{
		InputTokens:  au.InputTokens,
		OutputTokens: au.OutputTokens,
		TotalTokens:  au.InputTokens + au.OutputTokens,
		InputTokensDetails: responses.InputDetails{
			CachedTokens: au.CacheReadInputTokens,
		},
		OutputTokensDetails: responses.OutputDetails{},
	}
}
