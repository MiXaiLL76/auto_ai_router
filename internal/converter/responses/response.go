package responses

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// generateResponseID generates a "resp_" prefixed unique ID.
func generateResponseID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "resp_" + hex.EncodeToString(b)
}

// generateItemID generates a short unique ID with the given prefix (e.g. "msg_", "fc_").
func generateItemID(prefix string) string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return prefix + hex.EncodeToString(b)
}

// ChatToResponse converts a Chat Completions response body to Responses API format.
func ChatToResponse(body []byte) ([]byte, error) {
	var ccResp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			TotalTokens         int `json:"total_tokens"`
			PromptTokensDetails *struct {
				CachedTokens int `json:"cached_tokens,omitempty"`
			} `json:"prompt_tokens_details,omitempty"`
			CompletionTokensDetails *struct {
				ReasoningTokens int `json:"reasoning_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
		} `json:"usage,omitempty"`
	}

	if err := json.Unmarshal(body, &ccResp); err != nil {
		return nil, fmt.Errorf("failed to parse chat completions response: %w", err)
	}

	// Build output items
	var output []OutputItem
	status := "completed"
	var incompleteDetails interface{}

	if len(ccResp.Choices) > 0 {
		choice := ccResp.Choices[0]

		// Map finish_reason to status
		switch choice.FinishReason {
		case "stop":
			status = "completed"
		case "length":
			status = "incomplete"
			incompleteDetails = map[string]interface{}{
				"reason": "max_output_tokens",
			}
		case "content_filter":
			status = "incomplete"
			incompleteDetails = map[string]interface{}{
				"reason": "content_filter",
			}
		case "tool_calls":
			status = "completed"
		default:
			status = "completed"
		}

		// Add message output item if there's text content
		if choice.Message.Content != "" {
			msgItem := OutputItem{
				Type:   "message",
				ID:     generateItemID("msg_"),
				Status: "completed",
				Role:   "assistant",
				Content: []OutputContent{
					{
						Type:        "output_text",
						Text:        choice.Message.Content,
						Annotations: []interface{}{},
					},
				},
			}
			output = append(output, msgItem)
		}

		// Add function_call output items for each tool call
		for _, tc := range choice.Message.ToolCalls {
			fcItem := OutputItem{
				Type:      "function_call",
				ID:        generateItemID("fc_"),
				Status:    "completed",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}
			output = append(output, fcItem)
		}
	}

	// If no output items were created, add an empty message
	if len(output) == 0 {
		output = []OutputItem{
			{
				Type:   "message",
				ID:     generateItemID("msg_"),
				Status: "completed",
				Role:   "assistant",
				Content: []OutputContent{
					{
						Type:        "output_text",
						Text:        "",
						Annotations: []interface{}{},
					},
				},
			},
		}
	}

	// Build usage
	var usage *Usage
	if ccResp.Usage != nil {
		usage = &Usage{
			InputTokens:  ccResp.Usage.PromptTokens,
			OutputTokens: ccResp.Usage.CompletionTokens,
			TotalTokens:  ccResp.Usage.TotalTokens,
			InputTokensDetails: &InputDetails{
				CachedTokens: 0,
			},
			OutputTokensDetails: &OutputDetails{
				ReasoningTokens: 0,
			},
		}
		if ccResp.Usage.PromptTokensDetails != nil {
			usage.InputTokensDetails.CachedTokens = ccResp.Usage.PromptTokensDetails.CachedTokens
		}
		if ccResp.Usage.CompletionTokensDetails != nil {
			usage.OutputTokensDetails.ReasoningTokens = ccResp.Usage.CompletionTokensDetails.ReasoningTokens
		}
	}

	resp := Response{
		ID:                 generateResponseID(),
		Object:             "response",
		CreatedAt:          ccResp.Created,
		Model:              ccResp.Model,
		Status:             status,
		Output:             output,
		Usage:              usage,
		Error:              nil,
		IncompleteDetails:  incompleteDetails,
		Metadata:           map[string]string{},
		Tools:              []Tool{},
		ParallelToolCalls:  true,
		Instructions:       nil,
		PreviousResponseID: nil,
		Store:              false,
	}

	result, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal responses API response: %w", err)
	}
	return result, nil
}
