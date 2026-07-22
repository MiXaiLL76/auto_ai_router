package responses

import (
	"encoding/json"
	"fmt"
	"strings"
)

// chatToResponseConfig holds optional parameters for ChatToResponse.
type chatToResponseConfig struct {
	extraOutputItems []OutputItem
}

// ChatToResponseOption is a functional option for ChatToResponse.
type ChatToResponseOption func(*chatToResponseConfig)

// WithExtraOutputItems injects additional OutputItems into the converted response.
// Used by provider-specific post-processing (e.g. Vertex web search grounding → web_search_call).
func WithExtraOutputItems(items []OutputItem) ChatToResponseOption {
	return func(c *chatToResponseConfig) {
		c.extraOutputItems = append(c.extraOutputItems, items...)
	}
}

// ChatToResponse converts a Chat Completions response body to Responses API format.
func ChatToResponse(body []byte, opts ...ChatToResponseOption) ([]byte, error) {
	cfg := &chatToResponseConfig{}
	for _, o := range opts {
		o(cfg)
	}
	var ccResp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string      `json:"role"`
				Content interface{} `json:"content"`
				Refusal string      `json:"refusal,omitempty"`
				Images  []struct {
					B64JSON  string `json:"b64_json,omitempty"`
					ImageURL *struct {
						URL string `json:"url"`
					} `json:"image_url,omitempty"`
				} `json:"images,omitempty"`
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
				ImageTokens     int `json:"image_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
		} `json:"usage,omitempty"`
	}

	if err := json.Unmarshal(body, &ccResp); err != nil {
		return nil, fmt.Errorf("failed to parse chat completions response: %w", err)
	}

	// Build output items
	var output []OutputItem
	status := "completed"
	var incompleteDetails *IncompleteDetails

	if len(ccResp.Choices) > 0 {
		for _, choice := range ccResp.Choices {
			// Map finish_reason to status (use the first non-completed as overall status)
			switch choice.FinishReason {
			case "length":
				status = "incomplete"
				incompleteDetails = &IncompleteDetails{Reason: "max_output_tokens"}
			case "content_filter":
				status = "incomplete"
				incompleteDetails = &IncompleteDetails{Reason: "content_filter"}
			}

			// Add message output item if there's content or refusal
			msgContent := convertChatMessageContent(choice.Message.Content)
			if choice.Message.Refusal != "" {
				msgContent = append(msgContent, OutputContent{
					Type:    "output_refusal",
					Refusal: choice.Message.Refusal,
				})
			}
			if len(msgContent) > 0 {
				msgItem := OutputItem{
					Type:    "message",
					ID:      GenerateItemID("msg_"),
					Status:  "completed",
					Role:    "assistant",
					Content: msgContent,
				}
				output = append(output, msgItem)
			}

			for _, image := range choice.Message.Images {
				result, outputFormat := responseImageResult(image.B64JSON, image.ImageURL)
				if result == "" {
					continue
				}
				output = append(output, OutputItem{
					Type:         "image_generation_call",
					ID:           GenerateItemID("ig_"),
					Status:       "completed",
					Result:       result,
					OutputFormat: outputFormat,
				})
			}

			// Add function_call output items for each tool call
			for _, tc := range choice.Message.ToolCalls {
				fcItem := OutputItem{
					Type:      "function_call",
					ID:        GenerateItemID("fc_"),
					Status:    "completed",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				}
				output = append(output, fcItem)
			}
		}
	}

	// If no output items were created, add an empty message
	if len(output) == 0 {
		output = []OutputItem{
			{
				Type:   "message",
				ID:     GenerateItemID("msg_"),
				Status: "completed",
				Role:   "assistant",
				Content: []OutputContent{
					{
						Type:        "output_text",
						Text:        "",
						Annotations: []Annotation{},
					},
				},
			},
		}
	}

	// Build usage
	var usage *Usage
	if ccResp.Usage != nil {
		usage = &Usage{
			InputTokens:         ccResp.Usage.PromptTokens,
			OutputTokens:        ccResp.Usage.CompletionTokens,
			TotalTokens:         ccResp.Usage.TotalTokens,
			InputTokensDetails:  InputDetails{},
			OutputTokensDetails: OutputDetails{},
		}
		if ccResp.Usage.PromptTokensDetails != nil {
			usage.InputTokensDetails.CachedTokens = ccResp.Usage.PromptTokensDetails.CachedTokens
		}
		if ccResp.Usage.CompletionTokensDetails != nil {
			usage.OutputTokensDetails.ReasoningTokens = ccResp.Usage.CompletionTokensDetails.ReasoningTokens
			usage.OutputTokensDetails.ImageTokens = ccResp.Usage.CompletionTokensDetails.ImageTokens
		}
	}

	output = append(output, cfg.extraOutputItems...)

	resp := NewResponse(ResponseParams{
		ID:                GenerateResponseID(),
		Model:             ccResp.Model,
		CreatedAt:         ccResp.Created,
		Status:            status,
		IncompleteDetails: incompleteDetails,
		Output:            output,
		Usage:             usage,
		ToolChoice:        "auto",
		Store:             false,
	})

	result, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal responses API response: %w", err)
	}
	return result, nil
}

func responseImageResult(b64JSON string, imageURL *struct {
	URL string `json:"url"`
}) (result, outputFormat string) {
	if b64JSON != "" {
		return b64JSON, ""
	}
	if imageURL == nil {
		return "", ""
	}
	url := imageURL.URL
	comma := strings.IndexByte(url, ',')
	if comma < 0 || !strings.HasPrefix(strings.ToLower(url), "data:image/") {
		return "", ""
	}
	header := url[:comma]
	if !strings.Contains(strings.ToLower(header), ";base64") {
		return "", ""
	}
	mimeType := strings.TrimPrefix(strings.SplitN(header, ";", 2)[0], "data:")
	format := strings.TrimPrefix(strings.ToLower(mimeType), "image/")
	if format == "jpg" {
		format = "jpeg"
	}
	return url[comma+1:], format
}

func convertChatMessageContent(content interface{}) []OutputContent {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil
		}
		return []OutputContent{
			{
				Type:        "output_text",
				Text:        c,
				Annotations: []Annotation{},
			},
		}
	case []interface{}:
		var out []OutputContent
		for _, part := range c {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := partMap["type"].(string)
			switch partType {
			case "text":
				text, _ := partMap["text"].(string)
				out = append(out, OutputContent{
					Type:        "output_text",
					Text:        text,
					Annotations: []Annotation{},
				})
			}
		}
		return out
	default:
		return nil
	}
}
