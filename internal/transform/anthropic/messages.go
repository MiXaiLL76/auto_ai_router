package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// AnthropicRequest represents the Anthropic Messages API request format
type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	System        interface{}        `json:"system,omitempty"` // string or []TextBlock
	MaxTokens     int                `json:"max_tokens"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	Tools         []interface{}      `json:"tools,omitempty"`
	ToolChoice    interface{}        `json:"tool_choice,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ContentBlock
}

type TextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ImageBlock struct {
	Type   string      `json:"type"`
	Source ImageSource `json:"source"`
}

type ImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	Data      string `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	URL       string `json:"url,omitempty"`
}

// AnthropicResponse represents Anthropic Messages API response format
type AnthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage          `json:"usage"`
}

type AnthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// OpenAIToAnthropic converts OpenAI format request to Anthropic format
func OpenAIToAnthropic(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	anthropicReq := AnthropicRequest{
		Model:    openAIReq.Model,
		Stream:   openAIReq.Stream,
		Messages: make([]AnthropicMessage, 0),
	}

	// Set temperature and top_p if provided
	if openAIReq.Temperature != nil {
		anthropicReq.Temperature = openAIReq.Temperature
	}
	if openAIReq.TopP != nil {
		anthropicReq.TopP = openAIReq.TopP
	}

	// Set max_tokens (required for Anthropic)
	if openAIReq.MaxTokens != nil {
		anthropicReq.MaxTokens = *openAIReq.MaxTokens
	} else {
		anthropicReq.MaxTokens = 4096
	}

	// Handle stop sequences
	if openAIReq.Stop != nil {
		switch s := openAIReq.Stop.(type) {
		case string:
			anthropicReq.StopSequences = []string{s}
		case []interface{}:
			for _, item := range s {
				if str, ok := item.(string); ok {
					anthropicReq.StopSequences = append(anthropicReq.StopSequences, str)
				}
			}
		}
	}

	// Convert tools if present
	if len(openAIReq.Tools) > 0 {
		anthropicReq.Tools = openAIReq.Tools
	}
	if openAIReq.ToolChoice != nil {
		anthropicReq.ToolChoice = openAIReq.ToolChoice
	}

	// Process messages
	for _, msg := range openAIReq.Messages {
		if msg.Role == "system" {
			// System messages go to the top-level system field
			anthropicReq.System = extractTextContent(msg.Content)
		} else {
			// Convert message content
			anthropicReq.Messages = append(anthropicReq.Messages, AnthropicMessage{
				Role:    msg.Role,
				Content: convertOpenAIContentToAnthropic(msg.Content),
			})
		}
	}

	return json.Marshal(anthropicReq)
}

// AnthropicToOpenAI converts Anthropic response to OpenAI format
func AnthropicToOpenAI(anthropicBody []byte, model string) ([]byte, error) {
	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(anthropicBody, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	// Extract text content from Anthropic response
	var content string
	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	openAIResp := openai.OpenAIResponse{
		ID:      anthropicResp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.OpenAIChoice{
			{
				Index: 0,
				Message: openai.OpenAIResponseMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: mapAnthropicStopReason(anthropicResp.StopReason),
			},
		},
		Usage: &openai.OpenAIUsage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
	}

	return json.Marshal(openAIResp)
}

// Helper functions

func extractTextContent(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var parts []string
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockMap["type"] == "text" {
					if text, ok := blockMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func convertOpenAIContentToAnthropic(content interface{}) interface{} {
	// If it's a simple string, return as is
	if s, ok := content.(string); ok {
		return s
	}

	// If it's a complex array of content blocks
	if parts, ok := content.([]interface{}); ok {
		var anthropicBlocks []interface{}
		for _, p := range parts {
			part, ok := p.(map[string]interface{})
			if !ok {
				continue
			}

			contentType, _ := part["type"].(string)
			switch contentType {
			case "text":
				if text, ok := part["text"].(string); ok {
					anthropicBlocks = append(anthropicBlocks, map[string]interface{}{
						"type": "text",
						"text": text,
					})
				}
			case "image_url":
				if imageBlock, ok := convertImageURL(part); ok {
					anthropicBlocks = append(anthropicBlocks, imageBlock)
				}
			}
		}
		if len(anthropicBlocks) > 0 {
			return anthropicBlocks
		}
	}

	return content
}

func convertImageURL(part map[string]interface{}) (map[string]interface{}, bool) {
	urlObj, ok := part["image_url"].(map[string]interface{})
	if !ok {
		return nil, false
	}

	url, _ := urlObj["url"].(string)
	if url == "" {
		return nil, false
	}

	if strings.HasPrefix(url, "data:") {
		return convertDataURL(url)
	}

	// Handle regular URLs
	return map[string]interface{}{
		"type": "image",
		"source": map[string]interface{}{
			"type": "url",
			"url":  url,
		},
	}, true
}

func convertDataURL(dataURL string) (map[string]interface{}, bool) {
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return nil, false
	}

	header := parts[0] // "data:image/jpeg;base64"
	data := parts[1]   // base64 data

	mediaType := parseMediaType(header)
	if mediaType == "" {
		return nil, false
	}

	return map[string]interface{}{
		"type": "image",
		"source": map[string]interface{}{
			"type":       "base64",
			"media_type": mediaType,
			"data":       data,
		},
	}, true
}

func parseMediaType(header string) string {
	// Extract media type from header like "data:image/jpeg;base64"
	if strings.Contains(header, "image/png") {
		return "image/png"
	} else if strings.Contains(header, "image/gif") {
		return "image/gif"
	} else if strings.Contains(header, "image/webp") {
		return "image/webp"
	} else if strings.Contains(header, "image/") {
		start := strings.Index(header, "image/")
		if start >= 0 {
			end := strings.IndexAny(header[start:], ";")
			if end > 0 {
				return header[start : start+end]
			}
			return header[start:]
		}
	}
	return "image/jpeg" // default
}

func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "stop_sequence":
		return "stop"
	case "tool_use":
		return "tool_calls"
	default:
		return "stop"
	}
}
