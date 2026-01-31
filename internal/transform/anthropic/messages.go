package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// OpenAIToAnthropic converts OpenAI format request to Anthropic format using SDK types
func OpenAIToAnthropic(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	// Create request with SDK MessageNewParams type
	params := anthropic.MessageNewParams{
		Model:    anthropic.Model(openAIReq.Model),
		Messages: make([]anthropic.MessageParam, 0),
	}

	var systemBlocks []anthropic.TextBlockParam

	// Process messages
	for _, msg := range openAIReq.Messages {
		switch msg.Role {
		case "system":
			// System messages go to the top-level system field
			systemText := extractTextContent(msg.Content)
			if systemText != "" {
				systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
					Type: "text",
					Text: systemText,
				})
			}
		case "developer":
			// Developer messages are treated as system instruction
			textContent := extractTextContent(msg.Content)
			if textContent != "" {
				systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
					Type: "text",
					Text: textContent,
				})
			}
		default:
			// Convert message content
			contentBlocks := convertOpenAIContentToAnthropic(msg.Content)

			// Handle tool_calls for assistant messages
			if len(msg.ToolCalls) > 0 && msg.Role == "assistant" {
				contentBlocks = convertOpenAIToolCallsToAnthropic(contentBlocks, msg.ToolCalls)
			}

			msgParam := anthropic.MessageParam{
				Role:    anthropic.MessageParamRole(msg.Role),
				Content: contentBlocks,
			}
			params.Messages = append(params.Messages, msgParam)
		}
	}

	// Set max_tokens (required for Anthropic)
	maxTokens := 4096
	if openAIReq.MaxCompletionTokens != nil {
		maxTokens = *openAIReq.MaxCompletionTokens
	} else if openAIReq.MaxTokens != nil {
		maxTokens = *openAIReq.MaxTokens
	}
	params.MaxTokens = int64(maxTokens)

	// Set system if we have any system blocks
	if len(systemBlocks) > 0 {
		params.System = systemBlocks
	}

	// Set temperature and top_p if provided
	// Note: Anthropic API doesn't allow both temperature and top_p together
	// Prefer temperature if both are provided
	if openAIReq.Temperature != nil {
		params.Temperature = anthropic.Float(*openAIReq.Temperature)
	} else if openAIReq.TopP != nil {
		params.TopP = anthropic.Float(*openAIReq.TopP)
	}

	// Handle stop sequences
	if openAIReq.Stop != nil {
		switch s := openAIReq.Stop.(type) {
		case string:
			params.StopSequences = []string{s}
		case []interface{}:
			for _, item := range s {
				if str, ok := item.(string); ok {
					params.StopSequences = append(params.StopSequences, str)
				}
			}
		}
	}

	// Convert tools if present
	if len(openAIReq.Tools) > 0 {
		anthropicTools := convertOpenAIToolsToAnthropic(openAIReq.Tools)
		if len(anthropicTools) > 0 {
			// Convert []ToolParam to []ToolUnionParam for the SDK
			for _, tool := range anthropicTools {
				params.Tools = append(params.Tools, anthropic.ToolUnionParam{
					OfTool: &tool,
				})
			}
		}
	}

	// Handle tool choice - This would need to be converted to proper ToolChoiceUnionParam
	// For now, we skip it as it needs proper type handling

	// Handle metadata - MetadataParam only has UserID field in current SDK version
	// For other metadata, we would need custom handling
	if len(openAIReq.Metadata) > 0 {
		// If we have a user_id in metadata, set it
		if userID, ok := openAIReq.Metadata["user_id"]; ok {
			params.Metadata = anthropic.MetadataParam{
				UserID: anthropic.String(userID),
			}
		}
	}

	if openAIReq.ServiceTier != "" {
		params.ServiceTier = anthropic.MessageNewParamsServiceTier(openAIReq.ServiceTier)
	}

	// Handle response_format (JSON schema)
	if openAIReq.ResponseFormat != nil {
		if jsonSchema := convertOpenAIResponseFormatToAnthropic(openAIReq.ResponseFormat); jsonSchema != nil {
			params.OutputConfig = anthropic.OutputConfigParam{
				Format: *jsonSchema,
			}
		}
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Anthropic request: %w", err)
	}

	// If streaming is requested, add "stream": true to the JSON
	if openAIReq.Stream {
		var reqMap map[string]interface{}
		if err := json.Unmarshal(jsonData, &reqMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal for streaming: %w", err)
		}
		reqMap["stream"] = true
		jsonData, err = json.Marshal(reqMap)
		if err != nil {
			return nil, fmt.Errorf("failed to remarshal with stream: %w", err)
		}
	}

	return jsonData, nil
}

// AnthropicToOpenAI converts Anthropic response to OpenAI format
func AnthropicToOpenAI(anthropicBody []byte, model string) ([]byte, error) {
	var anthropicResp anthropic.Message
	if err := json.Unmarshal(anthropicBody, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	// Extract text content and tool calls from Anthropic response
	var content string
	var toolCalls []openai.OpenAIToolCall

	for _, block := range anthropicResp.Content {
		if block.Type == "text" && block.Text != "" {
			content += block.Text
		} else if block.Type == "tool_use" {
			toolCall := convertAnthropicToolUseToOpenAI(block)
			toolCalls = append(toolCalls, toolCall)
		}
	}

	message := openai.OpenAIResponseMessage{
		Role:    "assistant",
		Content: content,
	}

	// Only include tool_calls if there are any
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}

	// Map Anthropic stop reason to OpenAI format
	finishReason := mapAnthropicStopReason(string(anthropicResp.StopReason))

	openAIResp := openai.OpenAIResponse{
		ID:      anthropicResp.ID,
		Object:  "chat.completion",
		Created: openai.GetCurrentTimestamp(),
		Model:   model,
		Choices: []openai.OpenAIChoice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: &openai.OpenAIUsage{
			PromptTokens:     int(anthropicResp.Usage.InputTokens),
			CompletionTokens: int(anthropicResp.Usage.OutputTokens),
			TotalTokens:      int(anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens),
		},
	}

	return json.Marshal(openAIResp)
}

// Helper functions

func convertAnthropicToolUseToOpenAI(block anthropic.ContentBlockUnion) openai.OpenAIToolCall {
	// Extract tool_use information from the SDK ContentBlockUnion
	var toolID, toolName string
	var argsJSON string

	if block.Type == "tool_use" {
		toolID = block.ID
		toolName = block.Name

		// The Input field is json.RawMessage, marshal it
		if len(block.Input) > 0 {
			argsJSON = string(block.Input)
		}
	}

	return openai.OpenAIToolCall{
		ID:   toolID,
		Type: "function",
		Function: openai.OpenAIToolFunction{
			Name:      toolName,
			Arguments: argsJSON,
		},
	}
}

// convertOpenAIToolCallsToAnthropic converts OpenAI tool_calls to Anthropic tool_use blocks
// and integrates them into the content array
func convertOpenAIToolCallsToAnthropic(contentBlocks []anthropic.ContentBlockParamUnion, toolCalls []interface{}) []anthropic.ContentBlockParamUnion {
	// Convert tool_calls to tool_use blocks
	for _, toolCallInterface := range toolCalls {
		toolCallMap, ok := toolCallInterface.(map[string]interface{})
		if !ok {
			continue
		}

		// Extract function information
		var funcName string
		var argsStr string
		toolID := openai.GetString(toolCallMap, "id")

		// Try to get from function field first (standard OpenAI format)
		if funcObj, ok := toolCallMap["function"].(map[string]interface{}); ok {
			funcName = openai.GetString(funcObj, "name")
			argsStr = openai.GetString(funcObj, "arguments")
		}

		if funcName != "" && argsStr != "" {
			// Parse arguments from JSON string to map
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
				args = map[string]interface{}{"_error": "failed to parse arguments"}
			}

			toolUseBlock := anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					Type:  "tool_use",
					ID:    toolID,
					Name:  funcName,
					Input: args,
				},
			}

			contentBlocks = append(contentBlocks, toolUseBlock)
		}
	}

	return contentBlocks
}

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

func convertOpenAIContentToAnthropic(content interface{}) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion

	// If it's a simple string, create a text block
	if s, ok := content.(string); ok && s != "" {
		blocks = append(blocks, anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Text: s,
				Type: "text",
			},
		})
		return blocks
	}

	// If it's a complex array of content blocks
	if parts, ok := content.([]interface{}); ok {
		for _, p := range parts {
			part, ok := p.(map[string]interface{})
			if !ok {
				continue
			}

			contentType, _ := part["type"].(string)
			switch contentType {
			case "text":
				if text, ok := part["text"].(string); ok {
					blocks = append(blocks, anthropic.ContentBlockParamUnion{
						OfText: &anthropic.TextBlockParam{
							Text: text,
							Type: "text",
						},
					})
				}
			case "image_url":
				if imageBlock, ok := convertImageURLToSDK(part); ok {
					blocks = append(blocks, imageBlock)
				}
			}
		}
	}

	return blocks
}

func convertImageURLToSDK(part map[string]interface{}) (anthropic.ContentBlockParamUnion, bool) {
	urlObj, ok := part["image_url"].(map[string]interface{})
	if !ok {
		return anthropic.ContentBlockParamUnion{}, false
	}

	url, _ := urlObj["url"].(string)
	if url == "" {
		return anthropic.ContentBlockParamUnion{}, false
	}

	if strings.HasPrefix(url, "data:") {
		return convertDataURLToSDK(url)
	}

	// Handle regular URLs
	return anthropic.ContentBlockParamUnion{
		OfImage: &anthropic.ImageBlockParam{
			Type: "image",
			Source: anthropic.ImageBlockParamSourceUnion{
				OfURL: &anthropic.URLImageSourceParam{
					URL: url,
				},
			},
		},
	}, true
}

func convertDataURLToSDK(dataURL string) (anthropic.ContentBlockParamUnion, bool) {
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return anthropic.ContentBlockParamUnion{}, false
	}

	header := parts[0] // "data:image/jpeg;base64"
	data := parts[1]   // base64 data

	mediaType := parseMediaType(header)
	if mediaType == "" {
		return anthropic.ContentBlockParamUnion{}, false
	}

	mediaTypeEnum := anthropic.Base64ImageSourceMediaType(mediaType)
	return anthropic.ContentBlockParamUnion{
		OfImage: &anthropic.ImageBlockParam{
			Type: "image",
			Source: anthropic.ImageBlockParamSourceUnion{
				OfBase64: &anthropic.Base64ImageSourceParam{
					Data:      data,
					MediaType: mediaTypeEnum,
				},
			},
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

// convertOpenAIToolsToAnthropic converts OpenAI tools format to Anthropic SDK ToolParam
// OpenAI format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
// Anthropic format: ToolParam with Name, Description, InputSchema
func convertOpenAIToolsToAnthropic(openAITools []interface{}) []anthropic.ToolParam {
	if len(openAITools) == 0 {
		return nil
	}

	var anthropicTools []anthropic.ToolParam

	for _, toolInterface := range openAITools {
		toolMap, ok := toolInterface.(map[string]interface{})
		if !ok {
			continue
		}

		// Check if it's a function type tool
		if toolType, ok := toolMap["type"].(string); !ok || toolType != "function" {
			continue
		}

		// Extract function definition
		if functionObj, ok := toolMap["function"].(map[string]interface{}); ok {
			name := openai.GetString(functionObj, "name")
			if name == "" {
				continue
			}

			description := openai.GetString(functionObj, "description")
			tool := anthropic.ToolParam{
				Name: name,
			}

			// Set description if provided - use param.Opt for optional string
			if description != "" {
				tool.Description = anthropic.String(description)
			}

			// Convert parameters to input_schema
			if schemaObj, ok := functionObj["parameters"].(map[string]interface{}); ok {
				// Extract properties and required from the schema
				inputSchema := anthropic.ToolInputSchemaParam{
					Type: "object",
				}

				if props, ok := schemaObj["properties"]; ok {
					inputSchema.Properties = props
				}

				if req, ok := schemaObj["required"].([]interface{}); ok {
					required := make([]string, 0, len(req))
					for _, r := range req {
						if s, ok := r.(string); ok {
							required = append(required, s)
						}
					}
					inputSchema.Required = required
				}

				tool.InputSchema = inputSchema
			}

			anthropicTools = append(anthropicTools, tool)
		}
	}

	if len(anthropicTools) == 0 {
		return nil
	}

	return anthropicTools
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

// convertOpenAIResponseFormatToAnthropic converts OpenAI response_format to Anthropic JSON schema format
func convertOpenAIResponseFormatToAnthropic(responseFormat interface{}) *anthropic.JSONOutputFormatParam {
	// response_format can be:
	// 1. {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}
	// 2. {"type": "text"}
	// 3. nil

	if responseFormat == nil {
		return nil
	}

	switch rf := responseFormat.(type) {
	case map[string]interface{}:
		// Check if it's json_schema type
		if rfType, ok := rf["type"].(string); ok {
			switch rfType {
			case "json_schema":
				// Extract the json_schema field
				if jsonSchema, ok := rf["json_schema"].(map[string]interface{}); ok {
					if schema, ok := jsonSchema["schema"].(map[string]interface{}); ok {
						// Return Anthropic JSONOutputFormatParam with schema
						return &anthropic.JSONOutputFormatParam{
							Schema: schema,
							Type:   constant.JSONSchema("json_schema"),
						}
					}
					// If no nested schema, return the whole json_schema object as schema
					return &anthropic.JSONOutputFormatParam{
						Schema: jsonSchema,
						Type:   constant.JSONSchema("json_schema"),
					}
				}
			case "json_object":
				// For simple json_object type without specific schema,
				// return a minimal schema
				return &anthropic.JSONOutputFormatParam{
					Schema: map[string]any{
						"type": "object",
					},
					Type: constant.JSONSchema("json_schema"),
				}
			}
		}
	}

	return nil
}
