package vertex

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// VertexRequest represents the Vertex AI API request format
type VertexRequest struct {
	Contents          []VertexContent         `json:"contents"`
	GenerationConfig  *VertexGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *VertexContent          `json:"systemInstruction,omitempty"`
	Tools             []VertexTool            `json:"tools,omitempty"`
}

type VertexContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []VertexPart `json:"parts"`
}

type VertexPart struct {
	Text         string              `json:"text,omitempty"`
	InlineData   *InlineData         `json:"inlineData,omitempty"`
	FunctionCall *VertexFunctionCall `json:"functionCall,omitempty"`
}

type VertexFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type VertexGenerationConfig struct {
	Temperature        *float64 `json:"temperature,omitempty"`
	MaxOutputTokens    *int     `json:"maxOutputTokens,omitempty"`
	TopP               *float64 `json:"topP,omitempty"`
	TopK               *int     `json:"topK,omitempty"`
	StopSequences      []string `json:"stopSequences,omitempty"`
	ResponseMimeType   string   `json:"responseMimeType,omitempty"`
	ResponseModalities []string `json:"responseModalities,omitempty"`
	CandidateCount     *int     `json:"candidateCount,omitempty"`
	Seed               *int     `json:"seed,omitempty"`
	FrequencyPenalty   *float64 `json:"frequencyPenalty,omitempty"`
	PresencePenalty    *float64 `json:"presencePenalty,omitempty"`
}

// Tool-related types for function calling
type VertexTool struct {
	FunctionDeclarations []VertexFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type VertexFunctionDeclaration struct {
	Name        string                `json:"name"`
	Description string                `json:"description,omitempty"`
	Parameters  *VertexFunctionSchema `json:"parameters,omitempty"`
}

type VertexFunctionSchema struct {
	Type       string                              `json:"type,omitempty"`
	Properties map[string]VertexPropertyDefinition `json:"properties,omitempty"`
	Required   []string                            `json:"required,omitempty"`
}

type VertexPropertyDefinition struct {
	Type        string        `json:"type,omitempty"`
	Description string        `json:"description,omitempty"`
	Items       interface{}   `json:"items,omitempty"`
	Enum        []interface{} `json:"enum,omitempty"`
}

// OpenAIToVertex converts OpenAI format request to Vertex AI format
func OpenAIToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]VertexContent, 0),
	}

	// Handle generation config from extra_body or direct params
	if openAIReq.Temperature != nil || openAIReq.MaxTokens != nil || openAIReq.TopP != nil || openAIReq.ExtraBody != nil ||
		openAIReq.N != nil || openAIReq.Seed != nil || openAIReq.FrequencyPenalty != nil || openAIReq.PresencePenalty != nil || openAIReq.Stop != nil {
		vertexReq.GenerationConfig = &VertexGenerationConfig{
			Temperature:      openAIReq.Temperature,
			MaxOutputTokens:  openAIReq.MaxTokens,
			TopP:             openAIReq.TopP,
			CandidateCount:   openAIReq.N,
			Seed:             openAIReq.Seed,
			FrequencyPenalty: openAIReq.FrequencyPenalty,
			PresencePenalty:  openAIReq.PresencePenalty,
		}

		// Handle extra_body generation_config (takes precedence)
		if openAIReq.ExtraBody != nil {
			if genConfig, ok := openAIReq.ExtraBody["generation_config"].(map[string]interface{}); ok {
				// Only set response_mime_type for non-image models
				if mimeType, ok := genConfig["response_mime_type"].(string); ok {
					// Skip response_mime_type for image generation models
					if !strings.Contains(strings.ToLower(openAIReq.Model), "image") {
						vertexReq.GenerationConfig.ResponseMimeType = mimeType
					}
				}
				if modalities, ok := genConfig["response_modalities"].([]interface{}); ok {
					for _, m := range modalities {
						if mod, ok := m.(string); ok {
							vertexReq.GenerationConfig.ResponseModalities = append(vertexReq.GenerationConfig.ResponseModalities, mod)
						}
					}
				}
				if topK, ok := genConfig["top_k"].(float64); ok {
					topKInt := int(topK)
					vertexReq.GenerationConfig.TopK = &topKInt
				}
				// extra_body values override direct params
				if seed, ok := genConfig["seed"].(float64); ok {
					seedInt := int(seed)
					vertexReq.GenerationConfig.Seed = &seedInt
				}
				if temp, ok := genConfig["temperature"].(float64); ok {
					vertexReq.GenerationConfig.Temperature = &temp
				}
			}
			// Handle modalities at top level
			if modalities, ok := openAIReq.ExtraBody["modalities"].([]interface{}); ok {
				for _, m := range modalities {
					if mod, ok := m.(string); ok {
						vertexReq.GenerationConfig.ResponseModalities = append(vertexReq.GenerationConfig.ResponseModalities, strings.ToUpper(mod))
					}
				}
			}
		}

		// Handle stop sequences
		if openAIReq.Stop != nil {
			switch stop := openAIReq.Stop.(type) {
			case string:
				vertexReq.GenerationConfig.StopSequences = []string{stop}
			case []interface{}:
				stopSeqs := make([]string, 0, len(stop))
				for _, s := range stop {
					if str, ok := s.(string); ok {
						stopSeqs = append(stopSeqs, str)
					}
				}
				vertexReq.GenerationConfig.StopSequences = stopSeqs
			}
		}
	}

	// Convert messages
	for _, msg := range openAIReq.Messages {
		if msg.Role == "system" {
			// System messages become systemInstruction
			content := extractTextContent(msg.Content)
			vertexReq.SystemInstruction = &VertexContent{
				Parts: []VertexPart{{Text: content}},
			}
		} else {
			// Convert role mapping
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}

			parts := convertContentToParts(msg.Content)
			vertexReq.Contents = append(vertexReq.Contents, VertexContent{
				Role:  role,
				Parts: parts,
			})
		}
	}

	// Convert tools/function_calling if present
	if len(openAIReq.Tools) > 0 {
		vertexTools := convertOpenAIToolsToVertex(openAIReq.Tools)
		if len(vertexTools) > 0 {
			vertexReq.Tools = vertexTools
		}
	}

	return json.Marshal(vertexReq)
}

// VertexToOpenAI converts Vertex AI response to OpenAI format
func VertexToOpenAI(vertexBody []byte, model string) ([]byte, error) {
	var vertexResp VertexResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex response: %w", err)
	}

	openAIResp := openai.OpenAIResponse{
		ID:      openai.GenerateID(),
		Object:  "chat.completion",
		Created: openai.GetCurrentTimestamp(),
		Model:   model,
		Choices: make([]openai.OpenAIChoice, 0),
	}

	// Convert candidates to choices
	for i, candidate := range vertexResp.Candidates {
		var content string
		var images []openai.ImageData
		var toolCalls []openai.OpenAIToolCall

		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				content += part.Text
			}
			// Handle inline data (images) from Vertex response
			if part.InlineData != nil {
				images = append(images, openai.ImageData{
					B64JSON: part.InlineData.Data,
				})
			}
			// Handle function calls from Vertex response
			if part.FunctionCall != nil {
				toolCall := convertVertexFunctionCallToOpenAI(part.FunctionCall)
				toolCalls = append(toolCalls, toolCall)
			}
		}

		if content == "" && len(images) == 0 && len(toolCalls) == 0 {
			// Handle case where parts is empty but we have a finish reason
			if candidate.FinishReason == "MAX_TOKENS" {
				content = "[Response truncated due to max tokens limit]"
			} else {
				content = "[No content generated]"
			}
		}

		message := openai.OpenAIResponseMessage{
			Role:    "assistant",
			Content: content,
			Images:  images,
		}

		// Only include tool_calls if there are any
		if len(toolCalls) > 0 {
			message.ToolCalls = toolCalls
		}

		choice := openai.OpenAIChoice{
			Index:        i,
			Message:      message,
			FinishReason: mapFinishReason(candidate.FinishReason),
		}
		openAIResp.Choices = append(openAIResp.Choices, choice)
	}

	// Convert usage metadata
	if vertexResp.UsageMetadata != nil {
		openAIResp.Usage = &openai.OpenAIUsage{
			PromptTokens:     vertexResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: vertexResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      vertexResp.UsageMetadata.TotalTokenCount,
		}
	}

	return json.Marshal(openAIResp)
}

// VertexResponse represents Vertex AI response format
type VertexResponse struct {
	Candidates    []VertexCandidate    `json:"candidates"`
	UsageMetadata *VertexUsageMetadata `json:"usageMetadata,omitempty"`
}

type VertexCandidate struct {
	Content      VertexContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"`
}

type VertexUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

func mapFinishReason(vertexReason string) string {
	switch vertexReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY":
		return "content_filter"
	case "RECITATION":
		return "content_filter"
	case "TOOL_CALL":
		return "tool_calls"
	default:
		return "stop"
	}
}

// convertVertexFunctionCallToOpenAI converts Vertex function call to OpenAI tool call format
func convertVertexFunctionCallToOpenAI(vertexCall *VertexFunctionCall) openai.OpenAIToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if vertexCall.Args != nil {
		if data, err := json.Marshal(vertexCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	return openai.OpenAIToolCall{
		ID:   openai.GenerateID(),
		Type: "function",
		Function: openai.OpenAIToolFunction{
			Name:      vertexCall.Name,
			Arguments: argsJSON,
		},
	}
}

// Helper functions for multimodal content
func extractTextContent(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockMap["type"] == "text" {
					if text, ok := blockMap["text"].(string); ok {
						return text
					}
				}
			}
		}
	}
	return ""
}

// convertOpenAIToolsToVertex converts OpenAI tools format to Vertex tool format
func convertOpenAIToolsToVertex(openAITools []interface{}) []VertexTool {
	if len(openAITools) == 0 {
		return nil
	}

	var functionDeclarations []VertexFunctionDeclaration

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
			funcDecl := VertexFunctionDeclaration{
				Name:        openai.GetString(functionObj, "name"),
				Description: openai.GetString(functionObj, "description"),
			}

			// Convert parameters
			if params, ok := functionObj["parameters"].(map[string]interface{}); ok {
				funcDecl.Parameters = convertOpenAIParamsToVertex(params)
			}

			if funcDecl.Name != "" {
				functionDeclarations = append(functionDeclarations, funcDecl)
			}
		}
	}

	if len(functionDeclarations) == 0 {
		return nil
	}

	return []VertexTool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}
}

// convertOpenAIParamsToVertex converts OpenAI parameter schema to Vertex format
func convertOpenAIParamsToVertex(params map[string]interface{}) *VertexFunctionSchema {
	schema := &VertexFunctionSchema{
		Type:       "OBJECT",
		Properties: make(map[string]VertexPropertyDefinition),
	}

	if paramType, ok := params["type"].(string); ok {
		schema.Type = strings.ToUpper(paramType)
	}

	// Convert required fields
	if required, ok := params["required"].([]interface{}); ok {
		requiredFields := make([]string, 0, len(required))
		for _, req := range required {
			if field, ok := req.(string); ok {
				requiredFields = append(requiredFields, field)
			}
		}
		if len(requiredFields) > 0 {
			schema.Required = requiredFields
		}
	}

	// Convert properties
	if properties, ok := params["properties"].(map[string]interface{}); ok {
		for propName, propDef := range properties {
			if propMap, ok := propDef.(map[string]interface{}); ok {
				prop := VertexPropertyDefinition{
					Type:        strings.ToUpper(openai.GetString(propMap, "type")),
					Description: openai.GetString(propMap, "description"),
				}

				// Handle enum values
				if enumVals, ok := propMap["enum"].([]interface{}); ok {
					prop.Enum = enumVals
				}

				// Handle items for array types
				if items, ok := propMap["items"]; ok {
					prop.Items = items
				}

				schema.Properties[propName] = prop
			}
		}
	}

	return schema
}

func convertContentToParts(content interface{}) []VertexPart {
	switch c := content.(type) {
	case string:
		return []VertexPart{{Text: c}}
	case []interface{}:
		var parts []VertexPart
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				switch blockMap["type"] {
				case "text":
					if text, ok := blockMap["text"].(string); ok {
						parts = append(parts, VertexPart{Text: text})
					}
				case "image_url":
					if imageURL, ok := blockMap["image_url"].(map[string]interface{}); ok {
						if url, ok := imageURL["url"].(string); ok {
							if strings.HasPrefix(url, "data:") {
								// Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
								parts_split := strings.Split(url, ",")
								if len(parts_split) == 2 {
									header := parts_split[0] // data:image/jpeg;base64
									data := parts_split[1]   // base64 data

									// Extract mime type
									mimeType := "image/png" // default
									if start := strings.Index(header, "image/"); start >= 0 {
										// Look for semicolon or end of string
										if end := strings.Index(header[start:], ";"); end > 0 {
											mimeType = header[start : start+end]
										} else {
											// No semicolon, take from image/ to end
											mimeType = header[start:]
										}
									}

									parts = append(parts, VertexPart{
										InlineData: &InlineData{
											MimeType: mimeType,
											Data:     data,
										},
									})
								}
							}
						}
					}
				}
			}
		}
		return parts
	}
	return []VertexPart{{Text: fmt.Sprintf("%v", content)}}
}

// determineVertexPublisher determines the Vertex AI publisher based on the model ID
func determineVertexPublisher(modelID string) string {
	modelLower := strings.ToLower(modelID)
	if strings.Contains(modelLower, "claude") {
		return "anthropic"
	}
	// Default to Google for Gemini and other models
	return "google"
}

// buildVertexImageURL constructs the Vertex AI URL for image generation
// Format: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexImageURL(cred *config.CredentialConfig, modelID string) string {
	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// buildVertexURL constructs the Vertex AI URL dynamically
// Format: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{endpoint}
func BuildVertexURL(cred *config.CredentialConfig, modelID string, streaming bool) string {
	publisher := determineVertexPublisher(modelID)

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}

	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/%s/models/%s:%s",
			cred.ProjectID, publisher, modelID, endpoint,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/%s/models/%s:%s",
		cred.Location, cred.ProjectID, cred.Location, publisher, modelID, endpoint,
	)
}
