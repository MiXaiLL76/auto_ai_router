package vertex

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
	"google.golang.org/genai"
)

// VertexRequest represents the Vertex AI API request format
type VertexRequest struct {
	Contents          []*genai.Content        `json:"contents"`
	GenerationConfig  *genai.GenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *genai.Content          `json:"systemInstruction,omitempty"`
	Tools             []*genai.Tool           `json:"tools,omitempty"`
}

// OpenAIToVertex converts OpenAI format request to Vertex AI format
func OpenAIToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]*genai.Content, 0),
	}

	// Handle generation config from extra_body or direct params
	if openAIReq.Temperature != nil || openAIReq.MaxTokens != nil || openAIReq.MaxCompletionTokens != nil || openAIReq.TopP != nil || openAIReq.ExtraBody != nil ||
		openAIReq.N != nil || openAIReq.Seed != nil || openAIReq.FrequencyPenalty != nil || openAIReq.PresencePenalty != nil || openAIReq.Stop != nil ||
		len(openAIReq.Modalities) > 0 || openAIReq.ReasoningEffort != "" {

		genConfig := &genai.GenerationConfig{}

		// Convert direct parameters
		if openAIReq.Temperature != nil {
			temp := float32(*openAIReq.Temperature)
			genConfig.Temperature = &temp
		}
		if openAIReq.MaxTokens != nil {
			maxTokens := int32(*openAIReq.MaxTokens)
			genConfig.MaxOutputTokens = maxTokens
		}
		// max_completion_tokens takes precedence over max_tokens
		if openAIReq.MaxCompletionTokens != nil {
			maxTokens := int32(*openAIReq.MaxCompletionTokens)
			genConfig.MaxOutputTokens = maxTokens
		}
		if openAIReq.TopP != nil {
			topP := float32(*openAIReq.TopP)
			genConfig.TopP = &topP
		}
		if openAIReq.N != nil {
			candidates := int32(*openAIReq.N)
			genConfig.CandidateCount = candidates
		}
		if openAIReq.Seed != nil {
			seed := int32(*openAIReq.Seed)
			genConfig.Seed = &seed
		}
		if openAIReq.FrequencyPenalty != nil {
			freq := float32(*openAIReq.FrequencyPenalty)
			genConfig.FrequencyPenalty = &freq
		}
		if openAIReq.PresencePenalty != nil {
			pres := float32(*openAIReq.PresencePenalty)
			genConfig.PresencePenalty = &pres
		}

		// Handle extra_body generation_config (takes precedence)
		if openAIReq.ExtraBody != nil {
			if genConfigMap, ok := openAIReq.ExtraBody["generation_config"].(map[string]interface{}); ok {
				// Only set response_mime_type for non-image models
				if mimeType, ok := genConfigMap["response_mime_type"].(string); ok {
					// Skip response_mime_type for image generation models
					if !strings.Contains(strings.ToLower(openAIReq.Model), "image") {
						genConfig.ResponseMIMEType = mimeType
					}
				}
				if modalities, ok := genConfigMap["response_modalities"].([]interface{}); ok {
					for _, m := range modalities {
						if mod, ok := m.(string); ok {
							genConfig.ResponseModalities = append(genConfig.ResponseModalities, genai.Modality(mod))
						}
					}
				}
				if topK, ok := genConfigMap["top_k"].(float64); ok {
					topKVal := float32(topK)
					genConfig.TopK = &topKVal
				}
				// extra_body values override direct params
				if seed, ok := genConfigMap["seed"].(float64); ok {
					seedVal := int32(seed)
					genConfig.Seed = &seedVal
				}
				if temp, ok := genConfigMap["temperature"].(float64); ok {
					tempVal := float32(temp)
					genConfig.Temperature = &tempVal
				}
			}
			// Handle modalities at top level
			if modalities, ok := openAIReq.ExtraBody["modalities"].([]interface{}); ok {
				for _, m := range modalities {
					if mod, ok := m.(string); ok {
						genConfig.ResponseModalities = append(genConfig.ResponseModalities, genai.Modality(strings.ToUpper(mod)))
					}
				}
			}
		}

		// Handle stop sequences
		if openAIReq.Stop != nil {
			switch stop := openAIReq.Stop.(type) {
			case string:
				genConfig.StopSequences = []string{stop}
			case []interface{}:
				stopSeqs := make([]string, 0, len(stop))
				for _, s := range stop {
					if str, ok := s.(string); ok {
						stopSeqs = append(stopSeqs, str)
					}
				}
				genConfig.StopSequences = stopSeqs
			}
		}

		// Handle modalities
		if len(openAIReq.Modalities) > 0 {
			for _, mod := range openAIReq.Modalities {
				genConfig.ResponseModalities = append(genConfig.ResponseModalities, genai.Modality(strings.ToUpper(mod)))
			}
		}

		// Handle reasoning_effort (for reasoning models)
		if openAIReq.ReasoningEffort != "" {
			// Vertex uses thinking_config for reasoning models
			// Pass it through extra_body if explicitly set
			if openAIReq.ExtraBody == nil {
				openAIReq.ExtraBody = make(map[string]interface{})
			}
			if _, exists := openAIReq.ExtraBody["thinking_config"]; !exists {
				openAIReq.ExtraBody["reasoning_effort"] = openAIReq.ReasoningEffort
			}
		}

		vertexReq.GenerationConfig = genConfig
	}

	// Convert messages
	for _, msg := range openAIReq.Messages {
		switch msg.Role {
		case "system":
			// System messages become systemInstruction
			content := extractTextContent(msg.Content)
			vertexReq.SystemInstruction = &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: content},
				},
			}
		case "developer":
			// Developer messages are treated as system instruction
			content := extractTextContent(msg.Content)
			vertexReq.SystemInstruction = &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: content},
				},
			}
		case "tool":
			// Tool messages are sent as user messages with tool results
			// In Vertex, tool results are just text content from the user perspective
			content := extractTextContent(msg.Content)
			vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: content},
				},
			})
		default:
			// Convert role mapping
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}

			parts := convertContentToParts(msg.Content)

			// Handle tool_calls for assistant messages
			if len(msg.ToolCalls) > 0 && role == "model" {
				toolCallParts := convertToolCallsToGenaiParts(msg.ToolCalls)
				parts = append(parts, toolCallParts...)
			}

			vertexReq.Contents = append(vertexReq.Contents, &genai.Content{
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
	var vertexResp genai.GenerateContentResponse
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
	for _, candidate := range vertexResp.Candidates {
		var content string
		var images []openai.ImageData
		var toolCalls []openai.OpenAIToolCall

		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					content += part.Text
				}
				// Handle inline data (images) from Vertex response
				if part.InlineData != nil {
					images = append(images, openai.ImageData{
						B64JSON: string(part.InlineData.Data),
					})
				}
				// Handle function calls from Vertex response
				if part.FunctionCall != nil {
					toolCall := convertGenaiToOpenAIFunctionCall(part.FunctionCall)
					toolCalls = append(toolCalls, toolCall)
				}
			}
		}

		if content == "" && len(images) == 0 && len(toolCalls) == 0 {
			// Handle case where parts is empty but we have a finish reason
			if candidate.FinishReason == genai.FinishReasonMaxTokens {
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
			Index:        int(candidate.Index),
			Message:      message,
			FinishReason: mapFinishReason(string(candidate.FinishReason)),
		}
		openAIResp.Choices = append(openAIResp.Choices, choice)
	}

	// Convert usage metadata
	if vertexResp.UsageMetadata != nil {
		openAIResp.Usage = &openai.OpenAIUsage{
			PromptTokens:     int(vertexResp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(vertexResp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(vertexResp.UsageMetadata.TotalTokenCount),
		}
	}

	return json.Marshal(openAIResp)
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

// convertGenaiToOpenAIFunctionCall converts genai.FunctionCall to OpenAI tool call format
func convertGenaiToOpenAIFunctionCall(genaiCall *genai.FunctionCall) openai.OpenAIToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	return openai.OpenAIToolCall{
		ID:   openai.GenerateID(),
		Type: "function",
		Function: openai.OpenAIToolFunction{
			Name:      genaiCall.Name,
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

// convertContentToParts converts OpenAI content format to genai.Part slice
func convertContentToParts(content interface{}) []*genai.Part {
	switch c := content.(type) {
	case string:
		return []*genai.Part{{Text: c}}
	case []interface{}:
		var parts []*genai.Part
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				switch blockMap["type"] {
				case "text":
					if text, ok := blockMap["text"].(string); ok {
						parts = append(parts, &genai.Part{Text: text})
					}
				case "image_url":
					if imageURL, ok := blockMap["image_url"].(map[string]interface{}); ok {
						if url, ok := imageURL["url"].(string); ok {
							part := parseDataURLToPart(url)
							if part != nil {
								parts = append(parts, part)
							}
						}
					}
				case "file":
					if fileObj, ok := blockMap["file"].(map[string]interface{}); ok {
						if fileID, ok := fileObj["file_id"].(string); ok {
							part := parseDataURLToPart(fileID)
							if part != nil {
								parts = append(parts, part)
							}
						}
					}
				}
			}
		}
		return parts
	}
	return []*genai.Part{{Text: fmt.Sprintf("%v", content)}}
}

// parseDataURLToPart converts a data URL string to a genai.Part with inline data
// Handles formats like: data:image/jpeg;base64,/9j/4AAQ...
func parseDataURLToPart(dataURL string) *genai.Part {
	if !strings.HasPrefix(dataURL, "data:") {
		return nil
	}

	// Split: data:image/jpeg;base64,<data>
	parts := strings.Split(dataURL, ",")
	if len(parts) != 2 {
		return nil
	}

	header := parts[0] // data:image/jpeg;base64
	data := parts[1]   // base64 data

	// Extract mime type from header
	mimeType := extractMimeType(header)
	if mimeType == "" {
		return nil
	}

	return &genai.Part{
		InlineData: &genai.Blob{
			MIMEType: mimeType,
			Data:     []byte(data),
		},
	}
}

// extractMimeType extracts MIME type from data URL header
// Example: "data:image/jpeg;base64" -> "image/jpeg"
func extractMimeType(header string) string {
	// Find start of mime type (after "data:")
	start := strings.Index(header, "data:")
	if start < 0 {
		return ""
	}
	start += 5 // len("data:")

	// Find end of mime type (at ";" or end of string)
	end := strings.Index(header[start:], ";")
	if end > 0 {
		return header[start : start+end]
	}

	// No semicolon, take from start to end
	return header[start:]
}

// convertToolCallsToGenaiParts converts OpenAI tool_calls to genai.Part with FunctionCall
func convertToolCallsToGenaiParts(toolCalls []interface{}) []*genai.Part {
	if len(toolCalls) == 0 {
		return nil
	}

	var parts []*genai.Part

	for _, toolCallInterface := range toolCalls {
		toolCallMap, ok := toolCallInterface.(map[string]interface{})
		if !ok {
			continue
		}

		// Extract function information
		funcName := openai.GetString(toolCallMap, "name")
		if funcName == "" {
			if funcObj, ok := toolCallMap["function"].(map[string]interface{}); ok {
				funcName = openai.GetString(funcObj, "name")
				if funcName != "" {
					// Parse arguments
					argsStr := openai.GetString(funcObj, "arguments")
					var args map[string]interface{}
					if argsStr != "" {
						if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
							args = map[string]interface{}{"_error": "failed to parse arguments"}
						}
					}

					parts = append(parts, &genai.Part{
						FunctionCall: &genai.FunctionCall{
							Name: funcName,
							Args: args,
						},
					})
				}
			}
		}
	}

	return parts
}

// convertOpenAIToolsToVertex converts OpenAI tools format to genai.Tool slice
func convertOpenAIToolsToVertex(openAITools []interface{}) []*genai.Tool {
	if len(openAITools) == 0 {
		return nil
	}

	var functionDeclarations []*genai.FunctionDeclaration

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
			funcDecl := &genai.FunctionDeclaration{
				Name:        openai.GetString(functionObj, "name"),
				Description: openai.GetString(functionObj, "description"),
			}

			// Convert parameters
			if params, ok := functionObj["parameters"].(map[string]interface{}); ok {
				funcDecl.Parameters = convertOpenAIParamsToGenaiSchema(params)
			}

			if funcDecl.Name != "" {
				functionDeclarations = append(functionDeclarations, funcDecl)
			}
		}
	}

	if len(functionDeclarations) == 0 {
		return nil
	}

	return []*genai.Tool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}
}

// convertOpenAIParamsToGenaiSchema converts OpenAI parameter schema to genai.Schema format
func convertOpenAIParamsToGenaiSchema(params map[string]interface{}) *genai.Schema {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: make(map[string]*genai.Schema),
	}

	if paramType, ok := params["type"].(string); ok {
		schema.Type = genai.Type(strings.ToUpper(paramType))
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
				prop := &genai.Schema{
					Type:        genai.Type(strings.ToUpper(openai.GetString(propMap, "type"))),
					Description: openai.GetString(propMap, "description"),
				}

				// Handle enum values
				if enumVals, ok := propMap["enum"].([]interface{}); ok {
					enumStrs := make([]string, 0, len(enumVals))
					for _, e := range enumVals {
						if str, ok := e.(string); ok {
							enumStrs = append(enumStrs, str)
						}
					}
					if len(enumStrs) > 0 {
						prop.Enum = enumStrs
					}
				}

				// Handle items for array types
				if items, ok := propMap["items"]; ok {
					if itemsMap, ok := items.(map[string]interface{}); ok {
						prop.Items = &genai.Schema{
							Type: genai.Type(strings.ToUpper(openai.GetString(itemsMap, "type"))),
						}
					}
				}

				schema.Properties[propName] = prop
			}
		}
	}

	return schema
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

// BuildVertexImageURL constructs the Vertex AI URL for image generation
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

// BuildVertexURL constructs the Vertex AI URL dynamically
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
