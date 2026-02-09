package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/transform/common"
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
func OpenAIToVertex(openAIBody []byte, isImageGeneration bool, model string) ([]byte, error) {
	var openAIReq openai.OpenAIRequest

	if isImageGeneration {
		if strings.Contains(model, "gemini") {
			// For Gemini models, convert imagen request to chat request first
			chatBody, err := ImageRequestToOpenAIChatRequest(openAIBody)
			if err != nil {
				return nil, err
			}
			// Now process as normal chat request (no longer isImageGeneration, just normal chat)
			openAIBody = chatBody
		} else {
			// For non-Gemini models (like Imagen), use standard image conversion
			return OpenAIImageToVertex(openAIBody)
		}
	}

	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]*genai.Content, 0),
	}

	// Handle generation config from extra_body or direct params
	if openAIReq.Temperature != nil || openAIReq.MaxTokens != nil || openAIReq.MaxCompletionTokens != nil || openAIReq.TopP != nil || openAIReq.ExtraBody != nil ||
		openAIReq.N != nil || openAIReq.Seed != nil || openAIReq.FrequencyPenalty != nil || openAIReq.PresencePenalty != nil || openAIReq.Stop != nil ||
		len(openAIReq.Modalities) > 0 || openAIReq.ReasoningEffort != "" || openAIReq.ResponseFormat != nil {

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
				// Note: image_config parameters are passed through extra_body
				// and will be included in the final request JSON by the SDK
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

		// Handle response_format (JSON schema)
		if openAIReq.ResponseFormat != nil {
			schema := convertOpenAIResponseFormatToGenaiSchema(openAIReq.ResponseFormat)
			if schema != nil {
				genConfig.ResponseSchema = schema
			}
			// Also set ResponseMIMEType to application/json for JSON output
			if rfMap, ok := openAIReq.ResponseFormat.(map[string]interface{}); ok {
				if rfType, ok := rfMap["type"].(string); ok && (rfType == "json_schema" || rfType == "json_object") {
					genConfig.ResponseMIMEType = "application/json"
				}
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

	vertexBody, err := json.Marshal(vertexReq)
	return vertexBody, err
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
					b64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
					images = append(images, openai.ImageData{
						B64JSON: b64Data,
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
		openAIResp.Usage = convertVertexUsageMetadata(vertexResp.UsageMetadata)
	}
	return json.Marshal(openAIResp)
}

// convertVertexUsageMetadata converts Vertex AI usage metadata to OpenAI format.
func convertVertexUsageMetadata(meta *genai.GenerateContentResponseUsageMetadata) *openai.OpenAIUsage {
	// Include thinking/reasoning tokens in completion tokens for accurate conversion
	// Vertex AI reasoning models include thoughts_token_count which are part of the response
	completionTokens := int(meta.CandidatesTokenCount)
	if meta.ThoughtsTokenCount > 0 {
		completionTokens += int(meta.ThoughtsTokenCount)
	}

	usage := &openai.OpenAIUsage{
		PromptTokens:     int(meta.PromptTokenCount),
		CompletionTokens: completionTokens,
		TotalTokens:      int(meta.TotalTokenCount),
	}

	// Map Vertex thinking tokens to OpenAI reasoning_tokens
	if meta.ThoughtsTokenCount > 0 {
		if usage.CompletionTokensDetails == nil {
			usage.CompletionTokensDetails = &openai.CompletionTokenDetails{}
		}
		usage.CompletionTokensDetails.ReasoningTokens = int(meta.ThoughtsTokenCount)
	}

	if meta.CachedContentTokenCount > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		usage.PromptTokensDetails.CachedTokens = int(meta.CachedContentTokenCount)
	}

	if len(meta.CandidatesTokensDetails) > 0 {
		if usage.CompletionTokensDetails == nil {
			usage.CompletionTokensDetails = &openai.CompletionTokenDetails{}
		}
		for _, detail := range meta.CandidatesTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.CompletionTokensDetails.AudioTokens += int(detail.TokenCount)
			case genai.MediaModalityImage, genai.MediaModalityVideo:
				// Image/video tokens are already included in CompletionTokens total;
				// OpenAI format has no dedicated field for these modalities
			}
		}
	}

	if len(meta.PromptTokensDetails) > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		for _, detail := range meta.PromptTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.AudioTokens += int(detail.TokenCount)
			}
		}
	}

	return usage
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
	parts := common.ExtractTextBlocks(content)
	if len(parts) == 0 {
		return ""
	}
	return parts[0]
}

// convertContentToParts converts OpenAI content format to genai.Part slice
func convertContentToParts(content interface{}) []*genai.Part {
	switch c := content.(type) {
	case string:
		return []*genai.Part{{Text: c}}
	case []interface{}:
		var parts []*genai.Part
		type partHandler func(map[string]interface{}) *genai.Part

		handlers := map[string]partHandler{
			"text": func(block map[string]interface{}) *genai.Part {
				text, ok := block["text"].(string)
				if !ok {
					return nil
				}
				return &genai.Part{Text: text}
			},
			"image_url": func(block map[string]interface{}) *genai.Part {
				imageURL, ok := block["image_url"].(map[string]interface{})
				if !ok {
					return nil
				}
				url, ok := imageURL["url"].(string)
				if !ok {
					return nil
				}
				// Try to parse as data URL first, then as regular URL
				part := parseDataURLToPart(url)
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(url, imageURL)
				}
				return part
			},
			"input_audio": func(block map[string]interface{}) *genai.Part {
				audioData, ok := block["input_audio"].(map[string]interface{})
				if !ok {
					return nil
				}
				data, ok := audioData["data"].(string)
				if !ok {
					return nil
				}

				// Decode base64 audio data
				decodedData, err := base64.StdEncoding.DecodeString(data)
				if err != nil {
					return nil
				}

				// Determine MIME type from format field or default to wav
				mimeType := "audio/wav"
				if format, ok := audioData["format"].(string); ok && format != "" {
					mimeType = getAudioMimeType(format)
				}

				return &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: mimeType,
						Data:     decodedData,
					},
				}
			},
			"video_url": func(block map[string]interface{}) *genai.Part {
				videoURL, ok := block["video_url"].(map[string]interface{})
				if !ok {
					return nil
				}
				url, ok := videoURL["url"].(string)
				if !ok {
					return nil
				}

				// Determine MIME type from format field or URL extension
				mimeType := ""
				if format, ok := videoURL["format"].(string); ok && format != "" {
					mimeType = format
				} else {
					mimeType = getMimeTypeFromURL(url)
				}

				if mimeType == "" {
					// Default to mp4 if we can't determine
					mimeType = "video/mp4"
				}

				return &genai.Part{
					FileData: &genai.FileData{
						MIMEType: mimeType,
						FileURI:  url,
					},
				}
			},
			"file": func(block map[string]interface{}) *genai.Part {
				fileObj, ok := block["file"].(map[string]interface{})
				if !ok {
					return nil
				}
				fileID, ok := fileObj["file_id"].(string)
				if !ok {
					return nil
				}

				// Try to parse as data URL first, then as regular URL
				part := parseDataURLToPart(fileID)
				if part == nil {
					// If not a data URL, treat as regular URL (http/https)
					part = parseURLToPart(fileID, fileObj)
				}
				return part
			},
		}

		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			contentType, _ := blockMap["type"].(string)
			if handler, ok := handlers[contentType]; ok {
				if part := handler(blockMap); part != nil {
					parts = append(parts, part)
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

	header := parts[0]  // data:image/jpeg;base64
	b64Data := parts[1] // base64 data

	// Extract mime type from header
	mimeType := extractMimeType(header)
	if mimeType == "" {
		return nil
	}

	// Decode base64 data to binary
	decodedData, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return nil
	}

	return &genai.Part{
		InlineData: &genai.Blob{
			MIMEType: mimeType,
			Data:     decodedData,
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

// mimeTypeMap maps file extensions to MIME types
var mimeTypeMap = map[string]string{
	"jpg":  "image/jpeg",
	"jpeg": "image/jpeg",
	"png":  "image/png",
	"gif":  "image/gif",
	"webp": "image/webp",
	"mp4":  "video/mp4",
	"mpeg": "video/mpeg",
	"mov":  "video/quicktime",
	"avi":  "video/x-msvideo",
	"mkv":  "video/x-matroska",
	"webm": "video/webm",
	"flv":  "video/x-flv",
	"pdf":  "application/pdf",
	"txt":  "text/plain",
}

// parseURLToPart converts a regular URL or file reference to a genai.Part
// Supports http/https URLs and determines MIME type from format or file extension
func parseURLToPart(url string, fileObj map[string]interface{}) *genai.Part {
	if url == "" {
		return nil
	}

	// Check if it's a valid HTTP(S) URL or a file URI
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") && !strings.HasPrefix(url, "file://") {
		return nil
	}

	// Determine MIME type from format field or file extension
	mimeType := ""

	// Check for explicit format field
	if format, ok := fileObj["format"].(string); ok && format != "" {
		mimeType = format
	} else {
		// Try to determine from URL extension
		mimeType = getMimeTypeFromURL(url)
	}

	if mimeType == "" {
		// Default to application/octet-stream if we can't determine
		mimeType = "application/octet-stream"
	}

	// Return FileData with FileURI for external URLs
	return &genai.Part{
		FileData: &genai.FileData{
			MIMEType: mimeType,
			FileURI:  url,
		},
	}
}

// getMimeTypeFromURL determines MIME type from URL extension
func getMimeTypeFromURL(url string) string {
	// Extract extension from URL (before query parameters)
	urlPath := url
	if idx := strings.Index(urlPath, "?"); idx > 0 {
		urlPath = urlPath[:idx]
	}

	// Get extension
	ext := ""
	if idx := strings.LastIndex(urlPath, "."); idx > 0 {
		ext = strings.ToLower(urlPath[idx+1:])
	}

	if mimeType, ok := mimeTypeMap[ext]; ok {
		return mimeType
	}

	return ""
}

// getAudioMimeType maps audio format to MIME type
func getAudioMimeType(format string) string {
	formatLower := strings.ToLower(format)
	mimeTypes := map[string]string{
		"wav":  "audio/wav",
		"mp3":  "audio/mpeg",
		"ogg":  "audio/ogg",
		"opus": "audio/opus",
		"aac":  "audio/aac",
		"flac": "audio/flac",
		"m4a":  "audio/mp4",
		"weba": "audio/webp",
	}

	if mimeType, ok := mimeTypes[formatLower]; ok {
		return mimeType
	}

	// Default to wav if format is not recognized
	return "audio/wav"
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

// convertMapToGenaiSchema recursively converts a map[string]interface{} (JSON Schema) to *genai.Schema
// This is used for response schemas to maintain structured format instead of raw JSON
func convertMapToGenaiSchema(data map[string]interface{}) *genai.Schema {
	if data == nil {
		return nil
	}

	schema := &genai.Schema{}

	// Convert type field
	if typeVal, ok := data["type"].(string); ok {
		schema.Type = genai.Type(strings.ToUpper(typeVal))
	}

	// Convert title
	if title, ok := data["title"].(string); ok {
		schema.Title = title
	}

	// Convert description
	if desc, ok := data["description"].(string); ok {
		schema.Description = desc
	}

	// Convert enum
	if enumVals, ok := data["enum"].([]interface{}); ok {
		enumStrs := make([]string, 0, len(enumVals))
		for _, e := range enumVals {
			if str, ok := e.(string); ok {
				enumStrs = append(enumStrs, str)
			}
		}
		if len(enumStrs) > 0 {
			schema.Enum = enumStrs
		}
	}

	// Convert required array
	if required, ok := data["required"].([]interface{}); ok {
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

	// Convert properties (recursive)
	if properties, ok := data["properties"].(map[string]interface{}); ok {
		schema.Properties = make(map[string]*genai.Schema)
		for propName, propVal := range properties {
			if propMap, ok := propVal.(map[string]interface{}); ok {
				schema.Properties[propName] = convertMapToGenaiSchema(propMap)
			}
		}
	}

	// Convert items for array types
	if items, ok := data["items"].(map[string]interface{}); ok {
		schema.Items = convertMapToGenaiSchema(items)
	}

	// Convert anyOf
	if anyOf, ok := data["anyOf"].([]interface{}); ok {
		schemas := make([]*genai.Schema, 0, len(anyOf))
		for _, item := range anyOf {
			if itemMap, ok := item.(map[string]interface{}); ok {
				schemas = append(schemas, convertMapToGenaiSchema(itemMap))
			}
		}
		if len(schemas) > 0 {
			schema.AnyOf = schemas
		}
	}

	// Convert format (e.g., "email", "date", etc.)
	if format, ok := data["format"].(string); ok {
		schema.Format = format
	}

	// Convert pattern (regex for string validation)
	if pattern, ok := data["pattern"].(string); ok {
		schema.Pattern = pattern
	}

	// Convert numeric constraints
	if minimum, ok := data["minimum"].(float64); ok {
		schema.Minimum = &minimum
	}
	if maximum, ok := data["maximum"].(float64); ok {
		schema.Maximum = &maximum
	}
	if minLength, ok := data["minLength"].(float64); ok {
		minLengthInt := int64(minLength)
		schema.MinLength = &minLengthInt
	}
	if maxLength, ok := data["maxLength"].(float64); ok {
		maxLengthInt := int64(maxLength)
		schema.MaxLength = &maxLengthInt
	}

	// Convert array constraints
	if minItems, ok := data["minItems"].(float64); ok {
		minItemsInt := int64(minItems)
		schema.MinItems = &minItemsInt
	}
	if maxItems, ok := data["maxItems"].(float64); ok {
		maxItemsInt := int64(maxItems)
		schema.MaxItems = &maxItemsInt
	}

	// Convert property ordering
	if propOrdering, ok := data["propertyOrdering"].([]interface{}); ok {
		propOrderingStrs := make([]string, 0, len(propOrdering))
		for _, prop := range propOrdering {
			if str, ok := prop.(string); ok {
				propOrderingStrs = append(propOrderingStrs, str)
			}
		}
		if len(propOrderingStrs) > 0 {
			schema.PropertyOrdering = propOrderingStrs
		}
	}

	// Convert default value
	if def, ok := data["default"]; ok {
		schema.Default = def
	}

	// Convert example
	if example, ok := data["example"]; ok {
		schema.Example = example
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
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexImageURL(cred *config.CredentialConfig, modelID string) string {
	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// BuildVertexURL constructs the Vertex AI URL dynamically
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{endpoint}
func BuildVertexURL(cred *config.CredentialConfig, modelID string, streaming bool) string {
	publisher := determineVertexPublisher(modelID)

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}

	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/%s/models/%s:%s",
			cred.ProjectID, publisher, modelID, endpoint,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/%s/models/%s:%s",
		cred.Location, cred.ProjectID, cred.Location, publisher, modelID, endpoint,
	)
}

// convertOpenAIResponseFormatToGenaiSchema converts OpenAI response_format to Vertex AI structured schema
// Using ResponseSchema (structured) instead of ResponseJsonSchema (raw JSON) may produce
// different output formatting (compact vs pretty-printed JSON)
func convertOpenAIResponseFormatToGenaiSchema(responseFormat interface{}) *genai.Schema {
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
						// Include schema name from OpenAI format if present
						if schemaName, ok := jsonSchema["name"].(string); ok && schemaName != "" {
							// Add title field to preserve the schema name for Vertex
							schema["title"] = schemaName
						}
						// Convert map to structured *genai.Schema
						return convertMapToGenaiSchema(schema)
					}
					// If no nested schema, convert the whole json_schema object
					return convertMapToGenaiSchema(jsonSchema)
				}
			case "json_object":
				// For simple json_object type, Vertex doesn't need additional schema
				return nil
			}
		}
	}

	return nil
}
