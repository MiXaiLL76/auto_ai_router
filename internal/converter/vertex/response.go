package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// VertexToOpenAI converts Vertex AI response to OpenAI format
func VertexToOpenAI(vertexBody []byte, model string) ([]byte, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex response: %w", err)
	}

	openAIResp := openai.OpenAIResponse{
		ID:      converterutil.GenerateID(),
		Object:  "chat.completion",
		Created: converterutil.GetCurrentTimestamp(),
		Model:   model,
		Choices: make([]openai.OpenAIChoice, 0),
	}

	// Convert candidates to choices
	for _, candidate := range vertexResp.Candidates {
		var content string
		var reasoningContent string
		var images []openai.ImageData
		var toolCalls []openai.OpenAIToolCall

		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
				// Handle thinking/reasoning parts (Thought == true means this is a reasoning token)
				if part.Thought {
					reasoningContent += part.Text
					continue
				}
				if part.Text != "" {
					content += part.Text
				}
				// Handle inline data (images) from Vertex response
				if part.InlineData != nil {
					if imageData, ok := inlineDataToChatImage(len(images), part.InlineData); ok {
						images = append(images, imageData)
					}
				}
				// Handle function calls from Vertex response
				if part.FunctionCall != nil {
					toolCall := convertGenaiToOpenAIFunctionCall(part.FunctionCall, part.ThoughtSignature)
					toolCalls = append(toolCalls, toolCall)
				}
				// Handle code execution results (model executed code and returned output)
				if part.CodeExecutionResult != nil {
					if part.CodeExecutionResult.Output != "" {
						content += "\n```\n" + part.CodeExecutionResult.Output + "\n```"
					}
				}
				// Handle executable code (model-generated code to be executed)
				if part.ExecutableCode != nil {
					if part.ExecutableCode.Code != "" {
						lang := strings.ToLower(string(part.ExecutableCode.Language))
						if lang == "" || lang == "language_unspecified" {
							lang = "python" // Vertex default language
						}
						content += "\n```" + lang + "\n" + part.ExecutableCode.Code + "\n```"
					}
				}
			}
		}

		if content == "" && len(images) == 0 && len(toolCalls) == 0 && reasoningContent == "" {
			// Handle case where parts is empty but we have a finish reason
			if candidate.FinishReason == genai.FinishReasonMaxTokens {
				content = "[Response truncated due to max tokens limit]"
			} else if candidate.FinishReason != genai.FinishReasonSafety {
				content = "[No content generated]"
			}
		}

		message := openai.OpenAIResponseMessage{
			Role:    "assistant",
			Content: content,
			Images:  images,
		}

		// Set reasoning content if present (thinking/reasoning tokens)
		if reasoningContent != "" {
			message.ReasoningContent = reasoningContent
		}

		// Only include tool_calls if there are any
		if len(toolCalls) > 0 {
			message.ToolCalls = toolCalls
		}

		// Set refusal message when content is filtered for safety
		if candidate.FinishReason == genai.FinishReasonSafety && content == "" && len(toolCalls) == 0 {
			message.Refusal = "Content was filtered for safety reasons"
			message.Content = "" // ensure empty
		}

		finishReason := mapFinishReason(string(candidate.FinishReason))
		// Vertex API returns "STOP" even when there are function calls (Gemini 3+).
		// Override to "tool_calls" for OpenAI compatibility — clients rely on this
		// to detect that tool results need to be sent back.
		if len(toolCalls) > 0 && finishReason != "tool_calls" {
			finishReason = "tool_calls"
		}

		choice := openai.OpenAIChoice{
			Index:        int(candidate.Index),
			Message:      message,
			FinishReason: finishReason,
		}
		openAIResp.Choices = append(openAIResp.Choices, choice)
	}

	// Convert usage metadata
	if vertexResp.UsageMetadata != nil {
		openAIResp.Usage = convertVertexUsageMetadata(vertexResp.UsageMetadata)
	}
	return json.Marshal(openAIResp)
}

func inlineDataToChatImage(index int, blob *genai.Blob) (openai.ImageData, bool) {
	mimeType := blob.MIMEType
	if mimeType == "" {
		mimeType = http.DetectContentType(blob.Data)
	}
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		return openai.ImageData{}, false
	}

	b64Data := base64.StdEncoding.EncodeToString(blob.Data)
	return openai.ImageData{
		Type:  "image_url",
		Index: &index,
		ImageURL: &openai.ImageURL{
			URL: "data:" + mimeType + ";base64," + b64Data,
		},
	}, true
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
		PromptTokens:     int(meta.PromptTokenCount + meta.ToolUsePromptTokenCount),
		CompletionTokens: completionTokens,
		TotalTokens:      int(meta.PromptTokenCount+meta.ToolUsePromptTokenCount) + completionTokens,
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
				// LiteLLM supports image_tokens as an extension to the OpenAI
				// completion token details. Preserve the modality so generated
				// images are billed with output_cost_per_image_token instead of
				// the much lower text output rate.
				usage.CompletionTokensDetails.ImageTokens += int(detail.TokenCount)
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

	if len(meta.ToolUsePromptTokensDetails) > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		for _, detail := range meta.ToolUsePromptTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.AudioTokens += int(detail.TokenCount)
			}
		}
	}

	// Avoid double-charging cached modality tokens as regular audio.
	// Cached tokens are billed separately via CachedTokens.
	if len(meta.CacheTokensDetails) > 0 && usage.PromptTokensDetails != nil {
		for _, detail := range meta.CacheTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.AudioTokens -= int(detail.TokenCount)
				if usage.PromptTokensDetails.AudioTokens < 0 {
					usage.PromptTokensDetails.AudioTokens = 0
				}
			}
		}
	}

	return usage
}

// mapFinishReason maps Vertex AI finish reason to OpenAI finish reason
func mapFinishReason(vertexReason string) string {
	switch vertexReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	case "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	case "TOOL_CALL":
		return "tool_calls"
	default:
		return "stop"
	}
}

// convertGenaiToOpenAIFunctionCall converts genai.FunctionCall to OpenAI tool call format.
// Preserves thoughtSignature in provider_specific_fields for Gemini 3.x multi-turn conversations.
// Per litellm >= 1.80.5 and Google Gemini 3 requirements, thoughtSignature must be preserved
// when sending tool results back to maintain context and avoid 400 errors.
func convertGenaiToOpenAIFunctionCall(genaiCall *genai.FunctionCall, thoughtSignature []byte) openai.OpenAIToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	toolCall := openai.OpenAIToolCall{
		ID:   converterutil.GenerateID(),
		Type: "function",
		Function: openai.OpenAIToolFunction{
			Name:      genaiCall.Name,
			Arguments: argsJSON,
		},
	}

	// Preserve thoughtSignature in provider_specific_fields for Gemini 3 function calling.
	// This is required by Gemini API when sending tool results in subsequent requests.
	providerFields := make(map[string]interface{})

	if len(thoughtSignature) > 0 {
		// Store thoughtSignature as base64 string for JSON compatibility
		providerFields["thought_signature"] = converterutil.EncodeBase64(thoughtSignature)
	} else {
		// Per litellm and Google docs: use dummy validator when thought_signature is missing.
		// This allows seamless model switching (e.g., from gemini-2.5-flash to gemini-3-pro).
		providerFields["skip_thought_signature_validator"] = true
	}

	toolCall.ProviderSpecificFields = providerFields
	return toolCall
}
