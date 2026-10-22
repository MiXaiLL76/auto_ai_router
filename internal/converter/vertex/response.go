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
	var toolUse ToolUseSources
	toolUse.Add(vertexResp.Candidates, vertexResp.ModelVersion)
	keepCallIDs := UsesFunctionCallIDs(toolUse.AnsweringModel(model))

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
					toolCall := convertGenaiToOpenAIFunctionCall(part.FunctionCall, part.ThoughtSignature, keepCallIDs)
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

		// A candidate with no parts back at all (MAX_TOKENS entirely consumed
		// by hidden reasoning, an image-generation candidate that produced no
		// image -- NO_IMAGE, IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, etc. --
		// or any other empty finish) has no partial model output to show.
		// Leave content empty and let finish_reason carry the signal, same as
		// every other provider route: a synthetic English placeholder here
		// isn't real model output, is silently indistinguishable from one,
		// and (unlike an empty string) can't be detected by a client without
		// hardcoding an exact string.

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
		openAIResp.Usage = convertVertexUsageMetadata(BillableUsageMetadata(vertexResp.UsageMetadata, toolUse, model))
		if initial := InitialPromptTokens(vertexBody); initial > 0 {
			if openAIResp.Usage.PromptTokensDetails == nil {
				openAIResp.Usage.PromptTokensDetails = &openai.TokenDetails{}
			}
			openAIResp.Usage.PromptTokensDetails.InitialPromptTokens = int(initial)
		}
	}
	if webSearchRequests := CountWebSearchRequests(vertexResp.Candidates, toolUse.AnsweringModel(model)); webSearchRequests > 0 {
		if openAIResp.Usage == nil {
			openAIResp.Usage = &openai.OpenAIUsage{}
		}
		openAIResp.Usage.ServerToolUse = &openai.ServerToolUseDetails{
			WebSearchRequests: webSearchRequests,
		}
	}
	return json.Marshal(openAIResp)
}

func inlineDataToChatImage(index int, blob *genai.Blob) (openai.ImageData, bool) {
	mimeType := blob.MIMEType
	if mimeType == "" {
		mimeType = http.DetectContentType(blob.Data)
	}
	if !IsImageMIME(mimeType) {
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

// mapFinishReason maps Vertex AI finish reason to OpenAI finish reason
func mapFinishReason(vertexReason string) string {
	switch vertexReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "CONTINUATION":
		// Generation stopped at the per-request output cap and was not continued
		// (see continuation.go): the answer is cut short, as with MAX_TOKENS.
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	case "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	case "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
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
func convertGenaiToOpenAIFunctionCall(genaiCall *genai.FunctionCall, thoughtSignature []byte, keepID bool) openai.OpenAIToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	toolCall := openai.OpenAIToolCall{
		ID:   functionCallID(genaiCall, keepID),
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

// functionCallID returns the tool call ID the client sees for a function call. With
// keepID (see UsesFunctionCallIDs) it is the ID Gemini gave the call, or a new one
// when it gave none: the client sends it back as the tool call's and the tool
// result's ID, and Gemini 4 refuses a function response whose ID is not its call's.
// Without keepID it is always a new one.
func functionCallID(call *genai.FunctionCall, keepID bool) string {
	if keepID && call.ID != "" {
		return call.ID
	}
	return converterutil.GenerateID()
}
