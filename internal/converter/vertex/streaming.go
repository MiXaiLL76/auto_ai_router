package vertex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// TransformVertexStreamToOpenAI converts Vertex AI SSE stream to OpenAI SSE format
func TransformVertexStreamToOpenAI(vertexStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(vertexStream)
	chatID := converterutil.GenerateID()
	timestamp := converterutil.GetCurrentTimestamp()
	isFirstChunk := true

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines and non-data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		// Extract JSON data
		jsonData := strings.TrimPrefix(line, "data: ")
		if jsonData == "[DONE]" {
			// Write final done message
			_, _ = fmt.Fprintf(output, "data: [DONE]\n\n")
			break
		}

		// Parse Vertex AI chunk
		var vertexChunk VertexStreamingChunk
		if err := json.Unmarshal([]byte(jsonData), &vertexChunk); err != nil {
			continue // Skip malformed chunks
		}

		// Skip chunks with no candidates
		if len(vertexChunk.Candidates) == 0 {
			continue
		}

		// Convert to OpenAI format
		openAIChunk := openai.OpenAIStreamingChunk{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: timestamp,
			Model:   model,
			Choices: make([]openai.OpenAIStreamingChoice, 0),
		}

		// Process candidates
		for i, candidate := range vertexChunk.Candidates {
			choice := openai.OpenAIStreamingChoice{
				Index: i,
				Delta: openai.OpenAIStreamingDelta{},
			}

			// Set role only for first chunk (OpenAI convention)
			if isFirstChunk {
				choice.Delta.Role = "assistant"
			}

			// Extract content and function calls from parts
			var content string
			var reasoningContent string
			var toolCalls []openai.OpenAIStreamingToolCall
			toolCallIdx := 0

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
					// Handle function calls
					if part.FunctionCall != nil {
						toolCall := convertVertexFunctionCallToStreamingOpenAI(part.FunctionCall, part.ThoughtSignature, toolCallIdx)
						toolCalls = append(toolCalls, toolCall)
						toolCallIdx++
					}
					// Note: streaming doesn't support images in delta, only text
				}
			}

			choice.Delta.Content = content
			if reasoningContent != "" {
				choice.Delta.ReasoningContent = reasoningContent
			}
			if len(toolCalls) > 0 {
				choice.Delta.ToolCalls = toolCalls
			}

			// Handle finish reason
			if candidate.FinishReason != genai.FinishReasonUnspecified {
				finishReason := mapFinishReason(string(candidate.FinishReason))
				// Vertex returns "STOP" even with function calls (Gemini 3+).
				// Override for OpenAI compatibility.
				if len(toolCalls) > 0 && finishReason != "tool_calls" {
					finishReason = "tool_calls"
				}
				choice.FinishReason = &finishReason
			}

			openAIChunk.Choices = append(openAIChunk.Choices, choice)
		}

		// Convert usage metadata if present
		if vertexChunk.UsageMetadata != nil {
			//slog.Error("STREAMING_VERTEX_USAGE_CHUNK",
			//	"prompt_tokens", vertexChunk.UsageMetadata.PromptTokenCount,
			//	"candidates_tokens", vertexChunk.UsageMetadata.CandidatesTokenCount,
			//	"total_tokens", vertexChunk.UsageMetadata.TotalTokenCount,
			//	"cached_tokens", vertexChunk.UsageMetadata.CachedContentTokenCount,
			//)
			openAIChunk.Usage = convertVertexUsageMetadata(vertexChunk.UsageMetadata)
		}

		// Write OpenAI formatted chunk
		chunkJSON, err := json.Marshal(openAIChunk)
		if err != nil {
			continue
		}

		_, _ = fmt.Fprintf(output, "data: %s\n\n", chunkJSON)
		isFirstChunk = false
	}

	return scanner.Err()
}

// convertVertexFunctionCallToStreamingOpenAI converts Vertex function call to OpenAI streaming tool call format.
// Preserves thoughtSignature in provider_specific_fields for Gemini 3.x multi-turn streaming.
func convertVertexFunctionCallToStreamingOpenAI(genaiCall *genai.FunctionCall, thoughtSignature []byte, index int) openai.OpenAIStreamingToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	toolCall := openai.OpenAIStreamingToolCall{
		Index: index,
		ID:    converterutil.GenerateID(),
		Type:  "function",
		Function: &openai.OpenAIStreamingToolFunction{
			Name:      genaiCall.Name,
			Arguments: argsJSON,
		},
	}

	// Preserve thoughtSignature for streaming mode
	providerFields := make(map[string]interface{})
	if len(thoughtSignature) > 0 {
		providerFields["thought_signature"] = converterutil.EncodeBase64(thoughtSignature)
	} else {
		providerFields["skip_thought_signature_validator"] = true
	}
	toolCall.ProviderSpecificFields = providerFields

	return toolCall
}
