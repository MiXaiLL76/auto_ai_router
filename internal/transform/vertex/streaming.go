package vertex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
	"google.golang.org/genai"
)

// VertexStreamingChunk wraps genai types for streaming response
type VertexStreamingChunk struct {
	Candidates    []*genai.Candidate                          `json:"candidates,omitempty"`
	UsageMetadata *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata,omitempty"`
}

// TransformVertexStreamToOpenAI converts Vertex AI SSE stream to OpenAI SSE format
func TransformVertexStreamToOpenAI(vertexStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(vertexStream)
	chatID := openai.GenerateID()
	timestamp := openai.GetCurrentTimestamp()
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
			var toolCalls []openai.OpenAIStreamingToolCall
			toolCallIdx := 0

			if candidate.Content != nil && candidate.Content.Parts != nil {
				for _, part := range candidate.Content.Parts {
					if part.Text != "" {
						content += part.Text
					}
					// Handle function calls
					if part.FunctionCall != nil {
						toolCall := convertVertexFunctionCallToStreamingOpenAI(part.FunctionCall, toolCallIdx)
						toolCalls = append(toolCalls, toolCall)
						toolCallIdx++
					}
					// Note: streaming doesn't support images in delta, only text
				}
			}

			choice.Delta.Content = content
			if len(toolCalls) > 0 {
				choice.Delta.ToolCalls = toolCalls
			}

			// Handle finish reason
			if candidate.FinishReason != genai.FinishReasonUnspecified {
				finishReason := mapFinishReason(string(candidate.FinishReason))
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

// convertVertexFunctionCallToStreamingOpenAI converts Vertex function call to OpenAI streaming tool call format
func convertVertexFunctionCallToStreamingOpenAI(genaiCall *genai.FunctionCall, index int) openai.OpenAIStreamingToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	return openai.OpenAIStreamingToolCall{
		Index: index,
		ID:    openai.GenerateID(),
		Type:  "function",
		Function: &openai.OpenAIStreamingToolFunction{
			Name:      genaiCall.Name,
			Arguments: argsJSON,
		},
	}
}
