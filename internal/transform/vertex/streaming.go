package vertex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// VertexStreamingChunk represents a single chunk from Vertex AI streaming response
type VertexStreamingChunk struct {
	Candidates    []VertexCandidate    `json:"candidates,omitempty"`
	UsageMetadata *VertexUsageMetadata `json:"usageMetadata,omitempty"`
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

			// Extract content from parts
			var content string
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					content += part.Text
				}
				// Note: streaming doesn't support images in delta, only text
			}
			choice.Delta.Content = content

			// Handle finish reason
			if candidate.FinishReason != "" {
				finishReason := mapFinishReason(candidate.FinishReason)
				choice.FinishReason = &finishReason
			}

			openAIChunk.Choices = append(openAIChunk.Choices, choice)
		}

		// Convert usage metadata if present
		if vertexChunk.UsageMetadata != nil {
			openAIChunk.Usage = &openai.OpenAIUsage{
				PromptTokens:     vertexChunk.UsageMetadata.PromptTokenCount,
				CompletionTokens: vertexChunk.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      vertexChunk.UsageMetadata.TotalTokenCount,
			}
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
