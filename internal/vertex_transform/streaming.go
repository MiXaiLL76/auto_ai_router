package vertex_transform

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// VertexStreamingChunk represents a single chunk from Vertex AI streaming response
type VertexStreamingChunk struct {
	Candidates []VertexCandidate `json:"candidates,omitempty"`
}

// OpenAIStreamingChunk represents OpenAI streaming response format
type OpenAIStreamingChunk struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Choices []OpenAIStreamingChoice `json:"choices"`
}

type OpenAIStreamingChoice struct {
	Index        int                  `json:"index"`
	Delta        OpenAIStreamingDelta `json:"delta"`
	FinishReason *string              `json:"finish_reason"`
}

type OpenAIStreamingDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// TransformVertexStreamToOpenAI converts Vertex AI SSE stream to OpenAI SSE format
func TransformVertexStreamToOpenAI(vertexStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(vertexStream)
	chatID := generateID()
	timestamp := getCurrentTimestamp()

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

		// Convert to OpenAI format
		openAIChunk := OpenAIStreamingChunk{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: timestamp,
			Model:   model,
			Choices: make([]OpenAIStreamingChoice, 0),
		}

		// Process candidates
		for i, candidate := range vertexChunk.Candidates {
			choice := OpenAIStreamingChoice{
				Index: i,
				Delta: OpenAIStreamingDelta{},
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

		// Write OpenAI formatted chunk
		chunkJSON, err := json.Marshal(openAIChunk)
		if err != nil {
			continue
		}

		_, _ = fmt.Fprintf(output, "data: %s\n\n", chunkJSON)
	}

	return scanner.Err()
}
