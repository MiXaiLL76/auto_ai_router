package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// AnthropicStreamEvent represents a single event from Anthropic streaming response
type AnthropicStreamEvent struct {
	Type    string          `json:"type"`
	Index   *int            `json:"index,omitempty"`
	Delta   *AnthropicDelta `json:"delta,omitempty"`
	Message *AnthropicMsg   `json:"message,omitempty"`
	Usage   *AnthropicUsage `json:"usage,omitempty"`
}

type AnthropicDelta struct {
	Type         string  `json:"type"`
	Text         string  `json:"text,omitempty"`
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

type AnthropicMsg struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Role  string         `json:"role"`
	Usage AnthropicUsage `json:"usage"`
}

// TransformAnthropicStreamToOpenAI converts Anthropic SSE stream to OpenAI SSE format
func TransformAnthropicStreamToOpenAI(anthropicStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(anthropicStream)
	chatID := ""
	timestamp := time.Now().Unix()
	isFirstChunk := true

	for scanner.Scan() {
		line := scanner.Text()

		// Skip non-data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonData := strings.TrimPrefix(line, "data: ")

		var event AnthropicStreamEvent
		if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
			continue
		}

		// Extract ID from message_start event
		if event.Type == "message_start" && event.Message != nil {
			chatID = event.Message.ID
		}

		// Only process events that have content
		content := ""
		var finishReason *string

		switch event.Type {
		case "content_block_delta":
			if event.Delta != nil {
				content = event.Delta.Text
			}
		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != nil {
				reason := mapAnthropicStopReason(*event.Delta.StopReason)
				finishReason = &reason
			}
		case "message_stop":
			// End of stream
			continue
		default:
			// Skip other event types (message_start, content_block_start, content_block_stop)
			if isFirstChunk && event.Type == "message_start" {
				// Don't continue, send initial chunk with role
			} else {
				continue
			}
		}

		// Create OpenAI formatted chunk
		openAIChunk := openai.OpenAIStreamingChunk{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: timestamp,
			Model:   model,
			Choices: []openai.OpenAIStreamingChoice{
				{
					Index: 0,
					Delta: openai.OpenAIStreamingDelta{
						Content: content,
					},
					FinishReason: finishReason,
				},
			},
		}

		// Set role only on first chunk
		if isFirstChunk {
			openAIChunk.Choices[0].Delta.Role = "assistant"
			isFirstChunk = false
		}

		// Add usage info if available
		if event.Usage != nil {
			openAIChunk.Usage = &openai.OpenAIUsage{
				PromptTokens:     event.Usage.InputTokens,
				CompletionTokens: event.Usage.OutputTokens,
				TotalTokens:      event.Usage.InputTokens + event.Usage.OutputTokens,
			}
		}

		// Write chunk
		chunkJSON, err := json.Marshal(openAIChunk)
		if err != nil {
			continue
		}

		_, _ = fmt.Fprintf(output, "data: %s\n\n", chunkJSON)
	}

	// Send final done message
	_, _ = fmt.Fprintf(output, "data: [DONE]\n\n")

	return scanner.Err()
}
