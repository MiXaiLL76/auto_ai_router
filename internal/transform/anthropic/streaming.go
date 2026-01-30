package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// TransformAnthropicStreamToOpenAI converts Anthropic SSE stream to OpenAI SSE format
func TransformAnthropicStreamToOpenAI(anthropicStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(anthropicStream)
	chatID := ""
	timestamp := openai.GetCurrentTimestamp()
	isFirstChunk := true

	// Track tool_use state during streaming
	currentToolUse := &struct {
		id          string
		name        string
		inputBuffer strings.Builder
		isActive    bool
		toolCallIdx int
	}{
		toolCallIdx: 0,
	}

	for scanner.Scan() {
		line := scanner.Text()

		// Skip non-data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonData := strings.TrimPrefix(line, "data: ")

		var event anthropic.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
			continue
		}

		// Extract ID from message_start event
		if event.Type == "message_start" && event.Message.ID != "" {
			chatID = event.Message.ID
		}

		// Only process events that have content
		var content string
		var toolCalls []openai.OpenAIStreamingToolCall
		var finishReason *string

		switch event.Type {
		case "content_block_start":
			// Handle tool_use block start
			if event.ContentBlock.Type == "tool_use" {
				currentToolUse.isActive = true
				currentToolUse.id = event.ContentBlock.ID
				currentToolUse.name = event.ContentBlock.Name
				currentToolUse.inputBuffer.Reset()
			}

		case "content_block_delta":
			if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
				// Text content
				content = event.Delta.Text
			} else if event.Delta.Type == "input_json_delta" && event.Delta.PartialJSON != "" {
				// Accumulate tool_use input JSON
				if currentToolUse.isActive {
					currentToolUse.inputBuffer.WriteString(event.Delta.PartialJSON)
				}
			}

		case "content_block_stop":
			// Handle tool_use block end
			if currentToolUse.isActive {
				toolCall := openai.OpenAIStreamingToolCall{
					Index: currentToolUse.toolCallIdx,
					ID:    currentToolUse.id,
					Type:  "function",
					Function: &openai.OpenAIStreamingToolFunction{
						Name:      currentToolUse.name,
						Arguments: currentToolUse.inputBuffer.String(),
					},
				}
				toolCalls = append(toolCalls, toolCall)
				currentToolUse.isActive = false
				currentToolUse.toolCallIdx++
			}

		case "message_delta":
			if event.Delta.StopReason != "" {
				reason := mapAnthropicStopReason(string(event.Delta.StopReason))
				finishReason = &reason
			}

		case "message_stop":
			// End of stream
			continue

		default:
			// Skip other event types
			if isFirstChunk && event.Type == "message_start" {
				// Allow message_start on first chunk
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
						Content:   content,
						ToolCalls: toolCalls,
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
		if event.Usage.InputTokens > 0 || event.Usage.OutputTokens > 0 {
			openAIChunk.Usage = &openai.OpenAIUsage{
				PromptTokens:     int(event.Usage.InputTokens),
				CompletionTokens: int(event.Usage.OutputTokens),
				TotalTokens:      int(event.Usage.InputTokens + event.Usage.OutputTokens),
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
