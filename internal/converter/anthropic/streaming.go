package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// blockState tracks an in-progress content block during streaming.
type blockState struct {
	blockType   string // "text", "thinking", "tool_use"
	id          string // tool_use id
	name        string // tool_use name
	toolCallIdx int
}

// TransformAnthropicStreamToOpenAI reads an Anthropic SSE stream from anthropicStream and
// writes OpenAI-compatible SSE chunks to output.
//
// Supported Anthropic event types:
//
//	message_start        — captures the message ID and input token usage
//	content_block_start  — opens a new content block (text / thinking / tool_use)
//	content_block_delta  — streams incremental text, thinking, or tool JSON
//	content_block_stop   — closes the current content block
//	message_delta        — carries stop_reason and output token usage
//	message_stop         — signals end of stream ([DONE] is written after the loop)
func TransformAnthropicStreamToOpenAI(anthropicStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(anthropicStream)
	// Increase scanner buffer for large chunks (e.g. thinking blocks).
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	chatID := converterutil.GenerateID()
	timestamp := converterutil.GetCurrentTimestamp()
	isFirstChunk := true

	// Per-block state; Anthropic streams one block at a time.
	var current blockState
	toolCallIdx := 0

	// Usage accumulated across message_start / message_delta events.
	var promptTokens, completionTokens int

	for scanner.Scan() {
		line := scanner.Text()

		// Only process "data: " SSE lines.
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonData := strings.TrimPrefix(line, "data: ")

		var event AnthropicStreamEvent
		if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
			// Malformed chunk — skip silently.
			continue
		}

		switch event.Type {

		case "message_start":
			// Extract message ID and input token count.
			if event.Message != nil {
				if event.Message.ID != "" {
					chatID = event.Message.ID
				}
				promptTokens = event.Message.Usage.InputTokens
			}
			// Emit the first (role-only) chunk so the client knows the stream has started.
			if isFirstChunk {
				chunk := buildStreamChunk(chatID, model, timestamp, openai.OpenAIStreamingDelta{
					Role: "assistant",
				}, nil, nil)
				if err := writeChunk(output, chunk); err != nil {
					return err
				}
				isFirstChunk = false
			}

		case "content_block_start":
			if event.ContentBlock == nil {
				continue
			}
			current = blockState{
				blockType:   event.ContentBlock.Type,
				id:          event.ContentBlock.ID,
				name:        event.ContentBlock.Name,
				toolCallIdx: toolCallIdx,
			}
			// For tool_use blocks: emit the opening chunk with id + name immediately so
			// that OpenAI clients receive id/name before any argument deltas.
			if current.blockType == "tool_use" {
				tc := openai.OpenAIStreamingToolCall{
					Index: current.toolCallIdx,
					ID:    current.id,
					Type:  "function",
					Function: &openai.OpenAIStreamingToolFunction{
						Name:      current.name,
						Arguments: "",
					},
				}
				delta := openai.OpenAIStreamingDelta{
					ToolCalls: []openai.OpenAIStreamingToolCall{tc},
				}
				chunk := buildStreamChunk(chatID, model, timestamp, delta, nil, nil)
				if err := writeChunk(output, chunk); err != nil {
					return err
				}
			}

		case "content_block_delta":
			if event.Delta == nil {
				continue
			}
			switch event.Delta.Type {
			case "text_delta":
				if event.Delta.Text != "" {
					delta := openai.OpenAIStreamingDelta{Content: event.Delta.Text}
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil, nil)
					if err := writeChunk(output, chunk); err != nil {
						return err
					}
				}

			case "thinking_delta":
				if event.Delta.Thinking != "" {
					delta := openai.OpenAIStreamingDelta{ReasoningContent: event.Delta.Thinking}
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil, nil)
					if err := writeChunk(output, chunk); err != nil {
						return err
					}
				}

			case "input_json_delta":
				// Stream partial tool arguments to the client.
				if event.Delta.PartialJSON != "" {
					tc := openai.OpenAIStreamingToolCall{
						Index: current.toolCallIdx,
						Function: &openai.OpenAIStreamingToolFunction{
							Arguments: event.Delta.PartialJSON,
						},
					}
					delta := openai.OpenAIStreamingDelta{
						ToolCalls: []openai.OpenAIStreamingToolCall{tc},
					}
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil, nil)
					if err := writeChunk(output, chunk); err != nil {
						return err
					}
				}
			}

		case "content_block_stop":
			// Increment tool_call index when a tool_use block finishes.
			if current.blockType == "tool_use" {
				toolCallIdx++
			}
			// Reset current block state.
			current = blockState{toolCallIdx: toolCallIdx}

		case "message_delta":
			// Carries the stop_reason and final output token count.
			if event.Delta == nil {
				continue
			}
			if event.Delta.StopReason != "" {
				reason := mapAnthropicStopReason(event.Delta.StopReason)
				if event.Usage != nil {
					completionTokens = event.Usage.OutputTokens
				}
				usage := &openai.OpenAIUsage{
					PromptTokens:     promptTokens,
					CompletionTokens: completionTokens,
					TotalTokens:      promptTokens + completionTokens,
				}
				chunk := buildStreamChunk(chatID, model, timestamp, openai.OpenAIStreamingDelta{}, &reason, usage)
				if err := writeChunk(output, chunk); err != nil {
					return err
				}
			}

		case "message_stop":
			// End of stream; [DONE] is written after the loop.

		default:
			// Unknown event type — skip.
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("anthropic stream scanner error: %w", err)
	}

	// Signal end of stream.
	_, _ = fmt.Fprintf(output, "data: [DONE]\n\n")
	return nil
}

// buildStreamChunk constructs an OpenAI streaming chunk.
func buildStreamChunk(
	chatID, model string,
	timestamp int64,
	delta openai.OpenAIStreamingDelta,
	finishReason *string,
	usage *openai.OpenAIUsage,
) openai.OpenAIStreamingChunk {
	return openai.OpenAIStreamingChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: timestamp,
		Model:   model,
		Choices: []openai.OpenAIStreamingChoice{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}
}

// writeChunk marshals a streaming chunk and writes it as an SSE data line.
func writeChunk(output io.Writer, chunk openai.OpenAIStreamingChunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal streaming chunk: %w", err)
	}
	_, err = fmt.Fprintf(output, "data: %s\n\n", data)
	return err
}
