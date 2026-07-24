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

// streamUsage keeps provider field presence separate from numeric values.
// A non-nil usage object is observable even when every reported counter is zero.
type streamUsage struct {
	present bool
	value   AnthropicUsage
}

func (u *streamUsage) observeStart(value *AnthropicUsage) {
	if value == nil {
		return
	}
	u.present = true
	u.value = *value
}

func (u *streamUsage) observeDelta(value *AnthropicStreamUsage) {
	if value == nil {
		return
	}
	u.present = true
	setIfPresent(&u.value.InputTokens, value.InputTokens)
	setIfPresent(&u.value.OutputTokens, value.OutputTokens)
	setIfPresent(&u.value.CacheReadInputTokens, value.CacheReadInputTokens)
	setIfPresent(&u.value.CacheCreationInputTokens, value.CacheCreationInputTokens)
}

func (u *streamUsage) openAI() *openai.OpenAIUsage {
	if !u.present {
		return nil
	}
	return convertAnthropicUsageToOpenAI(&u.value)
}

func setIfPresent(destination *int, value *int) {
	if value != nil {
		*destination = *value
	}
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

	// Usage is accumulated across message_start / message_delta events.
	var usage streamUsage

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
					chatID = openAIChatCompletionID(event.Message.ID)
				}
				usage.observeStart(event.Message.Usage)
			}
			// Emit the first (role-only) chunk so the client knows the stream has started.
			if isFirstChunk {
				chunk := buildStreamChunk(chatID, model, timestamp, openai.OpenAIStreamingDelta{
					Role: "assistant",
				}, nil)
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
				chunk := buildStreamChunk(chatID, model, timestamp, delta, nil)
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
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil)
					if err := writeChunk(output, chunk); err != nil {
						return err
					}
				}

			case "thinking_delta":
				if event.Delta.Thinking != "" {
					delta := openai.OpenAIStreamingDelta{ReasoningContent: event.Delta.Thinking}
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil)
					if err := writeChunk(output, chunk); err != nil {
						return err
					}
				}

			case "signature_delta":
				// Anthropic sends this for multi-turn thinking verification.
				// No OpenAI equivalent; silently consumed to avoid unknown-delta errors.

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
					chunk := buildStreamChunk(chatID, model, timestamp, delta, nil)
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
			// Carries the stop_reason and final output token count. Track usage even
			// when a provider sends it separately from the stop_reason event.
			usage.observeDelta(event.Usage)
			if event.Delta == nil || event.Delta.StopReason == "" {
				continue
			}
			reason := mapAnthropicStopReason(event.Delta.StopReason)
			if err := writeTerminalChunks(output, chatID, model, timestamp, reason, usage.openAI()); err != nil {
				return err
			}

		case "message_stop":
			// End of stream; [DONE] is written after the loop.

		case "error": // handle Anthropic error events in stream
			// Anthropic streams an error event on overload, rate-limit, etc.
			// Error JSON: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}
			errMsg := "anthropic stream error"
			if event.Error != nil && event.Error.Message != "" {
				errMsg = event.Error.Message
			}
			reason := "stop"
			delta := openai.OpenAIStreamingDelta{Content: errMsg}
			chunk := buildStreamChunk(chatID, model, timestamp, delta, &reason)
			if err := writeChunk(output, chunk); err != nil {
				return err
			}

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

// writeTerminalChunks keeps the OpenAI terminal contract in one place:
// finish first, then an optional usage-only chunk, then the caller writes [DONE].
func writeTerminalChunks(
	output io.Writer,
	chatID, model string,
	timestamp int64,
	reason string,
	usage *openai.OpenAIUsage,
) error {
	if err := writeChunk(output, buildStreamChunk(
		chatID,
		model,
		timestamp,
		openai.OpenAIStreamingDelta{},
		&reason,
	)); err != nil {
		return err
	}
	if usage == nil {
		return nil
	}
	return writeChunk(output, buildUsageStreamChunk(chatID, model, timestamp, usage))
}

// buildUsageStreamChunk constructs the OpenAI terminal usage chunk. OpenAI
// requires this chunk to carry no choices and to follow the finish chunk.
func buildUsageStreamChunk(
	chatID, model string,
	timestamp int64,
	usage *openai.OpenAIUsage,
) openai.OpenAIStreamingChunk {
	return openai.OpenAIStreamingChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: timestamp,
		Model:   model,
		Choices: []openai.OpenAIStreamingChoice{},
		Usage:   usage,
	}
}

// buildStreamChunk constructs an OpenAI streaming chunk.
func buildStreamChunk(
	chatID, model string,
	timestamp int64,
	delta openai.OpenAIStreamingDelta,
	finishReason *string,
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
