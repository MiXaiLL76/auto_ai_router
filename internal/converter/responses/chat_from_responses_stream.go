package responses

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// responsesStreamEvent captures the fields TransformResponsesStreamToChat
// needs across every Responses API SSE event type it handles. Type acts as
// the discriminator; only the fields relevant to that type are populated by
// the provider.
type responsesStreamEvent struct {
	Type        string      `json:"type"`
	OutputIndex int         `json:"output_index"`
	Delta       string      `json:"delta"` // plain-string delta for output_text/reasoning_summary_text/function_call_arguments events
	Item        *OutputItem `json:"item,omitempty"`
	Response    *Response   `json:"response,omitempty"`
	Error       *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// TransformResponsesStreamToChat reads a Responses API SSE stream from
// reader and writes OpenAI Chat Completions SSE chunks to output -- the
// streaming mirror of ResponseToChat, used for a model marked
// responses_only. Text and function-call-argument deltas are relayed as
// they arrive (true incremental streaming); finish_reason and usage are
// only known once the terminal response.completed/incomplete/failed event
// arrives, since -- like OpenAI's own Responses API contract -- that event
// always repeats the complete, final Response object.
//
// Supported event types:
//
//	response.created / response.in_progress — captures the response ID, emits the role-only opening chunk
//	response.output_item.added              — announces a function_call's id/name (message items produce no chunk of their own)
//	response.output_text.delta              — streams assistant text
//	response.reasoning_summary_text.delta   — streams reasoning/thinking text
//	response.function_call_arguments.delta  — streams a function call's arguments
//	response.completed / .incomplete / .failed — carries the final status/usage; [DONE] is written after the loop
//	error / response.error                  — provider-side stream error, surfaced as content
func TransformResponsesStreamToChat(reader io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(reader)
	// Increase scanner buffer for large chunks (e.g. a long reasoning summary).
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var chatID string
	timestamp := converterutil.GetCurrentTimestamp()
	isFirstChunk := true

	// output_index -> chat tool_call slot index, assigned the first time a
	// function_call item is seen at that output_index. Chat Completions
	// indexes tool_calls among themselves, not among every output item, so a
	// message item sharing the output array never consumes a slot.
	toolCallSlots := make(map[int]int)
	nextToolCallIdx := 0

	ensureChatID := func() string {
		if chatID == "" {
			chatID = converterutil.GenerateID()
		}
		return chatID
	}
	writeFirstChunkOnce := func() error {
		if !isFirstChunk {
			return nil
		}
		isFirstChunk = false
		return writeChatStreamChunk(output, ensureChatID(), model, timestamp, openai.OpenAIStreamingDelta{Role: "assistant"}, nil)
	}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonData := strings.TrimPrefix(line, "data: ")
		if jsonData == "" || jsonData == "[DONE]" {
			continue
		}

		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
			// Malformed chunk — skip silently, matching every other stream
			// transformer in this codebase (anthropic.TransformAnthropicStreamToOpenAI).
			continue
		}

		switch event.Type {
		case "response.created", "response.in_progress":
			if event.Response != nil && event.Response.ID != "" {
				chatID = chatCompletionIDFromResponses(event.Response.ID)
			}
			if err := writeFirstChunkOnce(); err != nil {
				return err
			}

		case "response.output_item.added":
			if err := writeFirstChunkOnce(); err != nil {
				return err
			}
			if event.Item == nil || event.Item.Type != "function_call" {
				continue
			}
			idx := toolCallSlotFor(toolCallSlots, &nextToolCallIdx, event.OutputIndex)
			tc := openai.OpenAIStreamingToolCall{
				Index: idx,
				ID:    event.Item.CallID,
				Type:  "function",
				Function: &openai.OpenAIStreamingToolFunction{
					Name:      event.Item.Name,
					Arguments: "",
				},
			}
			delta := openai.OpenAIStreamingDelta{ToolCalls: []openai.OpenAIStreamingToolCall{tc}}
			if err := writeChatStreamChunk(output, ensureChatID(), model, timestamp, delta, nil); err != nil {
				return err
			}

		case "response.output_text.delta":
			if err := writeFirstChunkOnce(); err != nil {
				return err
			}
			if event.Delta == "" {
				continue
			}
			delta := openai.OpenAIStreamingDelta{Content: event.Delta}
			if err := writeChatStreamChunk(output, ensureChatID(), model, timestamp, delta, nil); err != nil {
				return err
			}

		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			if err := writeFirstChunkOnce(); err != nil {
				return err
			}
			if event.Delta == "" {
				continue
			}
			delta := openai.OpenAIStreamingDelta{ReasoningContent: event.Delta}
			if err := writeChatStreamChunk(output, ensureChatID(), model, timestamp, delta, nil); err != nil {
				return err
			}

		case "response.function_call_arguments.delta":
			if event.Delta == "" {
				continue
			}
			idx := toolCallSlotFor(toolCallSlots, &nextToolCallIdx, event.OutputIndex)
			tc := openai.OpenAIStreamingToolCall{
				Index:    idx,
				Function: &openai.OpenAIStreamingToolFunction{Arguments: event.Delta},
			}
			delta := openai.OpenAIStreamingDelta{ToolCalls: []openai.OpenAIStreamingToolCall{tc}}
			if err := writeChatStreamChunk(output, ensureChatID(), model, timestamp, delta, nil); err != nil {
				return err
			}

		case "response.completed", "response.incomplete", "response.failed":
			if event.Response == nil {
				continue
			}
			reason := responsesFinishReason(event.Response.Status, event.Response.IncompleteDetails, nextToolCallIdx > 0)
			usage := responsesUsageToChat(event.Response.Usage)
			if err := writeChatTerminalChunks(output, ensureChatID(), model, timestamp, reason, usage); err != nil {
				return err
			}

		case "error", "response.error":
			msg := "responses API stream error"
			if event.Error != nil && event.Error.Message != "" {
				msg = event.Error.Message
			}
			reason := "stop"
			delta := openai.OpenAIStreamingDelta{Content: msg}
			if err := writeChatStreamChunk(output, ensureChatID(), model, timestamp, delta, &reason); err != nil {
				return err
			}

		default:
			// response.output_item.done, response.content_part.*,
			// response.output_text.done, response.function_call_arguments.done,
			// etc.: bookkeeping-only events already reflected by the deltas
			// streamed above; nothing further for the client to see.
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("responses stream scanner error: %w", err)
	}

	_, _ = fmt.Fprintf(output, "data: [DONE]\n\n")
	return nil
}

// toolCallSlotFor returns the Chat Completions tool_calls[] index for a
// function_call item at the given Responses API output_index, allocating a
// new one on first sight. Guards against a provider that (unlike a
// spec-faithful one) never sends response.output_item.added before the
// first response.function_call_arguments.delta -- a slot is still allocated
// lazily so arguments stream correctly either way.
func toolCallSlotFor(slots map[int]int, next *int, outputIndex int) int {
	if idx, ok := slots[outputIndex]; ok {
		return idx
	}
	idx := *next
	slots[outputIndex] = idx
	*next++
	return idx
}

// writeChatStreamChunk marshals and writes one OpenAI Chat Completions
// streaming chunk as an SSE data line.
func writeChatStreamChunk(output io.Writer, chatID, model string, timestamp int64, delta openai.OpenAIStreamingDelta, finishReason *string) error {
	chunk := openai.OpenAIStreamingChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: timestamp,
		Model:   model,
		Choices: []openai.OpenAIStreamingChoice{
			{Index: 0, Delta: delta, FinishReason: finishReason},
		},
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal streaming chunk: %w", err)
	}
	_, err = fmt.Fprintf(output, "data: %s\n\n", data)
	return err
}

// writeChatTerminalChunks keeps the OpenAI terminal contract in one place:
// a finish_reason chunk, then an optional usage-only chunk (no choices) --
// the caller writes [DONE] after the stream loop ends.
func writeChatTerminalChunks(output io.Writer, chatID, model string, timestamp int64, reason string, usage *openai.OpenAIUsage) error {
	if err := writeChatStreamChunk(output, chatID, model, timestamp, openai.OpenAIStreamingDelta{}, &reason); err != nil {
		return err
	}
	if usage == nil {
		return nil
	}
	chunk := openai.OpenAIStreamingChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: timestamp,
		Model:   model,
		Choices: []openai.OpenAIStreamingChoice{},
		Usage:   usage,
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal usage chunk: %w", err)
	}
	_, err = fmt.Fprintf(output, "data: %s\n\n", data)
	return err
}
