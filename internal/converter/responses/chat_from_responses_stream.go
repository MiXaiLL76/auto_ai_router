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
	Delta       string      `json:"delta"` // plain-string delta for output_text/reasoning_summary_text/refusal/function_call_arguments events
	Item        *OutputItem `json:"item,omitempty"`
	Response    *Response   `json:"response,omitempty"`
	Message     string      `json:"message,omitempty"` // top-level on the standalone "error" event, not nested
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
//	response.refusal.delta                  — streams a model refusal
//	response.reasoning_summary_text.delta   — streams reasoning/thinking text
//	response.function_call_arguments.delta  — streams a function call's arguments
//	response.output_item.done               — emits a finished image_generation_call as delta.images
//	response.completed / .incomplete / .failed — carries the final status/usage; [DONE] is written after the loop
//	error / response.error                  — provider-side stream error, surfaced as content
func TransformResponsesStreamToChat(reader io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(reader)
	// Image output and terminal events can carry the entire generated image.
	scanner.Buffer(make([]byte, 1024*1024), converterutil.MaxSSELineBytes)
	stream := responsesToChatStream{
		output:        output,
		model:         model,
		timestamp:     converterutil.GetCurrentTimestamp(),
		toolCallSlots: make(map[int]int),
	}

	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data: ")
		if !ok || data == "" || data == "[DONE]" {
			continue
		}
		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			// Match the other stream converters: one malformed event does
			// not prevent subsequent events from reaching the client.
			continue
		}
		if err := stream.processEvent(&event); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("responses stream scanner error: %w", err)
	}

	_, _ = fmt.Fprintf(output, "data: [DONE]\n\n")
	return nil
}

type responsesToChatStream struct {
	output    io.Writer
	model     string
	timestamp int64
	chatID    string
	opened    bool

	// Responses indexes every output item; Chat indexes only tool calls.
	toolCallSlots   map[int]int
	nextToolCallIdx int
	imageIdx        int
	streamedContent bool

	// Standalone errors send their own finish_reason. A later terminal
	// event can still supply usage, but should not repeat the failure.
	streamErrorReported bool
}

func (s *responsesToChatStream) ensureChatID() string {
	if s.chatID == "" {
		s.chatID = converterutil.GenerateID()
	}
	return s.chatID
}

func (s *responsesToChatStream) open() error {
	if s.opened {
		return nil
	}
	s.opened = true
	return s.writeDelta(openai.OpenAIStreamingDelta{Role: "assistant"}, nil)
}

func (s *responsesToChatStream) writeDelta(delta openai.OpenAIStreamingDelta, reason *string) error {
	return writeChatStreamChunk(s.output, s.ensureChatID(), s.model, s.timestamp, delta, reason)
}

func (s *responsesToChatStream) processEvent(event *responsesStreamEvent) error {
	switch event.Type {
	case "response.created", "response.in_progress":
		if !s.opened && event.Response != nil && event.Response.ID != "" {
			s.chatID = chatCompletionIDFromResponses(event.Response.ID)
		}
		return s.open()
	case "response.output_item.added":
		return s.announceToolCall(event)
	case "response.output_text.delta", "response.refusal.delta",
		"response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		return s.relayTextDelta(event)
	case "response.function_call_arguments.delta":
		if event.Delta == "" {
			return nil
		}
		if err := s.open(); err != nil {
			return err
		}
		// Providers can omit output_item.added, so allocate the slot on first use.
		idx := toolCallSlotFor(s.toolCallSlots, &s.nextToolCallIdx, event.OutputIndex)
		return s.writeDelta(openai.OpenAIStreamingDelta{ToolCalls: []openai.OpenAIStreamingToolCall{{
			Index:    idx,
			Function: &openai.OpenAIStreamingToolFunction{Arguments: event.Delta},
		}}}, nil)
	case "response.output_item.done":
		return s.relayFinishedImage(event.Item)
	case "response.completed", "response.incomplete", "response.failed":
		return s.finish(event.Response)
	case "error", "response.error":
		return s.reportError(event.Message)
	}
	return nil
}

func (s *responsesToChatStream) announceToolCall(event *responsesStreamEvent) error {
	// The role-only opening also precedes non-tool items.
	if err := s.open(); err != nil {
		return err
	}
	if event.Item == nil || event.Item.Type != "function_call" {
		return nil
	}
	idx := toolCallSlotFor(s.toolCallSlots, &s.nextToolCallIdx, event.OutputIndex)
	return s.writeDelta(openai.OpenAIStreamingDelta{ToolCalls: []openai.OpenAIStreamingToolCall{{
		Index:    idx,
		ID:       event.Item.CallID,
		Type:     "function",
		Function: &openai.OpenAIStreamingToolFunction{Name: event.Item.Name},
	}}}, nil)
}

func (s *responsesToChatStream) relayTextDelta(event *responsesStreamEvent) error {
	// Empty text/refusal/reasoning deltas can still open the Chat stream.
	if err := s.open(); err != nil {
		return err
	}
	if event.Delta == "" {
		return nil
	}
	var delta openai.OpenAIStreamingDelta
	switch event.Type {
	case "response.output_text.delta":
		delta.Content = event.Delta
		s.streamedContent = true
	case "response.refusal.delta":
		delta.Refusal = event.Delta
		s.streamedContent = true
	default:
		delta.ReasoningContent = event.Delta
	}
	return s.writeDelta(delta, nil)
}

func (s *responsesToChatStream) relayFinishedImage(item *OutputItem) error {
	// Partial images are previews. Only a finished image enters Chat output.
	if item == nil || item.Type != "image_generation_call" || item.Result == "" {
		return nil
	}
	if err := s.open(); err != nil {
		return err
	}
	s.streamedContent = true
	idx := s.imageIdx
	s.imageIdx++
	return s.writeDelta(openai.OpenAIStreamingDelta{
		Images: []openai.ImageData{{Type: "image_url", Index: &idx, B64JSON: item.Result}},
	}, nil)
}

func (s *responsesToChatStream) finish(response *Response) error {
	if response == nil {
		return nil
	}
	if s.streamErrorReported {
		if usage := responsesUsageToChat(response.Usage); usage != nil {
			return writeChatUsageChunk(s.output, s.ensureChatID(), s.model, s.timestamp, usage)
		}
		return nil
	}

	// Surface a bare failure as text. Existing visible output or tool calls
	// already give the client a turn, so keep their content intact.
	if response.Status == "failed" && !s.streamedContent && s.nextToolCallIdx == 0 {
		if message := responsesErrorMessage(response.Error); message != "" {
			if err := s.open(); err != nil {
				return err
			}
			if err := s.writeDelta(openai.OpenAIStreamingDelta{Content: message}, nil); err != nil {
				return err
			}
		}
	}

	// The terminal response carries full reasoning items. Echo them as
	// thinking_blocks so the client can retain them in its next tool turn.
	if blocks := reasoningBlocksFromOutput(response.Output); len(blocks) > 0 {
		if err := s.open(); err != nil {
			return err
		}
		if err := writeThinkingBlocksChunk(s.output, s.ensureChatID(), s.model, s.timestamp, blocks); err != nil {
			return err
		}
	}
	reason := responsesFinishReason(response.Status, response.IncompleteDetails, s.nextToolCallIdx > 0)
	return writeChatTerminalChunks(s.output, s.ensureChatID(), s.model, s.timestamp, reason, responsesUsageToChat(response.Usage))
}

func (s *responsesToChatStream) reportError(message string) error {
	if message == "" {
		message = "responses API stream error"
	}
	if err := s.open(); err != nil {
		return err
	}
	reason := "stop"
	if err := s.writeDelta(openai.OpenAIStreamingDelta{Content: message}, &reason); err != nil {
		return err
	}
	s.streamErrorReported = true
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

// writeThinkingBlocksChunk writes a chunk carrying thinking_blocks on the
// delta, built by hand (it predates openai.OpenAIStreamingDelta.ThinkingBlocks).
func writeThinkingBlocksChunk(output io.Writer, chatID, model string, timestamp int64, blocks []responsesReasoningBlock) error {
	chunk := map[string]interface{}{
		"id": chatID, "object": "chat.completion.chunk", "created": timestamp, "model": model,
		"choices": []interface{}{
			map[string]interface{}{"index": 0, "delta": map[string]interface{}{"thinking_blocks": blocks}, "finish_reason": nil},
		},
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal thinking_blocks chunk: %w", err)
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
	return writeChatUsageChunk(output, chatID, model, timestamp, usage)
}

// writeChatUsageChunk writes the usage-only chunk (no choices) half of the terminal
// contract on its own -- split out of writeChatTerminalChunks for the
// streamErrorReported case above, which needs usage without a second finish_reason
// chunk (a standalone "error" event already sent one).
func writeChatUsageChunk(output io.Writer, chatID, model string, timestamp int64, usage *openai.OpenAIUsage) error {
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
