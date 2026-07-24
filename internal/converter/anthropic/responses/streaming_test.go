package anthropicresponses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildAnthropicSSEStream constructs a minimal Anthropic SSE stream for testing.
func buildAnthropicSSEStream(events []map[string]interface{}) string {
	var sb strings.Builder
	for _, e := range events {
		b, _ := json.Marshal(e)
		sb.WriteString("data: ")
		sb.Write(b)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

func parseSSEEvents(output string) []map[string]interface{} {
	var events []map[string]interface{}
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var e map[string]interface{}
		if json.Unmarshal([]byte(data), &e) == nil {
			events = append(events, e)
		}
	}
	return events
}

func TestTransformAnthropicStreamToResponses_TextStream(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 10, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text", "id": "", "name": ""},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "Hello"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": " world"},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 5},
		},
		{
			"type": "message_stop",
		},
	})

	var out bytes.Buffer
	var completedResp *responses.Response

	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream),
		&out,
		"claude-opus-4-5",
		"",
		nil,
		func(r *responses.Response) {
			completedResp = r
		},
	)
	require.NoError(t, err)
	require.NotNil(t, completedResp)

	events := parseSSEEvents(out.String())
	require.NotEmpty(t, events)

	// Find event types
	var eventTypes []string
	for _, e := range events {
		if et, ok := e["type"].(string); ok {
			eventTypes = append(eventTypes, et)
		}
	}

	assert.Contains(t, eventTypes, "response.created")
	assert.Contains(t, eventTypes, "response.in_progress")
	assert.Contains(t, eventTypes, "response.output_item.added")
	assert.Contains(t, eventTypes, "response.content_part.added")
	assert.Contains(t, eventTypes, "response.output_text.delta")
	assert.Contains(t, eventTypes, "response.output_text.done")
	assert.Contains(t, eventTypes, "response.content_part.done")
	assert.Contains(t, eventTypes, "response.output_item.done")
	assert.Contains(t, eventTypes, "response.completed")
}

func TestTransformAnthropicStreamToResponses_TextDeltaContent(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 5, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "Part1"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "Part2"},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 10},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "resp_test", nil, nil,
	)
	require.NoError(t, err)

	outputStr := out.String()

	// Verify both deltas appear
	assert.Contains(t, outputStr, "Part1")
	assert.Contains(t, outputStr, "Part2")

	// Find the done event and verify full text
	events := parseSSEEvents(outputStr)
	var fullText string
	for _, e := range events {
		if e["type"] == "response.output_text.done" {
			fullText, _ = e["text"].(string)
		}
	}
	assert.Equal(t, "Part1Part2", fullText)
}

func TestTransformAnthropicStreamToResponses_MessageEventsIncludeRequiredFields(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 5, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "hello"},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 3},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseSSEEvents(out.String())
	require.NotEmpty(t, events)

	var messageItemID string
	for _, e := range events {
		_, hasSeq := e["sequence_number"]
		assert.True(t, hasSeq, "every event must include sequence_number: %#v", e)

		typ, _ := e["type"].(string)
		if typ == "response.output_item.added" {
			item, _ := e["item"].(map[string]interface{})
			if item != nil && item["type"] == "message" {
				messageItemID, _ = item["id"].(string)
			}
		}
	}
	require.NotEmpty(t, messageItemID)

	for _, e := range events {
		typ, _ := e["type"].(string)
		switch typ {
		case "response.content_part.added", "response.output_text.delta", "response.output_text.done", "response.content_part.done":
			itemID, _ := e["item_id"].(string)
			assert.Equal(t, messageItemID, itemID, "event %s must include matching item_id", typ)
		}
	}
}

func TestTransformAnthropicStreamToResponses_ThinkingBlock(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 20, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "thinking"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "thinking_delta", "thinking": "I am reasoning"},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "My answer"},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 30},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	var completedResp *responses.Response
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil,
		func(r *responses.Response) { completedResp = r },
	)
	require.NoError(t, err)
	require.NotNil(t, completedResp)

	events := parseSSEEvents(out.String())
	var eventTypes []string
	for _, e := range events {
		if et, ok := e["type"].(string); ok {
			eventTypes = append(eventTypes, et)
		}
	}

	// Should have completion events
	assert.Contains(t, eventTypes, "response.completed")
}

func TestTransformAnthropicStreamToResponses_ToolUseBlock(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 15, "cache_read_input_tokens": 0},
			},
		},
		{
			"type": "content_block_start",
			"content_block": map[string]interface{}{
				"type": "tool_use",
				"id":   "call_xyz",
				"name": "get_weather",
			},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": `{"city`},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": `": "NYC"}`},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "tool_use"},
			"usage": map[string]interface{}{"output_tokens": 20},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseSSEEvents(out.String())
	var completedEvent map[string]interface{}
	var argumentsDoneEvent map[string]interface{}
	for _, e := range events {
		if e["type"] == "response.completed" {
			completedEvent = e
		}
		if e["type"] == "response.function_call_arguments.done" {
			argumentsDoneEvent = e
		}
	}
	require.NotNil(t, completedEvent)
	require.NotNil(t, argumentsDoneEvent)
	assert.Equal(t, "get_weather", argumentsDoneEvent["name"])
	assert.Equal(t, `{"city": "NYC"}`, argumentsDoneEvent["arguments"])

	respObj := completedEvent["response"].(map[string]interface{})
	output := respObj["output"].([]interface{})
	require.NotEmpty(t, output)

	// First output item should be function_call
	fc := output[0].(map[string]interface{})
	assert.Equal(t, "function_call", fc["type"])
	assert.Equal(t, "call_xyz", fc["call_id"])
	assert.Equal(t, "get_weather", fc["name"])
	assert.Contains(t, fc["arguments"].(string), "NYC")
}

func TestTransformAnthropicStreamToResponses_EmptyStream(t *testing.T) {
	// A minimal stream with no content
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 5, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 0},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseSSEEvents(out.String())
	var eventTypes []string
	for _, e := range events {
		if et, ok := e["type"].(string); ok {
			eventTypes = append(eventTypes, et)
		}
	}

	// Must always emit response.created + response.completed
	assert.Contains(t, eventTypes, "response.created")
	assert.Contains(t, eventTypes, "response.completed")
}

func TestTransformAnthropicStreamToResponses_UsageTokens(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 42, "cache_read_input_tokens": 10},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "hi"},
		},
		{"type": "content_block_stop"},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 7},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseSSEEvents(out.String())
	var completedEvent map[string]interface{}
	for _, e := range events {
		if e["type"] == "response.completed" {
			completedEvent = e
		}
	}
	require.NotNil(t, completedEvent)

	respObj := completedEvent["response"].(map[string]interface{})
	usage := respObj["usage"].(map[string]interface{})
	assert.Equal(t, float64(42), usage["input_tokens"])
	assert.Equal(t, float64(7), usage["output_tokens"])
	assert.Equal(t, float64(49), usage["total_tokens"])
}

func TestTransformAnthropicStreamToResponses_OnComplete(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 1, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "Done!"},
		},
		{"type": "content_block_stop"},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 3},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	var callbackCalled bool
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "resp_abc", nil,
		func(r *responses.Response) { callbackCalled = true },
	)
	require.NoError(t, err)
	assert.True(t, callbackCalled, "onComplete callback should have been called")
}

func TestTransformAnthropicStreamToResponses_TextDeltaOutputIndex(t *testing.T) {
	// Regression: text delta events must use output_index=0 for a simple text response.
	// the Python SDK to throw IndexError when accessing output[1] (array has only 1 item).
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{"type": "message_start", "message": map[string]interface{}{
			"usage": map[string]interface{}{"input_tokens": 5, "cache_read_input_tokens": 0},
		}},
		{"type": "content_block_start", "content_block": map[string]interface{}{"type": "text"}},
		{"type": "content_block_delta", "delta": map[string]interface{}{"type": "text_delta", "text": "hello"}},
		{"type": "content_block_stop"},
		{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": "end_turn"}, "usage": map[string]interface{}{"output_tokens": 3}},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil)
	require.NoError(t, err)

	for _, e := range parseSSEEvents(out.String()) {
		if e["type"] == "response.output_text.delta" {
			idx, ok := e["output_index"].(float64)
			require.True(t, ok, "output_index must be a number")
			assert.Equal(t, float64(0), idx, "text delta must reference output_index 0 for a simple response")
			return
		}
	}
	t.Fatal("no response.output_text.delta event found")
}

func TestTransformAnthropicStreamToResponses_ThinkingThenTextOutputIndex(t *testing.T) {
	// Regression: when reasoning precedes text, text deltas must use output_index=1
	// (reasoning is at 0, message at 1), not output_index=2.
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{"type": "message_start", "message": map[string]interface{}{
			"usage": map[string]interface{}{"input_tokens": 10, "cache_read_input_tokens": 0},
		}},
		{"type": "content_block_start", "content_block": map[string]interface{}{"type": "thinking"}},
		{"type": "content_block_delta", "delta": map[string]interface{}{"type": "thinking_delta", "thinking": "reasoning..."}},
		{"type": "content_block_stop"},
		{"type": "content_block_start", "content_block": map[string]interface{}{"type": "text"}},
		{"type": "content_block_delta", "delta": map[string]interface{}{"type": "text_delta", "text": "answer"}},
		{"type": "content_block_stop"},
		{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": "end_turn"}, "usage": map[string]interface{}{"output_tokens": 5}},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil)
	require.NoError(t, err)

	for _, e := range parseSSEEvents(out.String()) {
		if e["type"] == "response.output_text.delta" {
			idx, ok := e["output_index"].(float64)
			require.True(t, ok, "output_index must be a number")
			assert.Equal(t, float64(1), idx, "text delta must reference output_index 1 when reasoning item precedes it")
			return
		}
	}
	t.Fatal("no response.output_text.delta event found")
}

func TestTransformAnthropicStreamToResponses_ToolUseEmptyArgs(t *testing.T) {
	// Tool use with no arguments (empty partial_json deltas)
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 5, "cache_read_input_tokens": 0},
			},
		},
		{
			"type": "content_block_start",
			"content_block": map[string]interface{}{
				"type": "tool_use",
				"id":   "call_no_args",
				"name": "noop",
			},
		},
		// No input_json_delta events
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "tool_use"},
			"usage": map[string]interface{}{"output_tokens": 5},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	)
	require.NoError(t, err)

	events := parseSSEEvents(out.String())
	for _, e := range events {
		if e["type"] == "response.completed" {
			respObj := e["response"].(map[string]interface{})
			output := respObj["output"].([]interface{})
			require.NotEmpty(t, output)
			fc := output[0].(map[string]interface{})
			// Empty args should default to "{}"
			assert.Equal(t, "{}", fc["arguments"])
			return
		}
	}
	t.Fatal("no response.completed event found")
}

func TestTransformAnthropicStreamToResponses_ReturnsTerminalProviderError(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 5},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "partial"},
		},
		{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "overloaded_error",
				"message": "Overloaded",
			},
		},
	})

	var out bytes.Buffer
	completed := false
	err := TransformAnthropicStreamToResponses(
		strings.NewReader(stream),
		&out,
		"claude-opus-4-5",
		"resp_error",
		nil,
		func(*responses.Response) { completed = true },
	)

	require.ErrorContains(t, err, "anthropic stream error (overloaded_error): Overloaded")
	assert.False(t, completed)
	assert.NotContains(t, out.String(), "response.completed")
}
