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

func TestTransformAnthropicStreamToResponses_TextStreamWithCitations(t *testing.T) {
	// Regression test: the streaming path must accumulate citations_delta events
	// and attach them to output_text.annotations on block finalize, matching the
	// non-streaming path's webSearchCitationsToAnnotations behavior. Two citations
	// quote the same substring ("Paris") from two different places in the text, to
	// also cover that each resolves to its own occurrence instead of both
	// collapsing onto the first match.
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{"input_tokens": 10, "cache_read_input_tokens": 0},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "Note: Paris is lovely. Paris is the capital of France."},
		},
		{
			"type": "content_block_delta",
			"delta": map[string]interface{}{
				"type": "citations_delta",
				"citation": map[string]interface{}{
					"type":       "web_search_result_location",
					"cited_text": "Paris",
					"url":        "https://example.com/paris-1",
					"title":      "Paris travel guide",
				},
			},
		},
		{
			"type": "content_block_delta",
			"delta": map[string]interface{}{
				"type": "citations_delta",
				"citation": map[string]interface{}{
					"type":       "web_search_result_location",
					"cited_text": "Paris",
					"url":        "https://example.com/paris-2",
					"title":      "Paris - capital of France",
				},
			},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 12},
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
	output := respObj["output"].([]interface{})
	require.NotEmpty(t, output)

	msg := output[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	require.NotEmpty(t, content)
	textContent := content[0].(map[string]interface{})

	annotations := textContent["annotations"].([]interface{})
	require.Len(t, annotations, 2)

	first := annotations[0].(map[string]interface{})
	assert.Equal(t, "url_citation", first["type"])
	assert.Equal(t, "https://example.com/paris-1", first["url"])
	assert.EqualValues(t, 6, first["start_index"])
	assert.EqualValues(t, 11, first["end_index"])

	second := annotations[1].(map[string]interface{})
	assert.Equal(t, "https://example.com/paris-2", second["url"])
	assert.EqualValues(t, 23, second["start_index"])
	assert.EqualValues(t, 28, second["end_index"])
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

// TestTransformAnthropicStreamToResponses_ServerToolUseWebSearch verifies that
// a streamed server_tool_use (Anthropic's hosted web_search) block becomes a
// web_search_call output item, instead of being silently dropped (there was
// previously no case for "server_tool_use" at all in the streaming switch).
func TestTransformAnthropicStreamToResponses_ServerToolUseWebSearch(t *testing.T) {
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
				"type": "server_tool_use",
				"id":   "srvtoolu_01",
				"name": "web_search",
			},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": `{"query`},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": `": "weather NYC"}`},
		},
		{
			"type": "content_block_stop",
		},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
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
	var itemDoneEvent map[string]interface{}
	for _, e := range events {
		if e["type"] == "response.completed" {
			completedEvent = e
		}
		if e["type"] == "response.output_item.done" {
			itemDoneEvent = e
		}
		// A web_search_call must never stream as a function_call_arguments event.
		assert.NotEqual(t, "response.function_call_arguments.delta", e["type"])
		assert.NotEqual(t, "response.function_call_arguments.done", e["type"])
	}
	require.NotNil(t, completedEvent)
	require.NotNil(t, itemDoneEvent)
	doneItem := itemDoneEvent["item"].(map[string]interface{})
	assert.Equal(t, "web_search_call", doneItem["type"])
	assert.Equal(t, "completed", doneItem["status"])

	respObj := completedEvent["response"].(map[string]interface{})
	output := respObj["output"].([]interface{})
	require.NotEmpty(t, output)

	wsCall := output[0].(map[string]interface{})
	assert.Equal(t, "web_search_call", wsCall["type"])
	assert.Equal(t, "completed", wsCall["status"])
	queries := wsCall["queries"].([]interface{})
	require.Len(t, queries, 1)
	assert.Equal(t, "weather NYC", queries[0])
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
	assert.Equal(t, float64(52), usage["input_tokens"])
	assert.Equal(t, float64(7), usage["output_tokens"])
	assert.Equal(t, float64(59), usage["total_tokens"])
	details := usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(10), details["cached_tokens"])
}

func TestTransformAnthropicStreamToResponses_CacheUsageUsesInclusiveInputTotal(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{
					"input_tokens": 100, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20,
					"cache_creation": map[string]interface{}{"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 15},
				},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "ok"},
		},
		{"type": "content_block_stop"},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{"output_tokens": 10},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	))

	events := parseSSEEvents(out.String())
	var response map[string]interface{}
	for _, event := range events {
		if event["type"] == "response.completed" {
			response = event["response"].(map[string]interface{})
		}
	}
	require.NotNil(t, response)
	usage := response["usage"].(map[string]interface{})
	assert.Equal(t, float64(200), usage["input_tokens"])
	assert.Equal(t, float64(210), usage["total_tokens"])
	details := usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(80), details["cached_tokens"])
	assert.Equal(t, float64(20), details["cache_creation_tokens"])
	ttlDetails := details["cache_creation_token_details"].(map[string]interface{})
	assert.Equal(t, float64(5), ttlDetails["ephemeral_5m_input_tokens"])
	assert.Equal(t, float64(15), ttlDetails["ephemeral_1h_input_tokens"])
}

func TestTransformAnthropicStreamToResponses_ExplicitZeroCacheDeltaClearsPreviousUsage(t *testing.T) {
	stream := buildAnthropicSSEStream([]map[string]interface{}{
		{
			"type": "message_start",
			"message": map[string]interface{}{
				"usage": map[string]interface{}{
					"input_tokens": 100, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20,
					"cache_creation": map[string]interface{}{"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 15},
				},
			},
		},
		{
			"type":          "content_block_start",
			"content_block": map[string]interface{}{"type": "text"},
		},
		{
			"type":  "content_block_delta",
			"delta": map[string]interface{}{"type": "text_delta", "text": "ok"},
		},
		{"type": "content_block_stop"},
		{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
			"usage": map[string]interface{}{
				"output_tokens":               10,
				"cache_read_input_tokens":     0,
				"cache_creation_input_tokens": 0,
			},
		},
		{"type": "message_stop"},
	})

	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToResponses(
		strings.NewReader(stream), &out, "claude-opus-4-5", "", nil, nil,
	))

	events := parseSSEEvents(out.String())
	var response map[string]interface{}
	for _, event := range events {
		if event["type"] == "response.completed" {
			response = event["response"].(map[string]interface{})
		}
	}
	require.NotNil(t, response)
	usage := response["usage"].(map[string]interface{})
	assert.Equal(t, float64(100), usage["input_tokens"])
	assert.Equal(t, float64(110), usage["total_tokens"])
	details := usage["input_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(0), details["cached_tokens"])
	assert.NotContains(t, details, "cache_creation_tokens")
	assert.NotContains(t, details, "cache_creation_token_details")
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

// TestTransformAnthropicStreamToResponses_InputTokensFromMessageDelta covers Anthropic-
// compatible providers that fill usage the other way round from Anthropic itself: they send
// message_start with a placeholder usage:{input_tokens:0} and only report the real input
// count in message_delta. The accumulator used to keep that placeholder and report the
// cached count — or zero — as the whole input, under-billing every streamed Responses
// request served by such a provider.
func TestTransformAnthropicStreamToResponses_InputTokensFromMessageDelta(t *testing.T) {
	t.Run("message_delta_supplies_input_tokens", func(t *testing.T) {
		stream := buildAnthropicSSEStream([]map[string]interface{}{
			{
				"type": "message_start",
				"message": map[string]interface{}{
					"usage": map[string]interface{}{"input_tokens": 0, "output_tokens": 0},
				},
			},
			{
				"type":          "content_block_start",
				"content_block": map[string]interface{}{"type": "text", "id": "", "name": ""},
			},
			{
				"type":  "content_block_delta",
				"delta": map[string]interface{}{"type": "text_delta", "text": "OK"},
			},
			{"type": "content_block_stop"},
			{
				"type":  "message_delta",
				"delta": map[string]interface{}{"stop_reason": "end_turn"},
				"usage": map[string]interface{}{"input_tokens": 3695, "output_tokens": 2},
			},
			{"type": "message_stop"},
		})

		var out bytes.Buffer
		var completedResp *responses.Response
		err := TransformAnthropicStreamToResponses(
			strings.NewReader(stream), &out, "anthropic-compatible-model", "", nil,
			func(r *responses.Response) { completedResp = r },
		)
		require.NoError(t, err)
		require.NotNil(t, completedResp)

		assert.Equal(t, 3695, completedResp.Usage.InputTokens, "input tokens must come from message_delta when message_start only carries a placeholder")
		assert.Equal(t, 2, completedResp.Usage.OutputTokens)
		assert.Equal(t, 3697, completedResp.Usage.TotalTokens)
	})

	t.Run("message_start_value_kept_when_delta_omits_input_tokens", func(t *testing.T) {
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
			{"type": "content_block_stop"},
			{
				"type":  "message_delta",
				"delta": map[string]interface{}{"stop_reason": "end_turn"},
				"usage": map[string]interface{}{"output_tokens": 5},
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

		assert.Equal(t, 10, completedResp.Usage.InputTokens, "Anthropic reports input tokens in message_start and omits them later; that value must survive")
		assert.Equal(t, 5, completedResp.Usage.OutputTokens)
	})

	t.Run("zero_in_message_delta_does_not_clobber_message_start", func(t *testing.T) {
		stream := buildAnthropicSSEStream([]map[string]interface{}{
			{
				"type": "message_start",
				"message": map[string]interface{}{
					"usage": map[string]interface{}{"input_tokens": 42, "cache_read_input_tokens": 128},
				},
			},
			{
				"type":          "content_block_start",
				"content_block": map[string]interface{}{"type": "text", "id": "", "name": ""},
			},
			{
				"type":  "content_block_delta",
				"delta": map[string]interface{}{"type": "text_delta", "text": "OK"},
			},
			{"type": "content_block_stop"},
			{
				"type":  "message_delta",
				"delta": map[string]interface{}{"stop_reason": "end_turn"},
				"usage": map[string]interface{}{"input_tokens": 0, "output_tokens": 2},
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

		assert.Equal(t, 170, completedResp.Usage.InputTokens, "42 fresh + 128 cached; a zero in message_delta must not wipe the message_start count")
		assert.Equal(t, 128, completedResp.Usage.InputTokensDetails.CachedTokens)
	})
}
