package anthropicresponses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// reasoningEventTypes returns, in order, the reasoning-related events of a
// stream (item added/done for reasoning items and every reasoning_summary_*).
func reasoningEventTypes(events []map[string]interface{}) []string {
	var out []string
	for _, e := range events {
		typ, _ := e["type"].(string)
		if strings.HasPrefix(typ, "response.reasoning_summary_") {
			out = append(out, typ)
			continue
		}
		if item, ok := e["item"].(map[string]interface{}); ok && item["type"] == "reasoning" {
			out = append(out, typ)
		}
	}
	return out
}

func thinkingStream(thinking ...string) string {
	events := []map[string]interface{}{
		{"type": "message_start", "message": map[string]interface{}{"usage": map[string]interface{}{"input_tokens": 1}}},
		{"type": "content_block_start", "content_block": map[string]interface{}{"type": "thinking"}},
	}
	for _, d := range thinking {
		events = append(events, map[string]interface{}{
			"type": "content_block_delta", "delta": map[string]interface{}{"type": "thinking_delta", "thinking": d},
		})
	}
	events = append(events,
		map[string]interface{}{"type": "content_block_stop"},
		map[string]interface{}{"type": "content_block_start", "content_block": map[string]interface{}{"type": "text"}},
		map[string]interface{}{"type": "content_block_delta", "delta": map[string]interface{}{"type": "text_delta", "text": "Answer"}},
		map[string]interface{}{"type": "content_block_stop"},
		map[string]interface{}{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": "end_turn"}, "usage": map[string]interface{}{"output_tokens": 3}},
		map[string]interface{}{"type": "message_stop"},
	)
	return buildAnthropicSSEStream(events)
}

// Thinking streams with the same reasoning_summary_* events the Chat and
// Vertex converters emit, all at the reasoning item's output_index.
func TestTransformAnthropicStreamToResponses_ThinkingStreamsSummaryEvents(t *testing.T) {
	var out bytes.Buffer
	var resp *responses.Response
	require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(thinkingStream("Let me ", "think.")),
		&out, "claude", "", nil, func(r *responses.Response) { resp = r }))
	events := parseSSEEvents(out.String())

	assert.Equal(t, []string{
		"response.output_item.added",
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done",
		"response.reasoning_summary_part.done",
		"response.output_item.done",
	}, reasoningEventTypes(events))

	var itemID string
	for _, e := range events {
		typ, _ := e["type"].(string)
		if item, ok := e["item"].(map[string]interface{}); ok && item["type"] == "reasoning" {
			itemID, _ = item["id"].(string)
			assert.Equal(t, float64(0), e["output_index"], typ)
		}
		if strings.HasPrefix(typ, "response.reasoning_summary_") {
			assert.Equal(t, float64(0), e["output_index"], typ)
			assert.Equal(t, itemID, e["item_id"], typ)
			assert.Equal(t, float64(0), e["summary_index"], typ)
		}
		if typ == "response.reasoning_summary_text.done" {
			assert.Equal(t, "Let me think.", e["text"])
		}
	}

	require.NotNil(t, resp)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "Let me think.", resp.Output[0].Summary[0].Text)
}

// An empty thinking block still closes every event it opened.
func TestTransformAnthropicStreamToResponses_EmptyThinkingEventsPaired(t *testing.T) {
	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(thinkingStream()),
		&out, "claude", "", nil, nil))

	assert.Equal(t, []string{
		"response.output_item.added",
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.done",
		"response.reasoning_summary_part.done",
		"response.output_item.done",
	}, reasoningEventTypes(parseSSEEvents(out.String())))
}

func anthropicMessages(t *testing.T, input string) []map[string]interface{} {
	t.Helper()
	result, err := ResponsesRequestToAnthropic([]byte(`{"model": "claude", "input": `+input+`}`), "claude")
	require.NoError(t, err)
	var ar struct {
		Messages []map[string]interface{} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(result, &ar))
	return ar.Messages
}

func findThinkingBlock(messages []map[string]interface{}) map[string]interface{} {
	for _, m := range messages {
		blocks, _ := m["content"].([]interface{})
		for _, b := range blocks {
			if block, _ := b.(map[string]interface{}); block["type"] == "thinking" {
				return block
			}
		}
	}
	return nil
}

// The signature is verified against the thinking text it was issued for, so a
// reasoning item carrying both sends both back.
func TestResponsesRequestToAnthropic_ReasoningSignatureTravelsWithText(t *testing.T) {
	block := findThinkingBlock(anthropicMessages(t, `[
		{"role": "user", "content": "q"},
		{"type": "reasoning", "summary": [{"type": "summary_text", "text": "full thinking"}], "encrypted_content": "sig"},
		{"role": "assistant", "content": "a"}
	]`))
	require.NotNil(t, block)
	assert.Equal(t, "full thinking", block["thinking"])
	assert.Equal(t, "sig", block["signature"])
}

// Raw reasoning_text content is read like in the Chat converter.
func TestResponsesRequestToAnthropic_ReasoningFromRawContent(t *testing.T) {
	block := findThinkingBlock(anthropicMessages(t, `[
		{"role": "user", "content": "q"},
		{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "raw"}]},
		{"role": "assistant", "content": "a"}
	]`))
	require.NotNil(t, block)
	assert.Equal(t, "raw", block["thinking"])
}

// A reasoning item with no text and no signature produces no thinking block.
func TestResponsesRequestToAnthropic_EmptyReasoningDropped(t *testing.T) {
	assert.Nil(t, findThinkingBlock(anthropicMessages(t, `[
		{"role": "user", "content": "q"},
		{"type": "reasoning", "summary": []},
		{"role": "assistant", "content": "a"}
	]`)))
}
