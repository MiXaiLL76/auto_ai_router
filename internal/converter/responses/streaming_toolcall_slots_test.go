package responses

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// toolCallDelta builds one tool_calls[] entry of a Chat stream delta; empty
// id/name/args are left out of the chunk, as providers do.
func toolCallDelta(index int, id, name, args string) map[string]interface{} {
	fn := map[string]interface{}{}
	if name != "" {
		fn["name"] = name
	}
	if args != "" {
		fn["arguments"] = args
	}
	tc := map[string]interface{}{"index": index, "function": fn}
	if id != "" {
		tc["id"] = id
		tc["type"] = "function"
	}
	return tc
}

// streamToolCalls runs a stream of tool_calls deltas (one chunk per entry)
// through the transformer and returns the events and the completed response.
func streamToolCalls(t *testing.T, deltas ...map[string]interface{}) ([]map[string]interface{}, *Response) {
	t.Helper()
	var input strings.Builder
	for _, d := range deltas {
		input.WriteString(buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{d}})))
	}
	stopReason := "tool_calls"
	input.WriteString(buildSSEChunk(buildChatChunk("", &stopReason)))
	input.WriteString("data: [DONE]\n\n")

	var resp *Response
	var output bytes.Buffer
	require.NotPanics(t, func() {
		require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input.String()), &output, "m",
			func(r *Response) { resp = r }))
	})
	require.NotNil(t, resp)
	events := parseSSEEvents(t, output.String())
	assertConsistentOutputIndices(t, events)
	assertItemsPaired(t, events, resp)
	return events, resp
}

// assertItemsPaired checks that every output item the client saw open is
// closed exactly once, nothing is closed that was never opened, the completed
// response lists exactly the opened items, and no function_call reaches the
// client without the call_id it must echo back next turn.
func assertItemsPaired(t *testing.T, events []map[string]interface{}, resp *Response) {
	t.Helper()
	added := map[string]int{}
	done := map[string]int{}
	for _, e := range events {
		item, ok := e["item"].(map[string]interface{})
		if !ok {
			continue
		}
		id, _ := item["id"].(string)
		switch e["_event"] {
		case "response.output_item.added":
			added[id]++
		case "response.output_item.done":
			done[id]++
		}
		if item["type"] == "function_call" {
			assert.NotEmpty(t, item["call_id"], "%s for %s without call_id", e["_event"], id)
		}
	}
	for id, n := range added {
		assert.Equal(t, 1, n, "item %s added %d times", id, n)
		assert.Equal(t, 1, done[id], "item %s added but closed %d times", id, done[id])
	}
	for id := range done {
		assert.Contains(t, added, id, "item %s closed but never added", id)
	}
	require.Len(t, resp.Output, len(added), "completed output must list exactly the opened items")
	for _, item := range resp.Output {
		assert.Contains(t, added, item.ID)
		if item.Type == "function_call" {
			assert.NotEmpty(t, item.CallID, "completed function_call %s without call_id", item.ID)
		}
	}
}

// Some OpenAI-compatible backends send index 0 for every parallel call; a new
// ID on an index already in use is a new call, not a replacement of the old.
func TestStreamTransform_ParallelToolCallsSharingIndex(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(0, "call_a", "f", ""),
		toolCallDelta(0, "", "", `{"a":`),
		toolCallDelta(0, "", "", `1}`),
		toolCallDelta(0, "call_b", "g", ""),
		toolCallDelta(0, "", "", `{"b":2}`),
	)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "call_a", resp.Output[0].CallID)
	assert.Equal(t, "f", resp.Output[0].Name)
	assert.Equal(t, `{"a":1}`, resp.Output[0].Arguments)
	assert.Equal(t, "call_b", resp.Output[1].CallID)
	assert.Equal(t, "g", resp.Output[1].Name)
	assert.Equal(t, `{"b":2}`, resp.Output[1].Arguments)
}

// Parallel calls on distinct indices may interleave their argument chunks.
func TestStreamTransform_ParallelToolCallsInterleavedByIndex(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(0, "call_a", "f", ""),
		toolCallDelta(1, "call_b", "g", ""),
		toolCallDelta(1, "", "", `{"b":`),
		toolCallDelta(0, "", "", `{"a":`),
		toolCallDelta(0, "", "", `1}`),
		toolCallDelta(1, "", "", `2}`),
	)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "call_a", resp.Output[0].CallID)
	assert.Equal(t, `{"a":1}`, resp.Output[0].Arguments)
	assert.Equal(t, "call_b", resp.Output[1].CallID)
	assert.Equal(t, `{"b":2}`, resp.Output[1].Arguments)
}

// A first call at index 1 leaves no empty slot 0 behind: no phantom
// function_call with an empty call_id, no done without an added.
func TestStreamTransform_ToolCallIndexGapNoPhantomItem(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(1, "call_b", "f", `{}`),
	)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "call_b", resp.Output[0].CallID)
	assert.Equal(t, `{}`, resp.Output[0].Arguments)
}

// A huge upstream index must not allocate the gap before it.
func TestStreamTransform_HugeToolCallIndex(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(1_000_000_000, "call_x", "f", `{`),
		toolCallDelta(1_000_000_000, "", "", `}`),
	)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "call_x", resp.Output[0].CallID)
	assert.Equal(t, `{}`, resp.Output[0].Arguments)
}

// A call whose chunks never carry an ID still gets a call_id, so the client
// can answer it.
func TestStreamTransform_ToolCallWithoutIDGetsGeneratedCallID(t *testing.T) {
	events, resp := streamToolCalls(t,
		toolCallDelta(0, "", "f", `{"a":`),
		toolCallDelta(0, "", "", `1}`),
	)
	require.Len(t, resp.Output, 1)
	assert.True(t, strings.HasPrefix(resp.Output[0].CallID, "call_"), "call_id %q", resp.Output[0].CallID)
	assert.Equal(t, "f", resp.Output[0].Name)
	assert.Equal(t, `{"a":1}`, resp.Output[0].Arguments)

	// The same call_id is announced up front and repeated at done.
	var ids []interface{}
	for _, e := range events {
		if item, ok := e["item"].(map[string]interface{}); ok && item["type"] == "function_call" {
			ids = append(ids, item["call_id"])
		}
	}
	assert.Equal(t, []interface{}{resp.Output[0].CallID, resp.Output[0].CallID}, ids)
}

// Arguments-only chunks with no ID on an unseen index still open a call, with
// the added event before the first arguments delta.
func TestStreamTransform_ToolCallArgsOnlyOpensCallBeforeDelta(t *testing.T) {
	events, resp := streamToolCalls(t,
		toolCallDelta(0, "", "", `{}`),
	)
	require.Len(t, resp.Output, 1)
	assert.NotEmpty(t, resp.Output[0].CallID)
	var order []string
	for _, e := range events {
		switch e["_event"] {
		case "response.output_item.added", "response.function_call_arguments.delta":
			order = append(order, e["_event"].(string))
		}
	}
	assert.Equal(t, []string{"response.output_item.added", "response.function_call_arguments.delta"}, order)
}

// A tool_calls entry with neither ID nor data (index only) opens nothing.
func TestStreamTransform_EmptyToolCallDeltaIgnored(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(0, "", "", ""),
		toolCallDelta(3, "", "", ""),
	)
	assert.Empty(t, resp.Output)
}

// Providers that repeat the call's ID on every chunk continue the same call
// rather than reopening it.
func TestStreamTransform_RepeatedIDDoesNotReopenCall(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(0, "call_a", "f", `{`),
		toolCallDelta(0, "call_a", "", `"x":1`),
		toolCallDelta(0, "call_a", "f", `}`),
	)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, `{"x":1}`, resp.Output[0].Arguments)
	assert.Equal(t, "f", resp.Output[0].Name)
}

// A name sent with the first chunk is not overwritten by a later chunk.
func TestStreamTransform_ToolCallNameNotOverwritten(t *testing.T) {
	_, resp := streamToolCalls(t,
		toolCallDelta(0, "call_a", "first", `{}`),
		toolCallDelta(0, "", "second", ``),
	)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "first", resp.Output[0].Name)
}

// The number of calls one stream may open is capped.
func TestStreamTransform_ToolCallLimit(t *testing.T) {
	deltas := make([]map[string]interface{}, 0, maxStreamToolCalls+5)
	for i := 0; i < maxStreamToolCalls+5; i++ {
		deltas = append(deltas, toolCallDelta(i, fmt.Sprintf("call_%d", i), "f", `{}`))
	}
	_, resp := streamToolCalls(t, deltas...)
	assert.Len(t, resp.Output, maxStreamToolCalls)
}

// Reasoning, then parallel calls sharing index 0: reasoning closes before the
// first call opens and keeps output_index 0.
func TestStreamTransform_ReasoningThenParallelToolCallsSharingIndex(t *testing.T) {
	stopReason := "tool_calls"
	input := buildSSEChunk(buildReasoningChunk("Need both.")) +
		buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{toolCallDelta(0, "call_a", "f", `{}`)}})) +
		buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{toolCallDelta(0, "call_b", "f", `{}`)}})) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		"data: [DONE]\n\n"

	var resp *Response
	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "m",
		func(r *Response) { resp = r }))
	events := parseSSEEvents(t, output.String())
	assertConsistentOutputIndices(t, events)
	require.NotNil(t, resp)
	assertItemsPaired(t, events, resp)

	require.Len(t, resp.Output, 3)
	assert.Equal(t, "reasoning", resp.Output[0].Type)
	assert.Equal(t, "call_a", resp.Output[1].CallID)
	assert.Equal(t, "call_b", resp.Output[2].CallID)
}

// Two tool_calls entries in one chunk, both at index 0 with different IDs.
func TestStreamTransform_TwoCallsSameIndexInOneChunk(t *testing.T) {
	stopReason := "tool_calls"
	input := buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{
		toolCallDelta(0, "call_a", "f", `{"a":1}`),
		toolCallDelta(0, "call_b", "g", `{"b":2}`),
	}})) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		"data: [DONE]\n\n"

	var resp *Response
	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "m",
		func(r *Response) { resp = r }))
	events := parseSSEEvents(t, output.String())
	assertConsistentOutputIndices(t, events)
	require.NotNil(t, resp)
	assertItemsPaired(t, events, resp)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, `{"a":1}`, resp.Output[0].Arguments)
	assert.Equal(t, `{"b":2}`, resp.Output[1].Arguments)
}

// Stream cut off without [DONE] mid tool call still closes every opened call.
func TestStreamTransform_ToolCallsWithoutDoneStillPaired(t *testing.T) {
	input := buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{toolCallDelta(0, "call_a", "f", `{"a":`)}})) +
		buildSSEChunk(buildDeltaChunk(map[string]interface{}{"tool_calls": []interface{}{toolCallDelta(0, "call_b", "f", `{}`)}}))

	var resp *Response
	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "m",
		func(r *Response) { resp = r }))
	events := parseSSEEvents(t, output.String())
	assertConsistentOutputIndices(t, events)
	require.NotNil(t, resp)
	assertItemsPaired(t, events, resp)
	require.Len(t, resp.Output, 2)
}

// An empty reasoning_content must not hide a populated reasoning field in a
// stream delta.
func TestStreamTransform_EmptyReasoningContentFallsBackToReasoning(t *testing.T) {
	stopReason := "stop"
	input := buildSSEChunk(buildDeltaChunk(map[string]interface{}{"reasoning_content": "", "reasoning": "Hmm."})) +
		buildSSEChunk(buildChatChunk("Ok.", &stopReason)) +
		"data: [DONE]\n\n"

	var resp *Response
	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "m",
		func(r *Response) { resp = r }))
	require.NotNil(t, resp)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "reasoning", resp.Output[0].Type)
	assert.Equal(t, "Hmm.", resp.Output[0].Summary[0].Text)
}
