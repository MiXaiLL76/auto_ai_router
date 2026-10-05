package vertexresponses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

func vertexChunk(parts ...map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{
		"candidates": []map[string]interface{}{{
			"content": map[string]interface{}{"role": "model", "parts": parts},
		}},
	}
}

// Thoughts stream live with reasoning_summary_* events (they used to be
// accumulated silently until the item closed), at the reasoning item's
// output_index, closed before output_item.done.
func TestTransformVertexStreamToResponses_ThoughtsStreamSummaryEvents(t *testing.T) {
	last := vertexChunk(map[string]interface{}{"text": "Answer"})
	last["candidates"].([]map[string]interface{})[0]["finishReason"] = "STOP"
	stream := buildVertexSSEStream([]map[string]interface{}{
		vertexChunk(map[string]interface{}{"text": "Let me ", "thought": true}),
		vertexChunk(map[string]interface{}{"text": "think.", "thought": true}),
		last,
	})

	var out bytes.Buffer
	var resp *responses.Response
	require.NoError(t, TransformVertexStreamToResponses(strings.NewReader(stream), &out, "gemini", "", nil,
		func(r *responses.Response) { resp = r }))
	events := parseVertexSSEEvents(out.String())

	var seq []string
	var itemID string
	for _, e := range events {
		typ, _ := e["type"].(string)
		item, _ := e["item"].(map[string]interface{})
		isReasoning := strings.HasPrefix(typ, "response.reasoning_summary_") || (item != nil && item["type"] == "reasoning")
		if !isReasoning {
			continue
		}
		seq = append(seq, typ)
		assert.Equal(t, float64(0), e["output_index"], typ)
		if item != nil {
			itemID, _ = item["id"].(string)
		} else {
			assert.Equal(t, itemID, e["item_id"], typ)
		}
		if typ == "response.reasoning_summary_text.done" {
			assert.Equal(t, "Let me think.", e["text"])
		}
	}
	assert.Equal(t, []string{
		"response.output_item.added",
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done",
		"response.reasoning_summary_part.done",
		"response.output_item.done",
	}, seq)

	require.NotNil(t, resp)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "reasoning", resp.Output[0].Type)
	assert.Equal(t, "Let me think.", resp.Output[0].Summary[0].Text)
}

func vertexContents(t *testing.T, input string) []map[string]interface{} {
	t.Helper()
	result, err := ResponsesRequestToVertex([]byte(`{"model": "gemini", "input": `+input+`}`), "gemini")
	require.NoError(t, err)
	var vr struct {
		Contents []map[string]interface{} `json:"contents"`
	}
	require.NoError(t, json.Unmarshal(result, &vr))
	return vr.Contents
}

// Reasoning items are not replayed to Gemini as visible "[Reasoning]: ..."
// model text the model would read back as an answer it already gave.
func TestResponsesRequestToVertex_ReasoningItemNotReplayedAsText(t *testing.T) {
	contents := vertexContents(t, `[
		{"role": "user", "content": "q"},
		{"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret thoughts"}]},
		{"role": "assistant", "content": "a"}
	]`)
	require.Len(t, contents, 2)
	assert.Equal(t, "user", contents[0]["role"])
	assert.Equal(t, "model", contents[1]["role"])
	raw, _ := json.Marshal(contents)
	assert.NotContains(t, string(raw), "secret thoughts")
	assert.NotContains(t, string(raw), "[Reasoning]")
}

// Reasoning items around function calls add no contents of their own.
func TestResponsesRequestToVertex_ReasoningAddsNoContentsAroundCalls(t *testing.T) {
	contents := vertexContents(t, `[
		{"role": "user", "content": "q"},
		{"type": "reasoning", "summary": [{"type": "summary_text", "text": "r1"}]},
		{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
		{"type": "reasoning", "summary": [{"type": "summary_text", "text": "r2"}]},
		{"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"},
		{"type": "function_call_output", "call_id": "c1", "output": "1"},
		{"type": "function_call_output", "call_id": "c2", "output": "2"}
	]`)
	var roles []interface{}
	for _, c := range contents {
		roles = append(roles, c["role"])
		if c["role"] != "model" {
			continue
		}
		parts, _ := c["parts"].([]interface{})
		for _, p := range parts {
			assert.NotContains(t, p, "text", "no text part may come from a reasoning item")
		}
	}
	assert.Equal(t, []interface{}{"user", "model", "model", "user"}, roles)
}
