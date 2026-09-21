package responses

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// collectSSEChunks parses "data: {...}" SSE lines out of raw output, decoding
// each into a generic map for assertion (skips the terminal "[DONE]" line).
func collectSSEChunks(t *testing.T, raw []byte) []map[string]interface{} {
	t.Helper()
	var chunks []map[string]interface{}
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
		chunks = append(chunks, chunk)
	}
	require.NoError(t, scanner.Err())
	return chunks
}

func sseLine(eventJSON string) string {
	return "data: " + eventJSON + "\n\n"
}

func TestTransformResponsesStreamToChat_TextDeltas(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_1","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`) +
			sseLine(`{"type":"response.output_text.delta","output_index":0,"delta":"lo"}`) +
			sseLine(`{"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))

	chunks := collectSSEChunks(t, out.Bytes())
	require.GreaterOrEqual(t, len(chunks), 4)

	// First chunk: role-only.
	firstDelta := chunks[0]["choices"].([]interface{})[0].(map[string]interface{})["delta"].(map[string]interface{})
	assert.Equal(t, "assistant", firstDelta["role"])

	// Content deltas, in order, reconstruct the full text.
	var text string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		if content, ok := delta["content"].(string); ok {
			text += content
		}
	}
	assert.Equal(t, "Hello", text)

	// Terminal finish_reason chunk.
	last := chunks[len(chunks)-2] // finish chunk precedes the usage-only chunk
	finishChoice := last["choices"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "stop", finishChoice["finish_reason"])

	// Usage-only terminal chunk (no choices).
	usageChunk := chunks[len(chunks)-1]
	assert.Empty(t, usageChunk["choices"])
	usage := usageChunk["usage"].(map[string]interface{})
	assert.Equal(t, float64(5), usage["prompt_tokens"])
	assert.Equal(t, float64(2), usage["completion_tokens"])
}

func TestTransformResponsesStreamToChat_FunctionCall(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_2","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_xyz","name":"get_weather"}}`) +
			sseLine(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}`) +
			sseLine(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"paris\"}"}`) +
			sseLine(`{"type":"response.completed","response":{"id":"resp_2","object":"response","status":"completed","output":[{"type":"function_call","call_id":"call_xyz","name":"get_weather","arguments":"{\"city\":\"paris\"}"}]}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))

	chunks := collectSSEChunks(t, out.Bytes())

	var toolCallID, name, args string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		tcRaw, ok := delta["tool_calls"]
		if !ok {
			continue
		}
		for _, tcAny := range tcRaw.([]interface{}) {
			tc := tcAny.(map[string]interface{})
			assert.Equal(t, float64(0), tc["index"])
			if id, ok := tc["id"].(string); ok && id != "" {
				toolCallID = id
			}
			fn := tc["function"].(map[string]interface{})
			if n, ok := fn["name"].(string); ok && n != "" {
				name = n
			}
			if a, ok := fn["arguments"].(string); ok {
				args += a
			}
		}
	}
	assert.Equal(t, "call_xyz", toolCallID)
	assert.Equal(t, "get_weather", name)
	assert.Equal(t, `{"city":"paris"}`, args)

	// finish_reason must be tool_calls once a function_call was streamed.
	var finishReason string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		if fr, ok := choices[0].(map[string]interface{})["finish_reason"].(string); ok {
			finishReason = fr
		}
	}
	assert.Equal(t, "tool_calls", finishReason)
}

// TestTransformResponsesStreamToChat_FailedSurfacesErrorAsContent reproduces
// a real gap: a stream that ends in response.failed with an embedded
// error.message and no other output must surface that message as content,
// exactly like ResponseToChat does non-streaming for the same status="failed"
// shape -- not silently hand back finish_reason:"stop" with an empty message.
func TestTransformResponsesStreamToChat_FailedSurfacesErrorAsContent(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_fail","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.failed","response":{"id":"resp_fail","object":"response","status":"failed","output":[],"error":{"message":"upstream rate limited","code":"rate_limit_exceeded"}}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))
	chunks := collectSSEChunks(t, out.Bytes())

	var text, finishReason string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		choice := choices[0].(map[string]interface{})
		delta := choice["delta"].(map[string]interface{})
		if content, ok := delta["content"].(string); ok {
			text += content
		}
		if fr, ok := choice["finish_reason"].(string); ok {
			finishReason = fr
		}
	}
	assert.Equal(t, "upstream rate limited", text)
	assert.Equal(t, "stop", finishReason)
}

func TestTransformResponsesStreamToChat_FailedDoesNotOverwriteStreamedContent(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_partial","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.output_text.delta","output_index":0,"delta":"partial answer"}`) +
			sseLine(`{"type":"response.failed","response":{"id":"resp_partial","object":"response","status":"failed","output":[],"error":{"message":"connection reset"}}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))
	chunks := collectSSEChunks(t, out.Bytes())

	var text string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		if content, ok := delta["content"].(string); ok {
			text += content
		}
	}
	assert.Equal(t, "partial answer", text)
}

func TestTransformResponsesStreamToChat_StandaloneErrorEvent(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_err","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"error","code":"server_error","message":"the model is overloaded"}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))
	chunks := collectSSEChunks(t, out.Bytes())

	var text, finishReason string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		choice := choices[0].(map[string]interface{})
		delta := choice["delta"].(map[string]interface{})
		if content, ok := delta["content"].(string); ok {
			text += content
		}
		if fr, ok := choice["finish_reason"].(string); ok {
			finishReason = fr
		}
	}
	assert.Equal(t, "the model is overloaded", text)
	assert.Equal(t, "stop", finishReason)
}

func TestTransformResponsesStreamToChat_RefusalDelta(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_refusal","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.refusal.delta","output_index":0,"delta":"I can't "}`) +
			sseLine(`{"type":"response.refusal.delta","output_index":0,"delta":"help with that."}`) +
			sseLine(`{"type":"response.completed","response":{"id":"resp_refusal","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))
	chunks := collectSSEChunks(t, out.Bytes())

	var refusal string
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		if r, ok := delta["refusal"].(string); ok {
			refusal += r
		}
	}
	assert.Equal(t, "I can't help with that.", refusal)
}

func TestTransformResponsesStreamToChat_ReasoningSurvivesAsThinkingBlocks(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_reason","object":"response","status":"in_progress"}}`) +
			sseLine(`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}`) +
			sseLine(`{"type":"response.completed","response":{"id":"resp_reason","object":"response","status":"completed","output":[`+
				`{"type":"reasoning","id":"rs_abc","encrypted_content":"opaque-blob","summary":[{"type":"summary_text","text":"thinking"}]},`+
				`{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{}"}`+
				`]}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "o3-pro", &out))
	chunks := collectSSEChunks(t, out.Bytes())

	var blocks []interface{}
	for _, c := range chunks {
		choices := c["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		if tb, ok := delta["thinking_blocks"].([]interface{}); ok {
			blocks = tb
		}
	}
	require.Len(t, blocks, 1, "the final response.completed event must carry the reasoning item forward as a thinking_blocks chunk")
	block := blocks[0].(map[string]interface{})
	assert.Equal(t, "responses_reasoning", block["type"])
	assert.Equal(t, "rs_abc", block["id"])
	assert.Equal(t, "opaque-blob", block["encrypted_content"])
}

func TestTransformResponsesStreamToChat_MalformedLineSkipped(t *testing.T) {
	input := strings.NewReader(
		sseLine(`{"type":"response.created","response":{"id":"resp_3","object":"response","status":"in_progress"}}`) +
			"data: {not valid json\n\n" +
			sseLine(`{"type":"response.output_text.delta","output_index":0,"delta":"ok"}`) +
			sseLine(`{"type":"response.completed","response":{"id":"resp_3","object":"response","status":"completed","output":[]}}`) +
			"data: [DONE]\n\n",
	)

	var out bytes.Buffer
	require.NoError(t, TransformResponsesStreamToChat(input, "gpt-5-pro", &out))
	assert.Contains(t, out.String(), `"content":"ok"`)
	assert.Contains(t, out.String(), "data: [DONE]")
}
