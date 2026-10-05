package responses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// replayAsChat feeds a converted Responses output back the way a Responses
// client continues a conversation — previous turn's items followed by the
// tool result — and returns the Chat Completions messages RequestToChat
// builds from it.
func replayAsChat(t *testing.T, output []OutputItem) []map[string]interface{} {
	t.Helper()
	next, err := PrependHistoryToInput(
		[]byte(`{"model": "deepseek-v4-pro", "input": [{"type": "function_call_output", "call_id": "call_1", "output": "Sunny"}]}`),
		json.RawMessage(`[{"role": "user", "content": "Weather in Paris?"}]`),
		output,
	)
	require.NoError(t, err)
	return chatMessages(t, string(next))
}

// assertAssistantTurnRestored checks the replayed history carries the
// original assistant turn as one Chat message with its reasoning intact.
func assertAssistantTurnRestored(t *testing.T, messages []map[string]interface{}, wantReasoning, wantText string) {
	t.Helper()
	require.Len(t, messages, 3, "user, assistant turn, tool result")
	assert.Equal(t, "user", messages[0]["role"])
	assert.Equal(t, "tool", messages[2]["role"])

	turn := messages[1]
	assert.Equal(t, "assistant", turn["role"])
	assert.Equal(t, wantReasoning, turn["reasoning_content"])
	if wantText != "" {
		content := turn["content"].([]interface{})
		require.Len(t, content, 1)
		assert.Equal(t, wantText, content[0].(map[string]interface{})["text"])
	}
	toolCalls := turn["tool_calls"].([]interface{})
	require.Len(t, toolCalls, 1)
	call := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "call_1", call["id"])
	fn := call["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
	assert.Equal(t, `{"city":"Paris"}`, fn["arguments"])
}

// TestChatResponsesRoundTrip_PreservesReasoning: a Chat Completions assistant
// turn converted to Responses output and replayed as the next request's input
// must reach the Chat provider again as the same assistant message —
// reasoning_content, content and tool_calls together.
func TestChatResponsesRoundTrip_PreservesReasoning(t *testing.T) {
	chatBody := `{
		"id": "chatcmpl-1", "object": "chat.completion", "created": 1700000000, "model": "deepseek-v4-pro",
		"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
			"role": "assistant",
			"content": "Checking the weather.",
			"reasoning_content": "The user asked about Paris; call get_weather.",
			"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]
		}}]
	}`
	respBody, err := ChatToResponse([]byte(chatBody))
	require.NoError(t, err)
	var resp Response
	require.NoError(t, json.Unmarshal(respBody, &resp))

	assertAssistantTurnRestored(t, replayAsChat(t, resp.Output),
		"The user asked about Paris; call get_weather.", "Checking the weather.")
}

// TestChatResponsesRoundTrip_StreamedReasoning is the streaming counterpart,
// with the "reasoning" spelling OpenRouter/vLLM stream.
func TestChatResponsesRoundTrip_StreamedReasoning(t *testing.T) {
	stopReason := "tool_calls"
	input := buildSSEChunk(buildDeltaChunk(map[string]interface{}{"role": "assistant", "reasoning": "The user asked about Paris; "})) +
		buildSSEChunk(buildDeltaChunk(map[string]interface{}{"reasoning": "call get_weather."})) +
		buildSSEChunk(buildToolCallStartChunk("call_1", "get_weather")) +
		buildSSEChunk(buildToolCallArgChunk(`{"city":"Paris"}`)) +
		buildSSEChunk(buildChatChunk("", &stopReason)) +
		"data: [DONE]\n\n"

	var resp *Response
	var output bytes.Buffer
	require.NoError(t, TransformChatStreamToResponses(strings.NewReader(input), &output, "deepseek-v4-pro",
		func(r *Response) { resp = r }))
	require.NotNil(t, resp)

	assertAssistantTurnRestored(t, replayAsChat(t, resp.Output),
		"The user asked about Paris; call get_weather.", "")
}
