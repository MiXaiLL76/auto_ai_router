package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponseToChat_BasicText(t *testing.T) {
	body := `{
		"id":"resp_123","object":"response","created_at":1700000000,"model":"gpt-5-pro",
		"status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant",
			"content":[{"type":"output_text","text":"Hello there","annotations":[]}]}],
		"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,
			"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}
	}`

	out, err := ResponseToChat([]byte(body))
	require.NoError(t, err)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &resp))

	assert.Equal(t, "chat.completion", resp["object"])
	assert.Equal(t, "gpt-5-pro", resp["model"])
	assert.Equal(t, "chatcmpl-resp_123", resp["id"])

	choices := resp["choices"].([]interface{})
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]interface{})
	assert.Equal(t, "stop", choice["finish_reason"])
	message := choice["message"].(map[string]interface{})
	assert.Equal(t, "assistant", message["role"])
	assert.Equal(t, "Hello there", message["content"])

	usage := resp["usage"].(map[string]interface{})
	assert.Equal(t, float64(10), usage["prompt_tokens"])
	assert.Equal(t, float64(4), usage["completion_tokens"])
	assert.Equal(t, float64(14), usage["total_tokens"])
}

func TestResponseToChat_FunctionCall(t *testing.T) {
	body := `{
		"id":"resp_456","object":"response","created_at":1700000000,"model":"gpt-5-pro",
		"status":"completed",
		"output":[
			{"type":"function_call","id":"fc_1","call_id":"call_abc123","name":"get_weather","arguments":"{\"city\":\"paris\"}","status":"completed"}
		]
	}`

	out, err := ResponseToChat([]byte(body))
	require.NoError(t, err)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &resp))

	choice := resp["choices"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "tool_calls", choice["finish_reason"])

	message := choice["message"].(map[string]interface{})
	toolCalls := message["tool_calls"].([]interface{})
	require.Len(t, toolCalls, 1)
	tc := toolCalls[0].(map[string]interface{})
	// The chat tool_call "id" must be the Responses item's call_id, not its
	// own item id -- the client echoes this exact value back as
	// tool_call_id, and chatMessagesToInput reads it back out as call_id.
	assert.Equal(t, "call_abc123", tc["id"])
	assert.Equal(t, "function", tc["type"])
	fn := tc["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
	assert.Equal(t, `{"city":"paris"}`, fn["arguments"])
}

func TestResponseToChat_IncompleteMaxOutputTokens(t *testing.T) {
	body := `{
		"id":"resp_789","object":"response","created_at":1700000000,"model":"gpt-5-pro",
		"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
		"output":[{"type":"message","id":"msg_1","status":"incomplete","role":"assistant",
			"content":[{"type":"output_text","text":"partial","annotations":[]}]}]
	}`

	out, err := ResponseToChat([]byte(body))
	require.NoError(t, err)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &resp))
	choice := resp["choices"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "length", choice["finish_reason"])
	message := choice["message"].(map[string]interface{})
	assert.Equal(t, "partial", message["content"])
}

func TestResponseToChat_ReasoningSummary(t *testing.T) {
	body := `{
		"id":"resp_r1","object":"response","created_at":1700000000,"model":"o3-pro",
		"status":"completed",
		"output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking it through"}]},
			{"type":"message","id":"msg_1","status":"completed","role":"assistant",
				"content":[{"type":"output_text","text":"42","annotations":[]}]}
		]
	}`

	out, err := ResponseToChat([]byte(body))
	require.NoError(t, err)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &resp))
	message := resp["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
	assert.Equal(t, "42", message["content"])
	assert.Equal(t, "thinking it through", message["reasoning_content"])
}

func TestResponseToChat_FailedWithError(t *testing.T) {
	body := `{
		"id":"resp_e1","object":"response","created_at":1700000000,"model":"gpt-5-pro",
		"status":"failed","error":{"type":"server_error","message":"something broke"},
		"output":[]
	}`

	out, err := ResponseToChat([]byte(body))
	require.NoError(t, err)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &resp))
	message := resp["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
	assert.Equal(t, "something broke", message["content"])
}
