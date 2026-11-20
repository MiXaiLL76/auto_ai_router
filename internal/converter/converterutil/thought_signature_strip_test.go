package converterutil

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStripThoughtSignaturesFromToolIDs(t *testing.T) {
	signed := EncodeToolCallIDWithSignature("call_1", []byte("signature"))
	body := []byte(`{
		"messages": [
			{"role":"assistant","tool_calls":[{"id":"` + signed + `","type":"function","function":{"name":"lookup"},"provider_specific_fields":{"thought_signature":"c2ln","other":"keep"}}]},
			{"role":"tool","tool_call_id":"` + signed + `","content":"ok"}
		],
		"input": [
			{"type":"function_call","id":"item_1","call_id":"` + signed + `","name":"lookup"},
			{"type":"function_call_output","call_id":"` + signed + `","output":"ok"}
		],
		"content": [
			{"type":"tool_use","id":"` + signed + `","name":"lookup"},
			{"type":"tool_result","tool_use_id":"` + signed + `","content":"ok"}
		],
		"max_tokens": 9007199254740993,
		"unrelated_id": "` + signed + `"
	}`)

	cleaned := StripThoughtSignaturesFromToolIDs(body)
	assert.NotContains(t, string(cleaned), `"max_tokens":9007199254740992`)
	assert.Contains(t, string(cleaned), `9007199254740993`)
	var got map[string]any
	require.NoError(t, json.Unmarshal(cleaned, &got))
	message := got["messages"].([]any)[0].(map[string]any)
	call := message["tool_calls"].([]any)[0].(map[string]any)
	assert.Equal(t, "call_1", call["id"])
	assert.Equal(t, map[string]any{"other": "keep"}, call["provider_specific_fields"])
	assert.Equal(t, "call_1", got["messages"].([]any)[1].(map[string]any)["tool_call_id"])
	assert.Equal(t, "call_1", got["input"].([]any)[0].(map[string]any)["call_id"])
	assert.Equal(t, "item_1", got["input"].([]any)[0].(map[string]any)["id"])
	assert.Equal(t, "call_1", got["input"].([]any)[1].(map[string]any)["call_id"])
	assert.Equal(t, "call_1", got["content"].([]any)[0].(map[string]any)["id"])
	assert.Equal(t, "call_1", got["content"].([]any)[1].(map[string]any)["tool_use_id"])
	assert.Equal(t, signed, got["unrelated_id"])
}

func TestStripThoughtSignaturesFromToolIDsInvalidSuffix(t *testing.T) {
	body := []byte(`{"messages":[{"role":"tool","tool_call_id":"call_1__thought__broken!"}]}`)
	assert.JSONEq(t, `{"messages":[{"role":"tool","tool_call_id":"call_1"}]}`, string(StripThoughtSignaturesFromToolIDs(body)))

	plain := []byte(`{ "messages": [], "max_tokens": 42 }`)
	assert.Equal(t, plain, StripThoughtSignaturesFromToolIDs(plain))
}
