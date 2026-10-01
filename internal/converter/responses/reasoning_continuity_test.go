package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReasoningContinuity_RoundTrip covers the gap this feature must not
// have: a reasoning model doing multi-turn tool calling in the
// stateless/store:false mode ChatRequestToResponses always uses. Without
// preserving the reasoning item's id/encrypted_content across the Chat
// Completions history round trip, the model loses the chain of thought that
// produced its function_call on the next turn -- see
// ResponseToChat/injectThinkingBlocks and
// chatAssistantMessageToInputItems/chatThinkingBlocksToReasoningItems.
func TestReasoningContinuity_RoundTrip(t *testing.T) {
	// Turn 1: a reasoning model's response includes both a reasoning item
	// and the function_call it informed.
	responsesBody := `{
		"id":"resp_1","object":"response","created_at":1700000000,"model":"o3-pro","status":"completed",
		"output":[
			{"type":"reasoning","id":"rs_abc","encrypted_content":"opaque-blob-xyz","summary":[{"type":"summary_text","text":"need the weather"}]},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"paris\"}"}
		]
	}`

	chatBody, err := ResponseToChat([]byte(responsesBody))
	require.NoError(t, err)

	var chatResp map[string]interface{}
	require.NoError(t, json.Unmarshal(chatBody, &chatResp))
	message := chatResp["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})

	thinkingBlocks, ok := message["thinking_blocks"].([]interface{})
	require.True(t, ok, "reasoning item must survive as thinking_blocks on the assistant message")
	require.Len(t, thinkingBlocks, 1)
	block := thinkingBlocks[0].(map[string]interface{})
	assert.Equal(t, "responses_reasoning", block["type"])
	assert.Equal(t, "rs_abc", block["id"])
	assert.Equal(t, "opaque-blob-xyz", block["encrypted_content"])

	toolCalls := message["tool_calls"].([]interface{})
	require.Len(t, toolCalls, 1)
	assert.Equal(t, "call_1", toolCalls[0].(map[string]interface{})["id"])

	// Turn 2: the client builds its next /v1/chat/completions request the
	// normal way -- assistant message exactly as received (thinking_blocks
	// included, since a well-behaved client preserves unknown fields), plus
	// the tool result.
	nextRequest := map[string]interface{}{
		"model":      "o3-pro",
		"max_tokens": 100,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "what's the weather in paris?"},
			message,
			map[string]interface{}{"role": "tool", "tool_call_id": "call_1", "content": "18C, cloudy"},
		},
	}
	nextRequestBody, err := json.Marshal(nextRequest)
	require.NoError(t, err)

	responsesRequest, err := ChatRequestToResponses(nextRequestBody)
	require.NoError(t, err)

	var reqRaw map[string]interface{}
	require.NoError(t, json.Unmarshal(responsesRequest, &reqRaw))
	input := reqRaw["input"].([]interface{})

	// The reasoning item must be reconstructed and precede the function_call
	// it informed, exactly as OpenAI's own multi-turn contract requires in
	// store:false mode.
	require.GreaterOrEqual(t, len(input), 3)
	reasoningItem := input[1].(map[string]interface{})
	assert.Equal(t, "reasoning", reasoningItem["type"])
	assert.Equal(t, "rs_abc", reasoningItem["id"])
	assert.Equal(t, "opaque-blob-xyz", reasoningItem["encrypted_content"])

	fc := input[2].(map[string]interface{})
	assert.Equal(t, "function_call", fc["type"])
	assert.Equal(t, "call_1", fc["call_id"])

	fco := input[3].(map[string]interface{})
	assert.Equal(t, "function_call_output", fco["type"])
	assert.Equal(t, "call_1", fco["call_id"])
	assert.Equal(t, "18C, cloudy", fco["output"])
}

// TestReasoningContinuity_RoundTrip_NoSummary is the actual bug scenario, distinct from
// TestReasoningContinuity_RoundTrip above: reasoning.summary is opt-in on OpenAI's side
// (this feature never requests it -- chatForceStatelessResponses only forces
// store:false + reasoning.encrypted_content), so the common case is a reasoning item
// with encrypted_content but no summary at all, not the summary-present case the other
// test covers. Before the fix, "summary" was omitempty on the client-facing
// thinking_blocks field, so turn 1's assistant message dropped the key entirely; a
// well-behaved client echoing that message back verbatim on turn 2 then had nothing for
// chatThinkingBlocksToReasoningItems to copy, and the reconstructed reasoning item
// reached OpenAI with no "summary" key -- rejected with "Invalid 'summary': summary is
// required and must be a list for reasoning".
func TestReasoningContinuity_RoundTrip_NoSummary(t *testing.T) {
	responsesBody := `{
		"id":"resp_1","object":"response","created_at":1700000000,"model":"o3-pro","status":"completed",
		"output":[
			{"type":"reasoning","id":"rs_abc","encrypted_content":"opaque-blob-xyz"},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"paris\"}"}
		]
	}`

	chatBody, err := ResponseToChat([]byte(responsesBody))
	require.NoError(t, err)

	// The client-facing JSON must carry "summary":[] -- present and an empty list, not
	// omitted and not null (a nil Go slice would marshal to "null", which still fails
	// OpenAI's "must be a list" check on the way back out).
	assert.Contains(t, string(chatBody), `"summary":[]`,
		"thinking_blocks must carry an explicit empty summary list, not omit the key or marshal it as null")

	var chatResp map[string]interface{}
	require.NoError(t, json.Unmarshal(chatBody, &chatResp))
	message := chatResp["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
	thinkingBlocks := message["thinking_blocks"].([]interface{})
	require.Len(t, thinkingBlocks, 1)
	block := thinkingBlocks[0].(map[string]interface{})
	summary, ok := block["summary"]
	require.True(t, ok, `"summary" key must be present even when empty`)
	assert.Equal(t, []interface{}{}, summary)

	// Turn 2: client echoes the message back verbatim, as a well-behaved client does.
	nextRequest := map[string]interface{}{
		"model":      "o3-pro",
		"max_tokens": 100,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "what's the weather in paris?"},
			message,
			map[string]interface{}{"role": "tool", "tool_call_id": "call_1", "content": "18C, cloudy"},
		},
	}
	nextRequestBody, err := json.Marshal(nextRequest)
	require.NoError(t, err)

	responsesRequest, err := ChatRequestToResponses(nextRequestBody)
	require.NoError(t, err)

	var reqRaw map[string]interface{}
	require.NoError(t, json.Unmarshal(responsesRequest, &reqRaw))
	input := reqRaw["input"].([]interface{})
	require.GreaterOrEqual(t, len(input), 2)
	reasoningItem := input[1].(map[string]interface{})
	assert.Equal(t, "reasoning", reasoningItem["type"])
	assert.Equal(t, "rs_abc", reasoningItem["id"])
	assert.Equal(t, "opaque-blob-xyz", reasoningItem["encrypted_content"])
	summaryOut, ok := reasoningItem["summary"]
	require.True(t, ok, `the reconstructed reasoning item sent to OpenAI must carry a "summary" key`)
	assert.Equal(t, []interface{}{}, summaryOut,
		`must be an empty list, not missing or null -- OpenAI rejects anything else with "summary is required and must be a list for reasoning"`)
}

// TestReasoningContinuity_UnrecognizedModelBareIDDoesNotReplay covers the 404
// this feature used to produce for any model outside openai.IsReasoningModel's
// known families (e.g. an Azure deployment named "prod-reasoner", or a custom
// model alias): chatRequestWantsReasoning never requested the
// "reasoning.encrypted_content" include for turn 1 (since the model name
// didn't match and the client never asked for reasoning either), so the
// provider's reasoning item in that response carries only an id -- no
// encrypted_content. chatForceStatelessResponses always pins store:false, so
// a bare id is never valid to replay: OpenAI's error is explicit ("Item with
// id '...' not found. Items are not persisted when store is set to false").
// The fix drops such a block instead of forwarding it.
func TestReasoningContinuity_UnrecognizedModelBareIDDoesNotReplay(t *testing.T) {
	responsesBody := `{
		"id":"resp_1","object":"response","created_at":1700000000,"model":"prod-reasoner","status":"completed",
		"output":[
			{"type":"reasoning","id":"rs_abc","summary":[]},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"paris\"}"}
		]
	}`

	chatBody, err := ResponseToChat([]byte(responsesBody))
	require.NoError(t, err)

	var chatResp map[string]interface{}
	require.NoError(t, json.Unmarshal(chatBody, &chatResp))
	message := chatResp["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})

	// ResponseToChat still surfaces the bare-id block on the client-facing
	// message (unchanged) -- the fix is entirely about not replaying it.
	thinkingBlocks, ok := message["thinking_blocks"].([]interface{})
	require.True(t, ok)
	require.Len(t, thinkingBlocks, 1)
	block := thinkingBlocks[0].(map[string]interface{})
	assert.Equal(t, "rs_abc", block["id"])
	assert.Empty(t, block["encrypted_content"])

	// Turn 2: client echoes the message back verbatim, as a well-behaved
	// client does.
	nextRequest := map[string]interface{}{
		"model":      "prod-reasoner",
		"max_tokens": 100,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "what's the weather in paris?"},
			message,
			map[string]interface{}{"role": "tool", "tool_call_id": "call_1", "content": "18C, cloudy"},
		},
	}
	nextRequestBody, err := json.Marshal(nextRequest)
	require.NoError(t, err)

	responsesRequest, err := ChatRequestToResponses(nextRequestBody)
	require.NoError(t, err)

	var reqRaw map[string]interface{}
	require.NoError(t, json.Unmarshal(responsesRequest, &reqRaw))
	for _, raw := range reqRaw["input"].([]interface{}) {
		item, ok := raw.(map[string]interface{})
		require.True(t, ok)
		assert.NotEqual(t, "reasoning", item["type"],
			"a bare-id reasoning item (no encrypted_content) must never be replayed under store:false — OpenAI 404s it")
	}
}

// TestChatThinkingBlocksToReasoningItems_MissingSummaryDefaultsToEmptyList covers the
// belt-and-suspenders side of the fix directly: even if a thinking_blocks entry has no
// "summary" key at all (an older client from before this fix, or any client/proxy that
// strips an empty-array field), the reconstructed reasoning item must still get
// "summary":[] rather than reproducing the missing-key 400.
func TestChatThinkingBlocksToReasoningItems_MissingSummaryDefaultsToEmptyList(t *testing.T) {
	blocks := []interface{}{
		map[string]interface{}{
			"type":              responsesReasoningBlockType,
			"id":                "rs_1",
			"encrypted_content": "blob",
			// no "summary" key at all
		},
	}
	items := chatThinkingBlocksToReasoningItems(blocks)
	require.Len(t, items, 1)
	item := items[0].(map[string]interface{})
	summary, ok := item["summary"]
	require.True(t, ok)
	assert.Equal(t, []interface{}{}, summary)
}
