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

// The blocks a Sonnet 5.5 tool turn returns under display "omitted": empty signed
// thinking, a redacted block, and a progress note, followed by text and a tool call.
var signedThinkingTurn = []map[string]interface{}{
	{"type": "content_block_start", "content_block": map[string]interface{}{"type": "thinking", "thinking": "", "signature": ""}},
	{"type": "content_block_delta", "delta": map[string]interface{}{"type": "signature_delta", "signature": "sig-1"}},
	{"type": "content_block_stop"},
	{"type": "content_block_start", "content_block": map[string]interface{}{"type": "redacted_thinking", "data": "opaque"}},
	{"type": "content_block_stop"},
	{"type": "content_block_start", "content_block": map[string]interface{}{"type": "text", "text": ""}},
	{"type": "content_block_delta", "delta": map[string]interface{}{"type": "text_delta", "text": "Checking."}},
	{"type": "content_block_stop"},
	{"type": "content_block_start", "content_block": map[string]interface{}{"type": "thinking", "thinking": "", "signature": ""}},
	{"type": "content_block_delta", "delta": map[string]interface{}{"type": "thinking_delta", "thinking": "progress note"}},
	{"type": "content_block_delta", "delta": map[string]interface{}{"type": "signature_delta", "signature": "sig-2"}},
	{"type": "content_block_stop"},
	{"type": "content_block_start", "content_block": map[string]interface{}{"type": "tool_use", "id": "toolu_1", "name": "f"}},
	{"type": "content_block_stop"},
	{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": "tool_use"}},
}

// replayedBlocks converts output back into an Anthropic request and returns the
// assistant turn's blocks as Anthropic would receive them.
func replayedBlocks(t *testing.T, output []responses.OutputItem) string {
	t.Helper()
	input, err := json.Marshal(output)
	require.NoError(t, err)
	var items []interface{}
	require.NoError(t, json.Unmarshal(input, &items))
	messages, err := inputToAnthropicMessages(items)
	require.NoError(t, err)
	var blocks []interface{}
	for _, message := range messages {
		raw, err := json.Marshal(message.Content)
		require.NoError(t, err)
		var content []interface{}
		require.NoError(t, json.Unmarshal(raw, &content))
		blocks = append(blocks, content...)
	}
	data, err := json.Marshal(blocks)
	require.NoError(t, err)
	return string(data)
}

const replayedTurnJSON = `[
	{"type":"thinking","thinking":"","signature":"sig-1"},
	{"type":"redacted_thinking","data":"opaque"},
	{"type":"text","text":"Checking."},
	{"type":"thinking","thinking":"progress note","signature":"sig-2"},
	{"type":"tool_use","id":"toolu_1","name":"f","input":{}}
]`

func TestTransformAnthropicStreamToResponses_KeepsSignedThinkingInOrder(t *testing.T) {
	var out bytes.Buffer
	var completed *responses.Response
	require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(buildAnthropicSSEStream(signedThinkingTurn)), &out,
		"claude-sonnet-5-5", "", nil, func(r *responses.Response) { completed = r }))
	require.NotNil(t, completed)

	var types []string
	for _, item := range completed.Output {
		types = append(types, item.Type)
	}
	assert.Equal(t, []string{"reasoning", "reasoning", "message", "reasoning", "function_call"}, types)
	assert.Equal(t, "sig-1", completed.Output[0].EncryptedContent)
	assert.Equal(t, "redacted_thinking:opaque", completed.Output[1].EncryptedContent)
	assert.Equal(t, "sig-2", completed.Output[3].EncryptedContent)

	// Each reasoning item's done event carries its encrypted_content as well.
	var doneEncrypted []string
	for _, event := range parseSSEEvents(out.String()) {
		if item, ok := event["item"].(map[string]interface{}); ok && event["type"] == "response.output_item.done" && item["type"] == "reasoning" {
			encrypted, _ := item["encrypted_content"].(string)
			doneEncrypted = append(doneEncrypted, encrypted)
			assert.IsType(t, []interface{}{}, item["summary"], "strict clients need the summary list")
		}
	}
	assert.Equal(t, []string{"sig-1", "redacted_thinking:opaque", "sig-2"}, doneEncrypted)
	assertOutputIndexes(t, out.String(), completed.Output)

	assert.JSONEq(t, replayedTurnJSON, replayedBlocks(t, completed.Output))
}

// Text after a later thinking block starts a new message, as for a whole response, so
// the replay keeps the progress note between the two texts.
func TestTransformAnthropicStreamToResponses_TextAfterLaterThinkingIsNewMessage(t *testing.T) {
	var turn []map[string]interface{}
	turn = append(turn, signedThinkingTurn[:3]...)   // thinking sig-1
	turn = append(turn, signedThinkingTurn[5:12]...) // text "Checking.", thinking sig-2
	turn = append(turn,
		map[string]interface{}{"type": "content_block_start", "content_block": map[string]interface{}{"type": "text", "text": ""}},
		map[string]interface{}{"type": "content_block_delta", "delta": map[string]interface{}{"type": "text_delta", "text": "Found it."}},
		map[string]interface{}{"type": "content_block_stop"},
	)
	turn = append(turn, signedThinkingTurn[12:]...) // tool_use, message_delta

	var out bytes.Buffer
	var completed *responses.Response
	require.NoError(t, TransformAnthropicStreamToResponses(strings.NewReader(buildAnthropicSSEStream(turn)), &out,
		"claude-sonnet-5-5", "", nil, func(r *responses.Response) { completed = r }))
	require.NotNil(t, completed)

	var types []string
	for _, item := range completed.Output {
		types = append(types, item.Type)
	}
	assert.Equal(t, []string{"reasoning", "message", "reasoning", "message", "function_call"}, types)
	assertOutputIndexes(t, out.String(), completed.Output)

	const replayed = `[
		{"type":"thinking","thinking":"","signature":"sig-1"},
		{"type":"text","text":"Checking."},
		{"type":"thinking","thinking":"progress note","signature":"sig-2"},
		{"type":"text","text":"Found it."},
		{"type":"tool_use","id":"toolu_1","name":"f","input":{}}
	]`
	assert.JSONEq(t, replayed, replayedBlocks(t, completed.Output))

	var textDone []string
	for _, event := range parseSSEEvents(out.String()) {
		if event["type"] == "response.output_text.done" {
			textDone = append(textDone, event["text"].(string))
		}
	}
	assert.Equal(t, []string{"Checking.", "Found it."}, textDone, "each message closes its own text")
}

// assertOutputIndexes checks that every event of an item uses one output_index, the
// item's position in the completed output.
func assertOutputIndexes(t *testing.T, stream string, output []responses.OutputItem) {
	t.Helper()
	indexes := map[string]map[int]bool{}
	for _, event := range parseSSEEvents(stream) {
		index, ok := event["output_index"].(float64)
		if !ok {
			continue
		}
		id, _ := event["item_id"].(string)
		if item, ok := event["item"].(map[string]interface{}); ok {
			id, _ = item["id"].(string)
		}
		if indexes[id] == nil {
			indexes[id] = map[int]bool{}
		}
		indexes[id][int(index)] = true
	}
	for position, item := range output {
		assert.Equal(t, map[int]bool{position: true}, indexes[item.ID], "output_index of %s %s", item.Type, item.ID)
	}
}

func TestAnthropicToResponses_KeepsSignedThinkingInOrder(t *testing.T) {
	resp, err := (&AnthropicResponses{}).ResponseTo([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[
		{"type":"thinking","thinking":"","signature":"sig-1"},
		{"type":"redacted_thinking","data":"opaque"},
		{"type":"text","text":"Checking."},
		{"type":"thinking","thinking":"progress note","signature":"sig-2"},
		{"type":"tool_use","id":"toolu_1","name":"f","input":{}}
	],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`), "claude-sonnet-5-5")
	require.NoError(t, err)
	assert.JSONEq(t, replayedTurnJSON, replayedBlocks(t, resp.Output))
}
