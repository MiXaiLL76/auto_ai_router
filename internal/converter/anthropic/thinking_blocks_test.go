package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Display "omitted" returns empty signed thinking; Sonnet 5.5 adds progress updates
// between tool calls. Every block must come back with its signature, in order.
const signedThinkingBlocksJSON = `[
	{"type":"thinking","thinking":"","signature":"sig-1"},
	{"type":"redacted_thinking","data":"opaque"},
	{"type":"thinking","thinking":"progress note","signature":"sig-2"}
]`

// The same blocks as a Chat client gets them in thinking_blocks, with their list index.
const chatThinkingBlocksJSON = `[
	{"index":0,"type":"thinking","thinking":"","signature":"sig-1"},
	{"index":1,"type":"redacted_thinking","data":"opaque"},
	{"index":2,"type":"thinking","thinking":"progress note","signature":"sig-2"}
]`

func TestAnthropicToOpenAI_ReturnsSignedThinkingBlocks(t *testing.T) {
	out, err := AnthropicToOpenAI([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[
		{"type":"thinking","thinking":"","signature":"sig-1"},
		{"type":"redacted_thinking","data":"opaque"},
		{"type":"thinking","thinking":"progress note","signature":"sig-2"},
		{"type":"tool_use","id":"toolu_1","name":"f","input":{}}
	],"stop_reason":"tool_use"}`), "claude-sonnet-5-5")
	require.NoError(t, err)

	var resp struct {
		Choices []struct {
			Message struct {
				ThinkingBlocks json.RawMessage `json:"thinking_blocks"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(out, &resp))
	assert.JSONEq(t, chatThinkingBlocksJSON, string(resp.Choices[0].Message.ThinkingBlocks))

	// Echoed back by the client, the blocks reach Anthropic unchanged, without the index.
	request := `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"","thinking_blocks":` +
		string(resp.Choices[0].Message.ThinkingBlocks) + `,"tool_calls":[{"id":"toolu_1","type":"function","function":{"name":"f","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"toolu_1","content":"ok"}]}`
	converted, err := OpenAIToAnthropic([]byte(request), "claude-sonnet-5-5", true)
	require.NoError(t, err)
	var anthropicReq struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(converted, &anthropicReq))
	require.Len(t, anthropicReq.Messages[1].Content, 4)
	replayed, _ := json.Marshal(anthropicReq.Messages[1].Content[:3])
	assert.JSONEq(t, signedThinkingBlocksJSON, string(replayed))
}

// A Chat message holds thinking, text and tool calls in separate fields, so the echo
// replays all thinking blocks first: an interleaved turn comes back reordered. This pins
// the known limitation of the Chat form (the Responses form keeps the order).
func TestAnthropicToOpenAI_InterleavedThinkingEchoesThinkingFirst(t *testing.T) {
	out, err := AnthropicToOpenAI([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[
		{"type":"thinking","thinking":"","signature":"sig-1"},
		{"type":"text","text":"Checking."},
		{"type":"thinking","thinking":"progress note","signature":"sig-2"},
		{"type":"tool_use","id":"toolu_1","name":"f","input":{}}
	],"stop_reason":"tool_use"}`), "claude-sonnet-5-5")
	require.NoError(t, err)
	var resp struct {
		Choices []struct {
			Message map[string]json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(out, &resp))
	message := resp.Choices[0].Message
	request := `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":` + string(message["content"]) +
		`,"thinking_blocks":` + string(message["thinking_blocks"]) + `,"tool_calls":` + string(message["tool_calls"]) + `},` +
		`{"role":"tool","tool_call_id":"toolu_1","content":"ok"}]}`
	converted, err := OpenAIToAnthropic([]byte(request), "claude-sonnet-5-5", true)
	require.NoError(t, err)
	var anthropicReq struct {
		Messages []struct {
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(converted, &anthropicReq))
	var order []string
	for _, block := range anthropicReq.Messages[1].Content {
		order = append(order, block.Type)
	}
	assert.Equal(t, []string{"thinking", "thinking", "text", "tool_use"}, order)
}

// An unsigned thinking block can't be echoed (Anthropic rejects it), so it stays out of
// thinking_blocks; its text still reaches the client as reasoning_content.
func TestAnthropicToOpenAI_UnsignedThinkingNotInThinkingBlocks(t *testing.T) {
	out, err := AnthropicToOpenAI([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[
		{"type":"thinking","thinking":"gateway thought"},
		{"type":"text","text":"hi"}
	],"stop_reason":"end_turn"}`), "claude-sonnet-5-5")
	require.NoError(t, err)
	var resp struct {
		Choices []struct {
			Message map[string]json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(out, &resp))
	assert.NotContains(t, resp.Choices[0].Message, "thinking_blocks")
	assert.JSONEq(t, `"gateway thought"`, string(resp.Choices[0].Message["reasoning_content"]))
}

const signedThinkingStream = `data: {"type":"message_start","message":{"id":"msg_1"}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}

data: {"type":"content_block_stop","index":1}

data: {"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":""}}

data: {"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"progress "}}

data: {"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"note"}}

data: {"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sig-2"}}

data: {"type":"content_block_stop","index":2}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

`

func TestTransformAnthropicStreamToOpenAI_ReturnsSignedThinkingBlocks(t *testing.T) {
	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(signedThinkingStream), "claude-sonnet-5-5", &out))

	var blocks []json.RawMessage
	for _, line := range strings.Split(out.String(), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ThinkingBlocks []json.RawMessage `json:"thinking_blocks"`
				} `json:"delta"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
		if len(chunk.Choices) > 0 {
			require.LessOrEqual(t, len(chunk.Choices[0].Delta.ThinkingBlocks), 1, "one finished block per chunk")
			blocks = append(blocks, chunk.Choices[0].Delta.ThinkingBlocks...)
		}
	}
	// Each entry carries its index: stream accumulators (the OpenAI SDK's among them)
	// merge list entries by it and fail on the second entry without one.
	all, _ := json.Marshal(blocks)
	assert.JSONEq(t, chatThinkingBlocksJSON, string(all))
}

// /v1/messages through the Chat adapter (Bedrock, passthrough off): the Messages
// stream must still carry every signature and the redacted block.
func TestTransformChatStreamToMessages_KeepsSignedThinkingBlocks(t *testing.T) {
	var chat, messages bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(signedThinkingStream), "claude-sonnet-5-5", &chat))
	require.NoError(t, TransformChatStreamToMessages(&chat, &messages, "claude-sonnet-5.5", MessagesAdapterMetadata{}, 0, 0))

	content := messagesStreamContent(t, messages.String())
	all, _ := json.Marshal([]interface{}{content[0], content[1], content[2]})
	assert.JSONEq(t, signedThinkingBlocksJSON, string(all))
	assert.Len(t, content, 3, "no empty text block after the last thinking block")
}

// A Chat producer that sends the thinking text only in thinking_blocks (no
// reasoning_content deltas): the signed block still gets the text its signature covers.
func TestTransformChatStreamToMessages_ThinkingTextFromThinkingBlocks(t *testing.T) {
	chat := `data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"Checking."}}]}` + "\n\n" +
		`data: {"id":"c","choices":[{"index":0,"delta":{"thinking_blocks":[{"type":"thinking","thinking":"plan","signature":"sig-1"}]}}]}` + "\n\n" +
		`data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	var messages bytes.Buffer
	require.NoError(t, TransformChatStreamToMessages(strings.NewReader(chat), &messages, "claude-sonnet-5.5", MessagesAdapterMetadata{}, 0, 0))

	content := messagesStreamContent(t, messages.String())
	all, _ := json.Marshal([]interface{}{content[0], content[1]})
	assert.JSONEq(t, `[{"type":"text","text":"Checking."},{"type":"thinking","thinking":"plan","signature":"sig-1"}]`, string(all))
}

// messagesStreamContent rebuilds the content blocks of a Messages API stream by index.
func messagesStreamContent(t *testing.T, stream string) map[int]map[string]interface{} {
	t.Helper()
	content := map[int]map[string]interface{}{}
	for _, line := range strings.Split(stream, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type         string                 `json:"type"`
			Index        int                    `json:"index"`
			ContentBlock map[string]interface{} `json:"content_block"`
			Delta        map[string]interface{} `json:"delta"`
		}
		require.NoError(t, json.Unmarshal([]byte(payload), &event))
		switch event.Type {
		case "content_block_start":
			content[event.Index] = event.ContentBlock
		case "content_block_delta":
			switch event.Delta["type"] {
			case "text_delta":
				content[event.Index]["text"] = content[event.Index]["text"].(string) + event.Delta["text"].(string)
			case "thinking_delta":
				content[event.Index]["thinking"] = content[event.Index]["thinking"].(string) + event.Delta["thinking"].(string)
			case "signature_delta":
				content[event.Index]["signature"] = event.Delta["signature"]
			}
		}
	}
	return content
}

// ChatToMessages copies the blocks, not the Chat entries: Anthropic rejects the index
// when the client echoes the block back.
func TestChatToMessages_ThinkingBlocksWithoutIndex(t *testing.T) {
	out, err := ChatToMessages([]byte(`{"id":"c","model":"m","choices":[{"finish_reason":"stop","message":{"content":"hi","thinking_blocks":`+
		chatThinkingBlocksJSON+`}}]}`), MessagesAdapterMetadata{}, 0, 0)
	require.NoError(t, err)
	var resp struct {
		Content []json.RawMessage `json:"content"`
	}
	require.NoError(t, json.Unmarshal(out, &resp))
	require.Len(t, resp.Content, 4)
	replayed, _ := json.Marshal(resp.Content[:3])
	assert.JSONEq(t, signedThinkingBlocksJSON, string(replayed))
}

func TestContentBlockMarshalKeepsEmptyThinking(t *testing.T) {
	data, err := json.Marshal([]ContentBlock{
		{Type: "thinking", Signature: "sig"},
		{Type: "redacted_thinking", Data: "opaque"},
		{Type: "text", Text: "hi"},
	})
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"thinking","thinking":"","signature":"sig"},{"type":"redacted_thinking","data":"opaque"},{"type":"text","text":"hi"}]`, string(data))
}

func TestPassthroughMessagesStream_RelaysEventsAndCompletesUsage(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-5-5","content":[],"input_transformations":[{"type":"thinking_dropped","path":"messages.1.content.0","reason":"prefix_binding_mismatch"}],"usage":{"input_tokens":100,"cache_read_input_tokens":10,"output_tokens":1}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40,"iterations":[]}}` + "\n\n"

	var out bytes.Buffer
	require.NoError(t, PassthroughMessagesStream(strings.NewReader(stream), &out, "claude-sonnet-5.5"))

	events := strings.Split(strings.TrimSuffix(out.String(), "\n\n"), "\n\n")
	require.Len(t, events, 5)
	var start struct {
		Message map[string]json.RawMessage `json:"message"`
	}
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(events[0], "event: message_start\ndata: ")), &start))
	assert.JSONEq(t, `"claude-sonnet-5.5"`, string(start.Message["model"]))
	assert.JSONEq(t, `[{"type":"thinking_dropped","path":"messages.1.content.0","reason":"prefix_binding_mismatch"}]`, string(start.Message["input_transformations"]))

	// Blocks and deltas pass byte for byte.
	assert.Equal(t, strings.Split(stream, "\n\n")[1:4], events[1:4])

	assert.JSONEq(t, `{"output_tokens":40,"iterations":[],"input_tokens":100,"cache_read_input_tokens":10,"server_tool_use":{"web_search_requests":1}}`,
		passthroughDeltaUsage(t, events[4]), "message_start's counts carried over; searches left out of usage counted from the blocks")
}

// The final usage keeps the provider's objects as sent: counters the router doesn't
// model survive (also when the search count is filled in from the blocks), zero
// objects from message_start aren't re-encoded into {}, and the cache-write total is
// filled in from its TTL split only when missing.
func TestPassthroughMessagesStream_KeepsProviderUsageObjects(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":5,"output_tokens":1,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":50}}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40,"server_tool_use":{"web_search_requests":0,"web_fetch_requests":1}}}` + "\n\n"

	var out bytes.Buffer
	require.NoError(t, PassthroughMessagesStream(strings.NewReader(stream), &out, "pub"))
	events := strings.Split(strings.TrimSuffix(out.String(), "\n\n"), "\n\n")
	require.Len(t, events, 3)
	assert.JSONEq(t, `{"output_tokens":40,"input_tokens":5,"server_tool_use":{"web_search_requests":1,"web_fetch_requests":1},
		"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":50},"cache_creation_input_tokens":50}`,
		passthroughDeltaUsage(t, events[2]))

	// A null usage is not a crash.
	out.Reset()
	require.NoError(t, PassthroughMessagesStream(strings.NewReader("event: message_delta\n"+
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":null}`+"\n\n"), &out, "pub"))
	assert.Contains(t, out.String(), `"usage":null`)
}

func passthroughDeltaUsage(t *testing.T, event string) string {
	t.Helper()
	var delta struct {
		Usage json.RawMessage `json:"usage"`
	}
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(event, "event: message_delta\ndata: ")), &delta))
	return string(delta.Usage)
}
