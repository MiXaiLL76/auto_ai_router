package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPassthroughMessagesRefusalPrivacy(t *testing.T) {
	for _, padding := range []int{0, 1024*1024 + 1} {
		for _, multiline := range []bool{false, true} {
			t.Run(fmt.Sprintf("padding=%d/multiline=%t", padding, multiline), func(t *testing.T) {
				payload := `{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"reasoning_extraction","explanation":"fixture-secret` + strings.Repeat("x", padding) + `","fallback_credit_token":"private-credit"}},"usage":{"output_tokens":0}}`
				if multiline {
					payload = strings.Replace(payload, `,"delta":`, ",\ndata: \"delta\":", 1)
				}
				var out bytes.Buffer
				require.NoError(t, PassthroughMessagesStream(strings.NewReader("event: message_delta\ndata: "+payload+"\n\n"), &out, "claude"))
				assert.Contains(t, out.String(), `"stop_reason":"refusal"`)
				assert.Contains(t, out.String(), "reasoning_extraction")
				assert.Contains(t, out.String(), "identified a request to reproduce internal reasoning")
				assert.NotContains(t, out.String(), "fixture-secret")
				assert.NotContains(t, out.String(), "private-credit")
			})
		}
	}
}

func TestPassthroughMessagesMultilineContentStillValid(t *testing.T) {
	stream := "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\n" +
		"data: \"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n"
	var out bytes.Buffer
	require.NoError(t, PassthroughMessagesStream(strings.NewReader(stream), &out, "claude"))
	var data []string
	for _, line := range strings.Split(out.String(), "\n") {
		if payload, ok := messagesDataLine([]byte(line)); ok {
			data = append(data, string(payload))
		}
	}
	require.Len(t, data, 1)
	assert.True(t, json.Valid([]byte(data[0])))
	assert.Contains(t, data[0], `"text":"hello"`)
}

func TestPassthroughMessagesRefusalFailsClosed(t *testing.T) {
	for _, payload := range []string{
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"fixture-secret"`,
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"fixture-secret` + strings.Repeat("x", 4*1024*1024) + `"}}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"broken"}`,
		`{"type":"content_block_delta"} {"stop_details":{"explanation":"fixture-secret"}}`,
		`[{"type":"content_block_delta"}]`,
	} {
		var out bytes.Buffer
		err := PassthroughMessagesStream(strings.NewReader("data: "+payload+"\n\n"), &out, "claude")
		require.Error(t, err)
		assert.NotContains(t, out.String(), "fixture-secret")
		assert.NotContains(t, err.Error(), "fixture-secret")
	}
}

func TestPassthroughMessagesOrdinaryEventPreserved(t *testing.T) {
	for _, payload := range []string{
		`{ "type": "content_block_delta", "index":0, "delta":{"type":"text_delta","text":"quote: \" and newline\n"} }`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"stop_details\":\"user-data\"}"}}`,
		`{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"do not rewrite model content"}}`,
		`{"type":"ping"}`,
	} {
		stream := ":keepalive\nevent: fixture\nid: 17\ndata: " + payload + "\n\n"
		var out bytes.Buffer
		require.NoError(t, PassthroughMessagesStream(strings.NewReader(stream), &out, "claude"))
		assert.Equal(t, stream, out.String())
	}
}

func TestPassthroughMessagesEscapedEnvelopeKeys(t *testing.T) {
	stream := "event: message_start\ndata: " + `{"t\u0079pe":"message_\u0073tart","message":{"model":"private-model","usage":{"input_tokens":9007199254740993}}}` + "\n\n" +
		"event: message_delta\ndata: " + `{"type":"message_delta","d\u0065lta":{"stop_\u0072eason":"refusal","\u0073top_details":{"category":"cyber","explanation":"fixture-secret"}},"usage":{"output_tokens":0}}` + "\n\n"
	var out bytes.Buffer
	require.NoError(t, PassthroughMessagesStream(strings.NewReader(stream), &out, "claude-public"))
	assert.NotContains(t, out.String(), "private-model")
	assert.NotContains(t, out.String(), "fixture-secret")
	assert.Contains(t, out.String(), "cybersecurity")
	assert.Contains(t, out.String(), `"input_tokens":9007199254740993`)
	assert.Contains(t, out.String(), `"model":"claude-public"`)
}
