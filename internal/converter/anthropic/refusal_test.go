package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropicRefusalChat(t *testing.T) {
	for _, tc := range []struct {
		name, reason, details, wantRefusal, wantFinish string
	}{
		{"explained", "refusal", `{"type":"refusal","category":"reasoning_extraction","explanation":"Request declined.","fallback_credit_token":"private-fixture-credit"}`, "The safety classifier identified a request to reproduce internal reasoning. (category: reasoning_extraction)", "content_filter"},
		{"no details", "refusal", `null`, "The model declined this request; no public reason was provided.", "content_filter"},
		{"null fields", "refusal", `{"category":null,"explanation":null}`, "The model declined this request; no public reason was provided.", "content_filter"},
		{"category only", "refusal", `{"category":"general_harms"}`, "The safety classifier identified a potential usage-policy violation. (category: general_harms)", "content_filter"},
		{"explanation only", "refusal", `{"explanation":"Request declined."}`, "The model declined this request; no public reason was provided.", "content_filter"},
		{"ordinary empty completion", "end_turn", `null`, "", "stop"},
		{"details without refusal", "end_turn", `{"category":"general_harms","explanation":"Ignored"}`, "", "stop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Sprintf(`{"id":"msg_fixture","content":[],"stop_reason":%q,"stop_details":%s,"usage":{"input_tokens":17,"output_tokens":0,"cache_read_input_tokens":3}}`, tc.reason, tc.details)
			converted, err := AnthropicToOpenAI([]byte(body), "claude-test")
			require.NoError(t, err)
			var response openai.OpenAIResponse
			require.NoError(t, json.Unmarshal(converted, &response))
			require.Len(t, response.Choices, 1)
			assert.Equal(t, tc.wantFinish, response.Choices[0].FinishReason)
			assert.Equal(t, tc.wantRefusal, response.Choices[0].Message.Refusal)
			assert.Empty(t, response.Choices[0].Message.Content)
			assert.Equal(t, 20, response.Usage.PromptTokens)
			assert.Zero(t, response.Usage.CompletionTokens)
			assert.NotContains(t, string(converted), "private-fixture-credit")

			stream := strings.Join([]string{
				`data: {"type":"message_start","message":{"id":"msg_fixture","usage":{"input_tokens":17,"cache_read_input_tokens":3}}}`,
				fmt.Sprintf(`data: {"type":"message_delta","delta":{"stop_reason":%q,"stop_details":%s},"usage":{"output_tokens":0}}`, tc.reason, tc.details),
				`data: {"type":"message_stop"}`,
			}, "\n\n")
			var out bytes.Buffer
			require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(stream), "claude-test", &out))
			chunks := refusalChatChunks(t, out.String())
			require.GreaterOrEqual(t, len(chunks), 2)
			var refusal string
			for _, chunk := range chunks {
				for _, choice := range chunk.Choices {
					refusal += choice.Delta.Refusal
				}
			}
			assert.Equal(t, tc.wantRefusal, refusal)
			finish := chunks[len(chunks)-2]
			require.Len(t, finish.Choices, 1)
			require.NotNil(t, finish.Choices[0].FinishReason)
			assert.Equal(t, tc.wantFinish, *finish.Choices[0].FinishReason)
			usage := chunks[len(chunks)-1]
			assert.Empty(t, usage.Choices)
			assert.Equal(t, response.Usage, usage.Usage)
			assert.True(t, strings.HasSuffix(out.String(), "data: [DONE]\n\n"))
			assert.NotContains(t, out.String(), "private-fixture-credit")
		})
	}
}

func TestAnthropicRefusalAfterPartialText(t *testing.T) {
	body := []byte(`{"content":[{"type":"text","text":"Partial answer"}],"stop_reason":"refusal","stop_details":{"explanation":"Stopped by a classifier."}}`)
	converted, err := AnthropicToOpenAI(body, "claude-test")
	require.NoError(t, err)
	var response openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &response))
	assert.Equal(t, "Partial answer", response.Choices[0].Message.Content)
	assert.Equal(t, "The model declined this request; no public reason was provided.", response.Choices[0].Message.Refusal)
	assert.Equal(t, "content_filter", response.Choices[0].FinishReason)

	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_fixture"}}`,
		`data: {"type":"content_block_start","content_block":{"type":"text"}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Partial answer"}}`,
		`data: {"type":"content_block_stop"}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"Stopped by a classifier."}}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n")
	var out bytes.Buffer
	require.NoError(t, TransformAnthropicStreamToOpenAI(strings.NewReader(stream), "claude-test", &out))
	var content, refusal string
	chunks := refusalChatChunks(t, out.String())
	for _, chunk := range chunks {
		for _, choice := range chunk.Choices {
			content += choice.Delta.Content
			refusal += choice.Delta.Refusal
		}
	}
	assert.Equal(t, response.Choices[0].Message.Content, content)
	assert.Equal(t, response.Choices[0].Message.Refusal, refusal)
	assert.Equal(t, "content_filter", *chunks[len(chunks)-1].Choices[0].FinishReason)
}

func refusalChatChunks(t *testing.T, stream string) []openai.OpenAIStreamingChunk {
	t.Helper()
	var chunks []openai.OpenAIStreamingChunk
	for _, line := range strings.Split(stream, "\n") {
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk))
		chunks = append(chunks, chunk)
	}
	return chunks
}
