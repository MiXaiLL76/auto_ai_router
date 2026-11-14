package proxy

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func aggregateSSE(t *testing.T, sse string) *chatStreamAggregator {
	t.Helper()
	a := newChatStreamAggregator()
	require.NoError(t, a.consume(iotest.HalfReader(strings.NewReader(sse)), 0))
	return a
}

func aggregatedJSON(t *testing.T, sse string) string {
	t.Helper()
	body, err := aggregateSSE(t, sse).completion()
	require.NoError(t, err)
	return string(body)
}

func TestChatStreamAggregator_TextReasoningUsage(t *testing.T) {
	sse := `data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[{"index":0,"delta":{"reasoning_content":"think "},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[{"index":0,"delta":{"reasoning_content":"more"},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[{"index":0,"delta":{"content":"Hello, "},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop","stop_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":7,"model":"qwen","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}

data: [DONE]

`
	assert.JSONEq(t, `{
		"id":"c1","object":"chat.completion","created":7,"model":"qwen",
		"choices":[{"index":0,"message":{"role":"assistant","content":"Hello, world","reasoning_content":"think more"},"logprobs":null,"finish_reason":"stop"}],
		"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}
	}`, aggregatedJSON(t, sse))
}

func TestChatStreamAggregator_ReasoningFieldNameIsKept(t *testing.T) {
	sse := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"r\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n"
	assert.JSONEq(t, `{"id":"c1","object":"chat.completion","created":0,"model":"",
		"choices":[{"index":0,"message":{"role":"assistant","content":"a","reasoning":"r"},"logprobs":null,"finish_reason":"stop"}]}`,
		aggregatedJSON(t, sse))
}

func TestChatStreamAggregator_ToolCallsByIndexAndNullContent(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"\n\n"}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_time","arguments":"{\"tz\":"}}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"UTC\"}"}}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"
	assert.JSONEq(t, `{"id":"c1","object":"chat.completion","created":0,"model":"",
		"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
			{"id":"call_b","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}
		]},"logprobs":null,"finish_reason":"tool_calls"}]}`, aggregatedJSON(t, sse))
}

func TestChatStreamAggregator_TextWithToolCallKeepsText(t *testing.T) {
	sse := `data: {"choices":[{"index":0,"delta":{"content":"Let me check."}}]}` + "\n\n" +
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n"
	body := aggregatedJSON(t, sse)
	assert.Contains(t, body, `"content":"Let me check."`)
	assert.Contains(t, body, `"type":"function"`, "type defaults to function")
}

func TestChatStreamAggregator_MultipleChoicesAndLogprobs(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c1","choices":[{"index":1,"delta":{"content":"B"},"logprobs":{"content":[{"token":"B","logprob":-0.1}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"A"},"logprobs":{"content":[{"token":"A","logprob":-0.2}]}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"a"},"logprobs":{"content":[{"token":"a","logprob":-0.3}]},"finish_reason":"stop"}]}`,
		`data: {"id":"c1","choices":[{"index":1,"delta":{},"finish_reason":"length"}]}`,
	}, "\n\n") + "\n\n"
	assert.JSONEq(t, `{"id":"c1","object":"chat.completion","created":0,"model":"","choices":[
		{"index":0,"message":{"role":"assistant","content":"Aa"},"logprobs":{"content":[{"token":"A","logprob":-0.2},{"token":"a","logprob":-0.3}]},"finish_reason":"stop"},
		{"index":1,"message":{"role":"assistant","content":"B"},"logprobs":{"content":[{"token":"B","logprob":-0.1}]},"finish_reason":"length"}
	]}`, aggregatedJSON(t, sse))
}

func TestChatStreamAggregator_ErrorEvent(t *testing.T) {
	for name, tc := range map[string]struct {
		event  string
		status int
	}{
		"vllm error object":   {`{"error":{"object":"error","message":"maximum context length is 8192 tokens","type":"BadRequestError","param":null,"code":400}}`, http.StatusBadRequest},
		"legacy object error": {`{"object":"error","message":"engine dead","type":"InternalServerError","code":500}`, http.StatusInternalServerError},
		"no code":             {`{"error":{"message":"rate limit exceeded","type":"rate_limit_exceeded"}}`, http.StatusTooManyRequests},
	} {
		t.Run(name, func(t *testing.T) {
			a := aggregateSSE(t, `data: {"id":"c1","choices":[{"index":0,"delta":{"content":"partial"}}]}`+"\n\ndata: "+tc.event+"\n\ndata: [DONE]\n\n")
			assert.Equal(t, tc.status, a.errStatus)
			assert.JSONEq(t, tc.event, string(a.errBody))
		})
	}
}

func TestChatStreamAggregator_LimitAndReadError(t *testing.T) {
	a := newChatStreamAggregator()
	err := a.consume(strings.NewReader(strings.Repeat(`data: {"choices":[{"index":0,"delta":{"content":"x"}}]}`+"\n\n", 100)), 500)
	assert.ErrorIs(t, err, ErrResponseBodyTooLarge)

	a = newChatStreamAggregator()
	err = a.consume(io.MultiReader(strings.NewReader("data: {\"choices\":[]}\n\n"), iotest.ErrReader(io.ErrUnexpectedEOF)), 0)
	assert.ErrorIs(t, err, io.ErrUnexpectedEOF)
}

func TestAggregatedResponse(t *testing.T) {
	upstream := &http.Response{
		StatusCode: http.StatusOK, Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
		Header: http.Header{"Content-Type": {"text/event-stream"}, "X-Request-Id": {"r1"}, "Content-Length": {"999"}},
	}
	resp := aggregatedResponse(upstream, http.StatusBadRequest, []byte(`{"e":1}`))
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Equal(t, "400 Bad Request", resp.Status)
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))
	assert.Equal(t, "r1", resp.Header.Get("X-Request-Id"))
	assert.Empty(t, resp.Header.Get("Content-Length"))
	assert.False(t, IsStreamingResponse(resp))
	body, _ := io.ReadAll(resp.Body)
	assert.Equal(t, `{"e":1}`, string(body))
	assert.Equal(t, "text/event-stream", upstream.Header.Get("Content-Type"), "the upstream response is not modified")
}

func TestChatStreamAggregator_ImagesWebSearchServiceTier(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c1","service_tier":"default","web_search":[{"title":"t","url":"https://example.com"}],"choices":[{"index":0,"delta":{"role":"assistant","content":"see"}}]}`,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},"finish_reason":"stop"}]}`,
	}, "\n\n") + "\n\n"
	assert.JSONEq(t, `{"id":"c1","object":"chat.completion","created":0,"model":"","service_tier":"default",
		"web_search":[{"title":"t","url":"https://example.com"}],
		"choices":[{"index":0,"message":{"role":"assistant","content":"see",
			"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},"logprobs":null,"finish_reason":"stop"}]}`,
		aggregatedJSON(t, sse))
}

func TestReassembleChatCompletion_ErrorEventIsNotReassembled(t *testing.T) {
	stream := `data: {"id":"c1","choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n" +
		`data: {"error":{"message":"engine dead","code":500}}` + "\n\ndata: [DONE]\n\n"
	assert.Nil(t, reassembleChatCompletion([]byte(stream)), "an error event fails the simulated response instead of a partial 200")
}
