package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// openAIChatSSE builds a Chat Completions SSE body from the given chunk JSON payloads.
func openAIChatSSE(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestReassembleChatCompletion_Basic(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	)

	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)

	var out struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Created int64  `json:"created"`
		Choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	assert.Equal(t, "chatcmpl-1", out.ID)
	assert.Equal(t, "chat.completion", out.Object)
	assert.Equal(t, "gpt-4o", out.Model)
	assert.Equal(t, int64(1700000000), out.Created)
	require.Len(t, out.Choices, 1)
	assert.Equal(t, 0, out.Choices[0].Index)
	assert.Equal(t, "assistant", out.Choices[0].Message.Role)
	assert.Equal(t, "Hello world", out.Choices[0].Message.Content)
	assert.Equal(t, "stop", out.Choices[0].FinishReason)
	assert.Equal(t, 5, out.Usage.PromptTokens)
	assert.Equal(t, 2, out.Usage.CompletionTokens)
	assert.Equal(t, 7, out.Usage.TotalTokens)
}

func TestReassembleChatCompletion_ToolCalls(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]},"finish_reason":"tool_calls"}]}`,
	)

	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)

	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "tool_calls", out.Choices[0].FinishReason)
	require.Len(t, out.Choices[0].Message.ToolCalls, 1)
	tc := out.Choices[0].Message.ToolCalls[0]
	assert.Equal(t, "call_1", tc.ID)
	assert.Equal(t, "function", tc.Type)
	assert.Equal(t, "get_weather", tc.Function.Name)
	assert.Equal(t, `{"city":"London"}`, tc.Function.Arguments)
}

func TestReassembleChatCompletion_MultiChoiceAndReasoning(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-3","object":"chat.completion.chunk","model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"},"finish_reason":null},{"index":1,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-3","object":"chat.completion.chunk","model":"deepseek-chat","choices":[{"index":1,"delta":{"content":"second"},"finish_reason":"stop"},{"index":0,"delta":{"content":"first"},"finish_reason":"stop"}]}`,
	)

	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)

	var out struct {
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Content          string  `json:"content"`
				ReasoningContent *string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 2)
	assert.Equal(t, 0, out.Choices[0].Index)
	assert.Equal(t, "first", out.Choices[0].Message.Content)
	require.NotNil(t, out.Choices[0].Message.ReasoningContent)
	assert.Equal(t, "think", *out.Choices[0].Message.ReasoningContent)
	assert.Equal(t, 1, out.Choices[1].Index)
	assert.Equal(t, "second", out.Choices[1].Message.Content)
}

func TestReassembleChatCompletion_EmptyStream(t *testing.T) {
	assert.Nil(t, reassembleChatCompletion([]byte("data: [DONE]\n\n")))
	assert.Nil(t, reassembleChatCompletion([]byte(": keepalive\n\n")))
	assert.Nil(t, reassembleChatCompletion([]byte("not sse at all")))
}

// TestHandleSimulatedNonStream_CapturesTTFT drives the simulated non-stream
// handler directly: upstream answers SSE to a forced stream, and the handler
// must reassemble the JSON body, stamp CompletionStartTime (TTFT) on the first
// content delta and serve the response as application/json.
func TestHandleSimulatedNonStream_CapturesTTFT(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	stream := openAIChatSSE(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(stream)),
	}

	start := time.Now().Add(-time.Second)
	logCtx := &RequestLogContext{StartTime: start, RequestID: "test-req", Request: req}
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}
	prepared := &orchestratedRequest{streaming: false}
	w := httptest.NewRecorder()

	prx.handleSimulatedNonStream(w, req, resp, prepared, cred, "gpt-4o", "gpt-4o",
		logCtx, nil, "http://upstream", []byte(`{"model":"gpt-4o","messages":[]}`))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	require.False(t, logCtx.CompletionStartTime.IsZero())
	assert.GreaterOrEqual(t, logCtx.CompletionStartTime, start)

	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "chat.completion", out.Object)
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "Hello world", out.Choices[0].Message.Content)
	assert.Equal(t, "stop", out.Choices[0].FinishReason)
}

// TestProxyRequest_ForceStreamForTTFT_E2E verifies the full request path with
// the toggle on: the outbound body must carry stream:true and
// stream_options.include_usage, and the client — who asked for a non-streaming
// response — must receive reassembled application/json instead of SSE.
func TestProxyRequest_ForceStreamForTTFT_E2E(t *testing.T) {
	type capturedRequest struct {
		body []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(openAIChatSSE(
			`{"id":"chatcmpl-e2e","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-e2e","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"E2E "},"finish_reason":null}]}`,
			`{"id":"chatcmpl-e2e","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
			`{"id":"chatcmpl-e2e","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		)))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithForceStreamForTTFT(true).
		Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	outbound := <-captured
	var outboundBody map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &outboundBody))
	assert.Equal(t, true, outboundBody["stream"], "outbound body must force stream:true")
	streamOptions, ok := outboundBody["stream_options"].(map[string]any)
	require.True(t, ok, "outbound body must carry stream_options")
	assert.Equal(t, true, streamOptions["include_usage"])

	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "chat.completion", out.Object)
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "E2E answer", out.Choices[0].Message.Content)
}

// TestProxyRequest_ForceStreamForTTFT_ToggleOff verifies the toggle leaves
// existing behavior untouched: with it off, a non-streaming request is
// forwarded as-is and an SSE upstream response is streamed to the client.
func TestProxyRequest_ForceStreamForTTFT_ToggleOff(t *testing.T) {
	var capturedBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(openAIChatSSE(
			`{"id":"chatcmpl-off","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"streamed"},"finish_reason":"stop"}]}`,
		)))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		Build() // toggle off (default)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "data: ")
	assert.Equal(t, false, capturedBody["stream"], "outbound body must stay untouched with toggle off")
}

// TestProxyRequest_ForceStreamForTTFT_MessagesE2E covers /v1/messages: the
// reassembled Chat-shaped body is converted back to the Messages API format the
// client asked for, and the upstream Anthropic body carries stream:true with no
// OpenAI-only stream_options leak.
func TestProxyRequest_ForceStreamForTTFT_MessagesE2E(t *testing.T) {
	type capturedRequest struct {
		body []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("anthropic", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
		WithForceStreamForTTFT(true).
		Build()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-3-5-sonnet",
		"max_tokens":1024,
		"messages":[{"role":"user","content":"hi"}],
		"stream":false
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	outbound := <-captured
	var outboundBody map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &outboundBody))
	assert.Equal(t, true, outboundBody["stream"], "outbound Anthropic body must force stream:true")
	assert.NotContains(t, outboundBody, "stream_options", "OpenAI-only stream_options must not leak into the Anthropic body")

	var out struct {
		Type       string `json:"type"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "message", out.Type)
	assert.Equal(t, "end_turn", out.StopReason)
	require.Len(t, out.Content, 1)
	assert.Equal(t, "text", out.Content[0].Type)
	assert.Equal(t, "Hi there", out.Content[0].Text)
}
