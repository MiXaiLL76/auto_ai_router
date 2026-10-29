package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/models"
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

// errReader returns up to len(data) bytes, then a transport error instead of
// io.EOF — simulating a mid-stream upstream connection drop.
type errReader struct {
	data []byte
	err  error
	pos  int
}

func (r *errReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// TestHandleSimulatedNonStream_MidStreamTransportError verifies a real
// mid-stream error (not EOF) is classified as a failure and returns a masked
// 502 instead of silently serving the truncated partial content as 200.
func TestHandleSimulatedNonStream_MidStreamTransportError(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer master-key")

	sse := openAIChatSSE(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`,
	)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(&errReader{data: []byte(sse), err: io.ErrUnexpectedEOF}),
	}
	start := time.Now().Add(-time.Second)
	logCtx := &RequestLogContext{StartTime: start, RequestID: "test-req", Request: req}
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI}
	prepared := &orchestratedRequest{streaming: false}
	w := httptest.NewRecorder()

	prx.handleSimulatedNonStream(w, req, resp, prepared, cred, "gpt-4o", "gpt-4o",
		logCtx, nil, "http://upstream", []byte(`{"model":"gpt-4o","messages":[]}`))

	require.Equal(t, http.StatusBadGateway, w.Code)
	assert.Equal(t, "failure", logCtx.Status)
	assert.Equal(t, "stream_error", logCtx.StreamOutcome)
	// Final in-memory state only -- writeSimulatedError unconditionally sets
	// logCtx.HTTPStatus itself right after finalizeStreamingLog runs, so this
	// passes either way and is NOT the regression guard for the spend-log bug
	// (see TestFinalizeStreamingLogArgument_UsesClassifiedFailureStatus below,
	// which catches the actual mechanism: what finalizeStreamingLog bakes into
	// the spend log *before* that overwrite ever happens).
	assert.Equal(t, http.StatusBadGateway, logCtx.HTTPStatus)
}

// The spend log is written synchronously inside finalizeStreamingLog, before
// writeSimulatedError's later (client-facing only) assignment to
// logCtx.HTTPStatus -- so the status finalizeStreamingLog is CALLED WITH is
// what actually lands in spend logs/ClickHouse, independent of whatever
// happens to logCtx afterward. This is the real regression guard for the bug
// TestHandleSimulatedNonStream_MidStreamTransportError's own HTTPStatus
// assertion can't catch (writeSimulatedError papers over it in-memory
// either way): previously all three simulated handlers called
// finalizeStreamingLog with resp.StatusCode (the upstream's original 2xx)
// instead of the classified failure status, so a request failing 40+ seconds
// in still logged a clean 200 even though the client got 502/504.
func TestFinalizeStreamingLogArgument_UsesClassifiedFailureStatus(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := &RequestLogContext{
		Request:       httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		StreamOutcome: "stream_error", // what markStreamFailure would have set
	}

	// What handleSimulatedNonStream (and its two siblings) must now pass as
	// finalizeStreamingLog's statusCode argument for a transport failure --
	// simulatedFailureStatus's own result, not resp.StatusCode.
	prx.finalizeStreamingLog(logCtx, 0, nil, "openai", simulatedFailureStatus(context.DeadlineExceeded), false)
	assert.Equal(t, http.StatusGatewayTimeout, logCtx.HTTPStatus, "a timeout-shaped failure must bake 504 into the spend log")

	logCtx2 := &RequestLogContext{
		Request:       httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		StreamOutcome: "stream_error",
	}
	prx.finalizeStreamingLog(logCtx2, 0, nil, "openai", simulatedFailureStatus(io.ErrUnexpectedEOF), false)
	assert.Equal(t, http.StatusBadGateway, logCtx2.HTTPStatus, "a plain connection-drop failure must bake 502 into the spend log")
}

// A timeout-shaped transport failure classifies as 504, not a bare 502 -- for
// both the client response and the spend log, mirroring the isTimeoutError
// convention used elsewhere (e.g. the "all attempts exhausted" tail).
func TestHandleSimulatedNonStream_TimeoutClassifiesAsGatewayTimeout(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer master-key")

	sse := openAIChatSSE(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`,
	)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(&errReader{data: []byte(sse), err: context.DeadlineExceeded}),
	}
	logCtx := &RequestLogContext{StartTime: time.Now(), RequestID: "test-req-timeout", Request: req}
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI}
	prepared := &orchestratedRequest{streaming: false}
	w := httptest.NewRecorder()

	prx.handleSimulatedNonStream(w, req, resp, prepared, cred, "gpt-4o", "gpt-4o",
		logCtx, nil, "http://upstream", []byte(`{"model":"gpt-4o","messages":[]}`))

	assert.Equal(t, http.StatusGatewayTimeout, w.Code)
	assert.Equal(t, http.StatusGatewayTimeout, logCtx.HTTPStatus)
}

// TestReassembleChatCompletion_Images verifies streamed image deltas (Gemini
// image generation) survive the reassembly, mirroring the non-stream message
// shape instead of being dropped.
func TestReassembleChatCompletion_Images(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-img","object":"chat.completion.chunk","model":"gemini-2.0-flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-img","object":"chat.completion.chunk","model":"gemini-2.0-flash","choices":[{"index":0,"delta":{"images":[{"b64_json":"AAAA","type":"image/png"}]},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)

	var out struct {
		Choices []struct {
			Message struct {
				Images []map[string]any `json:"images"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 1)
	require.Len(t, out.Choices[0].Message.Images, 1)
	assert.Equal(t, "AAAA", out.Choices[0].Message.Images[0]["b64_json"])
	assert.Equal(t, "image/png", out.Choices[0].Message.Images[0]["type"])
}

// TestReassembleChatCompletion_WebSearch verifies the top-level web_search
// payload of a chunk survives reassembly (it has no field on
// OpenAIStreamingChunk and would otherwise be silently dropped).
func TestReassembleChatCompletion_WebSearch(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-ws","object":"chat.completion.chunk","model":"gpt-4o","web_search":[{"title":"Result title","url":"https://example.com"}],"choices":[{"index":0,"delta":{"content":"found"},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)

	var out map[string]any
	require.NoError(t, json.Unmarshal(body, &out))
	ws, ok := out["web_search"].([]any)
	require.True(t, ok, "web_search must survive reassembly")
	require.Len(t, ws, 1)
}

// TestHandleSimulatedNonStream_WebSearchHidden verifies HideWebSearchResults
// strips web_search (mirroring the plain non-stream path) and that
// ClientResponseID is captured for billing correlation.
func TestHandleSimulatedNonStream_WebSearchHiddenAndClientResponseID(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer master-key")

	stream := openAIChatSSE(
		`{"id":"chatcmpl-ws","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","web_search":[{"title":"Result title"}],"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-ws","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"found"},"finish_reason":"stop"}]}`,
	)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(stream)),
	}
	logCtx := &RequestLogContext{StartTime: time.Now().Add(-time.Second), RequestID: "test-req", Request: req, HideWebSearchResults: true}
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI}
	prepared := &orchestratedRequest{streaming: false}
	w := httptest.NewRecorder()

	prx.handleSimulatedNonStream(w, req, resp, prepared, cred, "gpt-4o", "gpt-4o",
		logCtx, nil, "http://upstream", []byte(`{"model":"gpt-4o","messages":[]}`))

	require.Equal(t, http.StatusOK, w.Code)
	assert.NotContains(t, w.Body.String(), "web_search", "hidden web_search must be stripped")
	assert.Equal(t, "chatcmpl-ws", logCtx.ClientResponseID, "ClientResponseID must be captured for billing correlation")
}

// TestReassembleChatCompletion_NoNullFinishReason verifies a choice that never
// received a terminal finish_reason is emitted without a null value, matching
// the string-only non-stream contract.
func TestReassembleChatCompletion_NoNullFinishReason(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"chatcmpl-nf","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	assert.NotContains(t, string(body), `"finish_reason":null`)
}

// TestSetStreamTrue_PreservesLargeIntegers verifies the JSON round-trip used by
// applyForceStream does not round integers beyond float64 precision (it must
// only add stream/stream_options, never rewrite other fields).
func TestSetStreamTrue_PreservesLargeIntegers(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","seed":9007199254740993,"stream":false}`)
	out := setStreamTrue(body)
	assert.Contains(t, string(out), `"seed":9007199254740993`, "large integer must survive byte-for-byte")
	var raw map[string]any
	require.NoError(t, json.Unmarshal(out, &raw))
	assert.Equal(t, true, raw["stream"])

	withOptions := injectStreamOptions(out)
	assert.Contains(t, string(withOptions), `"seed":9007199254740993`, "injectStreamOptions must preserve large integers too")
}

// TestApplyForceStream_SkipsProxyLike verifies AIR-to-AIR chains are never
// force-streamed: their response path streams SSE straight to the client
// without reassembly, which would break the non-stream contract.
func TestApplyForceStream_SkipsProxyLike(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)
	cred := &config.CredentialConfig{Name: "air", Type: config.ProviderTypeAIR, BaseURL: "http://peer"}

	out := prx.applyForceStream(credentialPreparedRequest{body: body, proxyBody: body, path: "/v1/chat/completions"}, false, cred, req)
	assert.Equal(t, string(body), string(out.body), "proxy-like bodies must be left untouched")
	assert.NotContains(t, string(out.body), `"stream":true`)
}

// TestApplyForceStream_NativeResponses verifies that a native-Responses
// credential (Anthropic/Vertex) is now force-streamed too: the outbound body
// must carry stream:true and the forcedNativeNonStream marker must be set.
func TestApplyForceStream_NativeResponses(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
	body := []byte(`{"model":"claude-haiku-4.5","input":"hi","stream":false}`)
	cred := &config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, BaseURL: "http://upstream"}

	out := prx.applyForceStream(credentialPreparedRequest{
		body:            body,
		proxyBody:       body,
		path:            "/v1/responses",
		nativeResponses: true,
	}, false, cred, req)

	assert.True(t, out.forcedNativeNonStream, "native Responses request must be marked as forced")
	assert.Contains(t, string(out.body), `"stream":true`, "outbound native body must force stream:true")
}

// TestApplyForceStream_NativeResponses_ToggleOff verifies the toggle still gates
// the native path: with the feature off, the native body is left untouched.
func TestApplyForceStream_NativeResponses_ToggleOff(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(false).Build()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
	body := []byte(`{"model":"claude-haiku-4.5","input":"hi","stream":false}`)
	cred := &config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, BaseURL: "http://upstream"}

	out := prx.applyForceStream(credentialPreparedRequest{
		body:            body,
		proxyBody:       body,
		path:            "/v1/responses",
		nativeResponses: true,
	}, false, cred, req)

	assert.False(t, out.forcedNativeNonStream)
	assert.NotContains(t, string(out.body), `"stream":true`)
}

// stubProviderResponses is a minimal ProviderResponses used to drive the
// simulated native non-stream handler without registering a real provider
// converter (which would change the behavior of unrelated orchestrator tests).
// Its StreamTo reads the provider SSE (ignored), writes one Responses SSE text
// delta, and reports the terminal Response via onComplete.
type stubProviderResponses struct {
	final *responses.Response
}

func (s *stubProviderResponses) RequestFrom(body []byte) ([]byte, string, error) {
	return body, "application/json", nil
}

func (s *stubProviderResponses) ResponseTo(body []byte, displayModelID string) (*responses.Response, error) {
	return s.final, nil
}

func (s *stubProviderResponses) StreamTo(reader io.Reader, writer io.Writer, displayModelID string,
	meta *responses.ResponsesMetadata, onComplete func(*responses.Response)) error {
	// Emit a Responses-shaped content delta so the TTFT scanner sees real content.
	_, _ = writer.Write([]byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"))
	_, _ = reader.Read(make([]byte, 1)) // consume a byte so the upstream body is read
	if onComplete != nil {
		onComplete(s.final)
	}
	return nil
}

func (s *stubProviderResponses) BuildURL(cred *config.CredentialConfig) string {
	return "http://upstream"
}

// TestHandleSimulatedNativeNonStream_CapturesTTFT drives the native simulated
// handler directly: the provider answers SSE, the converter folds it into a
// terminal Responses object via onComplete, and the handler must serve
// application/json with the full text, stamping CompletionStartTime (TTFT) on
// the first content delta.
func TestHandleSimulatedNativeNonStream_CapturesTTFT(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"claude-haiku-4.5","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	provResponses := &stubProviderResponses{final: &responses.Response{
		ID:     "resp_1",
		Object: "response",
		Model:  "claude-haiku-4.5",
		Status: "in_progress",
		Output: []responses.OutputItem{{
			Type:   "message",
			ID:     "msg_1",
			Status: "completed",
			Role:   "assistant",
			Content: []responses.OutputContent{
				{Type: "output_text", Text: "Hello world"},
			},
		}},
		Usage: &responses.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader("event: message_start\ndata: {\"type\":\"message_start\"}\n\n")),
	}

	start := time.Now().Add(-time.Second)
	logCtx := &RequestLogContext{StartTime: start, RequestID: "test-native-req", Request: req}
	cred := &config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, BaseURL: "http://upstream"}
	prepared := &orchestratedRequest{streaming: false, nativeResponses: true, forcedNativeNonStream: true, body: []byte(`{"model":"claude-haiku-4.5","input":"hi"}`)}
	w := httptest.NewRecorder()

	prx.handleSimulatedNativeNonStream(w, req, resp, prepared, cred, provResponses, "claude-haiku-4.5",
		logCtx, nil, "http://upstream")

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	require.False(t, logCtx.CompletionStartTime.IsZero(), "TTFT must be stamped on the first content delta")
	assert.GreaterOrEqual(t, logCtx.CompletionStartTime, start)

	var out struct {
		Object string `json:"object"`
		Status string `json:"status"`
		Output []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "response", out.Object)
	assert.Equal(t, "completed", out.Status)
	require.NotEmpty(t, out.Output)
	require.NotEmpty(t, out.Output[0].Content)
	assert.Equal(t, "Hello world", out.Output[0].Content[0].Text)
}

// TestHandleSimulatedNativeNonStream_SavesResponse verifies store: true on a
// forced native Responses request still persists the terminal object for
// later GET /v1/responses/{id} retrieval — saveResponseFn was previously
// never called on this path, unlike the chat-shaped and passthrough
// reassembly paths.
func TestHandleSimulatedNativeNonStream_SavesResponse(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"claude-haiku-4.5","input":"hi","store":true}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	provResponses := &stubProviderResponses{final: &responses.Response{
		ID:     "resp_1",
		Object: "response",
		Model:  "claude-haiku-4.5",
		Status: "in_progress",
		Output: []responses.OutputItem{{
			Type:   "message",
			ID:     "msg_1",
			Status: "completed",
			Role:   "assistant",
			Content: []responses.OutputContent{
				{Type: "output_text", Text: "Hello world"},
			},
		}},
		Usage: &responses.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader("event: message_start\ndata: {\"type\":\"message_start\"}\n\n")),
	}

	logCtx := &RequestLogContext{StartTime: time.Now(), RequestID: "test-native-store-req", Request: req}
	cred := &config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, BaseURL: "http://upstream"}
	prepared := &orchestratedRequest{streaming: false, nativeResponses: true, forcedNativeNonStream: true, body: []byte(`{"model":"claude-haiku-4.5","input":"hi","store":true}`)}
	w := httptest.NewRecorder()

	var saved *responses.Response
	saveResponseFn := func(r *responses.Response) { saved = r }

	prx.handleSimulatedNativeNonStream(w, req, resp, prepared, cred, provResponses, "claude-haiku-4.5",
		logCtx, saveResponseFn, "http://upstream")

	require.Equal(t, http.StatusOK, w.Code)
	require.NotNil(t, saved, "saveResponseFn must be called so store: true responses are retrievable later")
	assert.Equal(t, "resp_1", saved.ID)
	assert.Equal(t, "completed", saved.Status)
}

// responsesSSE builds a Responses API SSE body whose terminal event carries the
// full final Response object — the wire format a codex/OpenAI passthrough
// upstream returns.
func responsesSSE(finalJSON string) string {
	var b strings.Builder
	b.WriteString("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"gpt-4o-mini\",\"status\":\"in_progress\"}}\n\n")
	b.WriteString("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n")
	b.WriteString("event: response.completed\ndata: " + finalJSON + "\n\n")
	return b.String()
}

// TestApplyForceStream_PassthroughResponses verifies a codex/OpenAI passthrough
// Responses request is now force-streamed too, with its own marker.
func TestApplyForceStream_PassthroughResponses(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
	body := []byte(`{"model":"gpt-4o-mini","input":"hi","stream":false}`)
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}

	out := prx.applyForceStream(credentialPreparedRequest{
		body:                 body,
		proxyBody:            body,
		path:                 "/v1/responses",
		passthroughResponses: true,
	}, false, cred, req)

	assert.True(t, out.forcedPassthroughNonStream, "passthrough Responses request must be marked as forced")
	assert.Contains(t, string(out.body), `"stream":true`, "outbound body must force stream:true")
	assert.False(t, out.forcedNativeNonStream, "passthrough must not take the native marker")
}

// TestHandleSimulatedPassthroughNonStream_CapturesTTFT drives the passthrough
// simulated handler directly: the upstream SSE is Responses-shaped and its
// terminal event carries the whole Response, which the handler must write back
// as non-stream JSON while stamping TTFT.
func TestHandleSimulatedPassthroughNonStream_CapturesTTFT(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"gpt-4o-mini","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	finalJSON := `{"type":"response.completed","response":{"id":"resp_1","object":"response","model":"gpt-4o-mini","status":"completed","output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(responsesSSE(finalJSON))),
	}

	start := time.Now().Add(-time.Second)
	logCtx := &RequestLogContext{StartTime: start, RequestID: "test-passthrough-req", Request: req}
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}
	prepared := &orchestratedRequest{streaming: false, passthroughResponses: true, forcedPassthroughNonStream: true, body: []byte(`{"model":"gpt-4o-mini","input":"hi"}`)}
	w := httptest.NewRecorder()

	prx.handleSimulatedPassthroughNonStream(w, req, resp, prepared, cred, "gpt-4o-mini", logCtx, nil, "http://upstream")

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	require.False(t, logCtx.CompletionStartTime.IsZero(), "TTFT must be stamped on the first content delta")
	assert.GreaterOrEqual(t, logCtx.CompletionStartTime, start)

	var out struct {
		Object string `json:"object"`
		Status string `json:"status"`
		Output []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "response", out.Object)
	assert.Equal(t, "completed", out.Status)
	require.NotEmpty(t, out.Output)
	require.NotEmpty(t, out.Output[0].Content)
	assert.Equal(t, "Hello world", out.Output[0].Content[0].Text)
}

// responsesOnlySSETransport simulates a responses_only upstream (only /v1/responses)
// answering with a Responses API SSE stream — the shape force_stream_for_ttft now
// handles for convertedToResponses requests.
type responsesOnlySSETransport struct {
	t *testing.T
}

func (tr responsesOnlySSETransport) RoundTrip(r *http.Request) (*http.Response, error) {
	assert.Equal(tr.t, "/v1/responses", r.URL.Path,
		"AIR must call the provider's native /v1/responses for a responses_only model")
	body, err := io.ReadAll(r.Body)
	require.NoError(tr.t, err)
	var reqBody map[string]interface{}
	require.NoError(tr.t, json.Unmarshal(body, &reqBody))
	assert.Contains(tr.t, reqBody, "input", "outbound body must be Responses-API-shaped")
	assert.NotContains(tr.t, reqBody, "messages")
	assert.Equal(tr.t, true, reqBody["stream"], "force_stream_for_ttft must set stream:true")

	sse := "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"gpt-5-pro\",\"status\":\"in_progress\"}}\n\n" +
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"4\"}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"4\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":9,\"output_tokens\":1,\"total_tokens\":10}}}\n\n"
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(sse)),
		Request:    r,
	}, nil
}

// TestProxyRequest_ForceStream_ResponsesOnly_ChatEndpoint verifies the
// convertedToResponses path now measures TTFT: the client asks for a
// non-streaming /v1/chat/completions against a responses_only model, the
// outbound body is converted to Responses shape and forced to stream, and the
// client still receives reassembled Chat-Completions JSON.
func TestProxyRequest_ForceStream_ResponsesOnly_ChatEndpoint(t *testing.T) {
	credential := config.CredentialConfig{Name: "openai_main", Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com", APIKey: "provider-key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).WithForceStreamForTTFT(true).Build()
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-5-pro", ResponsesOnly: true, Credential: "openai_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx.client.Transport = responsesOnlySSETransport{t: t}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-pro","max_tokens":64,"messages":[{"role":"user","content":"2+2?"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp), "client must receive Chat-Completions JSON, got: %s", w.Body.String())
	assert.Equal(t, "chat.completion", resp["object"])
	choices := resp["choices"].([]interface{})
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]interface{})
	assert.Equal(t, "stop", choice["finish_reason"])
	assert.Equal(t, "4", choice["message"].(map[string]interface{})["content"])
}

// completionsSSE builds a legacy OpenAI Completions SSE body
// (choices[].text, object "text_completion").
func completionsSSE(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestReassembleCompletions_Basic(t *testing.T) {
	stream := completionsSSE(
		`{"id":"cmpl-1","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"Hel","index":0,"finish_reason":null}]}`,
		`{"id":"cmpl-1","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"lo","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	)
	body := reassembleCompletions([]byte(stream))
	require.NotNil(t, body)
	var out struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int    `json:"index"`
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	assert.Equal(t, "cmpl-1", out.ID)
	assert.Equal(t, "text_completion", out.Object)
	assert.Equal(t, "gpt-3.5-turbo-instruct", out.Model)
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "Hello", out.Choices[0].Text)
	assert.Equal(t, "stop", out.Choices[0].FinishReason)
	assert.Equal(t, 5, out.Usage.TotalTokens)
}

// TestProxyRequest_ForceStream_Completions verifies the legacy /v1/completions
// path: with force_stream_for_ttft on, the outbound body is forced to stream and
// the client still receives a text_completion JSON body (not a chat-completion
// one, and not an empty message.content).
func TestProxyRequest_ForceStream_Completions(t *testing.T) {
	var gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(completionsSSE(
			`{"id":"cmpl-9","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"PONG","index":0,"finish_reason":"stop"}]}`,
		)))
	}))
	defer upstream.Close()

	p := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
		WithForceStreamForTTFT(true).Build()
	req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"model":"gpt-3.5-turbo-instruct","prompt":"say hi","stream":false}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	p.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	var outbound map[string]any
	require.NoError(t, json.Unmarshal([]byte(gotBody), &outbound))
	assert.Equal(t, true, outbound["stream"], "outbound completions body must force stream:true")

	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "text_completion", out.Object, "client must receive text_completion, got: %s", w.Body.String())
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "PONG", out.Choices[0].Text)
}

// TestApplyForceStream_ChatSetsForcedMarker verifies a normal chat request
// forced to stream carries the forcedChatNonStream marker, which the converter
// call site uses to build a streaming upstream URL (Vertex/Gemini).
func TestApplyForceStream_ChatSetsForcedMarker(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`)
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}

	out := prx.applyForceStream(credentialPreparedRequest{body: body, proxyBody: body, path: "/v1/chat/completions"}, false, cred, req)

	assert.True(t, out.forcedChatNonStream, "chat request must carry the forced marker")
	assert.Contains(t, string(out.body), `"stream":true`)
}

// TestForceStream_GeminiChat_UsesStreamingURL verifies that with
// force_stream_for_ttft on, a non-streaming gemini chat request reaches the
// upstream's streaming URL (streamGenerateContent), so the simulated path can
// reassemble it and measure TTFT.
func TestForceStream_GeminiChat_UsesStreamingURL(t *testing.T) {
	var gotPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4}}\n\n"))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("gemini", config.ProviderTypeGemini, upstream.URL, "k").
		WithForceStreamForTTFT(true).
		Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, gotPath, "streamGenerateContent", "gemini upstream must be called on the streaming URL")
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
}

// TestReassembleChatCompletion_Logprobs verifies per-token logprobs streamed by
// the upstream (one content entry per chunk) survive reassembly into the
// non-stream choice, matching the plain non-stream response shape.
func TestReassembleChatCompletion_Logprobs(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"logprobs":{"content":[],"refusal":null}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.1,"top_logprobs":[{"token":"Hi","logprob":-0.1}]}],"refusal":null}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"logprobs":{"content":[{"token":"!","logprob":-0.5,"top_logprobs":[{"token":"!","logprob":-0.5}]}],"refusal":null}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Logprobs *struct {
				Content []struct {
					Token string `json:"token"`
				} `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "Hi!", out.Choices[0].Message.Content)
	require.NotNil(t, out.Choices[0].Logprobs, "logprobs must survive reassembly")
	require.Len(t, out.Choices[0].Logprobs.Content, 2)
	assert.Equal(t, "Hi", out.Choices[0].Logprobs.Content[0].Token)
	assert.Equal(t, "!", out.Choices[0].Logprobs.Content[1].Token)
}

// TestReassembleChatCompletion_NoLogprobsWhenAbsent verifies the logprobs field
// is omitted entirely (not emitted as null) for an ordinary stream that never
// carried it, keeping the plain non-stream shape.
func TestReassembleChatCompletion_NoLogprobsWhenAbsent(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	assert.NotContains(t, string(body), `"logprobs"`)
}

// TestReassembleChatCompletion_Annotations verifies url_citation annotations
// streamed on delta.annotations survive reassembly into message.annotations,
// matching the non-stream shape.
func TestReassembleChatCompletion_Annotations(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"see "}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"here","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"t"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	var out struct {
		Choices []struct {
			Message struct {
				Content     string `json:"content"`
				Annotations []struct {
					Type string `json:"type"`
				} `json:"annotations"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "see here", out.Choices[0].Message.Content)
	require.Len(t, out.Choices[0].Message.Annotations, 1, "annotations must survive reassembly")
	assert.Equal(t, "url_citation", out.Choices[0].Message.Annotations[0].Type)
}

// TestReassembleChatCompletion_MessageAnnotationsCarrier verifies the
// non-standard choices[].message.annotations carrier some aggregators
// (Requesty) use instead of delta.annotations also survives reassembly.
func TestReassembleChatCompletion_MessageAnnotationsCarrier(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"message":{"annotations":[{"type":"url_citation"}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	var out struct {
		Choices []struct {
			Message struct {
				Annotations []struct {
					Type string `json:"type"`
				} `json:"annotations"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &out))
	require.Len(t, out.Choices, 1)
	require.Len(t, out.Choices[0].Message.Annotations, 1, "message.annotations carrier must survive reassembly")
	assert.Equal(t, "url_citation", out.Choices[0].Message.Annotations[0].Type)
}

// TestReassembleChatCompletion_NoAnnotationsWhenAbsent verifies the
// annotations field is omitted entirely for an ordinary stream that never
// carried any, keeping the plain non-stream shape.
func TestReassembleChatCompletion_NoAnnotationsWhenAbsent(t *testing.T) {
	stream := openAIChatSSE(
		`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
	)
	body := reassembleChatCompletion([]byte(stream))
	require.NotNil(t, body)
	assert.NotContains(t, string(body), `"annotations"`)
}

// TestApplyForceStream_SkipsNonChatPaths is a regression guard: force_stream_for_ttft
// must only touch chat-shaped endpoints. Images and embeddings previously got a
// stray "stream": true and their providers reject it (gpt-image-1-mini 400
// "unknown parameter: stream"), breaking a request that worked with the toggle off.
func TestApplyForceStream_SkipsNonChatPaths(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{}`))

	for _, c := range []struct{ path, body string }{
		{"/v1/images/generations", `{"model":"gpt-image-1","prompt":"x"}`},
		{"/v1/images/edits", `{"model":"gpt-image-1","prompt":"x"}`},
		{"/v1/embeddings", `{"model":"text-embedding-3-small","input":"hi"}`},
	} {
		out := prx.applyForceStream(credentialPreparedRequest{body: []byte(c.body), proxyBody: []byte(c.body), path: c.path}, false, cred, req)
		assert.NotContains(t, string(out.body), `"stream":true`, "path %s must not be force-streamed", c.path)
	}
}

// TestApplyForceStream_TouchesChatPaths confirms the chat-shaped endpoints are
// still forced (the fix must not disable the feature).
func TestApplyForceStream_TouchesChatPaths(t *testing.T) {
	prx := NewTestProxyBuilder().WithForceStreamForTTFT(true).Build()
	cred := &config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, BaseURL: "http://upstream"}

	for _, path := range []string{"/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions"} {
		req := httptest.NewRequest("POST", path, strings.NewReader(`{}`))
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		out := prx.applyForceStream(credentialPreparedRequest{body: body, proxyBody: body, path: path}, false, cred, req)
		assert.Contains(t, string(out.body), `"stream":true`, "path %s must be force-streamed", path)
	}
}

// TestSimulatedPassthrough_DataWithoutSpace covers upstreams that emit SSE
// frames as "data:{...}" (no space after the colon, which the SSE spec allows
// and e.g. Alibaba DashScope through a proxy does). The reassembly must still
// find the terminal response.completed event and return the non-stream JSON.
func TestSimulatedPassthrough_DataWithoutSpace(t *testing.T) {
	sse := "id:1\n" +
		"event:response.created\n" +
		"data:{\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n" +
		"event:response.output_text.delta\n" +
		"data:{\"type\":\"response.output_text.delta\",\"delta\":\"PONG\"}\n\n" +
		"event:response.completed\n" +
		"data:{\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"m1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"PONG\"}]}],\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n" +
		"data:[DONE]\n\n"

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(sse)),
	}
	prx := NewTestProxyBuilder().Build()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer master-key")
	logCtx := &RequestLogContext{Request: req}
	cred := &config.CredentialConfig{Name: "intl", Type: config.ProviderTypeOpenAI, BaseURL: "http://x"}
	prepared := &orchestratedRequest{passthroughResponses: true, forcedPassthroughNonStream: true, body: []byte(`{}`)}
	w := httptest.NewRecorder()

	prx.handleSimulatedPassthroughNonStream(w, req, resp, prepared, cred, "m", logCtx, nil, "http://x")

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	var out struct {
		Object string `json:"object"`
		Status string `json:"status"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	assert.Equal(t, "response", out.Object)
	assert.Equal(t, "completed", out.Status)
}
