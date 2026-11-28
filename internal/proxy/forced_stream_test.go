package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

// forcedSSE frames payloads as SSE data events, without the closing [DONE].
func forcedSSE(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	return b.String()
}

func forcedStreamRequest(path, body string) *http.Request {
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestForcedStreamBlocker(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"plain chat":                {`{"model":"m","messages":[]}`, ""},
		"best_of above 1":           {`{"model":"m","prompt":"x","best_of":3}`, "best_of"},
		"best_of 1":                 {`{"model":"m","prompt":"x","best_of":1}`, ""},
		"best_of null":              {`{"model":"m","prompt":"x","best_of":null}`, ""},
		"beam search":               {`{"model":"m","messages":[],"use_beam_search":true}`, "use_beam_search"},
		"beam search off":           {`{"model":"m","messages":[],"use_beam_search":false}`, ""},
		"the words in a message":    {`{"model":"m","messages":[{"role":"user","content":"what is \"best_of\" and \"use_beam_search\"?"}]}`, ""},
		"a nested best_of":          {`{"model":"m","messages":[],"metadata":{"best_of":5}}`, ""},
		"not a JSON object":         {`"best_of"`, ""},
		"both, best_of comes first": {`{"best_of":2,"use_beam_search":true}`, "best_of"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, forcedStreamBlocker([]byte(tc.body)))
		})
	}
}

func TestForcedStreamRequestBody(t *testing.T) {
	// The client's own stream_options are kept unless the caller replaces them.
	got, ok := forcedStreamRequestBody([]byte(`{"model":"m","prompt":"x","stream":false,"stream_options":{"extra":1},"seed":9007199254740993}`),
		forcedStreamOptions{includeUsage: true})
	require.True(t, ok)
	assert.JSONEq(t, `{"model":"m","prompt":"x","stream":true,"stream_options":{"extra":1,"include_usage":true},"seed":9007199254740993}`, string(got))
	assert.Contains(t, string(got), "9007199254740993", "large integers keep their digits")
	assert.Equal(t, 1, strings.Count(string(got), `"stream":`), "replaced, not duplicated")

	got, ok = forcedStreamRequestBody([]byte(`{"model":"m","stream_options":{"extra":1}}`),
		forcedStreamOptions{includeUsage: true, continuousUsage: true, replaceOptions: true})
	require.True(t, ok)
	assert.JSONEq(t, `{"model":"m","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}`, string(got))

	got, ok = forcedStreamRequestBody([]byte(`{"model":"m"}`), forcedStreamOptions{})
	require.True(t, ok)
	assert.JSONEq(t, `{"model":"m","stream":true}`, string(got))

	for _, body := range []string{`null`, `"x"`, `[1]`, `not json`} {
		got, ok = forcedStreamRequestBody([]byte(body), forcedStreamOptions{includeUsage: true})
		assert.False(t, ok, body)
		assert.Equal(t, body, string(got), "a body that is not a JSON object is returned as is")
	}
}

func TestReassembleForcedStream_Completeness(t *testing.T) {
	text := `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{"role":"assistant","content":"The answer is"},"finish_reason":null}]}`
	finish := `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`

	// Ended by a clean EOF mid-answer: no finish_reason, no [DONE].
	_, err := reassembleForcedStream([]byte(forcedSSE(text)), false)
	require.ErrorIs(t, err, errForcedStreamTruncated)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
	assert.Nil(t, reassembleChatCompletion([]byte(forcedSSE(text))))

	// A finish_reason for every choice is a complete answer without [DONE].
	body, err := reassembleForcedStream([]byte(forcedSSE(text, finish)), false)
	require.NoError(t, err)
	assert.Contains(t, string(body), `"finish_reason":"stop"`)

	// [DONE] is a complete answer even without a finish_reason: a provider
	// converter always closes its stream with it.
	body, err = reassembleForcedStream([]byte(forcedSSE(text)+"data: [DONE]\n\n"), false)
	require.NoError(t, err)
	assert.Contains(t, string(body), "The answer is")

	// Two choices, one of them never finished.
	second := `{"id":"c1","choices":[{"index":1,"delta":{"content":"other"},"finish_reason":null}]}`
	_, err = reassembleForcedStream([]byte(forcedSSE(text, second, finish)), false)
	require.ErrorIs(t, err, errForcedStreamTruncated)

	// Nothing at all is an empty stream, not a truncated one.
	_, err = reassembleForcedStream([]byte("data: [DONE]\n\n"), false)
	require.ErrorIs(t, err, errForcedStreamEmpty)
	_, err = reassembleForcedStream(nil, true)
	require.ErrorIs(t, err, errForcedStreamEmpty)

	// The legacy completions shape follows the same rule.
	_, err = reassembleForcedStream([]byte(forcedSSE(`{"id":"cmpl-1","choices":[{"index":0,"text":"Hel"}]}`)), true)
	require.ErrorIs(t, err, errForcedStreamTruncated)
	assert.Nil(t, reassembleCompletions([]byte(forcedSSE(`{"id":"cmpl-1","choices":[{"index":0,"text":"Hel"}]}`))))
}

func TestReassembleCompletions_LogprobsStopReasonUsage(t *testing.T) {
	stream := forcedSSE(
		`{"id":"cmpl-1","object":"text_completion","created":7,"model":"glm","choices":[{"index":0,"text":"Hello","logprobs":{"tokens":["Hello"],"token_logprobs":[-0.1],"top_logprobs":[{"Hello":-0.1}],"text_offset":[0]},"finish_reason":null,"stop_reason":null}]}`,
		`{"id":"cmpl-1","object":"text_completion","created":7,"model":"glm","choices":[{"index":0,"text":"!","logprobs":{"tokens":["!"],"token_logprobs":[-0.25],"top_logprobs":[{"!":-0.25}],"text_offset":[5]},"finish_reason":"stop","stop_reason":128001}]}`,
		`{"id":"cmpl-1","object":"text_completion","created":7,"model":"glm","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	) + "data: [DONE]\n\n"
	body := reassembleCompletions([]byte(stream))
	require.NotNil(t, body)
	assert.JSONEq(t, `{
		"id":"cmpl-1","object":"text_completion","created":7,"model":"glm",
		"choices":[{"index":0,"text":"Hello!",
			"logprobs":{"tokens":["Hello","!"],"token_logprobs":[-0.1,-0.25],"top_logprobs":[{"Hello":-0.1},{"!":-0.25}],"text_offset":[0,5]},
			"finish_reason":"stop","stop_reason":128001}],
		"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
	}`, string(body))

	// No logprobs asked for: null, as a non-streamed reply has it.
	plain := reassembleCompletions([]byte(forcedSSE(`{"id":"cmpl-2","choices":[{"index":0,"text":"hi","finish_reason":"stop"}]}`)))
	require.NotNil(t, plain)
	var out struct {
		Choices []map[string]json.RawMessage `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(plain, &out))
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "null", string(out.Choices[0]["logprobs"]))
	assert.NotContains(t, out.Choices[0], "stop_reason")
}

// A forced stream that ends cleanly before its answer is complete is a failed
// attempt: the next credential serves the request.
func TestProxyRequest_ForceStream_TruncatedStreamRetriesNextCredential(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(forcedSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"The answer is"},"finish_reason":null}]}`,
		)))
	}))
	defer primary.Close()
	secondary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		secondaryHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(openAIChatSSE(
			`{"id":"c2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"whole"},"finish_reason":"stop"}]}`,
		)))
	}))
	defer secondary.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: primary.URL, APIKey: "k1", RPM: 100, TPM: 100000},
		config.CredentialConfig{Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: secondary.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 100000},
	).WithForceStreamForTTFT(true).WithMaxProviderRetries(3).Build()

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`))

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(1), secondaryHits.Load(), "a truncated stream must retry on the next credential")
	assert.Contains(t, w.Body.String(), "whole")
	assert.NotContains(t, w.Body.String(), "The answer is", "the cut answer must not reach the client")
}

// With no credential left, the cut answer is an error, never a 200.
func TestProxyRequest_ForceStream_TruncatedStreamIsNotA200(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(forcedSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"The answer is"},"finish_reason":null}]}`,
		)))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
		WithForceStreamForTTFT(true).Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`))

	assert.Equal(t, http.StatusBadGateway, w.Code, "body: %s", w.Body.String())
	assert.NotContains(t, w.Body.String(), "The answer is")
}

// A request with an option the provider refuses with stream:true is sent as
// the client sent it.
func TestProxyRequest_ForceStream_SkipsStreamIncompatibleOptions(t *testing.T) {
	for name, tc := range map[string]struct{ path, body, reply string }{
		"best_of": {
			"/v1/completions",
			`{"model":"gpt-3.5-turbo-instruct","prompt":"hi","best_of":3,"stream":false}`,
			`{"id":"cmpl-1","object":"text_completion","model":"gpt-3.5-turbo-instruct","choices":[{"index":0,"text":"plain","finish_reason":"stop"}]}`,
		},
		"use_beam_search": {
			"/v1/chat/completions",
			`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"use_beam_search":true}`,
			`{"id":"c1","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"plain"},"finish_reason":"stop"}]}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var gotBody []byte
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotBody, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.reply))
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
				WithForceStreamForTTFT(true).Build()
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, forcedStreamRequest(tc.path, tc.body))

			require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
			assert.Contains(t, w.Body.String(), "plain")
			var outbound map[string]any
			require.NoError(t, json.Unmarshal(gotBody, &outbound))
			assert.NotEqual(t, true, outbound["stream"], "the request must not be forced to stream: %s", gotBody)
			assert.NotContains(t, outbound, "stream_options")
		})
	}
}

// The legacy completions endpoint asks for usage in its forced stream, and
// the client gets it in the rebuilt body.
func TestProxyRequest_ForceStream_CompletionsAsksForUsage(t *testing.T) {
	var gotBody []byte
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(completionsSSE(
			`{"id":"cmpl-9","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"PONG","index":0,"logprobs":{"tokens":["PONG"],"token_logprobs":[-0.5],"top_logprobs":[{"PONG":-0.5}],"text_offset":[0]},"finish_reason":"stop"}]}`,
			`{"id":"cmpl-9","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		)))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
		WithForceStreamForTTFT(true).Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/completions", `{"model":"gpt-3.5-turbo-instruct","prompt":"say hi","logprobs":1,"stream":false}`))

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	var outbound struct {
		Stream        bool `json:"stream"`
		StreamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	require.NoError(t, json.Unmarshal(gotBody, &outbound))
	assert.True(t, outbound.Stream)
	assert.True(t, outbound.StreamOptions.IncludeUsage, "outbound completions body must ask for usage: %s", gotBody)

	var out struct {
		Choices []struct {
			Text     string `json:"text"`
			Logprobs struct {
				Tokens []string `json:"tokens"`
			} `json:"logprobs"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	require.Len(t, out.Choices, 1)
	assert.Equal(t, "PONG", out.Choices[0].Text)
	assert.Equal(t, []string{"PONG"}, out.Choices[0].Logprobs.Tokens)
	assert.Equal(t, 3, out.Usage.TotalTokens)
}

// An upstream that hangs mid-stream fails the request at request_timeout
// instead of holding it until the client leaves.
func TestProxyRequest_ForceStream_HungStreamEndsAtRequestTimeout(t *testing.T) {
	release := make(chan struct{})
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(forcedSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`,
		)))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()
	defer close(release)

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
		WithForceStreamForTTFT(true).WithRequestTimeout(300 * time.Millisecond).WithMaxProviderRetries(0).Build()

	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		w := httptest.NewRecorder()
		prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`))
		done <- w
	}()
	select {
	case w := <-done:
		assert.Equal(t, http.StatusGatewayTimeout, w.Code, "body: %s", w.Body.String())
		assert.NotContains(t, w.Body.String(), "partial")
	case <-time.After(10 * time.Second):
		t.Fatal("the request was not ended by request_timeout")
	}
}

func TestStreamReadDeadline(t *testing.T) {
	// No timeout: the read error passes through.
	d := startStreamReadDeadline(io.NopCloser(strings.NewReader("")), 0)
	assert.NoError(t, d.stop(nil))
	assert.ErrorIs(t, d.stop(io.ErrUnexpectedEOF), io.ErrUnexpectedEOF)

	// Fired: the body is closed and the read reports the deadline, even when
	// the closed body read as a clean end.
	pr, pw := io.Pipe()
	defer func() { _ = pw.Close() }()
	d = startStreamReadDeadline(pr, 20*time.Millisecond)
	_, readErr := io.ReadAll(pr)
	require.Error(t, readErr, "the deadline closes the body")
	err := d.stop(nil)
	require.Error(t, err)
	assert.True(t, isTimeoutError(err))
}

// The loop gate's force_stream refuses a stream that ended early the same
// way: the client gets an error, not the cut answer.
func TestLoopGate_ForceStreamTruncatedStreamIsNotA200(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(forcedSSE(
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"Qwen/Qwen3.6-35B","choices":[{"index":0,"delta":{"role":"assistant","content":"The answer is"},"finish_reason":null}]}`,
		)))
	}))
	defer upstream.Close()

	prx := newLoopGateProxyWithGate(t, upstream.URL, forcedGate(0, 1))
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`))

	assert.GreaterOrEqual(t, w.Code, http.StatusInternalServerError, "body: %s", w.Body.String())
	assert.NotContains(t, w.Body.String(), "The answer is")
}

// The loop gate's force_stream leaves a request with a stream-incompatible
// option non-streamed.
func TestLoopGate_ForceStreamSkipsStreamIncompatibleOptions(t *testing.T) {
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","model":"Qwen/Qwen3.6-35B","choices":[{"index":0,"message":{"role":"assistant","content":"plain"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	prx := newLoopGateProxyWithGate(t, upstream.URL, forcedGate(0, 1))
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"use_beam_search":true}`))

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	assert.Contains(t, w.Body.String(), "plain")
	var outbound map[string]any
	require.NoError(t, json.Unmarshal(gotBody, &outbound))
	assert.NotEqual(t, true, outbound["stream"], "the request must not be streamed upstream: %s", gotBody)
}

// The TTFT of a forced stream is recorded per model, so one model served by
// several credentials is read as a single series.
func TestProxyRequest_ForceStream_RecordsTTFTPerModel(t *testing.T) {
	monitoring.TimeToFirstTokenSeconds.Reset()
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(openAIChatSSE(
			`{"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
		)))
	}))
	defer upstream.Close()

	builder := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "k").
		WithForceStreamForTTFT(true)
	builder.config.Metrics = monitoring.New(true)
	prx := builder.Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, forcedStreamRequest("/v1/chat/completions", `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())

	series, err := monitoring.TimeToFirstTokenSeconds.CurryWith(prometheus.Labels{
		"credential": "openai", "model": "gpt-4o", "endpoint": "/v1/chat/completions",
	})
	require.NoError(t, err)
	assert.Equal(t, 1, testutil.CollectAndCount(series), "one TTFT series for the model")
	assert.Equal(t, 1, testutil.CollectAndCount(monitoring.TimeToFirstTokenSeconds))
}
