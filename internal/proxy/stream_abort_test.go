package proxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// failAfterNBytesWriter — ResponseWriter + Flusher that returns a write error
// once totalWritten >= failAt. This simulates a client disconnecting mid-stream.
type failAfterNBytesWriter struct {
	header       http.Header
	statusCode   int
	totalWritten int
	failAt       int
}

func newFailAfterNBytesWriter(failAt int) *failAfterNBytesWriter {
	return &failAfterNBytesWriter{
		header:     make(http.Header),
		statusCode: 200,
		failAt:     failAt,
	}
}

func (f *failAfterNBytesWriter) Header() http.Header  { return f.header }
func (f *failAfterNBytesWriter) WriteHeader(code int) { f.statusCode = code }
func (f *failAfterNBytesWriter) Flush()               {}
func (f *failAfterNBytesWriter) Write(p []byte) (int, error) {
	if f.totalWritten >= f.failAt {
		return 0, fmt.Errorf("write: broken pipe")
	}
	n := len(p)
	f.totalWritten += n
	return n, nil
}

func TestStreamToClient_RecordAbortedMetric(t *testing.T) {
	monitoring.AbortedRequestsTotal.Reset()

	prx := NewTestProxyBuilder().Build()
	prx.metrics = monitoring.New(true)

	w := newFailAfterNBytesWriter(0)
	err := prx.streamToClient(context.Background(), w, strings.NewReader("data: hello\n\n"), "cred1", "gpt-4o", "/v1/chat/completions", nil, nil, nil)
	require.Error(t, err)

	assert.Equal(t, 1.0, testutil.ToFloat64(monitoring.AbortedRequestsTotal.WithLabelValues("cred1", "gpt-4o", "/v1/chat/completions")))
}

type unexpectedEOFReader struct {
	read bool
}

func (r *unexpectedEOFReader) Read(p []byte) (int, error) {
	if r.read {
		return 0, io.EOF
	}
	r.read = true
	return copy(p, "data: partial\n\n"), io.ErrUnexpectedEOF
}

func TestStreamToClientPropagatesUnexpectedReadError(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()

	err := prx.streamToClient(
		context.Background(),
		w,
		&unexpectedEOFReader{},
		"cred1",
		"gpt-4o",
		"/v1/chat/completions",
		nil,
		nil,
		nil,
	)

	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
	assert.Equal(t, "data: partial\n\n", w.Body.String())
}

// TestHandleStreamingWithTokens_AbortLogsTokens verifies that when the client
// disconnects mid-stream (before the usage chunk), the handler drains the upstream
// to capture the real usage chunk and logs accurate token counts.
func TestHandleStreamingWithTokens_AbortLogsTokens(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		// Content chunks (no usage yet)
		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"world "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"from OpenRouter"}}]}` + "\n\n",
			// Usage chunk — client disconnects before this, but drain captures it
			`data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}` + "\n\n",
			"data: [DONE]\n\n",
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		WithDrainUpstreamOnAbort(true). // drain enabled: expect real usage chunk
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	logCtx := &RequestLogContext{
		RequestID:            "abort-test-1",
		PromptTokensEstimate: 15,
		Credential:           &config.CredentialConfig{Name: "test", Type: config.ProviderTypeOpenAI},
		TokenUsage:           &converter.TokenUsage{},
	}

	// Client disconnects after 10 bytes (before the usage chunk)
	w := newFailAfterNBytesWriter(10)

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	assert.Error(t, err, "should return error when client disconnects")

	// Key assertion: logged even though stream was aborted
	assert.True(t, logCtx.Logged, "finalizeStreamingLog must be called even on abort")

	// Drain captured real usage chunk: completion_tokens=10, prompt_tokens=20
	assert.Equal(t, 10, logCtx.TokenUsage.CompletionTokens,
		"completion tokens must match real usage chunk captured during drain")
	assert.Equal(t, 20, logCtx.TokenUsage.PromptTokens,
		"prompt tokens must come from real usage chunk captured during drain")
	assert.Equal(t, "client_aborted", logCtx.StreamOutcome)
	assert.Equal(t, "failure", logCtx.Status)
	assert.Equal(t, "provider", logCtx.UsageSource)

	t.Logf("Abort logging result: prompt=%d completion=%d",
		logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens)
}

// TestHandleStreamingWithTokens_AbortEstimatesWithoutDrain verifies the default
// (drain_upstream_on_abort=false) path: when client disconnects, token counts are
// estimated from delta text received before the abort — no upstream drain.
func TestHandleStreamingWithTokens_AbortEstimatesWithoutDrain(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"world"}}]}` + "\n\n",
			`data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	// drainUpstreamOnAbort = false (default)
	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	logCtx := &RequestLogContext{
		RequestID:            "abort-no-drain-test",
		PromptTokensEstimate: 15,
		Credential:           &config.CredentialConfig{Name: "test", Type: config.ProviderTypeOpenAI},
		TokenUsage:           &converter.TokenUsage{},
	}

	w := newFailAfterNBytesWriter(10)

	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	assert.Error(t, err)
	assert.True(t, logCtx.Logged, "must log even without drain")

	// Without drain no usage chunk arrives — tokens come from delta-text estimation
	assert.Greater(t, logCtx.TokenUsage.CompletionTokens, 0,
		"completion tokens must be counted from delta text (no drain)")
	// Real usage chunk NOT captured — prompt comes from PromptTokensEstimate
	assert.Equal(t, 15, logCtx.TokenUsage.PromptTokens,
		"prompt tokens must come from PromptTokensEstimate when drain is disabled")
	assert.Equal(t, "client_aborted", logCtx.StreamOutcome)
	assert.Equal(t, "failure", logCtx.Status)
	assert.Equal(t, "estimated", logCtx.UsageSource)

	t.Logf("No-drain abort: prompt=%d completion=%d (estimated)",
		logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens)
}

// TestHandleStreamingWithTokens_ProviderEOFWithoutUsage verifies that when the
// provider closes the connection without ever sending a usage chunk (EOF path,
// not a write error), completion tokens are estimated from delta text.
func TestHandleStreamingWithTokens_ProviderEOFWithoutUsage(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		// Only content chunks, no usage chunk — simulates provider closing early
		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"world"}}]}` + "\n\n",
			// No usage chunk, no [DONE]
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
		}
		// Handler returns — connection closes (EOF)
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	logCtx := &RequestLogContext{
		RequestID:            "eof-no-usage-test",
		PromptTokensEstimate: 10,
		Credential:           &config.CredentialConfig{Name: "test", Type: config.ProviderTypeOpenAI},
		TokenUsage:           &converter.TokenUsage{},
	}

	w := httptest.NewRecorder()
	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err, "EOF is not an error")

	assert.True(t, logCtx.Logged, "must be logged even without usage chunk")

	assert.Greater(t, logCtx.TokenUsage.CompletionTokens, 0,
		"completion tokens must be counted from delta text when no usage chunk")

	// Prompt tokens from estimate
	assert.Equal(t, 10, logCtx.TokenUsage.PromptTokens,
		"prompt tokens from PromptTokensEstimate when no usage chunk present")

	t.Logf("EOF-no-usage result: prompt=%d completion=%d",
		logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens)
}

// TestHandleStreamingWithTokens_NormalCompletion verifies that when the stream
// completes normally (usage chunk arrives), the real token counts win over
// the local tokenizer estimate. The usage and [DONE] lines are flushed together
// so they arrive in the same buffer read, ensuring lastChunk contains usage data.
func TestHandleStreamingWithTokens_NormalCompletion(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		// Content chunk flushed first
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Hi"}}]}`+"\n\n")
		flusher.Flush()

		// Usage + [DONE] flushed together — guarantees same buffer read so lastChunk
		// contains usage data and CompletionTokens is overridden from total_tokens.
		_, _ = fmt.Fprint(w,
			`data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":50,"completion_tokens":25,"total_tokens":75}}`+"\n\n"+
				"data: [DONE]\n\n",
		)
		flusher.Flush()
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	logCtx := &RequestLogContext{
		RequestID:            "normal-completion-test",
		PromptTokensEstimate: 1, // Tiny estimate; real values (50/25) must win
		Credential:           &config.CredentialConfig{Name: "test", Type: config.ProviderTypeOpenAI},
		TokenUsage:           &converter.TokenUsage{},
	}

	w := httptest.NewRecorder()
	err = prx.handleStreamingWithTokens(w, resp, "test", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	assert.True(t, logCtx.Logged)

	// Prompt tokens: real value from usage chunk (50) must beat estimate (1)
	assert.Equal(t, 50, logCtx.TokenUsage.PromptTokens,
		"real prompt tokens from usage chunk must override estimate")

	// Completion tokens: usage chunk has completion_tokens=25, total_tokens=75.
	// When usage+[DONE] arrive together, lastChunk contains usage so
	// completion_tokens (25) overrides total_tokens (75).
	assert.Equal(t, 25, logCtx.TokenUsage.CompletionTokens,
		"completion_tokens from usage chunk overrides total_tokens when lastChunk has usage data")
}

// TestHandleTransformedStreaming_AbortLogsTokens verifies the same behaviour
// for transformed streams (Vertex/Anthropic path via handleTransformedStreaming).
func TestHandleTransformedStreaming_AbortLogsTokens(t *testing.T) {
	// Upstream serves already-transformed OpenAI SSE (like tokenCapturingWriter produces)
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Some "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"response "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"text from Vertex"}}]}` + "\n\n",
			// Usage chunk comes last — client will disconnect before this
			`data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":30,"completion_tokens":15,"total_tokens":45}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	logCtx := &RequestLogContext{
		RequestID:            "transformed-abort-test",
		PromptTokensEstimate: 20,
		Credential:           &config.CredentialConfig{Name: "test", Type: config.ProviderTypeOpenAI},
		TokenUsage:           &converter.TokenUsage{},
	}

	// Fail after receiving first chunk (before usage chunk arrives)
	w := newFailAfterNBytesWriter(10)

	// Use passthrough transformer (identity) to exercise handleTransformedStreaming directly
	transformer := func(r io.Reader, id string, ww io.Writer) error {
		_, err := io.Copy(ww, r)
		return err
	}

	err = prx.handleTransformedStreaming(w, resp, "test", "gemini-2.5-flash", "Vertex AI", transformer, logCtx)
	assert.Error(t, err, "should return write error on client disconnect")

	assert.True(t, logCtx.Logged, "finalizeStreamingLog must be called on transform abort")

	assert.Greater(t, logCtx.TokenUsage.CompletionTokens, 0,
		"completion tokens must be estimated from delta text")

	assert.Greater(t, logCtx.TokenUsage.PromptTokens, 0,
		"prompt tokens must come from PromptTokensEstimate")

	t.Logf("Transformed abort result: prompt=%d completion=%d",
		logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens)
}

// TestExtractCompletionDeltaText_ChatCompletions verifies the delta text extractor.
func TestExtractCompletionDeltaText_ChatCompletions(t *testing.T) {
	tests := []struct {
		name     string
		chunk    []byte
		wantText string
	}{
		{
			name:     "single delta",
			chunk:    []byte(`data: {"choices":[{"delta":{"content":"Hello"}}]}` + "\n\n"),
			wantText: "Hello",
		},
		{
			name: "multiple SSE lines in chunk",
			chunk: []byte(
				`data: {"choices":[{"delta":{"content":"Hi "}}]}` + "\n\n" +
					`data: {"choices":[{"delta":{"content":"there"}}]}` + "\n\n",
			),
			wantText: "Hi there",
		},
		{
			name:     "no content field",
			chunk:    []byte(`data: {"choices":[{"finish_reason":"stop","delta":{}}]}` + "\n\n"),
			wantText: "",
		},
		{
			name:     "DONE marker",
			chunk:    []byte("data: [DONE]\n\n"),
			wantText: "",
		},
		{
			name:     "usage-only chunk",
			chunk:    []byte(`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}` + "\n\n"),
			wantText: "",
		},
		{
			name:     "empty chunk",
			chunk:    []byte(""),
			wantText: "",
		},
		{
			name:     "multi-word content",
			chunk:    []byte(`data: {"choices":[{"delta":{"content":"Hello world!"}}]}` + "\n\n"),
			wantText: "Hello world!",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractCompletionDeltaText(tt.chunk)
			assert.Equal(t, tt.wantText, got)
		})
	}
}

// TestWriteProxyStreamingResponseWithTokens_Abort verifies that
// writeProxyStreamingResponseWithTokens returns estimated usage when the stream
// is cut before the usage chunk arrives (proxy-type credential path, e.g. OpenRouter).
func TestWriteProxyStreamingResponseWithTokens_Abort(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"OpenRouter "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"reply text"}}]}` + "\n\n",
			// No usage chunk — simulates OpenRouter not sending usage before abort
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	proxyResp := &ProxyResponse{
		StatusCode:  200,
		Headers:     resp.Header,
		StreamBody:  resp.Body,
		IsStreaming: true,
	}

	// Client disconnects after 10 bytes
	w := newFailAfterNBytesWriter(10)
	logCtx := &RequestLogContext{}

	streamUsage, err := prx.writeProxyStreamingResponseWithTokens(w, proxyResp, &http.Request{Header: make(http.Header)}, "test", "test-model", "gpt-4o-mini", logCtx)
	assert.Error(t, err, "should return write error")

	// Even on abort, estimated usage must be returned
	require.NotNil(t, streamUsage, "writeProxyStreamingResponseWithTokens must return partial usage on abort")
	assert.Greater(t, streamUsage.CompletionTokens, 0,
		"completion tokens must be counted from 'OpenRouter reply text' delta text")
	assert.Equal(t, "client_aborted", logCtx.StreamOutcome)
	assert.Equal(t, "estimated", logCtx.UsageSource)

	t.Logf("Proxy abort estimated completion tokens: %d", streamUsage.CompletionTokens)
}

// TestWriteProxyStreamingResponseWithTokens_DrainCapturesUsage verifies that
// writeProxyStreamingResponseWithTokens drains the upstream after a client
// disconnect and returns the real usage from the provider's usage chunk.
// This covers the chain scenario: user→Router1→Router2→Provider where Router1
// keeps reading from Router2 even after the user drops, getting the real counts.
func TestWriteProxyStreamingResponseWithTokens_DrainCapturesUsage(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		// Content chunks, then a real usage chunk — client disconnects before usage
		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"world"}}]}` + "\n\n",
			// Usage arrives after abort — drain must capture this
			`data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		WithDrainUpstreamOnAbort(true). // drain enabled: expect real usage chunk
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	proxyResp := &ProxyResponse{
		StatusCode:  200,
		Headers:     resp.Header,
		StreamBody:  resp.Body,
		IsStreaming: true,
	}

	// Client disconnects after 10 bytes (before usage chunk)
	w := newFailAfterNBytesWriter(10)
	logCtx := &RequestLogContext{}

	streamUsage, err := prx.writeProxyStreamingResponseWithTokens(w, proxyResp, &http.Request{Header: make(http.Header)}, "test", "test-model", "gpt-4o-mini", logCtx)
	assert.Error(t, err, "should return write error")

	// Drain must have captured the real usage chunk
	require.NotNil(t, streamUsage, "must return usage after drain")
	assert.Equal(t, 7, streamUsage.CompletionTokens,
		"completion tokens must come from real usage chunk captured during drain")
	assert.Equal(t, 42, streamUsage.PromptTokens,
		"prompt tokens must come from real usage chunk captured during drain")
	assert.Equal(t, "client_aborted", logCtx.StreamOutcome)
	assert.Equal(t, "provider", logCtx.UsageSource)

	t.Logf("Drain captured usage: prompt=%d completion=%d",
		streamUsage.PromptTokens, streamUsage.CompletionTokens)
}

// TestWriteProxyStreamingResponseWithTokens_NoUsageChunk verifies estimation
// when the stream completes (EOF) without ever sending a usage chunk.
func TestWriteProxyStreamingResponseWithTokens_NoUsageChunk(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Response without usage"}}]}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	proxyResp := &ProxyResponse{
		StatusCode:  200,
		Headers:     resp.Header,
		StreamBody:  resp.Body,
		IsStreaming: true,
	}

	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{}
	streamUsage, err := prx.writeProxyStreamingResponseWithTokens(w, proxyResp, &http.Request{Header: make(http.Header)}, "test", "test-model", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	// Should return estimated usage from streamed text when no usage chunk is present.
	require.NotNil(t, streamUsage, "must return estimated usage when no usage chunk")
	assert.Greater(t, streamUsage.CompletionTokens, 0,
		"completion tokens estimated from delta text")
	assert.Equal(t, "completed", logCtx.StreamOutcome)
	assert.Equal(t, "estimated", logCtx.UsageSource)
	assert.Equal(t, "no", w.Header().Get(accelBufferingHeader))

	t.Logf("No-usage-chunk estimated completion tokens: %d", streamUsage.CompletionTokens)
}

func TestWriteProxyStreamingResponseWithTokens_DoesNotDisableBufferingForPreStreamError(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	proxyResp := &ProxyResponse{
		StatusCode:  http.StatusTooManyRequests,
		Headers:     http.Header{"Content-Type": {"application/json"}},
		StreamBody:  io.NopCloser(strings.NewReader(`{"error":{"message":"rate limited"}}`)),
		IsStreaming: true,
	}
	w := httptest.NewRecorder()

	_, err := prx.writeProxyStreamingResponseWithTokens(
		w,
		proxyResp,
		&http.Request{Header: make(http.Header)},
		"test",
		"test-model",
		"gpt-4o-mini",
		nil,
	)

	require.NoError(t, err)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)
	assert.Empty(t, w.Header().Get(accelBufferingHeader))
}
