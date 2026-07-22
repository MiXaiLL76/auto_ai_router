package proxy

import (
	"context"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// chunkReader returns each element of chunks on a separate Read call, to
// simulate content arriving across multiple separate network reads instead of
// one single Read returning the whole stream at once (as strings.Reader would
// for a short string). An optional delay is slept before returning any chunk
// after the first, so tests can assert CompletionStartTime reflects which
// chunk actually carried the timestamp.
type chunkReader struct {
	chunks [][]byte
	delay  time.Duration
	pos    int
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.chunks) {
		return 0, io.EOF
	}
	if r.pos > 0 && r.delay > 0 {
		time.Sleep(r.delay)
	}
	n := copy(p, r.chunks[r.pos])
	r.pos++
	return n, nil
}

// TestStreamToClient_CapturesTTFT verifies that streamToClient sets
// RequestLogContext.CompletionStartTime once a real content delta is read —
// this is the TTFT capture point shared by every streaming handler.
func TestStreamToClient_CapturesTTFT(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()

	logCtx := &RequestLogContext{StartTime: time.Now().Add(-50 * time.Millisecond)}
	reader := strings.NewReader(
		`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n" +
			`data: {"choices":[{"delta":{"content":" world"}}]}` + "\n\n",
	)

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", nil, nil, logCtx)
	require.NoError(t, err)

	assert.False(t, logCtx.CompletionStartTime.IsZero(), "CompletionStartTime should be set after a real content delta")
	assert.True(t, logCtx.CompletionStartTime.After(logCtx.StartTime), "TTFT timestamp should be after the request start time")
}

// TestStreamToClient_TTFTIgnoresContentFreeDeltas verifies that role-only
// deltas and SSE ping/comment lines arriving before the first real content
// delta do not get mistaken for TTFT — this is the TTFB-vs-TTFT distinction
// from the review: the first non-empty Read is not necessarily the first
// content token.
func TestStreamToClient_TTFTIgnoresContentFreeDeltas(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()

	start := time.Now()
	logCtx := &RequestLogContext{StartTime: start}
	reader := &chunkReader{
		delay: 20 * time.Millisecond,
		chunks: [][]byte{
			[]byte(`data: {"choices":[{"delta":{"role":"assistant"}}]}` + "\n\n"),
			[]byte(": ping\n\n"),
			[]byte(`data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n\n"),
		},
	}

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", nil, nil, logCtx)
	require.NoError(t, err)

	require.False(t, logCtx.CompletionStartTime.IsZero(), "CompletionStartTime should be set once real content arrives")
	assert.GreaterOrEqual(t, logCtx.CompletionStartTime.Sub(start), 15*time.Millisecond,
		"CompletionStartTime should be captured on the content delta, not the earlier role-only/ping chunks")
}

// TestStreamToClient_TTFTNotOverwrittenOnSubsequentChunks verifies the
// IsZero() guard: once CompletionStartTime is set, later chunks (or repeat
// calls, as happens on fallback/retry paths) must not overwrite it.
func TestStreamToClient_TTFTNotOverwrittenOnSubsequentChunks(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()

	preset := time.Now().Add(-time.Hour)
	logCtx := &RequestLogContext{StartTime: preset.Add(-time.Minute), CompletionStartTime: preset}
	reader := strings.NewReader(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n")

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", nil, nil, logCtx)
	require.NoError(t, err)

	assert.Equal(t, preset, logCtx.CompletionStartTime, "an already-set CompletionStartTime must not be overwritten")
}

// TestStreamToClient_NilLogCtx verifies streaming still works when no
// RequestLogContext is supplied (e.g. raw proxy passthrough paths that don't
// track TTFT).
func TestStreamToClient_NilLogCtx(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()
	reader := strings.NewReader("data: hello\n\n")

	assert.NotPanics(t, func() {
		err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", nil, nil, nil)
		require.NoError(t, err)
	})
}
