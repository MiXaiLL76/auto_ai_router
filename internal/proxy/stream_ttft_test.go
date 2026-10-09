package proxy

import (
	"bytes"
	"context"
	"io"
	"net/http"
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

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, logCtx)
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

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, logCtx)
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

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, logCtx)
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
		err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, nil)
		require.NoError(t, err)
	})
}

// TestTTFTScanState_ResponsesEchoBytesNotChargedToLimit verifies that the
// response.created / response.in_progress metadata echoes — which repeat the
// full request, instructions and tools — do not consume the
// streamTTFTDetectionLimit byte budget, so a long agent instruction cannot
// exhaust the budget before the first real content delta exists.
func TestTTFTScanState_ResponsesEchoBytesNotChargedToLimit(t *testing.T) {
	var s ttftScanState

	// Long agent instruction, well beyond the 64KB detection limit.
	createdLine := []byte(`data: {"type":"response.created","response":{"instructions":"`)
	createdLine = append(createdLine, bytes.Repeat([]byte("long-instruction-segment-"), 4096)...)
	createdLine = append(createdLine, []byte(`","tools":[]}}`)...)
	createdLine = append(createdLine, '\n')
	require.Greater(t, len(createdLine), streamTTFTDetectionLimit,
		"metadata echo must exceed the detection limit for this test to be meaningful")

	buf := createdLine
	for len(buf) > 0 {
		n := min(len(buf), 8192) // feed in 8KB pieces, as reader.Read would
		assert.False(t, s.observe(buf[:n]), "metadata echo must not be mistaken for content")
		buf = buf[n:]
	}
	assert.Zero(t, s.total, "response.created echo bytes must not count toward the TTFT limit")

	delta := []byte(`data: {"type":"response.output_text.delta","delta":"hi"}` + "\n")
	assert.True(t, s.observe(delta), "first real content delta must be detected after the echo")
}

// TestTTFTScanState_ContentFreeLinesStillCountTowardLimit guards that the
// Responses metadata exemption did not disable the byte budget for other
// content-free streams: chat keep-alive deltas and SSE comments must still
// count, so error-only / keep-alive-only streams keep bailing out at the cap.
func TestTTFTScanState_ContentFreeLinesStillCountTowardLimit(t *testing.T) {
	var s ttftScanState

	s.observe([]byte(`data: {"choices":[{"delta":{"role":"assistant"}}]}` + "\n"))
	require.Positive(t, s.total, "content-free chat deltas must still count toward the limit")

	s.observe([]byte(": ping\n"))
	require.Positive(t, s.total, "SSE comment lines must still count toward the limit")
}

// TestStreamToClient_CapturesTTFT_ResponsesLongInstruction reproduces the
// reported gap: a Responses API stream whose response.created / in_progress
// events echo an instruction larger than streamTTFTDetectionLimit. TTFT must
// be stamped on the first real content delta even though the metadata exceeded
// the former byte budget on its own.
func TestStreamToClient_CapturesTTFT_ResponsesLongInstruction(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := httptest.NewRecorder()

	start := time.Now()
	logCtx := &RequestLogContext{StartTime: start}
	echo := strings.Repeat("long-instruction-segment-", 4096) // > 64KB
	stream := "data: {\"type\":\"response.created\",\"response\":{\"instructions\":\"" + echo +
		"\",\"tools\":[]}}\n\n" +
		"data: {\"type\":\"response.in_progress\",\"response\":{}}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"
	reader := strings.NewReader(stream)

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/responses", http.StatusOK, nil, nil, logCtx)
	require.NoError(t, err)

	require.False(t, logCtx.CompletionStartTime.IsZero(),
		"TTFT must be stamped for a Responses stream whose metadata echo exceeds the 64KB detection limit")
	assert.False(t, logCtx.CompletionStartTime.Before(start), "TTFT timestamp should be after the request start time")
}
