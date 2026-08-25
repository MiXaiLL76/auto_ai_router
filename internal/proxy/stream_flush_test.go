package proxy

import (
	"context"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// flushTrackingWriter separates "written" from "flushed" bytes, unlike
// httptest.ResponseRecorder (whose Write goes straight into its Body buffer
// with no distinct buffering layer, so it can't detect a skipped Flush()).
// Flush() snapshots everything written so far as flushed; bytes written after
// the last Flush() call are the "unflushed tail" a test can assert on. Guarded
// by a mutex since tests sample written/flushed lengths from a separate
// goroutine while streamToClient is still running.
type flushTrackingWriter struct {
	mu         sync.Mutex
	header     http.Header
	written    []byte
	flushed    []byte
	flushCalls int
	writeCalls int
}

func (w *flushTrackingWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *flushTrackingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writeCalls++
	w.written = append(w.written, p...)
	return len(p), nil
}

func (w *flushTrackingWriter) WriteHeader(int) {}

func (w *flushTrackingWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.flushCalls++
	w.flushed = append(w.flushed[:0:0], w.written...)
}

func (w *flushTrackingWriter) snapshot() (written, flushed int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.written), len(w.flushed)
}

// TestStreamToClient_FlushesTailOnEOF is a regression test for the
// flush-coalescing bug: the flush-decision block used to live entirely
// inside "if n > 0", so a terminal Read() returning (0, io.EOF) — the normal
// end of a chunked HTTP body — never triggered a flush, leaving any bytes
// written just before EOF, inside the coalescing window, stuck unflushed.
func TestStreamToClient_FlushesTailOnEOF(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := &flushTrackingWriter{}

	// Two data chunks ~1ms apart (well inside streamFlushCoalesceWindow),
	// then a bare (0, io.EOF) terminal read — the second write's flush would
	// have been skipped for coalescing, and nothing else ever forces it out.
	reader := &timedChunkReader{
		chunks: [][]byte{
			[]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n"),
			[]byte(`data: {"choices":[{"delta":{"content":" world"}}]}` + "\n\n"),
		},
		delay: time.Millisecond,
	}

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, nil)
	require.NoError(t, err)

	written, flushed := w.snapshot()
	require.Equal(t, written, flushed,
		"all written bytes must be flushed by the time streamToClient returns; got %d written, %d flushed (%d flush calls)",
		written, flushed, w.flushCalls)
	assert.Equal(t, string(w.written), string(w.flushed))
}

// TestStreamToClient_FlushesTailOnMidStreamPause documents a KNOWN, NOT-YET-FIXED
// limitation (deliberately left out of scope — see the flushPending fix's own
// comment in streamToClient): a chunk written right before a live upstream
// pause (reasoning preamble, tool-call boundary, provider hiccup) can still
// sit unflushed for the whole pause, because streamToClient is blocked
// synchronously in reader.Read() and nothing forces a flush until the next
// Read() call returns. The flushPending fix only guarantees a flush on loop
// EXIT (stream ended / errored) — it fixed the confirmed "tail lost forever"
// bug (see TestStreamToClient_FlushesTailOnEOF) but not this live-pause case,
// which would need a timer-driven background flush with its own
// synchronization against concurrent Write() — a materially bigger, riskier
// change intentionally deferred pending a product decision.
func TestStreamToClient_FlushesTailOnMidStreamPause(t *testing.T) {
	t.Skip("known limitation, not fixed by the flushPending change — see comment above; needs a timer-driven background flush")
	prx := NewTestProxyBuilder().Build()
	w := &flushTrackingWriter{}

	reader := &timedChunkReader{
		chunks: [][]byte{
			// Chunk 0 always flushes immediately (lastFlush.IsZero() covers the
			// very first write even with the bug) — not the interesting one.
			[]byte(`data: {"choices":[{"delta":{"content":"first"}}]}` + "\n\n"),
			// Chunk 1, ~1ms after chunk 0 (inside the coalescing window): under
			// the bug this flush gets skipped and nothing forces it out until
			// chunk 2 arrives — this is the one the assertion targets.
			[]byte(`data: {"choices":[{"delta":{"content":"second"}}]}` + "\n\n"),
			// Chunk 2 arrives only after a long pause.
			[]byte(`data: {"choices":[{"delta":{"content":"third"}}]}` + "\n\n"),
		},
		delay:      time.Millisecond,
		pauseAfter: 2, // pause before reading chunk index 2 ("third")
		pause:      300 * time.Millisecond,
	}

	done := make(chan error, 1)
	go func() {
		done <- prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusOK, nil, nil, nil)
	}()

	// Sample mid-pause, well after "first" was written (~1ms in) but well
	// before "second" arrives (300ms in) and long before EOF.
	time.Sleep(50 * time.Millisecond)
	writtenMidPause, flushedMidPause := w.snapshot()

	require.NoError(t, <-done)

	require.Greater(t, writtenMidPause, 0, "test setup: \"first\" should already be written by the sample point")
	// This is the actual regression signal: with the bug, "first"'s flush can
	// be skipped for coalescing and nothing forces it out until "second"
	// arrives 300ms later — the fix guarantees it's flushed within the
	// coalescing window instead of held for the whole pause.
	assert.Equal(t, writtenMidPause, flushedMidPause,
		"a chunk written before a pause must be flushed within the coalescing window, not held back until the next chunk arrives")
}

// timedChunkReader is like chunkReader in stream_ttft_test.go but sleeps
// before every chunk (including the first), to simulate real inter-chunk
// timing rather than an instant back-to-back sequence. pauseAfter/pause
// optionally insert one longer sleep before a specific chunk index, to
// simulate an upstream mid-stream stall.
type timedChunkReader struct {
	chunks     [][]byte
	delay      time.Duration
	pauseAfter int // chunk index before which the long pause is inserted; -1 = none
	pause      time.Duration
	pos        int
}

func (r *timedChunkReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.chunks) {
		return 0, io.EOF
	}
	if r.pause > 0 && r.pos == r.pauseAfter {
		time.Sleep(r.pause)
	} else if r.delay > 0 {
		time.Sleep(r.delay)
	}
	n := copy(p, r.chunks[r.pos])
	r.pos++
	return n, nil
}
