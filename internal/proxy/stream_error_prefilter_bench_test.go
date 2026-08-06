package proxy

import "testing"

// --- Frozen pre-fix baseline ---
//
// oldObserveFrame reproduces proxyStreamErrorCapture.Observe's per-frame body
// before the byte-level prefilter was added: extractStreamErrorEvent runs
// unconditionally on every assembled frame.
func oldObserveFrame(frame []byte) string {
	return extractStreamErrorEvent(frame)
}

// newObserveFrame is the fixed version: frameMayCarryStreamError gates the
// call, skipping the parse entirely for a frame that can't possibly be an
// error event.
func newObserveFrame(frame []byte) string {
	if !frameMayCarryStreamError(frame) {
		return ""
	}
	return extractStreamErrorEvent(frame)
}

// A realistic ~180-byte content-only SSE frame (matches this plan's
// established mock-go per-frame shape) — the overwhelming majority of frames
// in a real stream, and the case the prefilter is meant to skip entirely.
var errorPrefilterBenchContentFrame = []byte(`data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"the quick brown fox jumps"},"finish_reason":null}]}` + "\n\n")

// A real error-event frame, to confirm the prefilter doesn't change the
// (much rarer) positive-match cost meaningfully.
var errorPrefilterBenchErrorFrame = []byte(`data: {"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}` + "\n\n")

func BenchmarkObserveErrorDetection_Old_ContentFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		oldObserveFrame(errorPrefilterBenchContentFrame)
	}
}

func BenchmarkObserveErrorDetection_New_ContentFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		newObserveFrame(errorPrefilterBenchContentFrame)
	}
}

func BenchmarkObserveErrorDetection_Old_ErrorFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		oldObserveFrame(errorPrefilterBenchErrorFrame)
	}
}

func BenchmarkObserveErrorDetection_New_ErrorFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		newObserveFrame(errorPrefilterBenchErrorFrame)
	}
}
