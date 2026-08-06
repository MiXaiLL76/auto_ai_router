package proxy

import (
	"fmt"
	"testing"
)

// oldTTFTDetect reproduces the pre-fix streamToClient TTFT detection: append
// every chunk to a growing buffer and re-run extractCompletionDeltaText over
// the *entire* accumulated buffer after every chunk. Kept only in this
// benchmark to measure the O(n^2) cost the ttftScanState rewrite replaces.
func oldTTFTDetect(chunks [][]byte, limit int) bool {
	var buf []byte
	for _, c := range chunks {
		buf = append(buf, c...)
		if extractCompletionDeltaText(buf) != "" {
			return true
		}
		if len(buf) > limit {
			return false
		}
	}
	return false
}

func newTTFTDetect(chunks [][]byte, limit int) bool {
	var s ttftScanState
	for _, c := range chunks {
		if s.observe(c) {
			return true
		}
		if s.total > limit {
			return false
		}
	}
	return false
}

// sseChunks builds n content-free SSE keep-alive/ping frames (~180 bytes each,
// matching mock-go's real SSE frame size per pprof_plan_test.md) followed by
// one real content delta, or, if withContent is false, no content delta at
// all (worst case: scan runs until the detection cap gives up).
func sseChunks(n int, withContent bool) [][]byte {
	chunks := make([][]byte, 0, n+1)
	for i := 0; i < n; i++ {
		chunks = append(chunks, []byte(fmt.Sprintf(
			`data: {"id":"chatcmpl-%d","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":null}]}`+"\n\n",
			i)))
	}
	if withContent {
		chunks = append(chunks, []byte(`data: {"choices":[{"delta":{"content":"hi"}}]}`+"\n\n"))
	}
	return chunks
}

func BenchmarkTTFTDetect_Old(b *testing.B) {
	for _, n := range []int{10, 50, 200, 355} { // 355*~180B ~= 64KB cap
		b.Run(fmt.Sprintf("preamble=%d/match", n), func(b *testing.B) {
			chunks := sseChunks(n, true)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldTTFTDetect(chunks, streamTTFTDetectionLimit)
			}
		})
		b.Run(fmt.Sprintf("preamble=%d/nomatch-cap", n), func(b *testing.B) {
			chunks := sseChunks(n, false)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldTTFTDetect(chunks, streamTTFTDetectionLimit)
			}
		})
	}
}

func BenchmarkTTFTDetect_New(b *testing.B) {
	for _, n := range []int{10, 50, 200, 355} {
		b.Run(fmt.Sprintf("preamble=%d/match", n), func(b *testing.B) {
			chunks := sseChunks(n, true)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				newTTFTDetect(chunks, streamTTFTDetectionLimit)
			}
		})
		b.Run(fmt.Sprintf("preamble=%d/nomatch-cap", n), func(b *testing.B) {
			chunks := sseChunks(n, false)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				newTTFTDetect(chunks, streamTTFTDetectionLimit)
			}
		})
	}
}
