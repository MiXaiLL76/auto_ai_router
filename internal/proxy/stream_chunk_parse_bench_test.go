package proxy

import (
	"bytes"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// --- Frozen pre-fix (plan items C/D/E) baselines ---
//
// These are exact copies of the code as it existed before this change, kept
// only so the benchmarks below measure this fix's own before/after numbers
// against a fixed target instead of a moving one.

// oldSplitSSEPayloads reproduces the original extractJSONPayloadsFromStreamChunk:
// string(chunk) copy, strings.Split, and a fresh []byte(payload) allocation
// per SSE line.
func oldSplitSSEPayloads(chunk []byte) [][]byte {
	trimmed := strings.TrimSpace(string(chunk))
	if trimmed == "" {
		return nil
	}
	if !strings.Contains(trimmed, "data:") {
		return [][]byte{[]byte(trimmed)}
	}
	lines := strings.Split(trimmed, "\n")
	payloads := make([][]byte, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		payloads = append(payloads, []byte(payload))
	}
	return payloads
}

// oldExtractTokensFromStreamingChunk reproduces the original
// extractTokensFromStreamingChunk(chunk string) int: its own independent
// string(chunk) + strings.Split, unconditionally (no "usage" prefilter).
func oldExtractTokensFromStreamingChunk(chunk string) int {
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}
			tokens := extractOpenAITotalTokens([]byte(jsonData))
			if tokens > 0 {
				return tokens
			}
		}
	}
	return 0
}

// oldExtractTokenUsageFromStreamingChunkWithOptions reproduces the original
// extractTokenUsageFromStreamingChunkWithOptions(chunk string, ...): a third
// independent string(chunk) + strings.Split + full ~60-field unmarshal
// attempt, unconditionally (no "usage" prefilter).
func oldExtractTokenUsageFromStreamingChunkWithOptions(chunk string, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}
			if usage := converter.ExtractTokenUsageWithOptions([]byte(jsonData), opts); usage != nil {
				return usage
			}
		}
	}
	return nil
}

// oldRememberLastStreamDataChunk reproduces the original
// rememberLastStreamDataChunk: a string(chunk) copy just to compare against
// ""/"[DONE]", then a fresh append([]byte(nil), chunk...) allocation.
func oldRememberLastStreamDataChunk(dst *[]byte, chunk []byte) {
	trimmed := strings.TrimSpace(string(chunk))
	if trimmed == "" || trimmed == "data: [DONE]" || trimmed == "[DONE]" {
		return
	}
	*dst = append([]byte(nil), chunk...)
}

// oldOnChunkPipeline reproduces the full pre-fix onChunk sequence from
// handleStreamingWithTokens: three independent copy+split+unmarshal passes
// over the same chunk (usage, total_tokens, completion delta text) plus a
// fourth copy for rememberLastStreamDataChunk. This is exactly the
// "~4 copies, 3 splits, 2-3 unmarshals" pattern pprof_fix_plan.md's item C
// describes.
func oldOnChunkPipeline(chunk []byte, totalTokens *int, text *strings.Builder, lastChunk *[]byte) {
	if usage := oldExtractTokenUsageFromStreamingChunkWithOptions(string(chunk), converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}); usage != nil {
		_ = usage
	}
	if tokens := oldExtractTokensFromStreamingChunk(string(chunk)); tokens > 0 {
		*totalTokens += tokens
	}
	for _, payload := range oldSplitSSEPayloads(chunk) {
		appendDeltaTextForPayload(text, payload)
	}
	oldRememberLastStreamDataChunk(lastChunk, chunk)
}

// newOnChunkPipeline reproduces the post-fix handleStreamingWithTokens.onChunk
// sequence: one zero-copy split (buffer reused via payloadBuf), a byte-level
// "usage" prefilter gating the usage/total_tokens unmarshal attempts, and a
// buffer-reusing rememberLastStreamDataChunk.
func newOnChunkPipeline(chunk []byte, payloadBuf *[][]byte, totalTokens *int, acc *completionTokenAccumulator, lastChunk *[]byte) {
	*payloadBuf = splitSSEPayloads(chunk, *payloadBuf)
	if bytes.Contains(chunk, sseUsageNeedle) {
		if usage := extractTokenUsageFromPayloads(*payloadBuf, converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}); usage != nil {
			_ = usage
		}
		if tokens := extractTokensFromPayloads(*payloadBuf); tokens > 0 {
			*totalTokens += tokens
		}
	}
	acc.AddPayloads(*payloadBuf)
	rememberLastStreamDataChunk(lastChunk, chunk)
}

// --- Realistic mock-go chunk fixtures ---
//
// Byte-for-byte shaped like mock-go's actual streamed frames (mock-go/main.go
// buildChat/sse): a single "token " content delta frame is the unit mock-go
// writes per trickle tick (~180 bytes per pprof_plan_test.md); the batched
// fixture concatenates 10 of them, matching --tokens-per-flush 10 (~1.8KB per
// Read, also per pprof_plan_test.md).

var (
	benchContentChunk = []byte(`data: {"id":"chatcmpl-stub","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"token "},"finish_reason":null}]}` + "\n\n")

	benchUsageChunk = []byte(`data: {"id":"chatcmpl-stub","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5632,"completion_tokens":563,"total_tokens":6195}}` + "\n\n")

	benchDoneChunk = []byte("data: [DONE]\n\n")

	benchBatchedChunk = buildBenchBatchedChunk(10)
)

func buildBenchBatchedChunk(n int) []byte {
	var buf bytes.Buffer
	for i := 0; i < n; i++ {
		buf.Write(benchContentChunk)
	}
	return buf.Bytes()
}

func init() {
	// Sanity-check the fixtures actually match the sizes documented in
	// pprof_plan_test.md (~180B single frame, ~1.8KB batched read) so a
	// future edit to these literals doesn't silently drift from what's
	// claimed in the benchmark comments/report.
	if len(benchContentChunk) < 150 || len(benchContentChunk) > 220 {
		panic("benchContentChunk size drifted from the ~180B mock-go frame shape")
	}
	if len(benchBatchedChunk) < 1500 || len(benchBatchedChunk) > 2100 {
		panic("benchBatchedChunk size drifted from the ~1.8KB batched-read shape")
	}
}

// --- C: splitSSEPayloads vs the old extractJSONPayloadsFromStreamChunk ---

func BenchmarkSplitSSEPayloads_Old(b *testing.B) {
	cases := map[string][]byte{
		"content-180B":  benchContentChunk,
		"usage-chunk":   benchUsageChunk,
		"batched-1.8KB": benchBatchedChunk,
		"done-sentinel": benchDoneChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = oldSplitSSEPayloads(chunk)
			}
		})
	}
}

func BenchmarkSplitSSEPayloads_New(b *testing.B) {
	cases := map[string][]byte{
		"content-180B":  benchContentChunk,
		"usage-chunk":   benchUsageChunk,
		"batched-1.8KB": benchBatchedChunk,
		"done-sentinel": benchDoneChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			var dst [][]byte
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst = splitSSEPayloads(chunk, dst)
			}
		})
	}
}

// --- D: byte-level "usage" prefilter, isolated ---

func BenchmarkExtractTokensFromStreamingChunk_Old(b *testing.B) {
	cases := map[string]string{
		"content-no-usage": string(benchContentChunk),
		"usage-chunk":      string(benchUsageChunk),
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldExtractTokensFromStreamingChunk(chunk)
			}
		})
	}
}

func BenchmarkExtractTokensFromStreamingChunk_New(b *testing.B) {
	cases := map[string][]byte{
		"content-no-usage": benchContentChunk,
		"usage-chunk":      benchUsageChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Mirrors the onChunk-level prefilter: skip the call entirely
				// when the chunk can't possibly contain usage/total_tokens.
				if bytes.Contains(chunk, sseUsageNeedle) {
					extractTokensFromStreamingChunk(chunk)
				}
			}
		})
	}
}

func BenchmarkExtractTokenUsageFromStreamingChunkWithOptions_Old(b *testing.B) {
	opts := converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
	cases := map[string]string{
		"content-no-usage": string(benchContentChunk),
		"usage-chunk":      string(benchUsageChunk),
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldExtractTokenUsageFromStreamingChunkWithOptions(chunk, opts)
			}
		})
	}
}

func BenchmarkExtractTokenUsageFromStreamingChunkWithOptions_New(b *testing.B) {
	opts := converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
	cases := map[string][]byte{
		"content-no-usage": benchContentChunk,
		"usage-chunk":      benchUsageChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if bytes.Contains(chunk, sseUsageNeedle) {
					extractTokenUsageFromStreamingChunkWithOptions(chunk, opts)
				}
			}
		})
	}
}

// --- E: rememberLastStreamDataChunk buffer reuse ---

func BenchmarkRememberLastStreamDataChunk_Old(b *testing.B) {
	cases := map[string][]byte{
		"content-180B":  benchContentChunk,
		"batched-1.8KB": benchBatchedChunk,
		"done-sentinel": benchDoneChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			var dst []byte
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldRememberLastStreamDataChunk(&dst, chunk)
			}
		})
	}
}

func BenchmarkRememberLastStreamDataChunk_New(b *testing.B) {
	cases := map[string][]byte{
		"content-180B":  benchContentChunk,
		"batched-1.8KB": benchBatchedChunk,
		"done-sentinel": benchDoneChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			var dst []byte
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rememberLastStreamDataChunk(&dst, chunk)
			}
		})
	}
}

// --- C+D+E combined: the full onChunk pipeline, old vs new ---
//
// This is the end-to-end number: pprof_fix_plan.md's item C table lists five
// per-chunk operations (usage extraction, total_tokens extraction, completion
// delta-text extraction, rememberLastStreamDataChunk, plus the already-fixed
// incremental error observer). These benchmarks reproduce the first four for
// both a content-only chunk (the overwhelmingly common case — usage only ever
// arrives in the final chunk) and the terminal usage chunk.

func BenchmarkOnChunkPipeline_Old(b *testing.B) {
	cases := map[string][]byte{
		"content-only (common case)": benchContentChunk,
		"final usage chunk":          benchUsageChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			var totalTokens int
			var text strings.Builder
			var lastChunk []byte
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				oldOnChunkPipeline(chunk, &totalTokens, &text, &lastChunk)
			}
		})
	}
}

func BenchmarkOnChunkPipeline_New(b *testing.B) {
	cases := map[string][]byte{
		"content-only (common case)": benchContentChunk,
		"final usage chunk":          benchUsageChunk,
	}
	for name, chunk := range cases {
		b.Run(name, func(b *testing.B) {
			var totalTokens int
			acc := &completionTokenAccumulator{}
			var lastChunk []byte
			var payloadBuf [][]byte
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				newOnChunkPipeline(chunk, &payloadBuf, &totalTokens, acc, &lastChunk)
			}
		})
	}
}
