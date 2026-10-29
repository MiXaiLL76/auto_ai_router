package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// handlePassthroughResponsesStreaming used to reassemble SSE lines by string
// concatenation, recopying and rescanning the pending line on every 8 KB
// read, and to decode the terminal response.completed event five times: a
// typed decode of the whole Response (run on every data: line, deltas
// included), ExtractTokenUsageWithOptions, the chat and Responses passes of
// the stream usage extractor in finalizeStreamingLog and extractStreamErrorEvent.
// Now lines are split in linear time, only a line naming a terminal type is
// decoded, and the terminal event is decoded once (plus
// ExtractTokenUsageWithOptions for the usage extensions, and the whole
// Response only when it is stored).
//
// Measured on the whole handler, before -> after:
//
//	codex_100KB:  ~1.9 ms, 5.1 MB/op  -> ~1.5 ms, 3.8 MB/op
//	image_2MB:    ~142 ms, 570 MB/op  -> ~39 ms, 308 MB/op
//
// Most of what image_2MB still allocates is proxyStreamErrorCapture: past
// 256 KB of an unfinished frame it copies that tail on every read of the raw
// provider stream.

// oldSSELineAccumulate reproduces the pre-fix line reassembly of
// handlePassthroughResponsesStreaming. Kept only in this benchmark.
func oldSSELineAccumulate(partial *string, chunk []byte, onLine func(string)) {
	combined := *partial + string(chunk)
	*partial = ""
	lastNL := strings.LastIndex(combined, "\n")
	if lastNL < 0 {
		*partial = combined
		return
	}
	if lastNL < len(combined)-1 {
		*partial = combined[lastNL+1:]
	}
	for _, line := range strings.Split(combined[:lastNL+1], "\n") {
		onLine(strings.TrimRight(line, "\r"))
	}
}

// buildPassthroughTerminalStream builds a passthrough Responses stream whose
// terminal event carries reasoningBytes of encrypted reasoning and an
// imageBytes-long generated image.
func buildPassthroughTerminalStream(reasoningBytes, imageBytes int) []byte {
	var b strings.Builder
	b.WriteString("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
	for i := 0; i < 200; i++ {
		b.WriteString("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":1,\"content_index\":0,\"delta\":\"token \"}\n\n")
	}
	b.WriteString(`event: response.completed` + "\n" + `data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.1-codex","output":[`)
	b.WriteString(`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"` + strings.Repeat("A", reasoningBytes) + `"},`)
	if imageBytes > 0 {
		b.WriteString(`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"` + strings.Repeat("B", imageBytes) + `"},`)
	}
	b.WriteString(`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"` + strings.Repeat("token ", 200) + `"}]}],`)
	b.WriteString(`"usage":{"input_tokens":7168,"output_tokens":717,"total_tokens":7885,"input_tokens_details":{"cached_tokens":1024},"output_tokens_details":{"reasoning_tokens":300}}}}` + "\n\ndata: [DONE]\n\n")
	return []byte(b.String())
}

var passthroughTerminalStreams = []struct {
	name   string
	stream []byte
}{
	{"codex_100KB", buildPassthroughTerminalStream(100_000, 0)},
	{"image_2MB", buildPassthroughTerminalStream(50_000, 2_000_000)},
}

func benchmarkLineSplit(b *testing.B, feed func(chunk []byte)) {
	for _, s := range passthroughTerminalStreams {
		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for off := 0; off < len(s.stream); off += 8192 {
					feed(s.stream[off:min(off+8192, len(s.stream))])
				}
			}
		})
	}
}

func BenchmarkPassthroughLineSplit_Old(b *testing.B) {
	var partial string
	benchmarkLineSplit(b, func(chunk []byte) { oldSSELineAccumulate(&partial, chunk, func(string) {}) })
}

func BenchmarkPassthroughLineSplit_New(b *testing.B) {
	lines := newResponsesStreamLines()
	benchmarkLineSplit(b, func(chunk []byte) { lines.Observe(chunk, func([]byte) {}) })
}

type discardFlushWriter struct{ header http.Header }

func (w *discardFlushWriter) Header() http.Header         { return w.header }
func (w *discardFlushWriter) Write(p []byte) (int, error) { return len(p), nil }
func (w *discardFlushWriter) WriteHeader(int)             {}
func (w *discardFlushWriter) Flush()                      {}

func BenchmarkPassthroughResponsesStreaming_TerminalEvent(b *testing.B) {
	prx := NewTestProxyBuilder().Build()
	for _, s := range passthroughTerminalStreams {
		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{}`))
				logCtx := &RequestLogContext{Request: req, StartTime: time.Now()}
				resp := &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       io.NopCloser(bytes.NewReader(s.stream)),
				}
				if err := prx.handlePassthroughResponsesStreaming(&discardFlushWriter{header: http.Header{}}, resp, "openai", "gpt-5.1-codex", logCtx, nil); err != nil {
					b.Fatal(err)
				}
				if logCtx.TokenUsage == nil || logCtx.TokenUsage.PromptTokens != 7168 {
					b.Fatalf("usage not extracted: %+v", logCtx.TokenUsage)
				}
			}
		})
	}
}
