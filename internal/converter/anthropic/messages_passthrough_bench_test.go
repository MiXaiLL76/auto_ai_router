package anthropic

import (
	"fmt"
	"io"
	"strings"
	"testing"
)

// Exercise full streams, including the model/usage envelopes and their refusal
// boundary, rather than timing the JSON helper in isolation.
func BenchmarkPassthroughMessagesLongStream(b *testing.B) {
	for _, tc := range []struct {
		name, delta string
		count       int
	}{
		{"text", `{"type":"text_delta","text":"ordinary text with a quote: \" and newline\n"}`, 2048},
		{"tool", `{"type":"input_json_delta","partial_json":"` + strings.Repeat("x", 4096) + `"}`, 256},
	} {
		var stream strings.Builder
		stream.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"upstream-model\",\"usage\":{\"input_tokens\":123}}}\n\n")
		for range tc.count {
			fmt.Fprintf(&stream, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":%s}\n\n", tc.delta)
		}
		stream.WriteString("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"category\":\"cyber\",\"explanation\":\"private-fixture\"}},\"usage\":{\"output_tokens\":42}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		body := stream.String()
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				if err := PassthroughMessagesStream(strings.NewReader(body), io.Discard, "claude-public"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
