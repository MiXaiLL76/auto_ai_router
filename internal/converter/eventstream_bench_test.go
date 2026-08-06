package converter

import (
	"encoding/base64"
	stdjson "encoding/json"
	"testing"

	goccyjson "github.com/goccy/go-json"
)

// eventStreamPayload mirrors DecodeEventStreamToSSE's expected shape: a JSON
// envelope whose only meaningful field is a base64 blob of the actual
// Anthropic event (a typical content_block_delta, per the same shape used in
// internal/converter/anthropic's benchmarks).
var eventStreamPayload = func() []byte {
	inner := []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello world, this is a streamed token"}}`)
	encoded := base64.StdEncoding.EncodeToString(inner)
	return []byte(`{"bytes":"` + encoded + `","p":"AAAAAAAAAAAAAAAAAAAAAAAAAA=="}`)
}()

func BenchmarkEventStreamEnvelopeUnmarshal_Stdlib(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var envelope struct {
			Bytes string `json:"bytes"`
		}
		_ = stdjson.Unmarshal(eventStreamPayload, &envelope)
	}
}

func BenchmarkEventStreamEnvelopeUnmarshal_Goccy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var envelope struct {
			Bytes string `json:"bytes"`
		}
		_ = goccyjson.Unmarshal(eventStreamPayload, &envelope)
	}
}
