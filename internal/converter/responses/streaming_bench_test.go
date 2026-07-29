package responses

import (
	stdjson "encoding/json"
	"testing"

	goccyjson "github.com/goccy/go-json"
)

var chatStreamChunkPayload = []byte(`{"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hello world, this is a streamed token"},"finish_reason":null}]}`)

func BenchmarkChatStreamChunkUnmarshal_Stdlib(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var chunk chatStreamChunk
		_ = stdjson.Unmarshal(chatStreamChunkPayload, &chunk)
	}
}

func BenchmarkChatStreamChunkUnmarshal_Goccy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var chunk chatStreamChunk
		_ = goccyjson.Unmarshal(chatStreamChunkPayload, &chunk)
	}
}

// writeSSEDataSample is representative of what writeSSEWithSeq marshals on
// every emitted Responses API SSE event: a dynamic map[string]interface{},
// not a typed struct.
func writeSSEDataSample() map[string]interface{} {
	return map[string]interface{}{
		"type":            "response.output_text.delta",
		"item_id":         "item_abc123",
		"output_index":    0,
		"content_index":   0,
		"delta":           "hello world, this is a streamed token",
		"sequence_number": 42,
	}
}

func BenchmarkWriteSSEMarshal_Stdlib(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = stdjson.Marshal(writeSSEDataSample())
	}
}

func BenchmarkWriteSSEMarshal_Goccy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = goccyjson.Marshal(writeSSEDataSample())
	}
}
