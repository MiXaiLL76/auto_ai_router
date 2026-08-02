package vertex

import (
	stdjson "encoding/json"
	"reflect"
	"testing"

	goccyjson "github.com/goccy/go-json"
)

// vertexStreamingChunkPayload exercises the fields most likely to trip up a
// JSON engine swap for VertexStreamingChunk: usage metadata, grounding
// metadata, a function call, and a []byte field (ThoughtSignature — encoded/
// decoded as base64 by both encoding/json and a spec-compliant alternative).
var vertexStreamingChunkPayload = []byte(`{
	"candidates": [{
		"content": {
			"role": "model",
			"parts": [
				{"text": "hello world, this is a streamed token"},
				{"thought": true, "thoughtSignature": "aGVsbG8gd29ybGQ="},
				{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
			]
		},
		"finishReason": "STOP",
		"groundingMetadata": {
			"webSearchQueries": ["query one", "query two"]
		}
	}],
	"usageMetadata": {
		"promptTokenCount": 512,
		"candidatesTokenCount": 16,
		"totalTokenCount": 528,
		"thoughtsTokenCount": 4
	}
}`)

// TestVertexStreamingChunkUnmarshal_GoccyMatchesStdlib guards against a
// goccy swap silently changing decode results for VertexStreamingChunk, which
// wraps large nested types from the external google.golang.org/genai SDK
// (some with custom UnmarshalJSON, e.g. ThoughtSignature's []byte/base64
// handling) — higher risk than this session's other goccy swaps, which were
// all on structs owned by this repo.
func TestVertexStreamingChunkUnmarshal_GoccyMatchesStdlib(t *testing.T) {
	var stdChunk, goccyChunk VertexStreamingChunk
	if err := stdjson.Unmarshal(vertexStreamingChunkPayload, &stdChunk); err != nil {
		t.Fatalf("stdlib unmarshal failed: %v", err)
	}
	if err := goccyjson.Unmarshal(vertexStreamingChunkPayload, &goccyChunk); err != nil {
		t.Fatalf("goccy unmarshal failed: %v", err)
	}
	if !reflect.DeepEqual(stdChunk, goccyChunk) {
		t.Fatalf("goccy decode diverges from stdlib decode:\nstdlib: %#v\ngoccy:  %#v", stdChunk, goccyChunk)
	}
}

func BenchmarkVertexStreamingChunkUnmarshal_Stdlib(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var chunk VertexStreamingChunk
		_ = stdjson.Unmarshal(vertexStreamingChunkPayload, &chunk)
	}
}

func BenchmarkVertexStreamingChunkUnmarshal_Goccy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var chunk VertexStreamingChunk
		_ = goccyjson.Unmarshal(vertexStreamingChunkPayload, &chunk)
	}
}
