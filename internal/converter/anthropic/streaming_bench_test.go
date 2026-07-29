package anthropic

import (
	stdjson "encoding/json"
	"testing"

	goccyjson "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// Representative per-chunk Anthropic SSE payloads (content_block_delta is by
// far the highest-frequency event type in a real stream: one per token/word).
var (
	anthropicTextDeltaPayload = []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello world, this is a streamed token"}}`)

	anthropicMessageStartPayload = []byte(`{"type":"message_start","message":{"id":"msg_abc123","usage":{"input_tokens":512,"output_tokens":1,"cache_read_input_tokens":0}}}`)
)

func BenchmarkAnthropicStreamEventUnmarshal_Stdlib(b *testing.B) {
	b.Run("text-delta", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var event AnthropicStreamEvent
			_ = stdjson.Unmarshal(anthropicTextDeltaPayload, &event)
		}
	})
	b.Run("message-start", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var event AnthropicStreamEvent
			_ = stdjson.Unmarshal(anthropicMessageStartPayload, &event)
		}
	})
}

func BenchmarkAnthropicStreamEventUnmarshal_Goccy(b *testing.B) {
	b.Run("text-delta", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var event AnthropicStreamEvent
			_ = goccyjson.Unmarshal(anthropicTextDeltaPayload, &event)
		}
	})
	b.Run("message-start", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var event AnthropicStreamEvent
			_ = goccyjson.Unmarshal(anthropicMessageStartPayload, &event)
		}
	})
}

// outgoingChunk is representative of what writeChunk (streaming.go) marshals
// on every emitted delta: a small typed struct, not a large body.
var outgoingChunk = openai.OpenAIStreamingChunk{
	ID:      "chatcmpl-abc123",
	Object:  "chat.completion.chunk",
	Created: 1730000000,
	Model:   "claude-opus-4",
	Choices: []openai.OpenAIStreamingChoice{
		{Index: 0, Delta: openai.OpenAIStreamingDelta{Content: "hello world, this is a streamed token"}},
	},
}

func BenchmarkWriteChunkMarshal_Stdlib(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = stdjson.Marshal(outgoingChunk)
	}
}

func BenchmarkWriteChunkMarshal_Goccy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = goccyjson.Marshal(outgoingChunk)
	}
}
