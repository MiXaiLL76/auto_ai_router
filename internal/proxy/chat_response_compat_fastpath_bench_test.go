package proxy

import "testing"

// The rewrite of the model name on a streamed chunk runs once per chunk of
// every response, so its cost is the router's per-chunk cost. These
// benchmarks cover the shapes that actually arrive:
//
//   - the chat.completion.chunk every /v1/chat/completions stream sends,
//     which carries a "model" of its own;
//   - the same chunk without a "model", where the alias is ADDED because
//     the "object" names a chunk shape;
//   - a /v1/responses delta event, which the router itself generates
//     ("type":"response.output_text.delta") and which carries neither a
//     "model" nor an "object" nor a nested "response" object -- there is
//     nothing to rewrite in it, and the gate must skip it;
//   - a chunk with nothing to rewrite at all.
//
// The Responses delta is the one that must not get slower: a single
// response is thousands of them.
func BenchmarkNormalizeSSEDataLineModel(b *testing.B) {
	chatChunk := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"upstream-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n")
	chatChunkNoModel := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n")
	responsesDelta := []byte("data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_01Ab\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n")
	nothingToRewrite := []byte("data: {\"id\":\"c1\",\"choices\":[]}\n\n")

	cases := []struct {
		name string
		line []byte
	}{
		{"chat_chunk", chatChunk},
		{"chat_chunk_no_model", chatChunkNoModel},
		{"responses_delta", responsesDelta},
		{"nothing_to_rewrite", nothingToRewrite},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = normalizeSSEDataLineModelWithOptions(c.line, "public-alias", true)
			}
		})
	}
}
