package proxy

import (
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// Plan item G: a non-streaming response used to be independently re-decoded
// by extractTokensFromResponse (rate-limiter RPM/TPM accounting) and
// converter.ExtractTokenUsageWithOptions (spend logging/metrics) — two full
// typed-struct Unmarshals of the same bytes at every non-streaming call site
// (proxy.go's proxy-credential and direct-provider branches, retry.go's
// fallback branch). extractOpenAITokensAndUsage (response_helpers.go) now
// calls converter.ExtractTotalTokensAndUsageWithOptions, which decodes the
// body once into tokenUsageResponseShape (the literal "total_tokens" field
// folded in alongside the existing usage/choices/output/response fields) and
// derives both numbers from that single decode.
//
// An earlier version of this fix instead decoded into a generic
// map[string]goccyjson.RawMessage and ran several smaller sub-Unmarshal calls
// off of it; that measured SLOWER than the two-decode baseline for ordinary
// chat/Responses API bodies (a map decode must allocate an entry for every
// top-level key, wanted or not, and each extra Unmarshal call adds its own
// fixed overhead) even though it won on the embeddings shape. Folding
// total_tokens into the existing typed struct avoids that regression
// entirely — measured ~1.9-2x faster and ~1.9-2x less memory across all
// three shapes below.
//
// Bodies mirror this plan's established production shapes (pprof_plan_test.md):
// chat 5632 prompt / 563 completion tokens, Responses API 7168/717, and an
// embeddings response with a 1536-float vector — the shape
// normalizeSuccessfulResponseModel was originally optimized for, so it's
// worth re-confirming the token-extraction path doesn't regress on it either.

func repeatWords(word string, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(word)
	}
	return b.String()
}

// buildRealisticChatCompletionResponseBody mirrors mock-go's chat profile:
// 5632 prompt / 563 completion tokens.
func buildRealisticChatCompletionResponseBody() []byte {
	completion := repeatWords("token", 563)
	var b strings.Builder
	b.WriteString(`{"id":"chatcmpl-abc123","object":"chat.completion","created":1730000000,"model":"gpt-4o",`)
	b.WriteString(`"choices":[{"index":0,"message":{"role":"assistant","content":"`)
	b.WriteString(completion)
	b.WriteString(`"},"finish_reason":"stop"}],`)
	b.WriteString(`"usage":{"prompt_tokens":5632,"completion_tokens":563,"total_tokens":6195,`)
	b.WriteString(`"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}`)
	return []byte(b.String())
}

// buildRealisticResponsesAPIResponseBody mirrors mock-go's /v1/responses
// profile: 7168 input / 717 output tokens, Responses API shape (usage under
// input_tokens/output_tokens, content under "output").
func buildRealisticResponsesAPIResponseBody() []byte {
	completion := repeatWords("token", 717)
	var b strings.Builder
	b.WriteString(`{"id":"resp_abc123","object":"response","created_at":1730000000,"model":"gpt-4o",`)
	b.WriteString(`"output":[{"type":"message","id":"msg_abc123","status":"completed","role":"assistant","content":[{"type":"output_text","text":"`)
	b.WriteString(completion)
	b.WriteString(`"}]}],`)
	b.WriteString(`"usage":{"input_tokens":7168,"output_tokens":717,"total_tokens":7885,`)
	b.WriteString(`"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}`)
	return []byte(b.String())
}

// buildRealisticEmbeddingsResponseBody mirrors mock-go's /v1/embeddings
// profile: 2950 prompt tokens, a single 1536-float vector.
func buildRealisticEmbeddingsResponseBody() []byte {
	var vec strings.Builder
	for i := 0; i < 1536; i++ {
		if i > 0 {
			vec.WriteByte(',')
		}
		vec.WriteString("0.0123456789")
	}
	var b strings.Builder
	b.WriteString(`{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[`)
	b.WriteString(vec.String())
	b.WriteString(`]}],"usage":{"prompt_tokens":2950,"total_tokens":2950}}`)
	return []byte(b.String())
}

var (
	benchChatCompletionRespBody = buildRealisticChatCompletionResponseBody()
	benchResponsesAPIRespBody   = buildRealisticResponsesAPIResponseBody()
	benchEmbeddingsRespBody     = buildRealisticEmbeddingsResponseBody()

	benchUsageOptions = converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
)

// extractTokensAndUsageOld reproduces the pre-fix call sequence exactly: two
// independent full-body decodes via the still-exported, unchanged
// extractTokensFromResponse and converter.ExtractTokenUsageWithOptions.
// Frozen baseline, not the production code path anymore.
func extractTokensAndUsageOld(body []byte, opts converter.TokenUsageExtractionOptions) (int, *converter.TokenUsage) {
	tokens := extractTokensFromResponse(body, config.ProviderTypeOpenAI)
	usage := converter.ExtractTokenUsageWithOptions(body, opts)
	return tokens, usage
}

func BenchmarkNonStreamingTokenExtraction_Old(b *testing.B) {
	cases := []struct {
		name string
		body []byte
	}{
		{"chat-completion", benchChatCompletionRespBody},
		{"responses-api", benchResponsesAPIRespBody},
		{"embeddings-1536", benchEmbeddingsRespBody},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				extractTokensAndUsageOld(tc.body, benchUsageOptions)
			}
		})
	}
}

// BenchmarkNonStreamingTokenExtraction_New exercises the actual production
// helper (response_helpers.go) now wired into proxy.go/retry.go.
func BenchmarkNonStreamingTokenExtraction_New(b *testing.B) {
	cases := []struct {
		name string
		body []byte
	}{
		{"chat-completion", benchChatCompletionRespBody},
		{"responses-api", benchResponsesAPIRespBody},
		{"embeddings-1536", benchEmbeddingsRespBody},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				extractOpenAITokensAndUsage(tc.body, benchUsageOptions)
			}
		})
	}
}
