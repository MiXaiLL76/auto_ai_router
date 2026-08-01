package proxy

import (
	"reflect"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// TestExtractOpenAITokensAndUsage_MatchesOldSequence proves the plan-item-G
// consolidation (single decode into map[string]goccyjson.RawMessage feeding
// both consumers) computes byte-for-byte identical results to the old
// sequence of two independent full-body decodes
// (extractTokensFromResponse + converter.ExtractTokenUsageWithOptions), for
// every edge case the non-streaming proxy.go/retry.go branches rely on:
// a normal chat-completion body, a Responses API body, an embeddings body
// (no total-tokens-affecting fields beyond prompt tokens), an image-generation
// style body with no "usage" at all, and a malformed body.
func TestExtractOpenAITokensAndUsage_MatchesOldSequence(t *testing.T) {
	cases := []struct {
		name string
		body []byte
	}{
		{"chat-completion", benchChatCompletionRespBody},
		{"responses-api", benchResponsesAPIRespBody},
		{"embeddings-1536", benchEmbeddingsRespBody},
		{"no-usage-image-gen", []byte(`{"created":1730000000,"data":[{"url":"https://example.com/1.png"}]}`)},
		{"empty-object", []byte(`{}`)},
		{"malformed", []byte(`not json`)},
		{"empty-body", []byte(``)},
		{"nested-response-completed-event", []byte(`{"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}`)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			oldTokens, oldUsage := extractTokensAndUsageOld(tc.body, benchUsageOptions)
			newTokens, newUsage := extractOpenAITokensAndUsage(tc.body, benchUsageOptions)

			if oldTokens != newTokens {
				t.Errorf("total tokens mismatch: old=%d new=%d", oldTokens, newTokens)
			}
			if !reflect.DeepEqual(oldUsage, newUsage) {
				t.Errorf("TokenUsage mismatch:\nold=%#v\nnew=%#v", oldUsage, newUsage)
			}
		})
	}
}

// TestExtractOpenAITokensAndUsage_TotalTokensVsUsageTotal documents (rather
// than asserts a false equivalence) that extractOpenAITokensAndUsage's tokens
// return value and usage.Total() (PromptTokens+CompletionTokens) are
// DIFFERENT numbers computed from different JSON fields — the provider's own
// literal "total_tokens" vs. our own prompt+completion sum. They happen to
// agree in the well-formed cases in
// TestExtractOpenAITokensAndUsage_MatchesOldSequence, but nothing in this
// codebase guarantees that in general (e.g. a provider reporting
// total_tokens that includes something not separately broken out into
// prompt/completion). This is exactly why plan item G shares the *decode*,
// not the *computation*, between the two consumers — see
// converter.ExtractTotalTokensAndUsageWithOptions's doc comment.
func TestExtractOpenAITokensAndUsage_TotalTokensVsUsageTotal(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":999}}`)
	tokens, usage := extractOpenAITokensAndUsage(body, converter.TokenUsageExtractionOptions{})
	if tokens != 999 {
		t.Fatalf("expected literal total_tokens=999, got %d", tokens)
	}
	if usage == nil || usage.Total() != 150 {
		t.Fatalf("expected usage.Total()=150 (100+50), got %#v", usage)
	}
	if tokens == usage.Total() {
		t.Fatalf("test fixture should demonstrate divergence, but both are %d", tokens)
	}
}
