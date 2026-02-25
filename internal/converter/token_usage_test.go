package converter

import "testing"

func TestTokenUsageTotal(t *testing.T) {
	var nilUsage *TokenUsage
	if got := nilUsage.Total(); got != 0 {
		t.Fatalf("expected 0 for nil receiver, got %d", got)
	}

	tu := &TokenUsage{
		PromptTokens:             1,
		CompletionTokens:         2,
		AudioInputTokens:         3,
		AudioOutputTokens:        4,
		CachedInputTokens:        5,
		CachedOutputTokens:       6,
		ReasoningTokens:          7,
		AcceptedPredictionTokens: 8,
		RejectedPredictionTokens: 9,
		ImageTokens:              10,
	}

	// Total() returns PromptTokens + CompletionTokens only,
	// because specialty tokens are already included in those totals.
	want := 1 + 2
	if got := tu.Total(); got != want {
		t.Fatalf("expected total %d, got %d", want, got)
	}
}

func TestTokenUsageTotal_CacheCreationDoesNotAffect(t *testing.T) {
	tu := &TokenUsage{
		PromptTokens:        100,
		CompletionTokens:    50,
		CacheCreationTokens: 500,
	}

	want := 100 + 50
	if got := tu.Total(); got != want {
		t.Fatalf("expected CacheCreationTokens not to affect Total(): want %d, got %d", want, got)
	}
}
