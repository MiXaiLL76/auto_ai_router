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

	want := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
	if got := tu.Total(); got != want {
		t.Fatalf("expected total %d, got %d", want, got)
	}
}
