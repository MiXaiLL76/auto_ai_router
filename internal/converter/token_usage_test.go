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

func TestTokenUsageNormalize(t *testing.T) {
	tu := (&TokenUsage{
		PromptTokens:             -1,
		CompletionTokens:         -2,
		AudioInputTokens:         -3,
		AudioOutputTokens:        -4,
		CachedInputTokens:        -5,
		CachedAudioInputTokens:   6,
		CacheCreationTokens:      10,
		CacheCreation5mTokens:    8,
		CacheCreation1hTokens:    8,
		CachedOutputTokens:       -9,
		ReasoningTokens:          -10,
		AcceptedPredictionTokens: -11,
		RejectedPredictionTokens: -12,
		ImageCount:               -13,
		ImageTokens:              -14,
		OutputImageTokens:        -15,
	}).Normalize()

	if tu.PromptTokens != 0 || tu.CompletionTokens != 0 {
		t.Fatalf("expected totals to be clamped, got %+v", tu)
	}
	if tu.CachedInputTokens != 0 || tu.CachedAudioInputTokens != 0 {
		t.Fatalf("expected cached audio to be capped by sanitized cached tokens, got %+v", tu)
	}
	if tu.CacheCreationTokens != 10 || tu.CacheCreation5mTokens != 8 || tu.CacheCreation1hTokens != 2 {
		t.Fatalf("expected cache creation details to be capped at aggregate, got %+v", tu)
	}
	if tu.AudioInputTokens != 0 || tu.AudioOutputTokens != 0 || tu.CachedOutputTokens != 0 || tu.ReasoningTokens != 0 {
		t.Fatalf("expected specialized negative fields to be clamped, got %+v", tu)
	}
	if tu.AcceptedPredictionTokens != 0 || tu.RejectedPredictionTokens != 0 || tu.ImageCount != 0 || tu.ImageTokens != 0 || tu.OutputImageTokens != 0 {
		t.Fatalf("expected prediction/image negative fields to be clamped, got %+v", tu)
	}
}

func TestTokenUsageNormalize_CacheCreationAggregateFromTTL(t *testing.T) {
	tu := (&TokenUsage{
		CacheCreation5mTokens: 3,
		CacheCreation1hTokens: 7,
	}).Normalize()

	if tu.CacheCreationTokens != 10 || tu.CacheCreation5mTokens != 3 || tu.CacheCreation1hTokens != 7 {
		t.Fatalf("expected cache creation aggregate from TTL details, got %+v", tu)
	}
}
