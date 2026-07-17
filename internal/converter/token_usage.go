package converter

import "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"

// TokenUsage is a universal format for token usage across all providers.
// Used by converters to return usage data without circular dependencies.
type TokenUsage struct {
	PromptTokens             int
	CompletionTokens         int
	AudioInputTokens         int
	AudioOutputTokens        int
	CachedInputTokens        int
	CachedAudioInputTokens   int
	CacheCreationTokens      int
	CacheCreation5mTokens    int
	CacheCreation1hTokens    int
	CachedOutputTokens       int
	ReasoningTokens          int
	AcceptedPredictionTokens int
	RejectedPredictionTokens int
	ImageCount               int // Number of images to generate (1-10)
	ImageTokens              int // Input image/video tokens
	OutputImageTokens        int // Generated image/video tokens
}

func (tu *TokenUsage) Normalize() *TokenUsage {
	if tu == nil {
		return nil
	}
	tu.PromptTokens = nonNegativeTokenCount(tu.PromptTokens)
	tu.CompletionTokens = nonNegativeTokenCount(tu.CompletionTokens)
	tu.AudioInputTokens = nonNegativeTokenCount(tu.AudioInputTokens)
	tu.AudioOutputTokens = nonNegativeTokenCount(tu.AudioOutputTokens)
	tu.CachedInputTokens, tu.CachedAudioInputTokens = converterutil.NormalizeCachedAudioBreakdown(
		tu.CachedInputTokens,
		tu.CachedAudioInputTokens,
	)
	tu.CacheCreationTokens = nonNegativeTokenCount(tu.CacheCreationTokens)
	tu.CacheCreation5mTokens = nonNegativeTokenCount(tu.CacheCreation5mTokens)
	tu.CacheCreation1hTokens = nonNegativeTokenCount(tu.CacheCreation1hTokens)
	if tu.CacheCreationTokens == 0 {
		tu.CacheCreationTokens = tu.CacheCreation5mTokens + tu.CacheCreation1hTokens
	}
	if tu.CacheCreation5mTokens > tu.CacheCreationTokens {
		tu.CacheCreation5mTokens = tu.CacheCreationTokens
	}
	if tu.CacheCreation1hTokens > tu.CacheCreationTokens-tu.CacheCreation5mTokens {
		tu.CacheCreation1hTokens = tu.CacheCreationTokens - tu.CacheCreation5mTokens
	}
	tu.CachedOutputTokens = nonNegativeTokenCount(tu.CachedOutputTokens)
	tu.ReasoningTokens = nonNegativeTokenCount(tu.ReasoningTokens)
	tu.AcceptedPredictionTokens = nonNegativeTokenCount(tu.AcceptedPredictionTokens)
	tu.RejectedPredictionTokens = nonNegativeTokenCount(tu.RejectedPredictionTokens)
	tu.ImageCount = nonNegativeTokenCount(tu.ImageCount)
	tu.ImageTokens = nonNegativeTokenCount(tu.ImageTokens)
	tu.OutputImageTokens = nonNegativeTokenCount(tu.OutputImageTokens)
	return tu
}

// Total returns the sum of prompt and completion tokens.
func (tu *TokenUsage) Total() int {
	if tu == nil {
		return 0
	}
	return tu.PromptTokens + tu.CompletionTokens
}

// TokenCosts contains cost breakdown by token type
type TokenCosts struct {
	InputCost         float64
	OutputCost        float64
	AudioInputCost    float64
	AudioOutputCost   float64
	ReasoningCost     float64
	CachedInputCost   float64
	CacheCreationCost float64
	CachedOutputCost  float64
	PredictionCost    float64
	ImageCost         float64
	TotalCost         float64
}

func nonNegativeTokenCount(tokens int) int {
	if tokens < 0 {
		return 0
	}
	return tokens
}
