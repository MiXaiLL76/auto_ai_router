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
	WebSearchRequests        int // Built-in web search tool calls/requests
	WebSearchContextSize     string
}

func (tu *TokenUsage) Normalize() *TokenUsage {
	if tu == nil {
		return nil
	}
	tu.PromptTokens = converterutil.NonNegativeTokenCount(tu.PromptTokens)
	tu.CompletionTokens = converterutil.NonNegativeTokenCount(tu.CompletionTokens)
	tu.AudioInputTokens = converterutil.NonNegativeTokenCount(tu.AudioInputTokens)
	tu.AudioOutputTokens = converterutil.NonNegativeTokenCount(tu.AudioOutputTokens)
	tu.CachedInputTokens, tu.CachedAudioInputTokens = converterutil.NormalizeCachedAudioBreakdown(
		tu.CachedInputTokens,
		tu.CachedAudioInputTokens,
	)
	tu.CacheCreationTokens = converterutil.NonNegativeTokenCount(tu.CacheCreationTokens)
	tu.CacheCreation5mTokens = converterutil.NonNegativeTokenCount(tu.CacheCreation5mTokens)
	tu.CacheCreation1hTokens = converterutil.NonNegativeTokenCount(tu.CacheCreation1hTokens)
	if tu.CacheCreationTokens == 0 {
		tu.CacheCreationTokens = tu.CacheCreation5mTokens + tu.CacheCreation1hTokens
	}
	if tu.CacheCreation5mTokens > tu.CacheCreationTokens {
		tu.CacheCreation5mTokens = tu.CacheCreationTokens
	}
	if tu.CacheCreation1hTokens > tu.CacheCreationTokens-tu.CacheCreation5mTokens {
		tu.CacheCreation1hTokens = tu.CacheCreationTokens - tu.CacheCreation5mTokens
	}
	tu.CachedOutputTokens = converterutil.NonNegativeTokenCount(tu.CachedOutputTokens)
	tu.ReasoningTokens = converterutil.NonNegativeTokenCount(tu.ReasoningTokens)
	tu.AcceptedPredictionTokens = converterutil.NonNegativeTokenCount(tu.AcceptedPredictionTokens)
	tu.RejectedPredictionTokens = converterutil.NonNegativeTokenCount(tu.RejectedPredictionTokens)
	tu.ImageCount = converterutil.NonNegativeTokenCount(tu.ImageCount)
	tu.ImageTokens = converterutil.NonNegativeTokenCount(tu.ImageTokens)
	tu.OutputImageTokens = converterutil.NonNegativeTokenCount(tu.OutputImageTokens)
	tu.WebSearchRequests = converterutil.NonNegativeTokenCount(tu.WebSearchRequests)
	if tu.WebSearchRequests > 0 || tu.WebSearchContextSize != "" {
		tu.WebSearchContextSize = NormalizeWebSearchContextSize(tu.WebSearchContextSize)
	}
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
	WebSearchCost     float64
	TotalCost         float64
}

func NormalizeWebSearchContextSize(size string) string {
	switch size {
	case "low", "medium", "high":
		return size
	default:
		return "medium"
	}
}
