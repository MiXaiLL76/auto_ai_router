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
	OutputTextTokens         int
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
	tu.OutputTextTokens = converterutil.NonNegativeTokenCount(tu.OutputTextTokens)
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

// IsZero reports whether every billable field is at its zero value, i.e. no
// provider was ever actually contacted for this request (or its response
// carried no usage at all). Used to distinguish "nothing to bill" from
// "something was consumed but we can't price it" — only the latter should
// fail closed when no price is available.
func (tu *TokenUsage) IsZero() bool {
	if tu == nil {
		return true
	}
	return *tu == TokenUsage{}
}

// MergeNonZero copies every non-zero/non-empty field from src into tu,
// leaving tu's existing value in place wherever src's is zero/empty.
//
// This matters for streaming: a provider (or an upstream AIR/proxy-type
// credential relaying frames) can split usage-relevant fields across
// multiple SSE chunks — e.g. a WebSearchRequests-bearing chunk arriving
// separately from the chunk carrying prompt/completion tokens. A plain
// "*dst = *src" per-chunk overwrite silently loses whatever the earlier
// chunk had set (since a later chunk's zero value for that field replaces
// it) — this method fixes that by only ever raising a field, never
// resetting one to zero because a later read didn't happen to touch it.
func (tu *TokenUsage) MergeNonZero(src *TokenUsage) {
	if tu == nil || src == nil {
		return
	}
	if src.PromptTokens != 0 {
		tu.PromptTokens = src.PromptTokens
	}
	if src.CompletionTokens != 0 {
		tu.CompletionTokens = src.CompletionTokens
	}
	if src.AudioInputTokens != 0 {
		tu.AudioInputTokens = src.AudioInputTokens
	}
	if src.AudioOutputTokens != 0 {
		tu.AudioOutputTokens = src.AudioOutputTokens
	}
	if src.CachedInputTokens != 0 {
		tu.CachedInputTokens = src.CachedInputTokens
	}
	if src.CachedAudioInputTokens != 0 {
		tu.CachedAudioInputTokens = src.CachedAudioInputTokens
	}
	if src.CacheCreationTokens != 0 {
		tu.CacheCreationTokens = src.CacheCreationTokens
	}
	if src.CacheCreation5mTokens != 0 {
		tu.CacheCreation5mTokens = src.CacheCreation5mTokens
	}
	if src.CacheCreation1hTokens != 0 {
		tu.CacheCreation1hTokens = src.CacheCreation1hTokens
	}
	if src.CachedOutputTokens != 0 {
		tu.CachedOutputTokens = src.CachedOutputTokens
	}
	if src.OutputTextTokens != 0 {
		tu.OutputTextTokens = src.OutputTextTokens
	}
	if src.ReasoningTokens != 0 {
		tu.ReasoningTokens = src.ReasoningTokens
	}
	if src.AcceptedPredictionTokens != 0 {
		tu.AcceptedPredictionTokens = src.AcceptedPredictionTokens
	}
	if src.RejectedPredictionTokens != 0 {
		tu.RejectedPredictionTokens = src.RejectedPredictionTokens
	}
	if src.ImageCount != 0 {
		tu.ImageCount = src.ImageCount
	}
	if src.ImageTokens != 0 {
		tu.ImageTokens = src.ImageTokens
	}
	if src.OutputImageTokens != 0 {
		tu.OutputImageTokens = src.OutputImageTokens
	}
	if src.WebSearchRequests != 0 {
		tu.WebSearchRequests = src.WebSearchRequests
	}
	if src.WebSearchContextSize != "" {
		tu.WebSearchContextSize = src.WebSearchContextSize
	}
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
