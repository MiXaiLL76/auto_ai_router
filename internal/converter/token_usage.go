package converter

// TokenUsage is a universal format for token usage across all providers.
// Used by converters to return usage data without circular dependencies.
type TokenUsage struct {
	PromptTokens             int
	CompletionTokens         int
	AudioInputTokens         int
	AudioOutputTokens        int
	CachedInputTokens        int
	CacheCreationTokens      int
	CachedOutputTokens       int
	ReasoningTokens          int
	AcceptedPredictionTokens int
	RejectedPredictionTokens int
	ImageCount               int // Number of images to generate (1-10)
	ImageTokens              int // Token count for image processing
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
	InputCost        float64
	OutputCost       float64
	AudioInputCost   float64
	AudioOutputCost  float64
	ReasoningCost    float64
	CachedInputCost  float64
	CachedOutputCost float64
	PredictionCost   float64
	ImageCost        float64
	TotalCost        float64
}
