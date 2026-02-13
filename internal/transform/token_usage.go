package transform

// TokenUsage is a universal format for token usage across all providers
// Used to avoid circular dependencies between transform and models packages
// Note: This is NOT 'usage.go', it's 'token_usage.go' to match the pattern
type TokenUsage struct {
	PromptTokens             int
	CompletionTokens         int
	AudioInputTokens         int
	AudioOutputTokens        int
	CachedInputTokens        int
	CachedOutputTokens       int
	ReasoningTokens          int
	AcceptedPredictionTokens int
	RejectedPredictionTokens int
	ImageCount               int // Number of images to generate (1-10)
	ImageTokens              int // Token count for image processing
}

// Total returns the sum of all token counts
func (tu *TokenUsage) Total() int {
	if tu == nil {
		return 0
	}
	return tu.PromptTokens + tu.CompletionTokens +
		tu.AudioInputTokens + tu.AudioOutputTokens +
		tu.CachedInputTokens + tu.CachedOutputTokens +
		tu.ReasoningTokens + tu.AcceptedPredictionTokens +
		tu.RejectedPredictionTokens + tu.ImageTokens
}

// TokenCosts contains cost breakdown by token type
type TokenCosts struct {
	InputCost        float64 // Regular input tokens
	OutputCost       float64 // Regular output tokens
	AudioInputCost   float64 // Audio input tokens
	AudioOutputCost  float64 // Audio output tokens
	ReasoningCost    float64 // Reasoning tokens (output)
	CachedInputCost  float64 // Cached input tokens
	CachedOutputCost float64 // Cached output tokens
	PredictionCost   float64 // Prediction tokens (accepted)
	ImageCost        float64 // Image tokens
	TotalCost        float64 // Sum of all costs
}
