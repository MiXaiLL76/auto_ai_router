package models

import (
	"github.com/mixaill76/auto_ai_router/internal/transform"
)

// CalculateTokenCosts computes costs based on token usage and model pricing
// Returns nil if price is nil (model not found in pricing database)
func CalculateTokenCosts(usage *transform.TokenUsage, price *ModelPrice) *transform.TokenCosts {
	if usage == nil || price == nil {
		return nil
	}

	costs := &transform.TokenCosts{}

	// Regular tokens
	costs.InputCost = float64(usage.PromptTokens) * price.InputCostPerToken
	costs.OutputCost = float64(usage.CompletionTokens) * price.OutputCostPerToken

	// Audio tokens with fallback to regular tokens
	audioInputCost := price.InputCostPerAudioToken
	if audioInputCost == 0 {
		audioInputCost = price.InputCostPerToken
	}
	costs.AudioInputCost = float64(usage.AudioInputTokens) * audioInputCost

	audioOutputCost := price.OutputCostPerAudioToken
	if audioOutputCost == 0 {
		audioOutputCost = price.OutputCostPerToken
	}
	costs.AudioOutputCost = float64(usage.AudioOutputTokens) * audioOutputCost

	// Cached tokens with fallback
	cachedInputCost := price.InputCostPerCachedToken
	if cachedInputCost == 0 {
		cachedInputCost = price.InputCostPerToken
	}
	costs.CachedInputCost = float64(usage.CachedInputTokens) * cachedInputCost

	cachedOutputCost := price.OutputCostPerCachedToken
	if cachedOutputCost == 0 {
		cachedOutputCost = price.OutputCostPerToken
	}
	costs.CachedOutputCost = float64(usage.CachedOutputTokens) * cachedOutputCost

	// Reasoning tokens with fallback
	reasoningCost := price.OutputCostPerReasoningToken
	if reasoningCost == 0 {
		reasoningCost = price.OutputCostPerToken
	}
	costs.ReasoningCost = float64(usage.ReasoningTokens) * reasoningCost

	// Prediction tokens with fallback (accepted tokens)
	predictionCost := price.OutputCostPerPredictionToken
	if predictionCost == 0 {
		predictionCost = price.OutputCostPerToken
	}
	costs.PredictionCost = float64(usage.AcceptedPredictionTokens) * predictionCost

	// Rejected prediction tokens count as regular output tokens
	costs.PredictionCost += float64(usage.RejectedPredictionTokens) * price.OutputCostPerToken

	// Image cost calculation: supports both per-image and per-image-token pricing
	// Priority: 1) Per-image cost if available (typical for image generation APIs)
	//           2) Per-image-token cost as fallback (rarely used for image generation)
	//           3) Default: $0 if neither is configured
	if usage.ImageCount > 0 && price.OutputCostPerImage > 0 {
		// Per-image cost (e.g., $0.02 per image)
		costs.ImageCost = float64(usage.ImageCount) * price.OutputCostPerImage
	} else if usage.ImageCount > 0 && price.OutputCostPerImageToken > 0 {
		// Per-image-token cost fallback (rarely used for image generation)
		costs.ImageCost = float64(usage.ImageCount) * price.OutputCostPerImageToken
	}

	// Calculate total
	costs.TotalCost = costs.InputCost +
		costs.OutputCost +
		costs.AudioInputCost +
		costs.AudioOutputCost +
		costs.ReasoningCost +
		costs.CachedInputCost +
		costs.CachedOutputCost +
		costs.PredictionCost +
		costs.ImageCost

	return costs
}

// CalculateCost is a convenience method on ModelPrice that calculates total cost
func (p *ModelPrice) CalculateCost(usage *transform.TokenUsage) float64 {
	costs := CalculateTokenCosts(usage, p)
	if costs == nil {
		return 0.0
	}
	return costs.TotalCost
}
