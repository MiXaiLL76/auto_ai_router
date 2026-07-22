package models

import (
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

const (
	tokenTiering200kThreshold = 200_000
	tokenTiering272kThreshold = 272_000
)

// CalculateTokenCosts computes costs based on token usage and model pricing
// Returns nil if price is nil (model not found in pricing database)
//
// IMPORTANT: Handles two token counting semantics:
//  1. Vertex/OpenAI: Audio/Cached/Reasoning tokens are INCLUDED in PromptTokens/CompletionTokens
//     (Example: PromptTokens=100 includes AudioInputTokens=5 and CachedInputTokens=20)
//  2. Anthropic: Cached tokens are SEPARATE from InputTokens
//     (Example: InputTokens=100 + CachedInputTokens=20 = 120 total)
//
// Solution: Calculate "regular" tokens by subtracting detail breakdowns from totals,
// then add back specialized token costs. This works for both semantics:
// - Vertex/OpenAI: 100 - 5 - 20 = 75 regular, then add audio and cached separately
// - Anthropic: 100 - 0 - 0 = 100 regular (since those tokens are in separate fields)
func CalculateTokenCosts(usage *converter.TokenUsage, price *ModelPrice) *converter.TokenCosts {
	if usage == nil || price == nil {
		return nil
	}

	costs := &converter.TokenCosts{}
	longContext272k := usage.PromptTokens > tokenTiering272kThreshold
	inputCostPerToken := price.InputCostPerToken
	if longContext272k && price.InputCostPerTokenAbove272k > 0 {
		inputCostPerToken = price.InputCostPerTokenAbove272k
	}
	outputCostPerToken := price.OutputCostPerToken
	if longContext272k && price.OutputCostPerTokenAbove272k > 0 {
		outputCostPerToken = price.OutputCostPerTokenAbove272k
	}

	// Calculate "regular" input tokens by subtracting specialized token types.
	// Vertex/OpenAI: audio/cached tokens are included in PromptTokens; Anthropic: same + cache creation.
	regularInputTokens := usage.PromptTokens - usage.AudioInputTokens - usage.CachedInputTokens - usage.CacheCreationTokens - usage.ImageTokens
	if regularInputTokens < 0 {
		regularInputTokens = 0
	}

	// Regular input with 200k tiering
	if longContext272k && price.InputCostPerTokenAbove272k > 0 {
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	} else if price.InputCostPerTokenAbove200k > 0 && usage.PromptTokens > tokenTiering200kThreshold {
		above := usage.PromptTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularInputTokens) * int64(above) / int64(usage.PromptTokens))
		regularBelow := regularInputTokens - regularAbove
		costs.InputCost = float64(regularBelow)*price.InputCostPerToken +
			float64(regularAbove)*price.InputCostPerTokenAbove200k
	} else {
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	}

	// Calculate "regular" output tokens by subtracting specialized token types
	regularOutputTokens := usage.CompletionTokens - usage.AudioOutputTokens - usage.ReasoningTokens -
		usage.AcceptedPredictionTokens - usage.RejectedPredictionTokens - usage.OutputImageTokens
	if regularOutputTokens < 0 {
		regularOutputTokens = 0
	}

	// Regular output with 200k tiering
	if longContext272k && price.OutputCostPerTokenAbove272k > 0 {
		costs.OutputCost = float64(regularOutputTokens) * outputCostPerToken
	} else if price.OutputCostPerTokenAbove200k > 0 && usage.CompletionTokens > tokenTiering200kThreshold {
		above := usage.CompletionTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularOutputTokens) * int64(above) / int64(usage.CompletionTokens))
		regularBelow := regularOutputTokens - regularAbove
		costs.OutputCost = float64(regularBelow)*price.OutputCostPerToken +
			float64(regularAbove)*price.OutputCostPerTokenAbove200k
	} else {
		costs.OutputCost = float64(regularOutputTokens) * outputCostPerToken
	}

	// Audio tokens with fallback to regular tokens
	audioInputCost := price.InputCostPerAudioToken
	if audioInputCost == 0 {
		audioInputCost = inputCostPerToken
	}
	costs.AudioInputCost = float64(usage.AudioInputTokens) * audioInputCost

	audioOutputCost := price.OutputCostPerAudioToken
	if audioOutputCost == 0 {
		audioOutputCost = outputCostPerToken
	}
	costs.AudioOutputCost = float64(usage.AudioOutputTokens) * audioOutputCost

	// Cached read tokens: prefer explicit cached price, fall back to LiteLLM alias,
	// then fall back to regular rate (no discount known).
	cachedInputCost := price.InputCostPerCachedToken
	if cachedInputCost == 0 {
		cachedInputCost = price.CacheReadInputTokenCost
	}
	if longContext272k && price.CacheReadInputTokenCostAbove272k > 0 {
		cachedInputCost = price.CacheReadInputTokenCostAbove272k
	}
	if cachedInputCost == 0 {
		cachedInputCost = inputCostPerToken
	}
	costs.CachedInputCost = float64(usage.CachedInputTokens) * cachedInputCost

	cacheCreationCost := price.CacheCreationInputTokenCost
	if longContext272k && price.CacheCreationInputTokenCostAbove272k > 0 {
		cacheCreationCost = price.CacheCreationInputTokenCostAbove272k
	}
	if cacheCreationCost == 0 {
		cacheCreationCost = inputCostPerToken
	}
	costs.CacheCreationCost = float64(usage.CacheCreationTokens) * cacheCreationCost

	cachedOutputCost := price.OutputCostPerCachedToken
	if cachedOutputCost == 0 {
		cachedOutputCost = outputCostPerToken
	}
	costs.CachedOutputCost = float64(usage.CachedOutputTokens) * cachedOutputCost

	// Reasoning tokens with fallback
	reasoningCost := price.OutputCostPerReasoningToken
	if reasoningCost == 0 {
		reasoningCost = outputCostPerToken
	}
	costs.ReasoningCost = float64(usage.ReasoningTokens) * reasoningCost

	// Prediction tokens with fallback (accepted tokens)
	predictionCost := price.OutputCostPerPredictionToken
	if predictionCost == 0 {
		predictionCost = outputCostPerToken
	}
	costs.PredictionCost = float64(usage.AcceptedPredictionTokens) * predictionCost

	// Rejected prediction tokens count as regular output tokens
	costs.PredictionCost += float64(usage.RejectedPredictionTokens) * outputCostPerToken

	// Input image tokens are part of PromptTokens. Price them separately when a
	// modality-specific rate exists, otherwise keep the regular input rate.
	inputImageCost := price.InputCostPerImageToken
	if inputImageCost == 0 {
		inputImageCost = inputCostPerToken
	}
	costs.ImageCost = float64(usage.ImageTokens) * inputImageCost

	// Generated image tokens are part of CompletionTokens and must not also be
	// charged as text. Prefer the token-based image rate when the provider reports
	// a token breakdown; otherwise use the per-image price for Imagen-style APIs.
	if usage.OutputImageTokens > 0 {
		outputImageCost := price.OutputCostPerImageToken
		if outputImageCost == 0 {
			outputImageCost = outputCostPerToken
		}
		costs.ImageCost += float64(usage.OutputImageTokens) * outputImageCost
	} else if usage.ImageCount > 0 && price.OutputCostPerImage > 0 {
		costs.ImageCost += float64(usage.ImageCount) * price.OutputCostPerImage
	}

	// Calculate total
	costs.TotalCost = costs.InputCost +
		costs.OutputCost +
		costs.AudioInputCost +
		costs.AudioOutputCost +
		costs.ReasoningCost +
		costs.CachedInputCost +
		costs.CacheCreationCost +
		costs.CachedOutputCost +
		costs.PredictionCost +
		costs.ImageCost

	return costs
}

// CalculateCost is a convenience method on ModelPrice that calculates total cost
func (p *ModelPrice) CalculateCost(usage *converter.TokenUsage) float64 {
	costs := CalculateTokenCosts(usage, p)
	if costs == nil {
		return 0.0
	}
	return costs.TotalCost
}

// CalculateCosts returns the full cost breakdown for all token types.
func (p *ModelPrice) CalculateCosts(usage *converter.TokenUsage) *converter.TokenCosts {
	return CalculateTokenCosts(usage, p)
}
