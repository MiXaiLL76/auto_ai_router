package models

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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

	normalizedUsage := *usage
	usage = normalizedUsage.Normalize()

	costs := &converter.TokenCosts{}
	promptTokens := converterutil.NonNegativeTokenCount(usage.PromptTokens)
	completionTokens := converterutil.NonNegativeTokenCount(usage.CompletionTokens)
	audioInputTokens := converterutil.NonNegativeTokenCount(usage.AudioInputTokens)
	audioOutputTokens := converterutil.NonNegativeTokenCount(usage.AudioOutputTokens)
	imageTokens := converterutil.NonNegativeTokenCount(usage.ImageTokens)
	outputImageTokens := converterutil.NonNegativeTokenCount(usage.OutputImageTokens)
	cachedOutputTokens := converterutil.NonNegativeTokenCount(usage.CachedOutputTokens)
	reasoningTokens := converterutil.NonNegativeTokenCount(usage.ReasoningTokens)
	acceptedPredictionTokens := converterutil.NonNegativeTokenCount(usage.AcceptedPredictionTokens)
	rejectedPredictionTokens := converterutil.NonNegativeTokenCount(usage.RejectedPredictionTokens)
	imageCount := converterutil.NonNegativeTokenCount(usage.ImageCount)

	longContext272k := promptTokens > tokenTiering272kThreshold
	inputCostPerToken := price.InputCostPerToken
	if longContext272k && price.InputCostPerTokenAbove272k > 0 {
		inputCostPerToken = price.InputCostPerTokenAbove272k
	}
	outputCostPerToken := price.OutputCostPerToken
	if longContext272k && price.OutputCostPerTokenAbove272k > 0 {
		outputCostPerToken = price.OutputCostPerTokenAbove272k
	}

	cachedInputTokens := converterutil.NonNegativeTokenCount(usage.CachedInputTokens)
	cacheCreationTokens := converterutil.NonNegativeTokenCount(usage.CacheCreationTokens)

	// Calculate "regular" input tokens by subtracting specialized token types.
	// Vertex/OpenAI: audio/cached tokens are included in PromptTokens; Anthropic: same + cache creation.
	regularInputTokens := promptTokens - audioInputTokens - cachedInputTokens - cacheCreationTokens - imageTokens
	if regularInputTokens < 0 {
		regularInputTokens = 0
	}

	// Regular input with 200k tiering
	if longContext272k && price.InputCostPerTokenAbove272k > 0 {
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	} else if price.InputCostPerTokenAbove200k > 0 && promptTokens > tokenTiering200kThreshold {
		above := promptTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularInputTokens) * int64(above) / int64(promptTokens))
		regularBelow := regularInputTokens - regularAbove
		costs.InputCost = float64(regularBelow)*price.InputCostPerToken +
			float64(regularAbove)*price.InputCostPerTokenAbove200k
	} else {
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	}

	// Calculate "regular" output tokens by subtracting specialized token types
	regularOutputTokens := completionTokens - audioOutputTokens - reasoningTokens -
		acceptedPredictionTokens - rejectedPredictionTokens - outputImageTokens
	if regularOutputTokens < 0 {
		regularOutputTokens = 0
	}

	// Regular output with 200k tiering
	if longContext272k && price.OutputCostPerTokenAbove272k > 0 {
		costs.OutputCost = float64(regularOutputTokens) * outputCostPerToken
	} else if price.OutputCostPerTokenAbove200k > 0 && completionTokens > tokenTiering200kThreshold {
		above := completionTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularOutputTokens) * int64(above) / int64(completionTokens))
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
	costs.AudioInputCost = float64(audioInputTokens) * audioInputCost

	audioOutputCost := price.OutputCostPerAudioToken
	if audioOutputCost == 0 {
		audioOutputCost = outputCostPerToken
	}
	costs.AudioOutputCost = float64(audioOutputTokens) * audioOutputCost

	// Cached read tokens: prefer explicit cached price, fall back to LiteLLM alias,
	// then fall back to regular rate (no discount known).
	cachedInputCost := price.InputCostPerCachedToken
	if cachedInputCost == 0 {
		cachedInputCost = price.CacheReadInputTokenCost
	}
	if longContext272k && price.CacheReadInputTokenCostAbove272k > 0 {
		cachedInputCost = price.CacheReadInputTokenCostAbove272k
	} else if promptTokens > tokenTiering200kThreshold && price.CacheReadInputTokenCostAbove200k > 0 {
		cachedInputCost = price.CacheReadInputTokenCostAbove200k
	}
	if cachedInputCost == 0 {
		cachedInputCost = inputCostPerToken
	}
	cachedAudioTokens := converterutil.NonNegativeTokenCount(usage.CachedAudioInputTokens)
	if cachedAudioTokens > cachedInputTokens {
		cachedAudioTokens = cachedInputTokens
	}
	regularCachedTokens := cachedInputTokens - cachedAudioTokens
	cachedAudioCost := price.CacheReadInputAudioTokenCost
	if cachedAudioCost == 0 {
		cachedAudioCost = cachedInputCost
	}
	costs.CachedInputCost = float64(regularCachedTokens)*cachedInputCost +
		float64(cachedAudioTokens)*cachedAudioCost

	cacheCreationCost := price.CacheCreationInputTokenCost
	cacheCreationUses272kRate := false
	if longContext272k && price.CacheCreationInputTokenCostAbove272k > 0 {
		cacheCreationCost = price.CacheCreationInputTokenCostAbove272k
		cacheCreationUses272kRate = true
	} else if promptTokens > tokenTiering200kThreshold && price.CacheCreationInputTokenCostAbove200k > 0 {
		cacheCreationCost = price.CacheCreationInputTokenCostAbove200k
	}
	if cacheCreationCost == 0 {
		cacheCreationCost = inputCostPerToken
	}
	cacheCreation1hCost := cacheCreationCost
	if !cacheCreationUses272kRate {
		cacheCreation1hCost = price.CacheCreationInputTokenCostAbove1hr
		if promptTokens > tokenTiering200kThreshold && price.CacheCreationInputTokenCostAbove1hrAbove200k > 0 {
			cacheCreation1hCost = price.CacheCreationInputTokenCostAbove1hrAbove200k
		}
		if cacheCreation1hCost == 0 {
			cacheCreation1hCost = cacheCreationCost
		}
	}

	// The aggregate remains authoritative for compatibility. TTL details split it;
	// malformed details are capped so they can never bill more than the aggregate.
	cacheCreation5mTokens := converterutil.NonNegativeTokenCount(usage.CacheCreation5mTokens)
	if cacheCreation5mTokens > cacheCreationTokens {
		cacheCreation5mTokens = cacheCreationTokens
	}
	cacheCreation1hTokens := converterutil.NonNegativeTokenCount(usage.CacheCreation1hTokens)
	if cacheCreation1hTokens > cacheCreationTokens-cacheCreation5mTokens {
		cacheCreation1hTokens = cacheCreationTokens - cacheCreation5mTokens
	}
	unclassifiedCacheCreationTokens := cacheCreationTokens - cacheCreation5mTokens - cacheCreation1hTokens
	costs.CacheCreationCost = float64(unclassifiedCacheCreationTokens+cacheCreation5mTokens)*cacheCreationCost +
		float64(cacheCreation1hTokens)*cacheCreation1hCost

	cachedOutputCost := price.OutputCostPerCachedToken
	if cachedOutputCost == 0 {
		cachedOutputCost = outputCostPerToken
	}
	costs.CachedOutputCost = float64(cachedOutputTokens) * cachedOutputCost

	// Reasoning tokens with fallback
	reasoningCost := price.OutputCostPerReasoningToken
	if reasoningCost == 0 {
		reasoningCost = outputCostPerToken
	}
	costs.ReasoningCost = float64(reasoningTokens) * reasoningCost

	// Prediction tokens with fallback (accepted tokens)
	predictionCost := price.OutputCostPerPredictionToken
	if predictionCost == 0 {
		predictionCost = outputCostPerToken
	}
	costs.PredictionCost = float64(acceptedPredictionTokens) * predictionCost

	// Rejected prediction tokens count as regular output tokens
	costs.PredictionCost += float64(rejectedPredictionTokens) * outputCostPerToken

	// Input image tokens are part of PromptTokens. Price them separately when a
	// modality-specific rate exists, otherwise keep the regular input rate.
	inputImageCost := price.InputCostPerImageToken
	if inputImageCost == 0 {
		inputImageCost = inputCostPerToken
	}
	costs.ImageCost = float64(imageTokens) * inputImageCost

	// Generated image tokens are part of CompletionTokens and must not also be
	// charged as text. Prefer the token-based image rate when the provider reports
	// a token breakdown; otherwise use the per-image price for Imagen-style APIs.
	if outputImageTokens > 0 {
		outputImageCost := price.OutputCostPerImageToken
		if outputImageCost == 0 {
			outputImageCost = outputCostPerToken
		}
		costs.ImageCost += float64(outputImageTokens) * outputImageCost
	} else if imageCount > 0 && price.OutputCostPerImage > 0 {
		costs.ImageCost += float64(imageCount) * price.OutputCostPerImage
	}

	webSearchRequests := converterutil.NonNegativeTokenCount(usage.WebSearchRequests)
	if webSearchRequests > 0 {
		if price.webSearchBillingUnit() == "per_prompt" {
			webSearchRequests = 1
		}
		costs.WebSearchCost = float64(webSearchRequests) *
			price.webSearchCostPerQuery(usage.WebSearchContextSize)
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
		costs.ImageCost +
		costs.WebSearchCost

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

func (p *ModelPrice) webSearchCostPerQuery(contextSize string) float64 {
	if p == nil || len(p.SearchContextCostPerQuery) == 0 {
		return 0
	}
	key := "search_context_size_" + converter.NormalizeWebSearchContextSize(contextSize)
	return p.SearchContextCostPerQuery[key]
}

func (p *ModelPrice) webSearchBillingUnit() string {
	if p == nil {
		return ""
	}
	unit := strings.ToLower(strings.TrimSpace(p.WebSearchBillingUnit))
	if unit != "" {
		return unit
	}
	provider := strings.ToLower(strings.TrimSpace(p.LiteLLMProvider))
	if strings.Contains(provider, "gemini") || strings.Contains(provider, "vertex") {
		// LiteLLM treats Gemini 2.x pricing entries without an explicit unit as
		// per-prompt. Gemini 3.x entries explicitly opt into per-query billing.
		return "per_prompt"
	}
	return "per_query"
}
