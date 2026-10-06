package models

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

const (
	tokenTiering200kThreshold = 200_000
	tokenTiering32kThreshold  = 32_000
	tokenTiering128kThreshold = 128_000
	tokenTiering256kThreshold = 256_000
	tokenTiering272kThreshold = 272_000
	tokenTiering512kThreshold = 512_000
)

// ModelPrice.LongContextPricingMode values.
const (
	// LongContextFullRequest200kInclusive bills every token of a request
	// whose prompt reaches 200k tokens (>= 200,000, not >) at the
	// *_above_200k_tokens rates: xAI's long-context rule.
	LongContextFullRequest200kInclusive = "full_request_200k_inclusive"
)

// ReasoningTokensAccountingAuto is the ModelPrice.ReasoningTokensAccounting
// value that detects reasoning-token semantics per response.
const ReasoningTokensAccountingAuto = "auto"

// Canonical ModelPrice.ToolCostPerCall keys and the aliases accepted for them.
const (
	ToolCodeExecution     = "code_execution"
	ToolAttachmentSearch  = "attachment_search"
	ToolCollectionsSearch = "collections_search"
)

var toolCostAliases = map[string]string{
	"code_interpreter": ToolCodeExecution,
	"document_search":  ToolAttachmentSearch,
	"file_search":      ToolCollectionsSearch,
}

// fullSessionTier describes one "Alibaba Cloud style" pricing bracket: once
// the prompt exceeds threshold, the ENTIRE request (not just the excess) is
// billed at this tier's rates. Zero-value rates mean "not configured for
// this tier" and are skipped by fullSessionRate. An inclusive tier already
// applies when the prompt equals threshold.
type fullSessionTier struct {
	threshold             int
	inclusive             bool
	inputRate             float64
	outputRate            float64
	cacheReadRate         float64
	cacheCreationRate     float64
	explicitCacheReadRate float64
}

// fullSessionTiers returns the configured full-session tiers in descending
// threshold order, so the first matching entry is always the highest
// (most specific) bracket the prompt has crossed. 200k is included only for
// providers whose documented billing rule is "once input context exceeds
// 200k tokens, the entire request (input, output, and cache) is billed at
// the long-context rate" — Google's Gemini/Vertex AI family, detected via
// config.IsGoogleGeminiProvider (shared with LiteLLM DB provider-type
// routing so the two can't drift apart on which aliases count as Gemini).
// This differs from the default 200k tier semantics below, which bill only
// the excess above 200k — the generic tier shape kept as a fallback for any
// non-Gemini provider. It is not modeled on an observed real Claude Sonnet
// billing rule: Sonnet 4.5 doesn't support >200k context via the standard
// Anthropic API at all, and Sonnet 4.6 (up to 1M context) bills that
// extended context at the regular rate with no above-200k surcharge. The
// same *_above_200k_tokens price fields are reused by both tier shapes;
// only the billing style differs per provider. Otherwise 200k keeps its
// separate proportional-only handling further down in CalculateTokenCosts.
//
// A price row may also opt into a 200k rule explicitly through
// LongContextPricingMode, regardless of provider: xAI bills the whole request
// at the long-context rates once the prompt reaches 200k tokens, inclusive.
// The explicit mode wins over the Gemini default, and is the same for every
// credential serving the model (an aggregator route included), because the
// client pays one tariff whichever route served it.
func fullSessionTiers(price *ModelPrice) []fullSessionTier {
	tiers := []fullSessionTier{
		{
			threshold:         tokenTiering512kThreshold,
			inputRate:         price.InputCostPerTokenAbove512k,
			outputRate:        price.OutputCostPerTokenAbove512k,
			cacheReadRate:     price.CacheReadInputTokenCostAbove512k,
			cacheCreationRate: price.CacheCreationInputTokenCostAbove512k,
		},
		{
			threshold:         tokenTiering272kThreshold,
			inputRate:         price.InputCostPerTokenAbove272k,
			outputRate:        price.OutputCostPerTokenAbove272k,
			cacheReadRate:     price.CacheReadInputTokenCostAbove272k,
			cacheCreationRate: price.CacheCreationInputTokenCostAbove272k,
		},
		{
			threshold:             tokenTiering256kThreshold,
			inputRate:             price.InputCostPerTokenAbove256k,
			outputRate:            price.OutputCostPerTokenAbove256k,
			cacheReadRate:         price.CacheReadInputTokenCostAbove256k,
			cacheCreationRate:     price.CacheCreationInputTokenCostAbove256k,
			explicitCacheReadRate: price.ExplicitCacheReadInputTokenCostAbove256k,
		},
	}
	if price.longContextFullRequest200kInclusive() || config.IsGoogleGeminiProvider(price.LiteLLMProvider) {
		tiers = append(tiers, fullSessionTier{
			threshold:         tokenTiering200kThreshold,
			inclusive:         price.longContextFullRequest200kInclusive(),
			inputRate:         price.InputCostPerTokenAbove200k,
			outputRate:        price.OutputCostPerTokenAbove200k,
			cacheReadRate:     price.CacheReadInputTokenCostAbove200k,
			cacheCreationRate: price.CacheCreationInputTokenCostAbove200k,
		})
	}
	return append(tiers,
		fullSessionTier{
			threshold:             tokenTiering128kThreshold,
			inputRate:             price.InputCostPerTokenAbove128k,
			outputRate:            price.OutputCostPerTokenAbove128k,
			cacheReadRate:         price.CacheReadInputTokenCostAbove128k,
			cacheCreationRate:     price.CacheCreationInputTokenCostAbove128k,
			explicitCacheReadRate: price.ExplicitCacheReadInputTokenCostAbove128k,
		},
		fullSessionTier{
			threshold:             tokenTiering32kThreshold,
			inputRate:             price.InputCostPerTokenAbove32k,
			outputRate:            price.OutputCostPerTokenAbove32k,
			cacheReadRate:         price.CacheReadInputTokenCostAbove32k,
			cacheCreationRate:     price.CacheCreationInputTokenCostAbove32k,
			explicitCacheReadRate: price.ExplicitCacheReadInputTokenCostAbove32k,
		},
	)
}

// fullSessionRate walks tiers (already sorted by descending threshold) and
// returns the rate selected by pick for the first tier whose threshold is
// exceeded by promptTokens and whose selected rate is actually configured
// (non-zero). Returns ok=false when no full-session tier applies, in which
// case callers should fall back to the existing 200k-proportional/base rate.
func fullSessionRate(tiers []fullSessionTier, promptTokens int, pick func(fullSessionTier) float64) (rate float64, ok bool) {
	for _, tier := range tiers {
		if promptTokens < tier.threshold || (promptTokens == tier.threshold && !tier.inclusive) {
			continue
		}
		if r := pick(tier); r > 0 {
			return r, true
		}
	}
	return 0, false
}

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
//
// If the provider also reports an explicit OutputTextTokens count smaller than the
// derived regular output tokens, that smaller value is used instead (e.g. Qwen's
// completion_tokens can include tokens not captured by any of the other output
// breakdowns, so the derived subtraction alone would overcount billable text tokens).
func CalculateTokenCosts(usage *converter.TokenUsage, price *ModelPrice) *converter.TokenCosts {
	return CalculateTokenCostsWithResolver(usage, price, nil)
}

// PriceResolver returns another model's price row from the same price source
// as the billed model (the organization tariff when one applies, the default
// price list otherwise), or nil when it has none.
type PriceResolver func(modelID string) *ModelPrice

// CalculateTokenCostsWithResolver is CalculateTokenCosts with a way to price
// usage billed at another model's tariff: images produced by a built-in
// image_generation tool are priced at ImageGenerationToolModel's row. With a
// nil resolver such images are not billed.
func CalculateTokenCostsWithResolver(usage *converter.TokenUsage, price *ModelPrice, resolve PriceResolver) *converter.TokenCosts {
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
	outputTextTokens := converterutil.NonNegativeTokenCount(usage.OutputTextTokens)
	reasoningTokens := converterutil.NonNegativeTokenCount(usage.ReasoningTokens)
	acceptedPredictionTokens := converterutil.NonNegativeTokenCount(usage.AcceptedPredictionTokens)
	rejectedPredictionTokens := converterutil.NonNegativeTokenCount(usage.RejectedPredictionTokens)
	imageCount := converterutil.NonNegativeTokenCount(usage.ImageCount)

	// Full-session tiers (32k/128k/256k/272k): the highest exceeded threshold
	// whose rate is configured wins and applies to the WHOLE request, not
	// just the excess. Falls back to the 200k proportional tier (unchanged
	// below) when no full-session tier matches.
	tiers := fullSessionTiers(price)
	fullSessionInputRate, fullSessionInputMatched := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.inputRate })
	fullSessionOutputRate, fullSessionOutputMatched := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.outputRate })
	fullSessionCacheReadRate, fullSessionCacheReadMatched := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheReadRate })
	fullSessionCacheCreationRate, fullSessionCacheCreationMatched := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheCreationRate })

	inputCostPerToken := price.InputCostPerToken
	if fullSessionInputMatched {
		inputCostPerToken = fullSessionInputRate
	}
	outputCostPerToken := price.OutputCostPerToken
	if fullSessionOutputMatched {
		outputCostPerToken = fullSessionOutputRate
	}

	cachedInputTokens := converterutil.NonNegativeTokenCount(usage.CachedInputTokens)
	cacheCreationTokens := converterutil.NonNegativeTokenCount(usage.CacheCreationTokens)

	// Calculate "regular" input tokens by subtracting specialized token types.
	// Vertex/OpenAI: audio/cached tokens are included in PromptTokens; Anthropic: same + cache creation.
	regularInputTokens := promptTokens - audioInputTokens - cachedInputTokens - cacheCreationTokens - imageTokens
	if regularInputTokens < 0 {
		regularInputTokens = 0
	}

	// Image tokens live inside PromptTokens, but a provider that serves a cached
	// image reports the same tokens inside CachedInputTokens as well (some model
	// does: prompt=340, cached=320, image=81). Charging the whole image bucket on
	// top of the cached one then bills the overlap twice -- once at the cache-read
	// rate and once at the image rate -- and the negative clamp above hides it.
	billableImageTokens := imageTokens
	if promptTokens > 0 {
		overlap := audioInputTokens + cachedInputTokens + cacheCreationTokens + imageTokens - promptTokens
		if overlap > 0 {
			billableImageTokens = converterutil.NonNegativeTokenCount(imageTokens - overlap)
		}
	}

	// The default 200k shape bills only the excess at the higher rate; a row
	// with an explicit long-context mode is billed by that mode alone. An
	// unrecognized mode value keeps the default rather than disabling both.
	proportional200k := !price.longContextFullRequest200kInclusive()

	// Regular input with full-session or 200k tiering
	switch {
	case fullSessionInputMatched:
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	case proportional200k && price.InputCostPerTokenAbove200k > 0 && promptTokens > tokenTiering200kThreshold:
		above := promptTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularInputTokens) * int64(above) / int64(promptTokens))
		regularBelow := regularInputTokens - regularAbove
		costs.InputCost = float64(regularBelow)*price.InputCostPerToken +
			float64(regularAbove)*price.InputCostPerTokenAbove200k
	default:
		costs.InputCost = float64(regularInputTokens) * inputCostPerToken
	}

	// Reasoning tokens are subtracted back out here because, by default,
	// CompletionTokens already includes them (see ModelPrice.ReasoningTokensAdditive
	// doc comment). Providers that report reasoning on top of CompletionTokens
	// instead (ReasoningTokensAdditive=true) must skip this subtraction, or these
	// tokens would be double-excluded: once here and once by never having been
	// part of CompletionTokens to begin with.
	reasoningTokensToSubtract := reasoningTokens
	if price.reasoningTokensAdditive(usage) {
		reasoningTokensToSubtract = 0
	}
	regularOutputTokens := completionTokens - audioOutputTokens - reasoningTokensToSubtract -
		acceptedPredictionTokens - rejectedPredictionTokens - outputImageTokens
	// Explicit text tokens, when smaller, cap the derived value (see doc comment above).
	if outputTextTokens > 0 && outputTextTokens < regularOutputTokens {
		regularOutputTokens = outputTextTokens
	}
	if regularOutputTokens < 0 {
		regularOutputTokens = 0
	}

	// Regular output with full-session or 200k tiering
	switch {
	case fullSessionOutputMatched:
		costs.OutputCost = float64(regularOutputTokens) * outputCostPerToken
	case proportional200k && price.OutputCostPerTokenAbove200k > 0 && completionTokens > tokenTiering200kThreshold:
		above := completionTokens - tokenTiering200kThreshold
		// Distribute regular tokens proportionally between below/above threshold
		regularAbove := int(int64(regularOutputTokens) * int64(above) / int64(completionTokens))
		regularBelow := regularOutputTokens - regularAbove
		costs.OutputCost = float64(regularBelow)*price.OutputCostPerToken +
			float64(regularAbove)*price.OutputCostPerTokenAbove200k
	default:
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
	cachedInputCost := 0.0
	if !price.CacheReadInputTokensFree {
		cachedInputCost = price.InputCostPerCachedToken
		if cachedInputCost == 0 {
			cachedInputCost = price.CacheReadInputTokenCost
		}
		if fullSessionCacheReadMatched {
			cachedInputCost = fullSessionCacheReadRate
		} else if promptTokens > tokenTiering200kThreshold && price.CacheReadInputTokenCostAbove200k > 0 {
			cachedInputCost = price.CacheReadInputTokenCostAbove200k
		}
		if cachedInputCost == 0 {
			cachedInputCost = inputCostPerToken
		}
	}
	cachedAudioTokens := converterutil.NonNegativeTokenCount(usage.CachedAudioInputTokens)
	if cachedAudioTokens > cachedInputTokens {
		cachedAudioTokens = cachedInputTokens
	}
	regularCachedTokens := cachedInputTokens - cachedAudioTokens
	cachedAudioCost := 0.0
	if !price.CacheReadInputTokensFree {
		cachedAudioCost = price.CacheReadInputAudioTokenCost
		if cachedAudioCost == 0 {
			cachedAudioCost = cachedInputCost
		}
	}

	// Cached read tokens are priced differently depending on cache mode.
	// Alibaba marks a request as explicit cache when
	// usage.prompt_tokens_details.cache_type == "ephemeral" (TokenUsage.CacheType);
	// those cached reads use the model's explicit_cache_read_input_token_cost
	// tariff (with its own full-session tiers, and its own dedicated audio rate
	// CacheReadInputAudioTokenCost — never the implicit-cache-derived
	// cachedAudioCost computed above) instead of the implicit
	// cache_read_input_token_cost. Explicit and implicit cache are mutually
	// exclusive, so CacheType decides the tariff. CacheReadInputTokensFree
	// gates the explicit tariff exactly like it gates the implicit one: when
	// set, cached reads stay free regardless of cache mode. When the model has
	// no explicit tariff configured (and isn't free), cached reads fall back
	// to the implicit cache-read rate (which is what would have applied
	// without this feature).
	if usage.CacheType == converter.CacheTypeExplicit && cachedInputTokens > 0 {
		explicitCacheReadRate := 0.0
		if !price.CacheReadInputTokensFree {
			explicitCacheReadRate = price.ExplicitCacheReadInputTokenCost
			if rate, ok := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.explicitCacheReadRate }); ok {
				explicitCacheReadRate = rate
			}
		}
		if explicitCacheReadRate > 0 {
			explicitCachedAudioCost := price.CacheReadInputAudioTokenCost
			if explicitCachedAudioCost == 0 {
				explicitCachedAudioCost = explicitCacheReadRate
			}
			costs.ExplicitCachedInputCost = float64(regularCachedTokens)*explicitCacheReadRate +
				float64(cachedAudioTokens)*explicitCachedAudioCost
		} else {
			costs.ExplicitCachedInputCost = float64(regularCachedTokens)*cachedInputCost +
				float64(cachedAudioTokens)*cachedAudioCost
		}
	} else {
		costs.CachedInputCost = float64(regularCachedTokens)*cachedInputCost +
			float64(cachedAudioTokens)*cachedAudioCost
	}

	cacheCreationCost := price.CacheCreationInputTokenCost
	cacheCreationFullSession := false
	if fullSessionCacheCreationMatched {
		cacheCreationCost = fullSessionCacheCreationRate
		cacheCreationFullSession = true
	} else if promptTokens > tokenTiering200kThreshold && price.CacheCreationInputTokenCostAbove200k > 0 {
		cacheCreationCost = price.CacheCreationInputTokenCostAbove200k
	}
	if cacheCreationCost == 0 {
		cacheCreationCost = inputCostPerToken
	}
	cacheCreation1hCost := cacheCreationCost
	if !cacheCreationFullSession {
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
	costs.ImageCost = float64(billableImageTokens) * inputImageCost

	// Generated image tokens are part of CompletionTokens and must not also be
	// charged as text. Prefer the token-based image rate when the provider reports
	// a token breakdown; otherwise use the per-image price for Imagen-style APIs.
	if outputImageTokens > 0 {
		outputImageCost := price.OutputCostPerImageToken
		if outputImageCost == 0 {
			outputImageCost = outputCostPerToken
		}
		costs.ImageCost += float64(outputImageTokens) * outputImageCost
	} else if imageCount > 0 {
		costs.ImageCost += price.outputImageCost(imageCount, usage.ImageBilling)
	}
	// Source images are only billed when the request actually produced output.
	if imageCount > 0 || outputImageTokens > 0 {
		costs.ImageCost += price.inputImageCost(usage.ImageBilling)
	}

	webSearchRequests := converterutil.NonNegativeTokenCount(usage.WebSearchRequests)
	if webSearchRequests > 0 {
		if price.webSearchBillingUnit() == "per_prompt" {
			webSearchRequests = 1
		}
		costs.WebSearchCost = float64(webSearchRequests) *
			price.webSearchCostPerQuery(usage.WebSearchContextSize)
	}

	// Other built-in server-side tools: per unit, never scaled by the
	// long-context tier, which only applies to tokens.
	costs.XSearchCost = float64(converterutil.NonNegativeTokenCount(usage.XSearchPosts))*price.XSearchCostPerPost +
		float64(converterutil.NonNegativeTokenCount(usage.XSearchProfiles))*price.XSearchCostPerProfile
	costs.CodeExecutionCost = float64(converterutil.NonNegativeTokenCount(usage.CodeExecutionCalls)) *
		price.toolCostPerCall(ToolCodeExecution)
	costs.AttachmentSearchCost = float64(converterutil.NonNegativeTokenCount(usage.AttachmentSearchCalls)) *
		price.toolCostPerCall(ToolAttachmentSearch)
	costs.CollectionsSearchCost = float64(converterutil.NonNegativeTokenCount(usage.CollectionsSearchCalls)) *
		price.toolCostPerCall(ToolCollectionsSearch)
	costs.ImageGenerationToolCost = price.imageGenerationToolCost(usage, resolve)
	costs.ToolUsageCost = costs.WebSearchCost +
		costs.XSearchCost +
		costs.CodeExecutionCost +
		costs.AttachmentSearchCost +
		costs.CollectionsSearchCost +
		costs.ImageGenerationToolCost

	// Calculate total. Tool charges enter once, through ToolUsageCost (web
	// search included).
	costs.TotalCost = costs.InputCost +
		costs.OutputCost +
		costs.AudioInputCost +
		costs.AudioOutputCost +
		costs.ReasoningCost +
		costs.CachedInputCost +
		costs.ExplicitCachedInputCost +
		costs.CacheCreationCost +
		costs.CachedOutputCost +
		costs.PredictionCost +
		costs.ImageCost +
		costs.ToolUsageCost

	return costs
}

// longContextFullRequest200kInclusive reports whether the row opted into
// xAI's long-context rule (see LongContextFullRequest200kInclusive).
func (p *ModelPrice) longContextFullRequest200kInclusive() bool {
	return p != nil && strings.EqualFold(strings.TrimSpace(p.LongContextPricingMode), LongContextFullRequest200kInclusive)
}

// reasoningTokensAdditive reports whether usage.ReasoningTokens comes on top
// of usage.CompletionTokens for this response: the row's fixed setting, or,
// with reasoning_tokens_accounting "auto", what the response itself showed.
func (p *ModelPrice) reasoningTokensAdditive(usage *converter.TokenUsage) bool {
	if strings.EqualFold(strings.TrimSpace(p.ReasoningTokensAccounting), ReasoningTokensAccountingAuto) {
		switch usage.ReasoningAccounting {
		case converter.ReasoningAccountingAdditive:
			return true
		case converter.ReasoningAccountingIncluded:
			return false
		}
	}
	return p.ReasoningTokensAdditive
}

// toolCostPerCall returns the per-call price of a built-in tool, looked up by
// its canonical key first and then by its alias. A tool priced under both
// names is charged once, at the canonical key's price.
func (p *ModelPrice) toolCostPerCall(tool string) float64 {
	if len(p.ToolCostPerCall) == 0 {
		return 0
	}
	if cost, ok := p.ToolCostPerCall[tool]; ok {
		return max(cost, 0)
	}
	for alias, canonical := range toolCostAliases {
		if canonical != tool {
			continue
		}
		if cost, ok := p.ToolCostPerCall[alias]; ok {
			return max(cost, 0)
		}
	}
	return 0
}

// imageGenerationToolCost prices the images a built-in image_generation tool
// produced, at ImageGenerationToolModel's tariff with its default request
// parameters (the tool takes no size or quality). Edits use the tier for
// edits and pay for one source image each, as the images endpoint does.
func (p *ModelPrice) imageGenerationToolCost(usage *converter.TokenUsage, resolve PriceResolver) float64 {
	generations := converterutil.NonNegativeTokenCount(usage.ImageToolGenerations)
	edits := converterutil.NonNegativeTokenCount(usage.ImageToolEdits)
	model := strings.TrimSpace(p.ImageGenerationToolModel)
	if generations+edits == 0 || model == "" || resolve == nil {
		return 0
	}
	imagePrice := resolve(model)
	if imagePrice == nil || imagePrice == p {
		return 0
	}
	cost := imagePrice.outputImageCost(generations, &converter.ImageBillingDetails{
		Operation: converter.ImageOperationGeneration,
	})
	if edits > 0 {
		edit := &converter.ImageBillingDetails{Operation: converter.ImageOperationEdit, InputImages: edits}
		cost += imagePrice.outputImageCost(edits, edit) + imagePrice.inputImageCost(edit)
	}
	return cost
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

// CalculateCostsWithResolver is CalculateCosts that can also price usage
// billed at another model's tariff (see CalculateTokenCostsWithResolver).
func (p *ModelPrice) CalculateCostsWithResolver(usage *converter.TokenUsage, resolve PriceResolver) *converter.TokenCosts {
	return CalculateTokenCostsWithResolver(usage, p, resolve)
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
