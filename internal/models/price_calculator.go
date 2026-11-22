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
	tokenTiering100kThreshold = 100_000
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
	// ToolWebFetch is the built-in web fetch tool: a page fetch the router
	// executed, billed per successful call (one that returned page content).
	ToolWebFetch = "web_fetch"
	// ToolMoonshotSearch is the Moonshot Basic search the router ran on the
	// model's behalf: the provider's own search API, billed per successful
	// call (one that returned results), apart from the model's tokens.
	ToolMoonshotSearch = "moonshot_search"
	// ToolMoonshotSearchPro is the Moonshot Pro search, priced apart from
	// the Basic one.
	ToolMoonshotSearchPro = "moonshot_search_pro"
	// ToolMoonshotFetch is the Moonshot page fetch, billed per successful
	// call (one that returned page content).
	ToolMoonshotFetch = "moonshot_fetch"
)

var toolCostAliases = map[string]string{
	"code_interpreter": ToolCodeExecution,
	"document_search":  ToolAttachmentSearch,
	"file_search":      ToolCollectionsSearch,
	// "fetch" is the name DeepSeek's own price lists use for the web fetch
	// tool; "web_fetch" is the tool's type name in a request.
	"fetch": ToolWebFetch,
	// The Moonshot tools' own spellings: the tool price document names them
	// the way the request does.
	"moonshot_search":     ToolMoonshotSearch,
	"moonshot_search_pro": ToolMoonshotSearchPro,
	"moonshot_fetch":      ToolMoonshotFetch,
}

// fullSessionTier describes one "Alibaba Cloud style" pricing bracket: once
// the prompt exceeds threshold, the ENTIRE request (not just the excess) is
// billed at this tier's rates. Zero-value rates mean "not configured for
// this tier" and are skipped by fullSessionRate. An inclusive tier already
// applies when the prompt equals threshold.
//
// cacheCreation1hRate is the tier's own rate for 1-hour cache writes. A tier
// without one bills both TTLs at cacheCreationRate, as all tiers did before
// the 100k tier, whose provider prices the two TTLs differently.
//
// cacheCreation12hRate is the tier's own rate for OpenAI's 12-hour
// cache-retention writes, same shape as cacheCreation1hRate: a tier without
// one bills the 12h TTL at cacheCreationRate too.
type fullSessionTier struct {
	threshold             int
	inclusive             bool
	inputRate             float64
	outputRate            float64
	cacheReadRate         float64
	cacheCreationRate     float64
	cacheCreation1hRate   float64
	cacheCreation12hRate  float64
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
			threshold:            tokenTiering272kThreshold,
			inputRate:            price.InputCostPerTokenAbove272k,
			outputRate:           price.OutputCostPerTokenAbove272k,
			cacheReadRate:        price.CacheReadInputTokenCostAbove272k,
			cacheCreationRate:    price.CacheCreationInputTokenCostAbove272k,
			cacheCreation12hRate: price.CacheCreationInputTokenCost12hAbove272k,
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
			threshold:           tokenTiering100kThreshold,
			inputRate:           price.InputCostPerTokenAbove100k,
			outputRate:          price.OutputCostPerTokenAbove100k,
			cacheReadRate:       price.CacheReadInputTokenCostAbove100k,
			cacheCreationRate:   price.CacheCreationInputTokenCostAbove100k,
			cacheCreation1hRate: price.CacheCreationInputTokenCostAbove1hrAbove100k,
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
	tier, ok := fullSessionTierFor(tiers, promptTokens, pick)
	if !ok {
		return 0, false
	}
	return pick(tier), true
}

// fullSessionTierFor is fullSessionRate returning the matched tier itself, for
// callers that need a second rate from the same tier.
func fullSessionTierFor(tiers []fullSessionTier, promptTokens int, pick func(fullSessionTier) float64) (fullSessionTier, bool) {
	for _, tier := range tiers {
		if promptTokens < tier.threshold || (promptTokens == tier.threshold && !tier.inclusive) {
			continue
		}
		if pick(tier) > 0 {
			return tier, true
		}
	}
	return fullSessionTier{}, false
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

	// Full-session rates are selected independently for each token category
	// using the full prompt size. The regular output's proportional tier
	// instead uses the completion size, regardless of the prompt size.
	tiers := fullSessionTiers(price)
	inputRate, inputTier := fullSessionRate(tiers, usage.PromptTokens, func(t fullSessionTier) float64 { return t.inputRate })
	outputRate, outputTier := fullSessionRate(tiers, usage.PromptTokens, func(t fullSessionTier) float64 { return t.outputRate })
	inputRate = rateOr(inputRate, price.InputCostPerToken)
	outputRate = rateOr(outputRate, price.OutputCostPerToken)
	proportional := !price.longContextFullRequest200kInclusive()

	costs := &converter.TokenCosts{
		InputCost: regularTokenCost(regularInputTokens(usage), usage.PromptTokens, inputRate,
			price.InputCostPerTokenAbove200k, proportional && !inputTier),
		OutputCost: regularTokenCost(regularOutputTokens(usage, price), usage.CompletionTokens, outputRate,
			price.OutputCostPerTokenAbove200k, proportional && !outputTier),
		AudioInputCost:  float64(usage.AudioInputTokens) * rateOr(price.InputCostPerAudioToken, inputRate),
		AudioOutputCost: float64(usage.AudioOutputTokens) * rateOr(price.OutputCostPerAudioToken, outputRate),
	}
	price.calculateCacheReadCosts(costs, usage, tiers, inputRate)
	costs.CacheCreationCost = price.calculateCacheWriteCost(usage, tiers, inputRate)
	price.calculateOutputDetailCosts(costs, usage, outputRate, outputTier)
	price.calculateImageCosts(costs, usage, inputRate, outputRate)
	price.calculateToolCosts(costs, usage, resolve)

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
		costs.VideoInputCost +
		costs.ToolUsageCost
	return costs
}

// rateOr uses zero as the unset-price sentinel used in price rows.
func rateOr(rate, fallback float64) float64 {
	if rate == 0 {
		return fallback
	}
	return rate
}

// regularTokenCost allocates text tokens proportionally when the full input
// or output crosses 200k. Integer division keeps the fractional token in the
// base bracket. Keep the two products separate to preserve billing rounding.
func regularTokenCost(tokens, total int, baseRate, aboveRate float64, proportional bool) float64 {
	if proportional && aboveRate > 0 && total > tokenTiering200kThreshold {
		above := int(int64(tokens) * int64(total-tokenTiering200kThreshold) / int64(total))
		return float64(tokens-above)*baseRate + float64(above)*aboveRate
	}
	return float64(tokens) * baseRate
}

func regularInputTokens(usage *converter.TokenUsage) int {
	cacheCreationTokens := converterutil.NonNegativeTokenCount(usage.CacheCreationTokens)
	// Calculate "regular" input tokens by subtracting specialized token types.
	// Vertex/OpenAI: audio/cached tokens are included in PromptTokens; Anthropic: same + cache creation.
	// Image and video tokens are part of PromptTokens too, so only the text
	// remainder is priced at the regular input rate.
	regularInputTokens := usage.PromptTokens - usage.AudioInputTokens - usage.CachedInputTokens - cacheCreationTokens - usage.ImageTokens - usage.VideoInputTokens
	return max(0, regularInputTokens)
}

func regularOutputTokens(usage *converter.TokenUsage, price *ModelPrice) int {
	// Reasoning tokens are subtracted back out here because, by default,
	// CompletionTokens already includes them (see ModelPrice.ReasoningTokensAdditive
	// doc comment). Providers that report reasoning on top of CompletionTokens
	// instead (ReasoningTokensAdditive=true) must skip this subtraction, or these
	// tokens would be double-excluded: once here and once by never having been
	// part of CompletionTokens to begin with.
	reasoningTokensToSubtract := usage.ReasoningTokens
	if price.reasoningTokensAdditive(usage) {
		reasoningTokensToSubtract = 0
	}
	regularOutputTokens := usage.CompletionTokens - usage.AudioOutputTokens - reasoningTokensToSubtract -
		usage.AcceptedPredictionTokens - usage.RejectedPredictionTokens - usage.OutputImageTokens
	// Explicit text tokens, when smaller, cap the derived value (see doc comment above).
	if usage.OutputTextTokens > 0 && usage.OutputTextTokens < regularOutputTokens {
		regularOutputTokens = usage.OutputTextTokens
	}
	return max(0, regularOutputTokens)
}

func (p *ModelPrice) calculateCacheReadCosts(costs *converter.TokenCosts, usage *converter.TokenUsage, tiers []fullSessionTier, inputCostPerToken float64) {
	// Cached read tokens: prefer explicit cached price, fall back to LiteLLM alias,
	// then fall back to regular rate (no discount known).
	cachedInputTokens := usage.CachedInputTokens
	promptTokens := usage.PromptTokens
	cachedInputCost := 0.0
	if !p.CacheReadInputTokensFree {
		cachedInputCost = p.InputCostPerCachedToken
		if cachedInputCost == 0 {
			cachedInputCost = p.CacheReadInputTokenCost
		}
		if rate, matched := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheReadRate }); matched {
			cachedInputCost = rate
		} else if promptTokens > tokenTiering200kThreshold && p.CacheReadInputTokenCostAbove200k > 0 {
			cachedInputCost = p.CacheReadInputTokenCostAbove200k
		}
		if cachedInputCost == 0 {
			cachedInputCost = inputCostPerToken
		}
	}
	// Normalize already bounds cached audio by the total cached input.
	cachedAudioTokens := usage.CachedAudioInputTokens
	regularCachedTokens := cachedInputTokens - cachedAudioTokens
	cachedAudioCost := 0.0
	if !p.CacheReadInputTokensFree {
		cachedAudioCost = rateOr(p.CacheReadInputAudioTokenCost, cachedInputCost)
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
		if !p.CacheReadInputTokensFree {
			explicitCacheReadRate = p.ExplicitCacheReadInputTokenCost
			if rate, ok := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.explicitCacheReadRate }); ok {
				explicitCacheReadRate = rate
			}
		}
		if explicitCacheReadRate > 0 {
			cachedInputCost = explicitCacheReadRate
			cachedAudioCost = rateOr(p.CacheReadInputAudioTokenCost, explicitCacheReadRate)
		}
		costs.ExplicitCachedInputCost = float64(regularCachedTokens)*cachedInputCost + float64(cachedAudioTokens)*cachedAudioCost
		return
	}
	costs.CachedInputCost = float64(regularCachedTokens)*cachedInputCost + float64(cachedAudioTokens)*cachedAudioCost
}

func (p *ModelPrice) calculateCacheWriteCost(usage *converter.TokenUsage, tiers []fullSessionTier, inputCostPerToken float64) float64 {
	promptTokens := usage.PromptTokens
	cacheCreationTokens := converterutil.NonNegativeTokenCount(usage.CacheCreationTokens)
	fullSessionCacheCreationTier, fullSessionCacheCreationMatched := fullSessionTierFor(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheCreationRate })
	cacheCreationCost := p.CacheCreationInputTokenCost
	if fullSessionCacheCreationMatched {
		cacheCreationCost = fullSessionCacheCreationTier.cacheCreationRate
	} else if promptTokens > tokenTiering200kThreshold && p.CacheCreationInputTokenCostAbove200k > 0 {
		cacheCreationCost = p.CacheCreationInputTokenCostAbove200k
	}
	if cacheCreationCost == 0 {
		cacheCreationCost = inputCostPerToken
	}
	// 1-hour writes: a full-session tier bills them at its own 1-hour rate when
	// it has one (the 100k tier), otherwise at its single write rate.
	cacheCreation1hCost := cacheCreationCost
	if fullSessionCacheCreationMatched {
		if rate := fullSessionCacheCreationTier.cacheCreation1hRate; rate > 0 {
			cacheCreation1hCost = rate
		}
	} else {
		cacheCreation1hCost = p.CacheCreationInputTokenCostAbove1hr
		if rate, ok := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheCreation1hRate }); ok {
			// A tier priced only for 1-hour writes still applies to them.
			cacheCreation1hCost = rate
		} else if promptTokens > tokenTiering200kThreshold && p.CacheCreationInputTokenCostAbove1hrAbove200k > 0 {
			cacheCreation1hCost = p.CacheCreationInputTokenCostAbove1hrAbove200k
		}
		if cacheCreation1hCost == 0 {
			cacheCreation1hCost = cacheCreationCost
		}
	}

	// 12-hour cache-retention writes: their own tier, priced apart from the
	// 30m/1h writes, and resolved from the full-session tier list the same way
	// cacheCreation1hRate is above, rather than a threshold hardcoded to 272k
	// — so the right bracket's 12h rate applies (today, only the 272k tier has
	// one) regardless of which OTHER tier (if any) matches the base
	// cache-creation rate for the same request, and a model that later needs
	// its own 512k/256k/128k/100k 12h rate only has to add that price field
	// and wire it into fullSessionTiers, with no change needed here. Falls
	// back to the base write rate only when the model prices no 12h tier at
	// all — a model that does is never silently billed at the cheaper 30m
	// rate.
	cacheCreation12hCost := p.CacheCreationInputTokenCost12h
	if rate, ok := fullSessionRate(tiers, promptTokens, func(t fullSessionTier) float64 { return t.cacheCreation12hRate }); ok {
		cacheCreation12hCost = rate
	}
	if cacheCreation12hCost == 0 {
		cacheCreation12hCost = cacheCreationCost
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
	cacheCreation12hTokens := converterutil.NonNegativeTokenCount(usage.CacheCreation12hTokens)
	if cacheCreation12hTokens > cacheCreationTokens-cacheCreation5mTokens-cacheCreation1hTokens {
		cacheCreation12hTokens = cacheCreationTokens - cacheCreation5mTokens - cacheCreation1hTokens
	}
	unclassifiedCacheCreationTokens := cacheCreationTokens - cacheCreation5mTokens - cacheCreation1hTokens - cacheCreation12hTokens
	return float64(unclassifiedCacheCreationTokens+cacheCreation5mTokens)*cacheCreationCost +
		float64(cacheCreation1hTokens)*cacheCreation1hCost +
		float64(cacheCreation12hTokens)*cacheCreation12hCost
}

func (p *ModelPrice) calculateOutputDetailCosts(costs *converter.TokenCosts, usage *converter.TokenUsage, outputCostPerToken float64, fullSessionOutputMatched bool) {
	cachedOutputCost := rateOr(p.OutputCostPerCachedToken, outputCostPerToken)
	costs.CachedOutputCost = float64(usage.CachedOutputTokens) * cachedOutputCost

	// Reasoning tokens with fallback. A long-context mode row bills reasoning
	// with the rest of the output at the tier's output rate: there is no
	// long-context reasoning rate, and the base reasoning rate would undercut it.
	reasoningCost := p.OutputCostPerReasoningToken
	if reasoningCost == 0 || (fullSessionOutputMatched && p.longContextFullRequest200kInclusive()) {
		reasoningCost = outputCostPerToken
	}
	costs.ReasoningCost = float64(usage.ReasoningTokens) * reasoningCost

	// Prediction tokens with fallback (accepted tokens)
	predictionCost := rateOr(p.OutputCostPerPredictionToken, outputCostPerToken)
	costs.PredictionCost = float64(usage.AcceptedPredictionTokens) * predictionCost

	// Rejected prediction tokens count as regular output tokens
	costs.PredictionCost += float64(usage.RejectedPredictionTokens) * outputCostPerToken
}

func (p *ModelPrice) calculateImageCosts(costs *converter.TokenCosts, usage *converter.TokenUsage, inputCostPerToken, outputCostPerToken float64) {
	cacheCreationTokens := converterutil.NonNegativeTokenCount(usage.CacheCreationTokens)
	// Image tokens live inside PromptTokens, but a provider that serves a cached
	// image reports the same tokens inside CachedInputTokens as well (some model
	// does: prompt=340, cached=320, image=81). Charging the whole image bucket on
	// top of the cached one then bills the overlap twice -- once at the cache-read
	// rate and once at the image rate -- and the negative clamp above hides it.
	// Video tokens share the same bucket; the overlap is taken off images first.
	billableImageTokens := usage.ImageTokens
	billableVideoTokens := usage.VideoInputTokens
	if usage.PromptTokens > 0 {
		overlap := usage.AudioInputTokens + usage.CachedInputTokens + cacheCreationTokens + usage.ImageTokens + usage.VideoInputTokens - usage.PromptTokens
		if overlap > 0 {
			fromImages := min(overlap, usage.ImageTokens)
			billableImageTokens = usage.ImageTokens - fromImages
			billableVideoTokens = converterutil.NonNegativeTokenCount(usage.VideoInputTokens - (overlap - fromImages))
		}
	}

	// Input image tokens are part of PromptTokens. Price them separately when a
	// modality-specific rate exists, otherwise keep the regular input rate.
	inputImageCost := rateOr(p.InputCostPerImageToken, inputCostPerToken)
	costs.ImageCost = float64(billableImageTokens) * inputImageCost

	// Input video tokens: own rate when configured, otherwise the image rate
	// (video used to be reported inside the image counter).
	inputVideoCost := rateOr(p.InputCostPerVideoToken, inputImageCost)
	costs.VideoInputCost = float64(billableVideoTokens) * inputVideoCost

	// Generated image tokens are part of CompletionTokens and must not also be
	// charged as text. Prefer the token-based image rate when the provider reports
	// a token breakdown; otherwise use the per-image price for Imagen-style APIs.
	if usage.OutputImageTokens > 0 {
		outputImageCost := rateOr(p.OutputCostPerImageToken, outputCostPerToken)
		costs.ImageCost += float64(usage.OutputImageTokens) * outputImageCost
	} else if usage.ImageCount > 0 {
		costs.ImageCost += p.outputImageCost(usage.ImageCount, usage.ImageBilling)
	}
	// Source images are only billed when the request actually produced output.
	if usage.ImageCount > 0 || usage.OutputImageTokens > 0 {
		costs.ImageCost += p.inputImageCost(usage.ImageBilling)
	}
}

func (p *ModelPrice) calculateToolCosts(costs *converter.TokenCosts, usage *converter.TokenUsage, resolve PriceResolver) {
	webSearchRequests := usage.WebSearchRequests
	if webSearchRequests > 0 {
		if p.webSearchBillingUnit() == "per_prompt" {
			webSearchRequests = 1
		}
		costs.WebSearchCost = float64(webSearchRequests) *
			p.webSearchCostPerQuery(usage.WebSearchContextSize)
	}

	// Other built-in server-side tools: per unit, never scaled by the
	// long-context tier, which only applies to tokens.
	costs.XSearchCost = float64(usage.XSearchPosts)*p.XSearchCostPerPost +
		float64(usage.XSearchProfiles)*p.XSearchCostPerProfile
	costs.CodeExecutionCost = float64(usage.CodeExecutionCalls) *
		p.toolCostPerCall(ToolCodeExecution)
	costs.AttachmentSearchCost = float64(usage.AttachmentSearchCalls) *
		p.toolCostPerCall(ToolAttachmentSearch)
	costs.CollectionsSearchCost = float64(usage.CollectionsSearchCalls) *
		p.toolCostPerCall(ToolCollectionsSearch)
	// The built-in web fetch the router executed: billed per successful call
	// (one that returned page content), never by tokens. A failed or empty
	// fetch is never counted in usage.WebFetchRequests, so it costs nothing.
	costs.WebFetchCost = float64(usage.WebFetchRequests) *
		p.toolCostPerCall(ToolWebFetch)
	// The router's own Moonshot calls, billed at the tool price document's
	// per-call prices. Each is charged only for a call that actually
	// succeeded and produced something: the counters carry only those.
	//
	// The prices are the FINAL client prices -- the rate is already folded
	// into them and is not applied again.
	costs.MoonshotSearchCost = float64(usage.MoonshotSearchRequests) *
		p.toolCostPerCall(ToolMoonshotSearch)
	costs.MoonshotSearchProCost = float64(usage.MoonshotSearchProRequests) *
		p.toolCostPerCall(ToolMoonshotSearchPro)
	costs.MoonshotFetchCost = float64(usage.MoonshotFetchRequests) *
		p.toolCostPerCall(ToolMoonshotFetch)
	costs.ImageGenerationToolCost = p.imageGenerationToolCost(usage, resolve)
	costs.ToolUsageCost = costs.WebSearchCost +
		costs.WebFetchCost +
		costs.MoonshotSearchCost +
		costs.MoonshotSearchProCost +
		costs.MoonshotFetchCost +
		costs.XSearchCost +
		costs.CodeExecutionCost +
		costs.AttachmentSearchCost +
		costs.CollectionsSearchCost +
		costs.ImageGenerationToolCost
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
// parameters (the tool takes no size or quality). Each edit is priced as its
// own images-endpoint edit request: the tier for edits plus one source image,
// with the row's free source-image allowance applying to every edit.
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
		edit := &converter.ImageBillingDetails{Operation: converter.ImageOperationEdit, InputImages: 1}
		cost += imagePrice.outputImageCost(edits, edit) + float64(edits)*imagePrice.inputImageCost(edit)
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
