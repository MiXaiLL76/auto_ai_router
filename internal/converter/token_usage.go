package converter

import (
	"math"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// CacheTypeExplicit is the TokenUsage.CacheType value Alibaba/Qwen sets
// (usage.prompt_tokens_details.cache_type) when a request used an explicit
// cache marker (cache_control:{"type":"ephemeral"}), as opposed to implicit
// (automatic) caching, which reports no cache_type at all.
const CacheTypeExplicit = "ephemeral"

// TokenUsage.ReasoningAccounting values: how the provider counted reasoning
// tokens relative to completion_tokens in one response.
const (
	// ReasoningAccountingIncluded: completion_tokens already contains the
	// reasoning tokens (OpenAI semantics, total = prompt + completion).
	ReasoningAccountingIncluded = "included"
	// ReasoningAccountingAdditive: reasoning tokens come on top of
	// completion_tokens (xAI, total = prompt + completion + reasoning).
	ReasoningAccountingAdditive = "additive"
)

// TokenUsage is a universal format for token usage across all providers.
// Used by converters to return usage data without circular dependencies.
type TokenUsage struct {
	PromptTokens           int
	CompletionTokens       int
	AudioInputTokens       int
	AudioOutputTokens      int
	CachedInputTokens      int
	CachedAudioInputTokens int
	CacheCreationTokens    int
	CacheCreation5mTokens  int
	CacheCreation1hTokens  int
	// CacheType is the explicit-cache mode marker from
	// usage.prompt_tokens_details.cache_type (Alibaba/Qwen returns
	// CacheTypeExplicit when the request ran with an explicit cache marker).
	// Empty means the request did not use explicit cache — any cached_tokens
	// then come from implicit (automatic) caching. Explicit and implicit
	// cache are mutually exclusive per request/model; CacheType decides which
	// tariff cached read tokens are billed at (see models.CalculateTokenCosts).
	CacheType                string
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
	// ImageBilling carries per-image pricing inputs for image generation/edit
	// requests (nil otherwise). A pointer keeps TokenUsage comparable.
	ImageBilling *ImageBillingDetails

	// ServerToolUsageReported is true when the provider reported a
	// server-side tool usage object (xAI usage.server_side_tool_usage_details).
	// Its counts, zeros included, are then authoritative: a zero there means
	// no billable executions rather than missing data, so neither output items
	// nor citations may stand in for it, and WebSearchRequests comes from it.
	ServerToolUsageReported bool
	XSearchCalls            int // X Search calls (logged; X Search bills per fetched item)
	XSearchPosts            int // X posts fetched across all X Search calls, not de-duplicated
	XSearchProfiles         int // X user profiles fetched across all X Search calls
	CodeExecutionCalls      int // code_execution (alias code_interpreter)
	AttachmentSearchCalls   int // attachment_search over files attached to the request
	CollectionsSearchCalls  int // collections_search (alias file_search)
	MCPCalls                int // remote MCP calls (only their tokens are billed)
	// ImageToolGenerations and ImageToolEdits count the images a built-in
	// image_generation tool produced inside a chat/Responses request, priced
	// at the image model's own tariff (ModelPrice.ImageGenerationToolModel).
	ImageToolGenerations int
	ImageToolEdits       int
	// ReasoningAccounting is how this response counted reasoning tokens
	// (ReasoningAccountingIncluded/Additive), detected from its total_tokens;
	// empty when the response does not tell. Prices opt into using it with
	// reasoning_tokens_accounting: "auto".
	ReasoningAccounting string
	// ProviderCostUSD is the provider's own figure for what the request cost
	// us (xAI usage.cost_in_usd_ticks, aggregators' usage.cost). It is kept
	// for reconciliation only and never added to the billed price.
	ProviderCostUSD float64
}

// HasServerToolUsage reports whether any built-in tool other than web search
// was used, i.e. whether the per-tool counters are worth logging.
func (tu *TokenUsage) HasServerToolUsage() bool {
	if tu == nil {
		return false
	}
	return tu.ServerToolUsageReported || tu.XSearchCalls > 0 || tu.XSearchPosts > 0 ||
		tu.XSearchProfiles > 0 || tu.CodeExecutionCalls > 0 || tu.AttachmentSearchCalls > 0 ||
		tu.CollectionsSearchCalls > 0 || tu.MCPCalls > 0 || tu.ImageToolGenerations > 0 ||
		tu.ImageToolEdits > 0
}

// Image request operations used by per-image price tiers.
const (
	ImageOperationGeneration = "generation"
	ImageOperationEdit       = "edit"
)

// ImageBillingDetails holds the request and response facts that per-image
// price tiers depend on, beyond the plain ImageCount.
type ImageBillingDetails struct {
	// Operation is ImageOperationGeneration or ImageOperationEdit.
	Operation string
	// RequestParams are the request's short scalar parameters (e.g.
	// "resolution", "quality", "layer_decomposition"), keys and values lower-cased.
	RequestParams map[string]string
	// OutputPixels lists width×height of each delivered image in response
	// order; 0 means that image's size is unknown.
	OutputPixels []int64
	// InputImages is the number of source images the request carried.
	InputImages int
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
	tu.XSearchCalls = converterutil.NonNegativeTokenCount(tu.XSearchCalls)
	tu.XSearchPosts = converterutil.NonNegativeTokenCount(tu.XSearchPosts)
	tu.XSearchProfiles = converterutil.NonNegativeTokenCount(tu.XSearchProfiles)
	tu.CodeExecutionCalls = converterutil.NonNegativeTokenCount(tu.CodeExecutionCalls)
	tu.AttachmentSearchCalls = converterutil.NonNegativeTokenCount(tu.AttachmentSearchCalls)
	tu.CollectionsSearchCalls = converterutil.NonNegativeTokenCount(tu.CollectionsSearchCalls)
	tu.MCPCalls = converterutil.NonNegativeTokenCount(tu.MCPCalls)
	tu.ImageToolGenerations = converterutil.NonNegativeTokenCount(tu.ImageToolGenerations)
	tu.ImageToolEdits = converterutil.NonNegativeTokenCount(tu.ImageToolEdits)
	if tu.ReasoningAccounting != ReasoningAccountingIncluded && tu.ReasoningAccounting != ReasoningAccountingAdditive {
		tu.ReasoningAccounting = ""
	}
	if !(tu.ProviderCostUSD > 0) || math.IsInf(tu.ProviderCostUSD, 0) {
		tu.ProviderCostUSD = 0
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
	if src.CacheType != "" {
		tu.CacheType = src.CacheType
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
	tu.mergeToolUsage(src)
	if src.WebSearchContextSize != "" {
		tu.WebSearchContextSize = src.WebSearchContextSize
	}
	if src.ImageBilling != nil {
		tu.ImageBilling = src.ImageBilling
	}
	if src.ReasoningAccounting != "" {
		tu.ReasoningAccounting = src.ReasoningAccounting
	}
	if src.ProviderCostUSD != 0 {
		tu.ProviderCostUSD = src.ProviderCostUSD
	}
}

// MergeUsageExtensions merges from src only what the generic usage
// extraction derives beyond token counts: the built-in tool counters (with
// the same cumulative semantics as MergeNonZero), the reasoning accounting
// and the provider's own cost. It is for stream handlers that read token
// counts from a typed usage object and would otherwise drop these.
func (tu *TokenUsage) MergeUsageExtensions(src *TokenUsage) {
	if tu == nil || src == nil {
		return
	}
	tu.mergeToolUsage(src)
	if src.ReasoningAccounting != "" {
		tu.ReasoningAccounting = src.ReasoningAccounting
	}
	if src.ProviderCostUSD != 0 {
		tu.ProviderCostUSD = src.ProviderCostUSD
	}
}

// mergeToolUsage merges the built-in tool counters. A provider-reported tool
// usage object is cumulative for the whole request (xAI repeats it on every
// usage-bearing stream chunk, and the terminal event carries the total), so
// the latest reported object replaces the counters outright, zeros included:
// summing them would bill repeated stream events twice, and keeping an
// earlier non-zero value would override the provider's authoritative zero.
// Once such an object was seen, counts derived from output items or
// citations of later chunks no longer apply.
func (tu *TokenUsage) mergeToolUsage(src *TokenUsage) {
	switch {
	case src.ServerToolUsageReported:
		tu.ServerToolUsageReported = true
		tu.WebSearchRequests = src.WebSearchRequests
		tu.XSearchCalls = src.XSearchCalls
		tu.XSearchPosts = src.XSearchPosts
		tu.XSearchProfiles = src.XSearchProfiles
		tu.CodeExecutionCalls = src.CodeExecutionCalls
		tu.AttachmentSearchCalls = src.AttachmentSearchCalls
		tu.CollectionsSearchCalls = src.CollectionsSearchCalls
		tu.MCPCalls = src.MCPCalls
		tu.ImageToolGenerations = src.ImageToolGenerations
		tu.ImageToolEdits = src.ImageToolEdits
	case tu.ServerToolUsageReported:
		// Keep the authoritative counters.
	default:
		mergeNonZeroInt(&tu.WebSearchRequests, src.WebSearchRequests)
		mergeNonZeroInt(&tu.XSearchCalls, src.XSearchCalls)
		mergeNonZeroInt(&tu.XSearchPosts, src.XSearchPosts)
		mergeNonZeroInt(&tu.XSearchProfiles, src.XSearchProfiles)
		mergeNonZeroInt(&tu.CodeExecutionCalls, src.CodeExecutionCalls)
		mergeNonZeroInt(&tu.AttachmentSearchCalls, src.AttachmentSearchCalls)
		mergeNonZeroInt(&tu.CollectionsSearchCalls, src.CollectionsSearchCalls)
		mergeNonZeroInt(&tu.MCPCalls, src.MCPCalls)
		mergeNonZeroInt(&tu.ImageToolGenerations, src.ImageToolGenerations)
		mergeNonZeroInt(&tu.ImageToolEdits, src.ImageToolEdits)
	}
}

func mergeNonZeroInt(dst *int, src int) {
	if src != 0 {
		*dst = src
	}
}

// TokenCosts contains cost breakdown by token type
type TokenCosts struct {
	InputCost       float64
	OutputCost      float64
	AudioInputCost  float64
	AudioOutputCost float64
	ReasoningCost   float64
	CachedInputCost float64
	// ExplicitCachedInputCost is the cost of cached prompt tokens read in
	// explicit cache mode (Alibaba/Qwen CacheType == CacheTypeExplicit),
	// billed at the model's explicit_cache_read_input_token_cost tariff. When
	// the model has no explicit tariff configured (or it's free via
	// CacheReadInputTokensFree), this instead holds the cost computed at the
	// implicit cache-read rate — i.e. the same amount CachedInputCost would
	// have held without this feature. It is kept separate from CachedInputCost
	// (which stays zero whenever CacheType == CacheTypeExplicit) so spend logs
	// can attribute the two cache tariffs independently; both are added into
	// TotalCost.
	ExplicitCachedInputCost float64
	CacheCreationCost       float64
	CachedOutputCost        float64
	PredictionCost          float64
	ImageCost               float64
	WebSearchCost           float64
	// Built-in server-side tool charges other than web search. Each is billed
	// per unit (call, fetched X post/profile, generated image), never by
	// tokens, and is part of TotalCost exactly once.
	XSearchCost             float64
	CodeExecutionCost       float64
	AttachmentSearchCost    float64
	CollectionsSearchCost   float64
	ImageGenerationToolCost float64
	// ToolUsageCost is the sum of every built-in tool charge above, web search
	// included. It is a breakdown figure already contained in TotalCost.
	ToolUsageCost     float64
	TotalCost         float64
	MarginPercent     float64
	MarginFixedAmount float64
	MarginTotalAmount float64
}

func NormalizeWebSearchContextSize(size string) string {
	switch size {
	case "low", "medium", "high":
		return size
	default:
		return "medium"
	}
}
