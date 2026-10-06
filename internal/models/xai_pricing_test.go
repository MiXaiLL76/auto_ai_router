package models

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// grokTestPrices is a grok-4.7 row using every xAI billing field plus the
// image model its image_generation tool uses, decoded the same lenient way
// the price file is.
const grokTestPrices = `{
  "grok-4.7": {
    "input_cost_per_token": 0.0000026,
    "output_cost_per_token": 0.0000078,
    "cache_read_input_token_cost": 0.00000065,
    "input_cost_per_token_above_200k_tokens": 0.0000052,
    "output_cost_per_token_above_200k_tokens": 0.0000156,
    "cache_read_input_token_cost_above_200k_tokens": 0.0000013,
    "rate": 1.3,
    "supports_web_search": true,
    "search_context_cost_per_query": {"search_context_size_low": 0.0065, "search_context_size_medium": 0.0065, "search_context_size_high": 0.0065},
    "web_search_billing_unit": "per_query",
    "long_context_pricing_mode": "full_request_200k_inclusive",
    "reasoning_tokens_accounting": "auto",
    "reasoning_tokens_additive": true,
    "tool_cost_per_call": {"code_execution": 0.0065, "attachment_search": 0.013, "collections_search": 0.00325},
    "x_search_cost_per_post": 0.0065,
    "x_search_cost_per_profile": 0.013,
    "image_generation_tool_model": "grok-imagine-image-2.0"
  },
  "grok-imagine-image-2.0": {
    "output_cost_per_image": 0.104,
    "input_cost_per_image": 0.013,
    "image_request_defaults": {"resolution": "1k", "quality": "auto"},
    "output_cost_per_image_tiers": [
      {"operation": "edit", "when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.078},
      {"when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.052}
    ],
    "rate": 1.3
  }
}`

func loadGrokTestPrices(t *testing.T) map[string]*ModelPrice {
	t.Helper()
	var prices map[string]*ModelPrice
	require.NoError(t, json.Unmarshal([]byte(grokTestPrices), &prices))
	return prices
}

// grokExpectedTokenCost is the expected token bill: below 200k prompt
// tokens the base rates, from 200k on (inclusive) the long-context rates for
// every token — input, cached input, visible output and reasoning alike.
func grokExpectedTokenCost(prompt, cached, visibleOutput, reasoning int) float64 {
	input, cachedRate, output := 0.0000026, 0.00000065, 0.0000078
	if prompt >= 200_000 {
		input, cachedRate, output = 0.0000052, 0.0000013, 0.0000156
	}
	return float64(prompt-cached)*input + float64(cached)*cachedRate + float64(visibleOutput+reasoning)*output
}

func TestGrokPricing_LongContextBoundaryBothRoutes(t *testing.T) {
	price := loadGrokTestPrices(t)["grok-4.7"]
	const cached, visible, reasoning = 50_000, 1_000, 3_000

	for _, prompt := range []int{100_000, 199_999, 200_000, 200_001, 450_000} {
		want := grokExpectedTokenCost(prompt, cached, visible, reasoning)
		// xAI direct: reasoning on top of completion_tokens.
		xai := &converter.TokenUsage{
			PromptTokens: prompt, CompletionTokens: visible, CachedInputTokens: cached,
			ReasoningTokens: reasoning, ReasoningAccounting: converter.ReasoningAccountingAdditive,
		}
		// Requesty: reasoning inside completion_tokens.
		requesty := &converter.TokenUsage{
			PromptTokens: prompt, CompletionTokens: visible + reasoning, CachedInputTokens: cached,
			ReasoningTokens: reasoning, ReasoningAccounting: converter.ReasoningAccountingIncluded,
		}

		assert.InDelta(t, want, price.CalculateCost(xai), 1e-12, "xAI route, prompt %d", prompt)
		assert.InDelta(t, want, price.CalculateCost(requesty), 1e-12, "Requesty route, prompt %d", prompt)
	}
}

func TestGrokPricing_LongContextAppliesToEveryTokenCategory(t *testing.T) {
	price := loadGrokTestPrices(t)["grok-4.7"]
	usage := &converter.TokenUsage{
		PromptTokens: 200_000, CompletionTokens: 100, CachedInputTokens: 150_000,
		ReasoningTokens: 400, ReasoningAccounting: converter.ReasoningAccountingAdditive,
	}

	costs := price.CalculateCosts(usage)

	require.NotNil(t, costs)
	assert.InDelta(t, 50_000*0.0000052, costs.InputCost, 1e-12)
	assert.InDelta(t, 150_000*0.0000013, costs.CachedInputCost, 1e-12)
	assert.InDelta(t, 100*0.0000156, costs.OutputCost, 1e-12)
	assert.InDelta(t, 400*0.0000156, costs.ReasoningCost, 1e-12)
}

func TestGrokPricing_ReasoningAccountingFallsBackToFixedFlag(t *testing.T) {
	price := loadGrokTestPrices(t)["grok-4.7"]
	// No total_tokens-based verdict (e.g. a locally estimated aborted
	// stream): reasoning_tokens_additive decides.
	usage := &converter.TokenUsage{PromptTokens: 1_000, CompletionTokens: 50, ReasoningTokens: 200}

	assert.InDelta(t, 1_000*0.0000026+250*0.0000078, price.CalculateCost(usage), 1e-12)

	fixed := *price
	fixed.ReasoningTokensAccounting = ""
	fixed.ReasoningTokensAdditive = false
	usage.ReasoningAccounting = converter.ReasoningAccountingAdditive
	// Without "auto" the per-response verdict is ignored: the historical
	// included semantics apply (visible output 50 - 200 clamps to 0).
	assert.InDelta(t, 1_000*0.0000026+200*0.0000078, fixed.CalculateCost(usage), 1e-12)
}

func TestGrokPricing_ServerSideTools(t *testing.T) {
	prices := loadGrokTestPrices(t)
	price := prices["grok-4.7"]
	resolve := func(model string) *ModelPrice { return prices[model] }
	usage := &converter.TokenUsage{
		PromptTokens: 250_000, CompletionTokens: 10, ReasoningAccounting: converter.ReasoningAccountingAdditive,
		ServerToolUsageReported: true,
		WebSearchRequests:       2,
		XSearchCalls:            3,
		XSearchPosts:            44,
		XSearchProfiles:         3,
		CodeExecutionCalls:      1,
		AttachmentSearchCalls:   2,
		CollectionsSearchCalls:  4,
		MCPCalls:                5,
		ImageToolGenerations:    2,
		ImageToolEdits:          1,
	}

	costs := price.CalculateCostsWithResolver(usage, resolve)

	require.NotNil(t, costs)
	// Tool charges are per unit and do not double with the long-context tier.
	assert.InDelta(t, 2*0.0065, costs.WebSearchCost, 1e-12)
	assert.InDelta(t, 44*0.0065+3*0.013, costs.XSearchCost, 1e-12, "X Search bills fetched posts and profiles, not calls")
	assert.InDelta(t, 0.0065, costs.CodeExecutionCost, 1e-12)
	assert.InDelta(t, 2*0.013, costs.AttachmentSearchCost, 1e-12)
	assert.InDelta(t, 4*0.00325, costs.CollectionsSearchCost, 1e-12)
	assert.InDelta(t, 2*0.052+0.078+0.013, costs.ImageGenerationToolCost, 1e-12, "edits use the edit tier plus one source image")
	wantTools := costs.WebSearchCost + costs.XSearchCost + costs.CodeExecutionCost + costs.AttachmentSearchCost +
		costs.CollectionsSearchCost + costs.ImageGenerationToolCost
	assert.InDelta(t, wantTools, costs.ToolUsageCost, 1e-12)
	wantTokens := 250_000*0.0000052 + 10*0.0000156
	assert.InDelta(t, wantTokens+wantTools, costs.TotalCost, 1e-12, "tool charges enter the total exactly once")
}

func TestGrokPricing_ImageToolNeedsResolvablePrice(t *testing.T) {
	price := loadGrokTestPrices(t)["grok-4.7"]
	usage := &converter.TokenUsage{PromptTokens: 10, ImageToolGenerations: 1}

	assert.Zero(t, price.CalculateCosts(usage).ImageGenerationToolCost, "no resolver, no image price")
	assert.Zero(t, price.CalculateCostsWithResolver(usage, func(string) *ModelPrice { return nil }).ImageGenerationToolCost)
	assert.Zero(t, price.CalculateCostsWithResolver(usage, func(string) *ModelPrice { return price }).ImageGenerationToolCost,
		"a row resolving to itself is not an image tariff")

	withoutModel := *price
	withoutModel.ImageGenerationToolModel = ""
	prices := loadGrokTestPrices(t)
	assert.Zero(t, withoutModel.CalculateCostsWithResolver(usage, func(m string) *ModelPrice { return prices[m] }).ImageGenerationToolCost)
}

func TestGrokPricing_ToolAliasesPricedOnce(t *testing.T) {
	usage := &converter.TokenUsage{CodeExecutionCalls: 1, CollectionsSearchCalls: 1, AttachmentSearchCalls: 1}

	aliasOnly := &ModelPrice{ToolCostPerCall: map[string]float64{"code_interpreter": 0.5, "file_search": 0.25, "document_search": 2}}
	assert.InDelta(t, 2.75, aliasOnly.CalculateCost(usage), 1e-12)

	both := &ModelPrice{ToolCostPerCall: map[string]float64{"code_execution": 0.5, "code_interpreter": 0.5}}
	assert.InDelta(t, 0.5, both.CalculateCost(usage), 1e-12, "a tool listed under both names is charged once")
}

func TestGrokPricing_NoToolPricesNoToolCharges(t *testing.T) {
	// Other models: counters alone never produce a charge.
	price := &ModelPrice{InputCostPerToken: 1}
	usage := &converter.TokenUsage{PromptTokens: 1, XSearchPosts: 100, CodeExecutionCalls: 5, ImageToolGenerations: 3}

	costs := price.CalculateCostsWithResolver(usage, func(string) *ModelPrice { return &ModelPrice{OutputCostPerImage: 1} })

	assert.InDelta(t, 1, costs.TotalCost, 1e-12)
	assert.Zero(t, costs.ToolUsageCost)
}

func TestLongContextMode_DefaultsUnchanged(t *testing.T) {
	base := ModelPrice{
		InputCostPerToken:           1,
		OutputCostPerToken:          2,
		InputCostPerTokenAbove200k:  10,
		OutputCostPerTokenAbove200k: 20,
	}
	usage := &converter.TokenUsage{PromptTokens: 200_100, CompletionTokens: 10}

	// Generic rows keep the proportional "only the excess" shape...
	generic := base
	assert.InDelta(t, 200_000*1+100*10+10*2, generic.CalculateCost(usage), 1e-9)
	// ...an unknown mode value does not disable it...
	typo := base
	typo.LongContextPricingMode = "full_request_200k"
	assert.InDelta(t, generic.CalculateCost(usage), typo.CalculateCost(usage), 1e-9)
	// ...and Gemini rows keep their exclusive full-session threshold.
	gemini := base
	gemini.LiteLLMProvider = "vertex_ai"
	assert.InDelta(t, 200_000*1+10*2, gemini.CalculateCost(&converter.TokenUsage{PromptTokens: 200_000, CompletionTokens: 10}), 1e-9)
	inclusive := base
	inclusive.LiteLLMProvider = "vertex_ai"
	inclusive.LongContextPricingMode = LongContextFullRequest200kInclusive
	assert.InDelta(t, 200_000*10+10*20, inclusive.CalculateCost(&converter.TokenUsage{PromptTokens: 200_000, CompletionTokens: 10}), 1e-9,
		"the explicit mode wins over the Gemini default")
}

func TestLongContextMode_HigherTierStillWins(t *testing.T) {
	price := ModelPrice{
		InputCostPerToken:          1,
		InputCostPerTokenAbove200k: 10,
		InputCostPerTokenAbove256k: 100,
		LongContextPricingMode:     LongContextFullRequest200kInclusive,
	}
	assert.InDelta(t, 300_000*100, price.CalculateCost(&converter.TokenUsage{PromptTokens: 300_000}), 1e-6)
	assert.InDelta(t, 250_000*10, price.CalculateCost(&converter.TokenUsage{PromptTokens: 250_000}), 1e-6)
}

func TestStrictOrganizationTariff_AcceptsGrokRow(t *testing.T) {
	var rows map[string]json.RawMessage
	require.NoError(t, json.Unmarshal([]byte(grokTestPrices), &rows))
	// supports_web_search is not a router price field, so strict tariffs omit it.
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(rows["grok-4.7"], &fields))
	delete(fields, "supports_web_search")
	row, err := json.Marshal(fields)
	require.NoError(t, err)

	price, err := decodeStrictPriceRow("x-ai/grok-4.7", row)

	require.NoError(t, err)
	assert.Equal(t, LongContextFullRequest200kInclusive, price.LongContextPricingMode)
	assert.Equal(t, ReasoningTokensAccountingAuto, price.ReasoningTokensAccounting)
	assert.Equal(t, 0.013, price.ToolCostPerCall[ToolAttachmentSearch])
	assert.Equal(t, 0.0065, price.XSearchCostPerPost)
	assert.Equal(t, "grok-imagine-image-2.0", price.ImageGenerationToolModel)
}

func TestStrictOrganizationTariff_RejectsInvalidToolAndModeFields(t *testing.T) {
	tests := map[string]string{
		"unknown mode":         `{"input_cost_per_token":1,"long_context_pricing_mode":"full_request"}`,
		"unknown accounting":   `{"input_cost_per_token":1,"reasoning_tokens_accounting":"always"}`,
		"unknown tool":         `{"input_cost_per_token":1,"tool_cost_per_call":{"web_search":0.01}}`,
		"negative tool":        `{"input_cost_per_token":1,"tool_cost_per_call":{"code_execution":-1}}`,
		"conflicting alias":    `{"input_cost_per_token":1,"tool_cost_per_call":{"file_search":1,"collections_search":2}}`,
		"negative x search":    `{"input_cost_per_token":1,"x_search_cost_per_post":-1}`,
		"settings only":        `{"long_context_pricing_mode":"full_request_200k_inclusive","image_generation_tool_model":"m"}`,
		"tool cost not number": `{"input_cost_per_token":1,"tool_cost_per_call":{"code_execution":"1"}}`,
	}
	for name, row := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := decodeStrictPriceRow("m", json.RawMessage(row))
			assert.Error(t, err)
		})
	}

	_, err := decodeStrictPriceRow("m", json.RawMessage(`{"input_cost_per_token":1,"tool_cost_per_call":{"file_search":1,"collections_search":1}}`))
	assert.NoError(t, err, "an alias repeating the canonical price is not ambiguous")
}
