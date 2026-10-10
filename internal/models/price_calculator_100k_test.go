package models

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

func haiku55Price() *ModelPrice {
	return &ModelPrice{
		InputCostPerToken:                            0.10e-6,
		OutputCostPerToken:                           0.50e-6,
		CacheReadInputTokenCost:                      0.01e-6,
		CacheCreationInputTokenCost:                  0.125e-6,
		CacheCreationInputTokenCostAbove1hr:          0.20e-6,
		InputCostPerTokenAbove100k:                   0.50e-6,
		OutputCostPerTokenAbove100k:                  2.50e-6,
		CacheReadInputTokenCostAbove100k:             0.05e-6,
		CacheCreationInputTokenCostAbove100k:         0.625e-6,
		CacheCreationInputTokenCostAbove1hrAbove100k: 1.00e-6,
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low":    0.01,
			"search_context_size_medium": 0.01,
			"search_context_size_high":   0.01,
		},
		WebSearchBillingUnit: "per_query",
	}
}

func TestCalculateTokenCosts_100kTierThreshold(t *testing.T) {
	price := haiku55Price()
	cases := []struct {
		name         string
		promptTokens int
		inputRate    float64
		outputRate   float64
	}{
		{"99_999 stays on the base card", 99_999, 0.10e-6, 0.50e-6},
		{"exactly 100_000 stays on the base card", 100_000, 0.10e-6, 0.50e-6},
		{"100_001 moves the whole request", 100_001, 0.50e-6, 2.50e-6},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			usage := &converter.TokenUsage{PromptTokens: tc.promptTokens, CompletionTokens: 1_000}
			costs := CalculateTokenCosts(usage, price)
			require.NotNil(t, costs)
			assert.InDelta(t, float64(tc.promptTokens)*tc.inputRate, costs.InputCost, 1e-12)
			assert.InDelta(t, 1_000*tc.outputRate, costs.OutputCost, 1e-12)
		})
	}
}

func TestCalculateTokenCosts_100kTierAppliesToCacheReadAndBothWriteTTLs(t *testing.T) {
	price := haiku55Price()
	usage := &converter.TokenUsage{
		PromptTokens:          120_000,
		CachedInputTokens:     60_000,
		CacheCreationTokens:   30_000,
		CacheCreation5mTokens: 10_000,
		CacheCreation1hTokens: 20_000,
		CompletionTokens:      2_000,
	}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)

	assert.InDelta(t, 30_000*0.50e-6, costs.InputCost, 1e-12) // 120k - 60k read - 30k written
	assert.InDelta(t, 60_000*0.05e-6, costs.CachedInputCost, 1e-12)
	// 1-hour writes keep their own rate inside the tier instead of the 5-minute one.
	assert.InDelta(t, 10_000*0.625e-6+20_000*1.00e-6, costs.CacheCreationCost, 1e-12)
	assert.InDelta(t, 2_000*2.50e-6, costs.OutputCost, 1e-12)
}

func TestCalculateTokenCosts_100kTierBaseCardKeepsSeparateTTLRates(t *testing.T) {
	price := haiku55Price()
	usage := &converter.TokenUsage{
		PromptTokens:          100_000,
		CachedInputTokens:     40_000,
		CacheCreationTokens:   30_000,
		CacheCreation5mTokens: 10_000,
		CacheCreation1hTokens: 20_000,
	}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)

	assert.InDelta(t, 30_000*0.10e-6, costs.InputCost, 1e-12)
	assert.InDelta(t, 40_000*0.01e-6, costs.CachedInputCost, 1e-12)
	assert.InDelta(t, 10_000*0.125e-6+20_000*0.20e-6, costs.CacheCreationCost, 1e-12)
}

// The threshold is the whole prompt: on Anthropic's own usage input_tokens
// excludes cache reads and writes, so a request with few uncached tokens can
// still cross it.
func TestCalculateTokenCosts_100kTierCountsCacheInAnthropicUsage(t *testing.T) {
	price := haiku55Price()
	body := []byte(`{"type":"message","usage":{"input_tokens":40000,` +
		`"cache_read_input_tokens":50000,"cache_creation_input_tokens":10001,` +
		`"cache_creation":{"ephemeral_5m_input_tokens":1,"ephemeral_1h_input_tokens":10000},` +
		`"output_tokens":100}}`)

	usage := converter.ExtractTokenUsage(body)
	require.NotNil(t, usage)
	assert.Equal(t, 100_001, usage.PromptTokens)

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 40_000*0.50e-6, costs.InputCost, 1e-12)
	assert.InDelta(t, 50_000*0.05e-6, costs.CachedInputCost, 1e-12)
	assert.InDelta(t, 1*0.625e-6+10_000*1.00e-6, costs.CacheCreationCost, 1e-12)
	assert.InDelta(t, 100*2.50e-6, costs.OutputCost, 1e-12)
}

// An OpenAI-shaped aggregator response already counts cache reads and writes
// inside prompt_tokens; they must not be added a second time.
func TestCalculateTokenCosts_100kTierUsesInclusivePromptTokens(t *testing.T) {
	price := haiku55Price()
	below := []byte(`{"object":"chat.completion","usage":{"prompt_tokens":100000,"completion_tokens":10,` +
		`"total_tokens":100010,"prompt_tokens_details":{"cached_tokens":60000,"caching_tokens":30000,` +
		`"caching_token_details":{"caching_5m_tokens":0,"caching_1h_tokens":30000}}}}`)
	above := []byte(`{"object":"chat.completion","usage":{"prompt_tokens":100001,"completion_tokens":10,` +
		`"total_tokens":100011,"prompt_tokens_details":{"cached_tokens":60000,"caching_tokens":30000,` +
		`"caching_token_details":{"caching_5m_tokens":0,"caching_1h_tokens":30000}}}}`)

	belowUsage := converter.ExtractTokenUsage(below)
	require.NotNil(t, belowUsage)
	assert.Equal(t, 100_000, belowUsage.PromptTokens)
	belowCosts := CalculateTokenCosts(belowUsage, price)
	assert.InDelta(t, 10_000*0.10e-6, belowCosts.InputCost, 1e-12)
	assert.InDelta(t, 30_000*0.20e-6, belowCosts.CacheCreationCost, 1e-12)

	aboveUsage := converter.ExtractTokenUsage(above)
	require.NotNil(t, aboveUsage)
	assert.Equal(t, 100_001, aboveUsage.PromptTokens)
	aboveCosts := CalculateTokenCosts(aboveUsage, price)
	assert.InDelta(t, 10_001*0.50e-6, aboveCosts.InputCost, 1e-12)
	assert.InDelta(t, 60_000*0.05e-6, aboveCosts.CachedInputCost, 1e-12)
	assert.InDelta(t, 30_000*1.00e-6, aboveCosts.CacheCreationCost, 1e-12)
}

// Thinking is part of output_tokens and is billed once, at the tier's output
// rate; searches are billed per call and are not scaled by the tier.
func TestCalculateTokenCosts_100kTierReasoningAndWebSearch(t *testing.T) {
	price := haiku55Price()
	usage := &converter.TokenUsage{
		PromptTokens:      150_000,
		CompletionTokens:  3_000,
		ReasoningTokens:   2_000,
		WebSearchRequests: 3,
	}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)

	assert.InDelta(t, 1_000*2.50e-6, costs.OutputCost, 1e-12)
	assert.InDelta(t, 2_000*2.50e-6, costs.ReasoningCost, 1e-12)
	assert.InDelta(t, 0.03, costs.WebSearchCost, 1e-12)
	assert.InDelta(t, 150_000*0.50e-6+3_000*2.50e-6+0.03, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_100kTierLeavesOtherModelsAlone(t *testing.T) {
	// A model priced only from 128k up is still on its base card at 110k.
	price := &ModelPrice{
		InputCostPerToken:           1,
		OutputCostPerToken:          2,
		CacheCreationInputTokenCost: 3,
		InputCostPerTokenAbove128k:  10,
		OutputCostPerTokenAbove128k: 20,
	}
	usage := &converter.TokenUsage{PromptTokens: 110_000, CompletionTokens: 10}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 110_000, costs.InputCost, 1e-6)
	assert.InDelta(t, 20, costs.OutputCost, 1e-9)
}

func TestCalculateTokenCosts_HigherTierWinsOver100k(t *testing.T) {
	price := haiku55Price()
	price.InputCostPerTokenAbove128k = 0.70e-6
	usage := &converter.TokenUsage{PromptTokens: 130_000}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 130_000*0.70e-6, costs.InputCost, 1e-12)
}

func TestCalculateTokenCosts_TierWithOnly1hWriteRate(t *testing.T) {
	price := &ModelPrice{
		InputCostPerToken:                            1,
		CacheCreationInputTokenCost:                  2,
		CacheCreationInputTokenCostAbove1hr:          3,
		CacheCreationInputTokenCostAbove1hrAbove100k: 7,
	}
	usage := &converter.TokenUsage{
		PromptTokens:          100_001,
		CacheCreationTokens:   10,
		CacheCreation5mTokens: 4,
		CacheCreation1hTokens: 6,
	}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 4*2+6*7, costs.CacheCreationCost, 1e-9)
}
