package models

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func terraPrice() *ModelPrice {
	return &ModelPrice{
		InputCostPerToken:                       1.3e-6,
		OutputCostPerToken:                      6.5e-6,
		CacheReadInputTokenCost:                 0.065e-6,
		CacheCreationInputTokenCost:             1.625e-6,
		InputCostPerTokenAbove272k:              2.6e-6,
		OutputCostPerTokenAbove272k:             9.75e-6,
		CacheReadInputTokenCostAbove272k:        0.13e-6,
		CacheCreationInputTokenCostAbove272k:    3.25e-6,
		CacheCreationInputTokenCost12h:          3.25e-6,
		CacheCreationInputTokenCost12hAbove272k: 6.5e-6,
	}
}

// A 12h cache write is billed at its own rate, not the 30m one.
func TestCalculateTokenCosts_12hWriteUsesItsOwnRate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:           10_000,
		CacheCreationTokens:    10_000,
		CacheCreation12hTokens: 10_000,
	}
	costs := CalculateTokenCosts(usage, terraPrice())
	require.NotNil(t, costs)
	assert.InDelta(t, 10_000*3.25e-6, costs.CacheCreationCost, 1e-12,
		"12h writes must bill at cache_creation_input_token_cost_12h")
}

// A request may carry a 30m and a 12h write at once; each is billed at its own
// rate rather than one rate picked by the top TTL.
func TestCalculateTokenCosts_MixedTTLsBillSeparately(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:           20_000,
		CacheCreationTokens:    20_000,
		CacheCreation5mTokens:  8_000,
		CacheCreation12hTokens: 12_000,
	}
	costs := CalculateTokenCosts(usage, terraPrice())
	require.NotNil(t, costs)
	assert.InDelta(t, 8_000*1.625e-6+12_000*3.25e-6, costs.CacheCreationCost, 1e-12)
}

// Past 272k the whole request pays the long-context rate, and the 12h write
// switches to its own above-272k rate.
func TestCalculateTokenCosts_12hLongContextRate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:           272_001,
		CacheCreationTokens:    20_000,
		CacheCreation12hTokens: 20_000,
	}
	costs := CalculateTokenCosts(usage, terraPrice())
	require.NotNil(t, costs)
	assert.InDelta(t, 20_000*6.5e-6, costs.CacheCreationCost, 1e-12)
}

// The 12h rate is now resolved by walking the same full-session tier list as
// the 1h rate (fullSessionTier.cacheCreation12hRate, wired to the 272k tier),
// instead of a threshold check hardcoded to 272k. For today's price data the
// two give the same number for every promptTokens, because
// cache_creation_input_token_cost_12h_above_272k is still the only configured
// 12h tier rate — this pins that the switch to the tier-list lookup didn't
// change that, including once a higher tier (512k here) also matches the
// base cache-creation rate for the same request. A model that later needs
// its own 512k-specific 12h rate only has to add that ModelPrice field and
// wire it into fullSessionTiers; no change to the resolution code above would
// be needed, unlike before this fix.
func TestCalculateTokenCosts_12hRateIndependentOfHigherMatchedTier(t *testing.T) {
	price := terraPrice()
	price.InputCostPerTokenAbove512k = 5.2e-6
	// A rate clearly different from both the 272k base rate (3.25e-6) and the
	// 12h above-272k rate (6.5e-6), so the assertion actually distinguishes
	// "12h billed at its own rate" from "12h billed at whatever the base rate
	// resolved to" rather than passing by coincidence.
	price.CacheCreationInputTokenCostAbove512k = 10.4e-6

	usage := &converter.TokenUsage{
		PromptTokens:           600_000,
		CacheCreationTokens:    20_000,
		CacheCreation12hTokens: 20_000,
	}
	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 20_000*price.CacheCreationInputTokenCost12hAbove272k, costs.CacheCreationCost, 1e-12,
		"12h writes must bill at the 12h-above-272k rate even when a higher tier (512k) matches the base rate")

	// Sanity: an uncached (non-TTL) write at the same prompt size does use the
	// 512k tier's own (higher) base rate, proving the two are resolved
	// independently rather than one masking the other.
	plainUsage := &converter.TokenUsage{
		PromptTokens:        600_000,
		CacheCreationTokens: 20_000,
	}
	plainCosts := CalculateTokenCosts(plainUsage, price)
	require.NotNil(t, plainCosts)
	assert.InDelta(t, 20_000*price.CacheCreationInputTokenCostAbove512k, plainCosts.CacheCreationCost, 1e-12)
}
