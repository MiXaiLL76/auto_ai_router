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
