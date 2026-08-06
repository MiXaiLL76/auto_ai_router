package models

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCalculateTokenCosts_RegularTokensOnly(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	price := &ModelPrice{
		InputCostPerToken:  0.01,
		OutputCostPerToken: 0.02,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.Equal(t, 1.0, costs.InputCost)  // 100 * 0.01
	assert.Equal(t, 1.0, costs.OutputCost) // 50 * 0.02
	assert.Equal(t, 2.0, costs.TotalCost)  // 1.0 + 1.0
}

func TestCalculateTokenCosts_GeneratedImageTokensUseImageRate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:      22,
		CompletionTokens:  1120,
		OutputImageTokens: 1120,
	}
	price := &ModelPrice{
		InputCostPerToken:       0.00000045,
		OutputCostPerToken:      0.0000027,
		OutputCostPerImageToken: 0.000054,
	}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 0.0000099, costs.InputCost, 1e-12)
	assert.Zero(t, costs.OutputCost, "image tokens must not also be charged as text")
	assert.InDelta(t, 0.06048, costs.ImageCost, 1e-12)
	assert.InDelta(t, 0.0604899, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_ImagenUsesPerImageRate(t *testing.T) {
	usage := &converter.TokenUsage{ImageCount: 2}
	price := &ModelPrice{OutputCostPerImage: 0.018}

	costs := CalculateTokenCosts(usage, price)
	require.NotNil(t, costs)
	assert.InDelta(t, 0.036, costs.ImageCost, 1e-12)
	assert.InDelta(t, 0.036, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_Vertex_WithAudioAndCached(t *testing.T) {
	// Vertex semantic: audio and cached are INCLUDED in totals
	// promptTokenCount=100 includes AudioInputTokens=5 and CachedInputTokens=20
	usage := &converter.TokenUsage{
		PromptTokens:      100, // includes 5 audio + 20 cached
		CompletionTokens:  50,  // includes 2 audio
		AudioInputTokens:  5,
		AudioOutputTokens: 2,
		CachedInputTokens: 20,
	}

	price := &ModelPrice{
		InputCostPerToken:       0.01,   // $0.01 per regular token
		OutputCostPerToken:      0.02,   // $0.02 per regular token
		InputCostPerAudioToken:  0.001,  // $0.001 per audio token (10x cheaper)
		OutputCostPerAudioToken: 0.002,  // $0.002 per audio token
		InputCostPerCachedToken: 0.0001, // $0.0001 per cached token (100x cheaper)
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Regular input: 100 - 5 - 20 = 75 tokens at $0.01
	assert.Equal(t, 0.75, costs.InputCost)

	// Regular output: 50 - 2 = 48 tokens at $0.02
	assert.Equal(t, 0.96, costs.OutputCost)

	// Audio input: 5 tokens at $0.001
	assert.Equal(t, 0.005, costs.AudioInputCost)

	// Audio output: 2 tokens at $0.002
	assert.Equal(t, 0.004, costs.AudioOutputCost)

	// Cached input: 20 tokens at $0.0001
	assert.Equal(t, 0.002, costs.CachedInputCost)

	// Total: 0.75 + 0.96 + 0.005 + 0.004 + 0.002 = 1.721
	assert.InDelta(t, 1.721, costs.TotalCost, 0.0001)

	// VERIFY: Original bug would calculate 1.007 (double counting audio and cached)
	// New implementation correctly calculates 1.721
}

func TestCalculateTokenCosts_WithReasoning(t *testing.T) {
	// OpenAI semantic: reasoning tokens are INCLUDED in completion tokens
	// completionTokens=100 includes ReasoningTokens=30
	usage := &converter.TokenUsage{
		PromptTokens:     50,
		CompletionTokens: 100, // includes 30 reasoning tokens
		ReasoningTokens:  30,
	}

	price := &ModelPrice{
		InputCostPerToken:           0.01,
		OutputCostPerToken:          0.02,
		OutputCostPerReasoningToken: 0.1, // reasoning is more expensive
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Regular input: 50 tokens at $0.01
	assert.Equal(t, 0.5, costs.InputCost)

	// Regular output: 100 - 30 = 70 tokens at $0.02
	assert.InDelta(t, 1.4, costs.OutputCost, 0.0001)

	// Reasoning: 30 tokens at $0.1
	assert.Equal(t, 3.0, costs.ReasoningCost)

	// Total: 0.5 + 1.4 + 3.0 = 4.9
	assert.InDelta(t, 4.9, costs.TotalCost, 0.0001)
}

func TestCalculateTokenCosts_WithExplicitOutputTextTokens(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     16,
		CompletionTokens: 219,
		OutputTextTokens: 219,
		ReasoningTokens:  212,
	}
	price := &ModelPrice{
		InputCostPerToken:  0.00000005,
		OutputCostPerToken: 0.00000052,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.InDelta(t, 0.0000008, costs.InputCost, 1e-12)
	assert.InDelta(t, 0.00011388, costs.OutputCost+costs.ReasoningCost, 1e-12)
	assert.InDelta(t, 0.00011468, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_ClampsToExplicitOutputTextTokens(t *testing.T) {
	// Derived regular tokens (300-50=250) overstate billable text tokens here;
	// the smaller explicit OutputTextTokens must win.
	usage := &converter.TokenUsage{
		PromptTokens:     10,
		CompletionTokens: 300,
		OutputTextTokens: 100,
		ReasoningTokens:  50,
	}
	price := &ModelPrice{
		InputCostPerToken:  0.002,
		OutputCostPerToken: 0.001,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.InDelta(t, 0.1, costs.OutputCost, 1e-12)
	assert.InDelta(t, 0.05, costs.ReasoningCost, 1e-12)
	assert.InDelta(t, 0.02, costs.InputCost, 1e-12)
	assert.InDelta(t, 0.17, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_WithPrediction(t *testing.T) {
	// Prediction tokens (accepted and rejected) are included in completion tokens
	usage := &converter.TokenUsage{
		PromptTokens:             100,
		CompletionTokens:         100, // includes 20 accepted + 5 rejected prediction
		AcceptedPredictionTokens: 20,
		RejectedPredictionTokens: 5,
	}

	price := &ModelPrice{
		InputCostPerToken:            0.01,
		OutputCostPerToken:           0.02,
		OutputCostPerPredictionToken: 0.03, // prediction tokens cost more
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Regular input: 100 tokens at $0.01
	assert.Equal(t, 1.0, costs.InputCost)

	// Regular output: 100 - 20 - 5 = 75 tokens at $0.02
	assert.Equal(t, 1.5, costs.OutputCost)

	// Accepted prediction: 20 tokens at $0.03 = 0.6
	// Rejected prediction: 5 tokens at $0.02 (fallback) = 0.1
	// Total prediction: 0.6 + 0.1 = 0.7
	assert.InDelta(t, 0.7, costs.PredictionCost, 0.0001)

	// Total: 1.0 + 1.5 + 0.7 = 3.2
	assert.InDelta(t, 3.2, costs.TotalCost, 0.0001)
}

func TestCalculateTokenCosts_AudioFallbackToRegularPrice(t *testing.T) {
	// When audio price not set, should fall back to regular price
	usage := &converter.TokenUsage{
		PromptTokens:     100,
		AudioInputTokens: 10,
	}

	price := &ModelPrice{
		InputCostPerToken: 0.01,
		// AudioCostPerToken not set (0)
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Regular: (100 - 10) * 0.01 = 0.9
	assert.Equal(t, 0.9, costs.InputCost)

	// Audio fallback: 10 * 0.01 = 0.1
	assert.Equal(t, 0.1, costs.AudioInputCost)

	// Total: 1.0
	assert.Equal(t, 1.0, costs.TotalCost)
}

func TestCalculateTokenCosts_CachedFallbackToRegularPrice(t *testing.T) {
	// When cached price not set, should fall back to regular price
	usage := &converter.TokenUsage{
		PromptTokens:      100,
		CachedInputTokens: 20,
	}

	price := &ModelPrice{
		InputCostPerToken: 0.01,
		// CachedCostPerToken not set (0)
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Regular: (100 - 20) * 0.01 = 0.8
	assert.Equal(t, 0.8, costs.InputCost)

	// Cached fallback: 20 * 0.01 = 0.2
	assert.Equal(t, 0.2, costs.CachedInputCost)

	// Total: 1.0
	assert.Equal(t, 1.0, costs.TotalCost)
}

func TestCalculateTokenCosts_SafetyNegativeTokens(t *testing.T) {
	// Edge case: more audio tokens reported than total (shouldn't happen, but be safe)
	usage := &converter.TokenUsage{
		PromptTokens:      50,
		AudioInputTokens:  60, // more than total!
		CachedInputTokens: 10,
	}

	price := &ModelPrice{
		InputCostPerToken: 0.01,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Should use 0 for regular tokens (safety clamp)
	assert.Equal(t, 0.0, costs.InputCost)

	// Audio and cached should still be calculated
	assert.Equal(t, 0.6, costs.AudioInputCost)  // 60 * 0.01
	assert.Equal(t, 0.1, costs.CachedInputCost) // 10 * 0.01

	assert.Equal(t, 0.7, costs.TotalCost)
}

func TestCalculateTokenCosts_NegativeBreakdownFieldsAreIgnored(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:             10,
		CompletionTokens:         20,
		AudioInputTokens:         -10,
		AudioOutputTokens:        -20,
		CachedInputTokens:        -30,
		CachedAudioInputTokens:   -40,
		CacheCreationTokens:      -50,
		CacheCreation5mTokens:    -60,
		CacheCreation1hTokens:    -70,
		CachedOutputTokens:       -80,
		ReasoningTokens:          -90,
		AcceptedPredictionTokens: -100,
		RejectedPredictionTokens: -110,
		ImageCount:               -1,
		ImageTokens:              -120,
		OutputImageTokens:        -130,
	}
	price := &ModelPrice{
		InputCostPerToken:            1,
		OutputCostPerToken:           2,
		InputCostPerAudioToken:       3,
		OutputCostPerAudioToken:      4,
		InputCostPerCachedToken:      5,
		CacheReadInputAudioTokenCost: 6,
		CacheCreationInputTokenCost:  7,
		OutputCostPerCachedToken:     8,
		OutputCostPerReasoningToken:  9,
		OutputCostPerPredictionToken: 10,
		InputCostPerImageToken:       11,
		OutputCostPerImageToken:      12,
		OutputCostPerImage:           13,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.Equal(t, 10.0, costs.InputCost)
	assert.Equal(t, 40.0, costs.OutputCost)
	assert.Zero(t, costs.AudioInputCost)
	assert.Zero(t, costs.AudioOutputCost)
	assert.Zero(t, costs.CachedInputCost)
	assert.Zero(t, costs.CacheCreationCost)
	assert.Zero(t, costs.CachedOutputCost)
	assert.Zero(t, costs.ReasoningCost)
	assert.Zero(t, costs.PredictionCost)
	assert.Zero(t, costs.ImageCost)
	assert.Equal(t, 50.0, costs.TotalCost)
}

func TestCalculateTokenCosts_NilUsage(t *testing.T) {
	price := &ModelPrice{
		InputCostPerToken: 0.01,
	}

	costs := CalculateTokenCosts(nil, price)
	assert.Nil(t, costs)
}

func TestCalculateTokenCosts_NilPrice(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	costs := CalculateTokenCosts(usage, nil)
	assert.Nil(t, costs)
}

func TestModelPrice_CalculateCost(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	price := &ModelPrice{
		InputCostPerToken:  0.01,
		OutputCostPerToken: 0.02,
	}

	totalCost := price.CalculateCost(usage)
	assert.Equal(t, 2.0, totalCost) // (100 * 0.01) + (50 * 0.02)
}

func TestModelPrice_CalculateCost_NilUsage(t *testing.T) {
	price := &ModelPrice{
		InputCostPerToken: 0.01,
	}

	totalCost := price.CalculateCost(nil)
	assert.Equal(t, 0.0, totalCost)
}

func TestCalculateTokenCosts_Above200k_Input(t *testing.T) {
	// Test 300k input tokens with no specialized tokens
	// below200k = 200k, above200k = 100k
	// regularAbove = 100k, regularBelow = 200k
	usage := &converter.TokenUsage{
		PromptTokens:     300_000,
		CompletionTokens: 50,
	}

	price := &ModelPrice{
		InputCostPerToken:          0.001,
		OutputCostPerToken:         0.002,
		InputCostPerTokenAbove200k: 0.0005, // cheaper for tokens above 200k
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// below200k cost: 200_000 * 0.001 = 200.0
	// above200k cost: 100_000 * 0.0005 = 50.0
	// Total input: 250.0
	assert.InDelta(t, 250.0, costs.InputCost, 0.0001)

	// output cost: 50 * 0.002 = 0.1
	assert.InDelta(t, 0.1, costs.OutputCost, 0.0001)

	// Total: 250.1
	assert.InDelta(t, 250.1, costs.TotalCost, 0.0001)
}

func TestCalculateTokenCosts_Above200k_WithAudio(t *testing.T) {
	// Test 300k input tokens with 30k audio tokens
	// regularInputTokens = 300k - 30k = 270k
	// proportion above = (300k - 200k) / 300k = 100k / 300k = 1/3
	// regularAbove = 270k * 1/3 = 90k, regularBelow = 180k
	usage := &converter.TokenUsage{
		PromptTokens:     300_000,
		AudioInputTokens: 30_000,
		CompletionTokens: 50,
	}

	price := &ModelPrice{
		InputCostPerToken:          0.001,
		OutputCostPerToken:         0.002,
		InputCostPerTokenAbove200k: 0.0005,
		InputCostPerAudioToken:     0.0001,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// regularBelow cost: 180_000 * 0.001 = 180.0
	// regularAbove cost: 90_000 * 0.0005 = 45.0
	// Total regular input: 225.0
	assert.InDelta(t, 225.0, costs.InputCost, 0.0001)

	// audio input: 30_000 * 0.0001 = 3.0
	assert.InDelta(t, 3.0, costs.AudioInputCost, 0.0001)

	// output cost: 50 * 0.002 = 0.1
	assert.InDelta(t, 0.1, costs.OutputCost, 0.0001)

	// Total: 225.0 + 3.0 + 0.1 = 228.1
	assert.InDelta(t, 228.1, costs.TotalCost, 0.0001)
}

func TestCalculateTokenCosts_Above200k_Output(t *testing.T) {
	// Test 250k output tokens with no specialized tokens
	// below200k = 200k, above200k = 50k
	usage := &converter.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 250_000,
	}

	price := &ModelPrice{
		InputCostPerToken:           0.001,
		OutputCostPerToken:          0.002,
		OutputCostPerTokenAbove200k: 0.001, // cheaper for tokens above 200k
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// input cost: 100 * 0.001 = 0.1
	assert.InDelta(t, 0.1, costs.InputCost, 0.0001)

	// below200k cost: 200_000 * 0.002 = 400.0
	// above200k cost: 50_000 * 0.001 = 50.0
	// Total output: 450.0
	assert.InDelta(t, 450.0, costs.OutputCost, 0.0001)

	// Total: 0.1 + 450.0 = 450.1
	assert.InDelta(t, 450.1, costs.TotalCost, 0.0001)
}

func TestCalculateTokenCosts_Below200k_NoTiering(t *testing.T) {
	// Test that tiering is NOT applied when tokens are below 200k
	// Even if InputCostPerTokenAbove200k is set
	usage := &converter.TokenUsage{
		PromptTokens:     150_000,
		CompletionTokens: 50_000,
	}

	price := &ModelPrice{
		InputCostPerToken:           0.001,
		OutputCostPerToken:          0.002,
		InputCostPerTokenAbove200k:  0.0005,
		OutputCostPerTokenAbove200k: 0.001,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)

	// Tiering should NOT apply since 150k < 200k
	// input cost: 150_000 * 0.001 = 150.0 (only base price, not tiered)
	assert.InDelta(t, 150.0, costs.InputCost, 0.0001)

	// output cost: 50_000 * 0.002 = 100.0 (only base price, not tiered)
	assert.InDelta(t, 100.0, costs.OutputCost, 0.0001)

	// Total: 250.0
	assert.InDelta(t, 250.0, costs.TotalCost, 0.0001)
}

// TestCalculateTokenCosts_CacheReadInputTokenCostAlias verifies that
// cache_read_input_token_cost (LiteLLM JSON alias) is used when
// input_cost_per_cached_token is not set. This was the root cause of the
// gpt-5.1 overbilling bug.
func TestCalculateTokenCosts_CacheReadInputTokenCostAlias(t *testing.T) {
	// gpt-5.1 example: 4736 prompt, 4608 cached, 285 completion
	usage := &converter.TokenUsage{
		PromptTokens:      4736,
		CompletionTokens:  285,
		CachedInputTokens: 4608,
	}

	price := &ModelPrice{
		InputCostPerToken:       0.000001125,  // $1.125/1M
		OutputCostPerToken:      0.000009,     // $9/1M
		CacheReadInputTokenCost: 0.0000001125, // $0.1125/1M — LiteLLM alias, no InputCostPerCachedToken set
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	// Regular input: 4736 - 4608 = 128 tokens × $0.000001125 = $0.000144
	assert.InDelta(t, 0.000144, costs.InputCost, 1e-9)
	// Cached input via alias: 4608 × $0.0000001125 = $0.0005184
	assert.InDelta(t, 0.0005184, costs.CachedInputCost, 1e-9)
	// Output: 285 × $0.000009 = $0.002565
	assert.InDelta(t, 0.002565, costs.OutputCost, 1e-9)
	// Total ≈ $0.003227 (not $0.007893 from the bug)
	assert.InDelta(t, 0.0032274, costs.TotalCost, 1e-9)
}

func TestCalculateTokenCosts_CacheCreationTokens(t *testing.T) {
	// Anthropic: cache creation tokens are included in PromptTokens and billed separately
	usage := &converter.TokenUsage{
		PromptTokens:        1000, // input(500) + cache_read(300) + cache_creation(200)
		CompletionTokens:    100,
		CachedInputTokens:   300, // cache_read
		CacheCreationTokens: 200, // cache_creation (Anthropic prompt caching write)
	}

	price := &ModelPrice{
		InputCostPerToken:           0.000003,   // $3/1M
		OutputCostPerToken:          0.000015,   // $15/1M
		CacheReadInputTokenCost:     0.0000003,  // $0.3/1M
		CacheCreationInputTokenCost: 0.00000375, // $3.75/1M (more expensive for write)
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	// Regular: 1000 - 300 - 200 = 500 tokens × $0.000003 = $0.0015
	assert.InDelta(t, 0.0015, costs.InputCost, 1e-9)
	// Cache read: 300 × $0.0000003 = $0.00009
	assert.InDelta(t, 0.00009, costs.CachedInputCost, 1e-9)
	// Cache creation: 200 × $0.00000375 = $0.00075
	assert.InDelta(t, 0.00075, costs.CacheCreationCost, 1e-9)
	// Output: 100 × $0.000015 = $0.0015
	assert.InDelta(t, 0.0015, costs.OutputCost, 1e-9)
	// Total: 0.0015 + 0.00009 + 0.00075 + 0.0015 = $0.00384
	assert.InDelta(t, 0.00384, costs.TotalCost, 1e-9)
}

func TestCalculateTokenCosts_GPT56CacheWrite(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:        1000,
		CompletionTokens:    100,
		CachedInputTokens:   300,
		CacheCreationTokens: 600,
	}

	price := &ModelPrice{
		InputCostPerToken:           0.000005,
		OutputCostPerToken:          0.00003,
		CacheReadInputTokenCost:     0.0000005,
		CacheCreationInputTokenCost: 0.00000625,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.InDelta(t, 0.0005, costs.InputCost, 1e-9)
	assert.InDelta(t, 0.00015, costs.CachedInputCost, 1e-9)
	assert.InDelta(t, 0.00375, costs.CacheCreationCost, 1e-9)
	assert.InDelta(t, 0.003, costs.OutputCost, 1e-9)
	assert.InDelta(t, 0.0074, costs.TotalCost, 1e-9)
}

func TestCalculateTokenCosts_GPT56LongContext(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:        300_000,
		CompletionTokens:    10_000,
		CachedInputTokens:   60_000,
		CacheCreationTokens: 90_000,
	}

	price := &ModelPrice{
		InputCostPerToken:                    0.0000065,
		OutputCostPerToken:                   0.000039,
		CacheReadInputTokenCost:              0.00000065,
		CacheCreationInputTokenCost:          0.000008125,
		InputCostPerTokenAbove272k:           0.000013,
		OutputCostPerTokenAbove272k:          0.0000585,
		CacheReadInputTokenCostAbove272k:     0.0000013,
		CacheCreationInputTokenCostAbove272k: 0.00001625,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.InDelta(t, 1.95, costs.InputCost, 1e-9)
	assert.InDelta(t, 0.078, costs.CachedInputCost, 1e-9)
	assert.InDelta(t, 1.4625, costs.CacheCreationCost, 1e-9)
	assert.InDelta(t, 0.585, costs.OutputCost, 1e-9)
	assert.InDelta(t, 4.0755, costs.TotalCost, 1e-9)
}

func TestCalculateTokenCosts_GPT56ThresholdIsExclusive(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     tokenTiering272kThreshold,
		CompletionTokens: 1,
	}
	price := &ModelPrice{
		InputCostPerToken:           0.0000065,
		OutputCostPerToken:          0.000039,
		InputCostPerTokenAbove272k:  0.000013,
		OutputCostPerTokenAbove272k: 0.0000585,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.NotNil(t, costs)
	assert.InDelta(t, 1.768, costs.InputCost, 1e-9)
	assert.InDelta(t, 0.000039, costs.OutputCost, 1e-12)
}

func TestCalculateTokenCosts_CacheLongContextAbove200kAndTTL(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:          210_000,
		CachedInputTokens:     100,
		CacheCreationTokens:   60,
		CacheCreation5mTokens: 20,
		CacheCreation1hTokens: 40,
	}
	price := &ModelPrice{
		InputCostPerToken:                            0.01,
		CacheReadInputTokenCost:                      1,
		CacheReadInputTokenCostAbove200k:             2,
		CacheCreationInputTokenCost:                  3,
		CacheCreationInputTokenCostAbove200k:         4,
		CacheCreationInputTokenCostAbove1hr:          5,
		CacheCreationInputTokenCostAbove1hrAbove200k: 6,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 200, costs.CachedInputCost, 1e-9)
	assert.InDelta(t, 320, costs.CacheCreationCost, 1e-9) // 20*4 (5m) + 40*6 (1h)
}

func TestCalculateTokenCosts_CacheAbove272kTakesPrecedence(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:        300_000,
		CachedInputTokens:   100,
		CacheCreationTokens: 50,
	}
	price := &ModelPrice{
		InputCostPerToken:                    0.01,
		CacheReadInputTokenCost:              1,
		CacheReadInputTokenCostAbove200k:     2,
		CacheReadInputTokenCostAbove272k:     3,
		CacheCreationInputTokenCost:          4,
		CacheCreationInputTokenCostAbove200k: 5,
		CacheCreationInputTokenCostAbove272k: 6,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 300, costs.CachedInputCost, 1e-9)
	assert.InDelta(t, 300, costs.CacheCreationCost, 1e-9)
}

func TestCalculateTokenCosts_CacheCreationAbove272kOverrides1hTTLRate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:          300_000,
		CacheCreationTokens:   50,
		CacheCreation5mTokens: 20,
		CacheCreation1hTokens: 30,
	}
	price := &ModelPrice{
		InputCostPerToken:                            0.01,
		CacheCreationInputTokenCost:                  4,
		CacheCreationInputTokenCostAbove200k:         5,
		CacheCreationInputTokenCostAbove272k:         6,
		CacheCreationInputTokenCostAbove1hr:          7,
		CacheCreationInputTokenCostAbove1hrAbove200k: 8,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 300, costs.CacheCreationCost, 1e-9) // 50*6, including 1h tokens
}

func TestCalculateTokenCosts_CacheThresholdIsExclusive(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:        tokenTiering200kThreshold,
		CachedInputTokens:   10,
		CacheCreationTokens: 10,
	}
	price := &ModelPrice{
		InputCostPerToken:                    0.01,
		CacheReadInputTokenCost:              1,
		CacheReadInputTokenCostAbove200k:     2,
		CacheCreationInputTokenCost:          3,
		CacheCreationInputTokenCostAbove200k: 4,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 10, costs.CachedInputCost, 1e-9)
	assert.InDelta(t, 30, costs.CacheCreationCost, 1e-9)
}

func TestCalculateTokenCosts_CacheCreationTTLBreakdownIsCappedAtAggregate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:          100,
		CacheCreationTokens:   10,
		CacheCreation5mTokens: 8,
		CacheCreation1hTokens: 8,
	}
	price := &ModelPrice{
		InputCostPerToken:                   0.01,
		CacheCreationInputTokenCost:         2,
		CacheCreationInputTokenCostAbove1hr: 5,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 26, costs.CacheCreationCost, 1e-9) // 8*2 + capped 2*5
}

func TestCalculateTokenCosts_CacheCreationTTLBreakdownProvidesMissingAggregate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:          100,
		CacheCreation5mTokens: 8,
		CacheCreation1hTokens: 2,
	}
	price := &ModelPrice{
		InputCostPerToken:                   0.01,
		CacheCreationInputTokenCost:         2,
		CacheCreationInputTokenCostAbove1hr: 5,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.Zero(t, usage.CacheCreationTokens, "CalculateTokenCosts must not mutate caller-owned usage")
	assert.InDelta(t, 26, costs.CacheCreationCost, 1e-9) // 8*2 + 2*5
}

func TestCalculateTokenCosts_CacheCreation1hCostsMoreThan5mForSameVolume(t *testing.T) {
	price := &ModelPrice{
		InputCostPerToken:                   0.000003,
		CacheCreationInputTokenCost:         0.00000375,
		CacheCreationInputTokenCostAbove1hr: 0.000006,
	}
	fiveMinutes := &converter.TokenUsage{
		PromptTokens:          1_000,
		CacheCreationTokens:   1_000,
		CacheCreation5mTokens: 1_000,
	}
	oneHour := &converter.TokenUsage{
		PromptTokens:          1_000,
		CacheCreationTokens:   1_000,
		CacheCreation1hTokens: 1_000,
	}

	fiveMinuteCosts := CalculateTokenCosts(fiveMinutes, price)
	oneHourCosts := CalculateTokenCosts(oneHour, price)

	assert.InDelta(t, 0.00375, fiveMinuteCosts.TotalCost, 1e-12)
	assert.InDelta(t, 0.006, oneHourCosts.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_CachedAudioUsesDedicatedRate(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:           100,
		CachedInputTokens:      80,
		CachedAudioInputTokens: 30,
	}
	price := &ModelPrice{
		InputCostPerToken:            1,
		CacheReadInputTokenCost:      0.1,
		CacheReadInputAudioTokenCost: 0.3,
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 14, costs.CachedInputCost, 1e-9) // 50*0.1 + 30*0.3
	assert.InDelta(t, 20, costs.InputCost, 1e-9)
}

func TestCalculateTokenCosts_WebSearchUsesContextPricing(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:         10,
		CompletionTokens:     5,
		WebSearchRequests:    2,
		WebSearchContextSize: "high",
	}
	price := &ModelPrice{
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low":    0.10,
			"search_context_size_medium": 0.20,
			"search_context_size_high":   0.30,
		},
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 10, costs.InputCost, 1e-9)
	assert.InDelta(t, 10, costs.OutputCost, 1e-9)
	assert.InDelta(t, 0.60, costs.WebSearchCost, 1e-9)
	assert.InDelta(t, 20.60, costs.TotalCost, 1e-9)
	assert.Equal(t, 15, usage.Total())
}

func TestCalculateTokenCosts_WebSearchBillingUnit(t *testing.T) {
	tests := []struct {
		name       string
		price      ModelPrice
		wantSearch float64
	}{
		{
			name: "per query",
			price: ModelPrice{
				WebSearchBillingUnit: "per_query",
			},
			wantSearch: 0.60,
		},
		{
			name: "per prompt",
			price: ModelPrice{
				WebSearchBillingUnit: "per_prompt",
			},
			wantSearch: 0.20,
		},
		{
			name: "Gemini 2 pricing defaults to per prompt like LiteLLM",
			price: ModelPrice{
				LiteLLMProvider: "gemini",
			},
			wantSearch: 0.20,
		},
		{
			name: "non Gemini pricing keeps per query default",
			price: ModelPrice{
				LiteLLMProvider: "anthropic",
			},
			wantSearch: 0.60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.price.SearchContextCostPerQuery = map[string]float64{
				"search_context_size_medium": 0.20,
			}
			costs := CalculateTokenCosts(&converter.TokenUsage{
				WebSearchRequests: 3,
			}, &tt.price)

			assert.InDelta(t, tt.wantSearch, costs.WebSearchCost, 1e-9)
			assert.InDelta(t, tt.wantSearch, costs.TotalCost, 1e-9)
		})
	}
}

func TestCalculateTokenCosts_WebSearchDefaultsToMedium(t *testing.T) {
	usage := &converter.TokenUsage{
		WebSearchRequests: 1,
	}
	price := &ModelPrice{
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low":    0.10,
			"search_context_size_medium": 0.20,
			"search_context_size_high":   0.30,
		},
	}

	costs := CalculateTokenCosts(usage, price)

	assert.InDelta(t, 0.20, costs.WebSearchCost, 1e-9)
	assert.InDelta(t, 0.20, costs.TotalCost, 1e-9)
}

// --- Alibaba Cloud style full-session tiers: 32k / 128k / 256k ---

func TestCalculateTokenCosts_Above32kFullSession(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:     40_000,
		CompletionTokens: 1_000,
	}
	price := &ModelPrice{
		InputCostPerToken:          0.01,
		OutputCostPerToken:         0.02,
		InputCostPerTokenAbove32k:  0.005,
		OutputCostPerTokenAbove32k: 0.01,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	// Full session: the WHOLE prompt/completion is billed at the above-32k
	// rate, not just the tokens beyond 32k.
	assert.InDelta(t, 200.0, costs.InputCost, 1e-9) // 40_000 * 0.005
	assert.InDelta(t, 10.0, costs.OutputCost, 1e-9) // 1_000 * 0.01
	assert.InDelta(t, 210.0, costs.TotalCost, 1e-9)
}

func TestCalculateTokenCosts_32kThresholdIsExclusive(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens: tokenTiering32kThreshold,
	}
	price := &ModelPrice{
		InputCostPerToken:         0.01,
		InputCostPerTokenAbove32k: 0.005,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	// Exactly at the threshold: still base rate, tier only applies strictly above it.
	assert.InDelta(t, 320.0, costs.InputCost, 1e-9) // 32_000 * 0.01
}

func TestCalculateTokenCosts_Above128kFullSession(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 150_000}
	price := &ModelPrice{
		InputCostPerToken:          0.001,
		InputCostPerTokenAbove128k: 0.0006,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 90.0, costs.InputCost, 1e-9) // 150_000 * 0.0006
}

func TestCalculateTokenCosts_Above256kFullSession(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 300_000}
	price := &ModelPrice{
		InputCostPerToken:          0.001,
		InputCostPerTokenAbove256k: 0.0004,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 120.0, costs.InputCost, 1e-9) // 300_000 * 0.0004
}

func TestCalculateTokenCosts_MultiTierLadder_128kAnd256k(t *testing.T) {
	// Mirrors qwen3.5-plus: two brackets configured, base rate below both.
	price := &ModelPrice{
		InputCostPerToken:          0.002,
		InputCostPerTokenAbove128k: 0.0015,
		InputCostPerTokenAbove256k: 0.001,
	}

	tests := []struct {
		name         string
		promptTokens int
		wantInput    float64
	}{
		{"below_128k_uses_base_rate", 100_000, 200.0},            // 100_000 * 0.002
		{"between_128k_and_256k_uses_128k_rate", 150_000, 225.0}, // 150_000 * 0.0015
		{"above_256k_uses_256k_rate", 300_000, 300.0},            // 300_000 * 0.001
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			costs := CalculateTokenCosts(&converter.TokenUsage{PromptTokens: tt.promptTokens}, price)
			require.NotNil(t, costs)
			assert.InDelta(t, tt.wantInput, costs.InputCost, 1e-9)
		})
	}
}

func TestCalculateTokenCosts_LadderSkipsUnconfiguredTier(t *testing.T) {
	// Mirrors qwen3.7-flash: 32k and 256k configured, 128k intentionally left
	// unset. A prompt between the two configured thresholds must still fall
	// through to the 32k rate rather than the base rate.
	price := &ModelPrice{
		InputCostPerToken:          0.003,
		InputCostPerTokenAbove32k:  0.0025,
		InputCostPerTokenAbove256k: 0.001,
	}

	tests := []struct {
		name         string
		promptTokens int
		wantInput    float64
	}{
		{"between_32k_and_256k_uses_32k_rate", 150_000, 375.0}, // 150_000 * 0.0025
		{"above_256k_uses_256k_rate", 300_000, 300.0},          // 300_000 * 0.001
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			costs := CalculateTokenCosts(&converter.TokenUsage{PromptTokens: tt.promptTokens}, price)
			require.NotNil(t, costs)
			assert.InDelta(t, tt.wantInput, costs.InputCost, 1e-9)
		})
	}
}

func TestCalculateTokenCosts_272kTakesPrecedenceOver256k(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 280_000} // above both 256k and 272k
	price := &ModelPrice{
		InputCostPerToken:          0.01,
		InputCostPerTokenAbove256k: 0.001,
		InputCostPerTokenAbove272k: 0.0009,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 252.0, costs.InputCost, 1e-9) // 280_000 * 0.0009, not the 256k rate
}

func TestCalculateTokenCosts_CacheTierOverridesBaseCacheRate(t *testing.T) {
	// Mirrors qwen3-vl-flash: a non-zero BASE cache_read rate is configured
	// alongside a tiered one. The base rate must not "win" once the prompt
	// crosses the threshold — the fallback-to-base-rate path only applies
	// when no explicit cache rate is configured at all.
	usage := &converter.TokenUsage{
		PromptTokens:      40_000,
		CachedInputTokens: 1_000,
	}
	price := &ModelPrice{
		InputCostPerToken:               0.01,
		CacheReadInputTokenCost:         0.0001, // base cache rate (non-zero)
		CacheReadInputTokenCostAbove32k: 0.00005,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 0.05, costs.CachedInputCost, 1e-9) // 1_000 * 0.00005, NOT 1_000 * 0.0001
}

func TestCalculateTokenCosts_CacheAbove32kFullSession(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:        50_000,
		CachedInputTokens:   1_000,
		CacheCreationTokens: 500,
	}
	price := &ModelPrice{
		InputCostPerToken:                   0.01,
		CacheReadInputTokenCost:             0.001,
		CacheReadInputTokenCostAbove32k:     0.0005,
		CacheCreationInputTokenCost:         0.002,
		CacheCreationInputTokenCostAbove32k: 0.0015,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 0.5, costs.CachedInputCost, 1e-9)    // 1_000 * 0.0005
	assert.InDelta(t, 0.75, costs.CacheCreationCost, 1e-9) // 500 * 0.0015
}

// --- Provider-specific 200k semantics: Gemini full-session vs Anthropic proportional ---

func TestCalculateTokenCosts_Gemini200k_FullSession(t *testing.T) {
	// Google's documented Gemini rule: once input context exceeds 200k tokens,
	// the ENTIRE request (input, output, cache) bills at the long-context rate.
	usage := &converter.TokenUsage{
		PromptTokens:        300_000,
		CompletionTokens:    1_000,
		CachedInputTokens:   50_000,
		CacheCreationTokens: 10_000,
	}
	price := &ModelPrice{
		LiteLLMProvider:                      "gemini",
		InputCostPerToken:                    0.000001625,
		InputCostPerTokenAbove200k:           0.00000325,
		OutputCostPerToken:                   0.000013,
		OutputCostPerTokenAbove200k:          0.0000195,
		CacheReadInputTokenCost:              0.000000125,
		CacheReadInputTokenCostAbove200k:     0.00000025,
		CacheCreationInputTokenCost:          0.0000015625,
		CacheCreationInputTokenCostAbove200k: 0.00000025,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	// regular input = 300_000 - 50_000 (cached) - 10_000 (cache creation) = 240_000
	assert.InDelta(t, 240_000*0.00000325, costs.InputCost, 1e-9, "ALL regular input tokens at the long-context rate, not just the excess")
	assert.InDelta(t, 1_000*0.0000195, costs.OutputCost, 1e-9, "output also billed fully at the long-context rate")
	assert.InDelta(t, 50_000*0.00000025, costs.CachedInputCost, 1e-9, "cache read also billed fully at the long-context rate")
	assert.InDelta(t, 10_000*0.00000025, costs.CacheCreationCost, 1e-9, "cache creation also billed fully at the long-context rate")
}

func TestCalculateTokenCosts_Anthropic200k_StaysProportional(t *testing.T) {
	// This is the generic (non-Gemini) proportional-200k tier shape, not an
	// observed real Claude Sonnet billing rule: Sonnet 4.5 doesn't support
	// >200k context via the standard Anthropic API at all, and Sonnet 4.6
	// (which supports up to 1M context) bills it at the regular rate with no
	// above-200k surcharge -- no real Anthropic model exercises this branch
	// today. The scenario below is synthetic, only to confirm an
	// Anthropic-tagged price doesn't accidentally flip to Gemini's
	// full-session behavior just because *_above_200k_tokens is configured.
	usage := &converter.TokenUsage{
		PromptTokens:     300_000,
		CompletionTokens: 50,
	}
	price := &ModelPrice{
		LiteLLMProvider:            "anthropic",
		InputCostPerToken:          0.001,
		InputCostPerTokenAbove200k: 0.0005,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	// Unchanged from TestCalculateTokenCosts_Above200k_Input: 200_000*0.001 + 100_000*0.0005
	assert.InDelta(t, 250.0, costs.InputCost, 1e-9, "Anthropic must keep proportional billing at 200k")
}

func TestCalculateTokenCosts_UnknownProvider200k_StaysProportional(t *testing.T) {
	// No provider set at all (or a provider that isn't Gemini/Vertex): default
	// to the existing proportional behavior, same as before this change.
	usage := &converter.TokenUsage{
		PromptTokens:     300_000,
		CompletionTokens: 50,
	}
	price := &ModelPrice{
		InputCostPerToken:          0.001,
		InputCostPerTokenAbove200k: 0.0005,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 250.0, costs.InputCost, 1e-9)
}

func TestCalculateTokenCosts_VertexProvider200k_FullSession(t *testing.T) {
	// "vertex_ai" (Gemini served via Vertex) must also get full-session 200k.
	usage := &converter.TokenUsage{PromptTokens: 250_000}
	price := &ModelPrice{
		LiteLLMProvider:            "vertex_ai",
		InputCostPerToken:          0.000001625,
		InputCostPerTokenAbove200k: 0.00000325,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 250_000*0.00000325, costs.InputCost, 1e-9)
}

func TestCalculateTokenCosts_Gemini200kBelowThreshold_UsesBaseRate(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 150_000}
	price := &ModelPrice{
		LiteLLMProvider:            "gemini",
		InputCostPerToken:          0.000001625,
		InputCostPerTokenAbove200k: 0.00000325,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 150_000*0.000001625, costs.InputCost, 1e-9)
}

func TestCalculateTokenCosts_Gemini256kTakesPrecedenceOver200k(t *testing.T) {
	// If a Gemini-family model ever also configures a 256k full-session tier,
	// the higher (256k) threshold must still win over the provider-triggered
	// 200k tier, matching the existing descending-priority ladder.
	usage := &converter.TokenUsage{PromptTokens: 260_000}
	price := &ModelPrice{
		LiteLLMProvider:            "gemini",
		InputCostPerToken:          0.000001625,
		InputCostPerTokenAbove200k: 0.00000325,
		InputCostPerTokenAbove256k: 0.000004,
	}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.InDelta(t, 260_000*0.000004, costs.InputCost, 1e-9)
}

func TestCalculateTokenCosts_GoogleProviderAlias200k_FullSession(t *testing.T) {
	// LiteLLM DB config routes both "google" and "google_ai_studio" custom_llm_provider
	// values to the Gemini provider type (see mapProviderType), so billing must
	// recognize them too -- not just the literal "gemini"/"vertex" strings.
	for _, provider := range []string{"google", "google_ai_studio", "GoogleAI"} {
		t.Run(provider, func(t *testing.T) {
			usage := &converter.TokenUsage{PromptTokens: 300_000}
			price := &ModelPrice{
				LiteLLMProvider:            provider,
				InputCostPerToken:          0.000001625,
				InputCostPerTokenAbove200k: 0.00000325,
			}

			costs := CalculateTokenCosts(usage, price)

			require.NotNil(t, costs)
			assert.InDelta(t, 300_000*0.00000325, costs.InputCost, 1e-9, "provider %q must bill full-session at 200k", provider)
		})
	}
}
