package models

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/transform"
	"github.com/stretchr/testify/assert"
)

func TestCalculateTokenCosts_RegularTokensOnly(t *testing.T) {
	usage := &transform.TokenUsage{
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

func TestCalculateTokenCosts_Vertex_WithAudioAndCached(t *testing.T) {
	// Vertex semantic: audio and cached are INCLUDED in totals
	// promptTokenCount=100 includes AudioInputTokens=5 and CachedInputTokens=20
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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

func TestCalculateTokenCosts_WithPrediction(t *testing.T) {
	// Prediction tokens (accepted and rejected) are included in completion tokens
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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

func TestCalculateTokenCosts_NilUsage(t *testing.T) {
	price := &ModelPrice{
		InputCostPerToken: 0.01,
	}

	costs := CalculateTokenCosts(nil, price)
	assert.Nil(t, costs)
}

func TestCalculateTokenCosts_NilPrice(t *testing.T) {
	usage := &transform.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	costs := CalculateTokenCosts(usage, nil)
	assert.Nil(t, costs)
}

func TestModelPrice_CalculateCost(t *testing.T) {
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
	usage := &transform.TokenUsage{
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
