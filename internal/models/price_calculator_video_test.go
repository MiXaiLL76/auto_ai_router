package models

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// geminiEmbedding2Price is the VseLLM prices entry for gemini-embedding-2
// (rates already include the 1.3 markup).
const geminiEmbedding2Price = `{
	"input_cost_per_token": 0.00000026,
	"input_cost_per_image_token": 0.000000585,
	"input_cost_per_audio_token": 0.00000845,
	"input_cost_per_video_token": 0.0000156,
	"output_cost_per_token": 0,
	"rate": 1.3
}`

func TestModelPrice_DecodesInputCostPerVideoToken(t *testing.T) {
	var price ModelPrice
	require.NoError(t, json.Unmarshal([]byte(geminiEmbedding2Price), &price))
	assert.Equal(t, 0.00000026, price.InputCostPerToken)
	assert.Equal(t, 0.000000585, price.InputCostPerImageToken)
	assert.Equal(t, 0.00000845, price.InputCostPerAudioToken)
	assert.Equal(t, 0.0000156, price.InputCostPerVideoToken)
}

func TestCalculateTokenCosts_GeminiEmbedding2PricesEachModality(t *testing.T) {
	var price ModelPrice
	require.NoError(t, json.Unmarshal([]byte(geminiEmbedding2Price), &price))
	// text 132 + one PDF page/image 258 + 10 s audio 250 + 10 frames video 660.
	usage := &converter.TokenUsage{
		PromptTokens:     1300,
		ImageTokens:      258,
		AudioInputTokens: 250,
		VideoInputTokens: 660,
	}

	costs := CalculateTokenCosts(usage, &price)

	require.NotNil(t, costs)
	assert.InDelta(t, 132*0.00000026, costs.InputCost, 1e-15, "only the text remainder at the text rate")
	assert.InDelta(t, 258*0.000000585, costs.ImageCost, 1e-15)
	assert.InDelta(t, 250*0.00000845, costs.AudioInputCost, 1e-15)
	assert.InDelta(t, 660*0.0000156, costs.VideoInputCost, 1e-15)
	assert.Zero(t, costs.OutputCost, "embedding vectors are not output tokens")
	assert.InDelta(t, 132*0.00000026+258*0.000000585+250*0.00000845+660*0.0000156, costs.TotalCost, 1e-15)
}

func TestCalculateTokenCosts_TextOnlyEmbeddingUsesTextRate(t *testing.T) {
	var price ModelPrice
	require.NoError(t, json.Unmarshal([]byte(geminiEmbedding2Price), &price))
	costs := CalculateTokenCosts(&converter.TokenUsage{PromptTokens: 1_000_000}, &price)
	require.NotNil(t, costs)
	assert.InDelta(t, 0.26, costs.TotalCost, 1e-12)
}

func TestCalculateTokenCosts_VideoFallsBackToImageThenInputRate(t *testing.T) {
	usage := &converter.TokenUsage{PromptTokens: 100, ImageTokens: 20, VideoInputTokens: 30}

	withImageRate := CalculateTokenCosts(usage, &ModelPrice{InputCostPerToken: 1, InputCostPerImageToken: 2})
	require.NotNil(t, withImageRate)
	assert.Equal(t, 50.0, withImageRate.InputCost)
	assert.Equal(t, 40.0, withImageRate.ImageCost)
	assert.Equal(t, 60.0, withImageRate.VideoInputCost, "video billed like images before the split")
	assert.Equal(t, 150.0, withImageRate.TotalCost)

	textRateOnly := CalculateTokenCosts(usage, &ModelPrice{InputCostPerToken: 1})
	require.NotNil(t, textRateOnly)
	assert.Equal(t, 100.0, textRateOnly.TotalCost, "no media rates: the whole prompt at the input rate")
}

func TestCalculateTokenCosts_CachedMediaOverlapIsNotBilledTwice(t *testing.T) {
	// A provider that also reports the cached media inside CachedInputTokens:
	// prompt 100 = 60 cached + 50 media counted twice by 10. The overlap comes
	// off images first, then video.
	usage := &converter.TokenUsage{PromptTokens: 100, CachedInputTokens: 60, ImageTokens: 5, VideoInputTokens: 45}
	price := &ModelPrice{InputCostPerToken: 1, CacheReadInputTokenCost: 0.5, InputCostPerImageToken: 2, InputCostPerVideoToken: 3}

	costs := CalculateTokenCosts(usage, price)

	require.NotNil(t, costs)
	assert.Equal(t, 0.0, costs.InputCost)
	assert.Equal(t, 30.0, costs.CachedInputCost)
	assert.Equal(t, 0.0, costs.ImageCost)
	assert.Equal(t, 120.0, costs.VideoInputCost) // (45 - (10 - 5)) * 3
	assert.Equal(t, 150.0, costs.TotalCost)
}
