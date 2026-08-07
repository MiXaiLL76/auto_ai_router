package models_test

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiFlashLiteImageBillingFromProviderResponse(t *testing.T) {
	providerBody := []byte(`{
		"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]}}],
		"usageMetadata":{
			"promptTokenCount":1128,
			"candidatesTokenCount":1357,
			"totalTokenCount":2485,
			"promptTokensDetails":[
				{"modality":"TEXT","tokenCount":8},
				{"modality":"IMAGE","tokenCount":1120}
			]
		}
	}`)
	imageConverter := converter.New(config.ProviderTypeGemini, converter.RequestMode{
		IsImageGeneration: true,
		ModelID:           "gemini-3.1-flash-lite-image",
	})
	convertedBody, err := imageConverter.ResponseTo(providerBody)
	require.NoError(t, err)

	usage := imageConverter.UsageFromResponse(convertedBody)
	require.NotNil(t, usage)
	assert.Equal(t, 1120, usage.ImageTokens)
	assert.Equal(t, 1357, usage.OutputImageTokens)

	price := &models.ModelPrice{
		InputCostPerToken:       0.000000225,
		OutputCostPerToken:      0.00000135,
		OutputCostPerImageToken: 0.000027,
	}
	costs := price.CalculateCosts(usage)
	require.NotNil(t, costs)
	assert.InDelta(t, 0.0000018, costs.InputCost, 1e-12)
	assert.Zero(t, costs.OutputCost)
	assert.InDelta(t, 0.036891, costs.ImageCost, 1e-12)
	assert.InDelta(t, 0.0368928, costs.TotalCost, 1e-12)
}
