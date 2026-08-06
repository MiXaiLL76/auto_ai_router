package modeltable

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMapProviderType(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected config.ProviderType
	}{
		{"openai", "openai", config.ProviderTypeOpenAI},
		{"router", "router", config.ProviderTypeOpenAI},
		{"air", "air", config.ProviderTypeAIR},
		{"aar", "aar", config.ProviderTypeAIR},
		{"auto-ai-router", "auto-ai-router", config.ProviderTypeAIR},
		{"vertex", "VERTEX", config.ProviderTypeVertexAI},
		{"vertex_ai", "vertex_ai", config.ProviderTypeVertexAI},
		{"google", "GoogleAI", config.ProviderTypeGemini},
		{"google_ai_studio", "google_ai_studio", config.ProviderTypeGemini},
		{"cometapi", "cometapi", config.ProviderTypeCometAPI},
		{"comet-api", "comet-api", config.ProviderTypeCometAPI},
		{"proman", "proman", config.ProviderTypeProMan},
		{"pro-man", "pro-man", config.ProviderTypeProMan},
		{"xai", "xAI", config.ProviderTypeOpenAI},
		{"unknown", "some-other", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, mapProviderType(tt.input))
		})
	}
}

func TestConvertCredentialTableToConfig(t *testing.T) {
	apiKey := "key"
	apiBase := "http://example.com"
	project := "proj"
	location := "us-central1"
	credsJSON := `{"type":"service_account"}`
	provider := "vertex_ai"
	name := "cred-1"

	cfg := convertCredentialTableToConfig(queries.CredentialTable{
		CredentialName: &name,
		CredentialInfo: &queries.CredentialLiteLLMInfo{
			CustomLLMProvider: &provider,
		},
		CredentialParams: &queries.CredentialLiteLLMParams{
			APIKey:            &apiKey,
			APIBase:           &apiBase,
			VertexProject:     &project,
			VertexLocation:    &location,
			VertexCredentials: &credsJSON,
		},
	})

	assert.Equal(t, "cred-1", cfg.Name)
	assert.Equal(t, config.ProviderTypeVertexAI, cfg.Type)
	assert.Equal(t, "key", cfg.APIKey)
	assert.Equal(t, "http://example.com", cfg.BaseURL)
	assert.Equal(t, "proj", cfg.ProjectID)
	assert.Equal(t, "us-central1", cfg.Location)
	assert.Equal(t, credsJSON, cfg.CredentialsJSON)
	assert.Equal(t, -1, cfg.RPM)
	assert.Equal(t, -1, cfg.TPM)
}

func TestConvertInlineCredToConfig(t *testing.T) {
	t.Run("uses_top_level_provider", func(t *testing.T) {
		provider := "openai"
		apiKey := "key"
		apiBase := "https://api.example.com"

		cfg := convertInlineCredToConfig("inline-1", &queries.GenericLiteLLMParams{
			CustomLLMProvider: &provider,
			CredentialLiteLLMParams: queries.CredentialLiteLLMParams{
				APIKey:  &apiKey,
				APIBase: &apiBase,
			},
		})

		assert.Equal(t, "inline-1", cfg.Name)
		assert.Equal(t, config.ProviderTypeOpenAI, cfg.Type)
		assert.Equal(t, "key", cfg.APIKey)
		assert.Equal(t, "https://api.example.com", cfg.BaseURL)
		assert.Equal(t, -1, cfg.RPM)
		assert.Equal(t, -1, cfg.TPM)
	})

	t.Run("falls_back_to_embedded_provider", func(t *testing.T) {
		provider := "vertex_ai"
		project := "proj"

		cfg := convertInlineCredToConfig("inline-2", &queries.GenericLiteLLMParams{
			CredentialLiteLLMParams: queries.CredentialLiteLLMParams{
				CustomLLMProviderName: &provider,
				VertexProject:         &project,
			},
		})

		assert.Equal(t, config.ProviderTypeVertexAI, cfg.Type)
		assert.Equal(t, "proj", cfg.ProjectID)
	})
}

func TestHasInlineCredentials(t *testing.T) {
	empty := ""
	apiKey := "key"
	project := "proj"

	assert.False(t, hasInlineCredentials(nil))
	assert.False(t, hasInlineCredentials(&queries.CredentialLiteLLMParams{}))
	assert.False(t, hasInlineCredentials(&queries.CredentialLiteLLMParams{APIKey: &empty}))
	assert.True(t, hasInlineCredentials(&queries.CredentialLiteLLMParams{APIKey: &apiKey}))
	assert.True(t, hasInlineCredentials(&queries.CredentialLiteLLMParams{VertexProject: &project}))
}

func TestConvertPricingToModelPrice(t *testing.T) {
	input := 0.01
	output := 0.02
	outputReasoning := 0.03
	cacheRead := 0.04
	cacheCreation := 0.05
	outputImage := 0.5
	outputImageToken := 0.6
	inputAbove200k := 0.07

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		InputCostPerToken:                 &input,
		OutputCostPerToken:                &output,
		OutputCostPerReasoningToken:       &outputReasoning,
		CacheReadInputTokenCost:           &cacheRead,
		CacheCreationInputTokenCost:       &cacheCreation,
		OutputCostPerImage:                &outputImage,
		OutputCostPerImageToken:           &outputImageToken,
		InputCostPerTokenAbove200kTokens:  &inputAbove200k,
		OutputCostPerTokenAbove200kTokens: &output,
	})

	assert.NotNil(t, price)
	assert.Equal(t, input, price.InputCostPerToken)
	assert.Equal(t, output, price.OutputCostPerToken)
	assert.Equal(t, outputReasoning, price.OutputCostPerReasoningToken)
	assert.Equal(t, cacheRead, price.InputCostPerCachedToken)
	assert.Equal(t, cacheCreation, price.CacheCreationInputTokenCost)
	assert.Equal(t, outputImage, price.OutputCostPerImage)
	assert.Equal(t, outputImageToken, price.OutputCostPerImageToken)
	assert.Equal(t, inputAbove200k, price.InputCostPerTokenAbove200k)

	assert.Nil(t, convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{}))
	assert.Nil(t, convertPricingToModelPrice(nil))
}

func TestConvertPricingToModelPrice_AllFields(t *testing.T) {
	input := 0.01
	output := 0.02
	inputAbove32k := 0.021
	outputAbove32k := 0.022
	inputAbove128k := 0.023
	outputAbove128k := 0.024
	inputAbove200k := 0.03
	outputAbove200k := 0.04
	inputAbove256k := 0.041
	outputAbove256k := 0.042
	inputAbove272k := 0.045
	outputAbove272k := 0.047
	inputAudio := 0.05
	outputAudio := 0.06
	outputReasoning := 0.07
	cacheRead := 0.08
	cacheCreation := 0.085
	cacheReadAbove32k := 0.0811
	cacheCreationAbove32k := 0.0812
	cacheReadAbove128k := 0.0821
	cacheCreationAbove128k := 0.0822
	cacheReadAbove200k := 0.084
	cacheCreationAbove200k := 0.0845
	cacheReadAbove256k := 0.0846
	cacheCreationAbove256k := 0.0847
	cacheCreationAbove1hr := 0.0855
	cacheCreationAbove1hrAbove200k := 0.0857
	cacheReadAbove272k := 0.086
	cacheCreationAbove272k := 0.087
	cacheReadAudio := 0.088
	outputImage := 0.09
	outputImageToken := 0.10
	searchContextCost := map[string]float64{
		"search_context_size_low":    0.01,
		"search_context_size_medium": 0.02,
		"search_context_size_high":   0.03,
	}
	webSearchBillingUnit := "per_prompt"

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		InputCostPerToken:                                  &input,
		OutputCostPerToken:                                 &output,
		InputCostPerTokenAbove32kTokens:                    &inputAbove32k,
		OutputCostPerTokenAbove32kTokens:                   &outputAbove32k,
		InputCostPerTokenAbove128kTokens:                   &inputAbove128k,
		OutputCostPerTokenAbove128kTokens:                  &outputAbove128k,
		InputCostPerTokenAbove200kTokens:                   &inputAbove200k,
		OutputCostPerTokenAbove200kTokens:                  &outputAbove200k,
		InputCostPerTokenAbove256kTokens:                   &inputAbove256k,
		OutputCostPerTokenAbove256kTokens:                  &outputAbove256k,
		InputCostPerTokenAbove272kTokens:                   &inputAbove272k,
		OutputCostPerTokenAbove272kTokens:                  &outputAbove272k,
		InputCostPerAudioToken:                             &inputAudio,
		OutputCostPerAudioToken:                            &outputAudio,
		OutputCostPerReasoningToken:                        &outputReasoning,
		CacheReadInputTokenCost:                            &cacheRead,
		CacheCreationInputTokenCost:                        &cacheCreation,
		CacheReadInputTokenCostAbove32kTokens:              &cacheReadAbove32k,
		CacheCreationInputTokenCostAbove32kTokens:          &cacheCreationAbove32k,
		CacheReadInputTokenCostAbove128kTokens:             &cacheReadAbove128k,
		CacheCreationInputTokenCostAbove128kTokens:         &cacheCreationAbove128k,
		CacheReadInputTokenCostAbove200kTokens:             &cacheReadAbove200k,
		CacheCreationInputTokenCostAbove200kTokens:         &cacheCreationAbove200k,
		CacheReadInputTokenCostAbove256kTokens:             &cacheReadAbove256k,
		CacheCreationInputTokenCostAbove256kTokens:         &cacheCreationAbove256k,
		CacheCreationInputTokenCostAbove1hr:                &cacheCreationAbove1hr,
		CacheCreationInputTokenCostAbove1hrAbove200kTokens: &cacheCreationAbove1hrAbove200k,
		CacheReadInputTokenCostAbove272kTokens:             &cacheReadAbove272k,
		CacheCreationInputTokenCostAbove272kTokens:         &cacheCreationAbove272k,
		CacheReadInputAudioTokenCost:                       &cacheReadAudio,
		OutputCostPerImage:                                 &outputImage,
		OutputCostPerImageToken:                            &outputImageToken,
		SearchContextCostPerQuery:                          searchContextCost,
		WebSearchBillingUnit:                               &webSearchBillingUnit,
	})

	assert.NotNil(t, price)
	assert.Equal(t, input, price.InputCostPerToken)
	assert.Equal(t, output, price.OutputCostPerToken)
	assert.Equal(t, inputAbove32k, price.InputCostPerTokenAbove32k)
	assert.Equal(t, outputAbove32k, price.OutputCostPerTokenAbove32k)
	assert.Equal(t, inputAbove128k, price.InputCostPerTokenAbove128k)
	assert.Equal(t, outputAbove128k, price.OutputCostPerTokenAbove128k)
	assert.Equal(t, inputAbove200k, price.InputCostPerTokenAbove200k)
	assert.Equal(t, outputAbove200k, price.OutputCostPerTokenAbove200k)
	assert.Equal(t, inputAbove256k, price.InputCostPerTokenAbove256k)
	assert.Equal(t, outputAbove256k, price.OutputCostPerTokenAbove256k)
	assert.Equal(t, inputAbove272k, price.InputCostPerTokenAbove272k)
	assert.Equal(t, outputAbove272k, price.OutputCostPerTokenAbove272k)
	assert.Equal(t, inputAudio, price.InputCostPerAudioToken)
	assert.Equal(t, outputAudio, price.OutputCostPerAudioToken)
	assert.Equal(t, outputReasoning, price.OutputCostPerReasoningToken)
	assert.Equal(t, cacheRead, price.InputCostPerCachedToken)
	assert.Equal(t, cacheCreation, price.CacheCreationInputTokenCost)
	assert.Equal(t, cacheReadAbove32k, price.CacheReadInputTokenCostAbove32k)
	assert.Equal(t, cacheCreationAbove32k, price.CacheCreationInputTokenCostAbove32k)
	assert.Equal(t, cacheReadAbove128k, price.CacheReadInputTokenCostAbove128k)
	assert.Equal(t, cacheCreationAbove128k, price.CacheCreationInputTokenCostAbove128k)
	assert.Equal(t, cacheReadAbove200k, price.CacheReadInputTokenCostAbove200k)
	assert.Equal(t, cacheCreationAbove200k, price.CacheCreationInputTokenCostAbove200k)
	assert.Equal(t, cacheReadAbove256k, price.CacheReadInputTokenCostAbove256k)
	assert.Equal(t, cacheCreationAbove256k, price.CacheCreationInputTokenCostAbove256k)
	assert.Equal(t, cacheCreationAbove1hr, price.CacheCreationInputTokenCostAbove1hr)
	assert.Equal(t, cacheCreationAbove1hrAbove200k, price.CacheCreationInputTokenCostAbove1hrAbove200k)
	assert.Equal(t, cacheReadAbove272k, price.CacheReadInputTokenCostAbove272k)
	assert.Equal(t, cacheCreationAbove272k, price.CacheCreationInputTokenCostAbove272k)
	assert.Equal(t, cacheReadAudio, price.CacheReadInputAudioTokenCost)
	assert.Equal(t, outputImage, price.OutputCostPerImage)
	assert.Equal(t, outputImageToken, price.OutputCostPerImageToken)
	assert.Equal(t, searchContextCost, price.SearchContextCostPerQuery)
	assert.Equal(t, webSearchBillingUnit, price.WebSearchBillingUnit)
}

func TestConvertPricingToModelPrice_WebSearchOnly(t *testing.T) {
	searchContextCost := map[string]float64{
		"search_context_size_medium": 0.02,
	}

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		SearchContextCostPerQuery: searchContextCost,
	})

	require.NotNil(t, price)
	assert.Equal(t, searchContextCost, price.SearchContextCostPerQuery)
}
