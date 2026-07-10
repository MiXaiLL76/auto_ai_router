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
		{"vertex", "VERTEX", config.ProviderTypeVertexAI},
		{"google", "GoogleAI", config.ProviderTypeGemini},
		{"cometapi", "cometapi", config.ProviderTypeCometAPI},
		{"comet-api", "comet-api", config.ProviderTypeCometAPI},
		{"sosana", "sosana", config.ProviderTypeSosana},
		{"sosana-art", "sosana-art", config.ProviderTypeSosana},
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
	inputImage := 0.4
	outputImage := 0.5
	outputImageToken := 0.6
	inputAbove200k := 0.07

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		InputCostPerToken:                 &input,
		OutputCostPerToken:                &output,
		OutputCostPerReasoningToken:       &outputReasoning,
		CacheReadInputTokenCost:           &cacheRead,
		CacheCreationInputTokenCost:       &cacheCreation,
		InputCostPerImage:                 &inputImage,
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
	assert.Equal(t, inputImage, price.InputCostPerImage)
	assert.Equal(t, outputImage, price.OutputCostPerImage)
	assert.Equal(t, outputImageToken, price.OutputCostPerImageToken)
	assert.Equal(t, inputAbove200k, price.InputCostPerTokenAbove200k)

	assert.Nil(t, convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{}))
	assert.Nil(t, convertPricingToModelPrice(nil))
}

func TestConvertPricingToModelPrice_ImageOnly(t *testing.T) {
	outputImage := 0.5

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		OutputCostPerImage: &outputImage,
	})

	require.NotNil(t, price)
	assert.Equal(t, outputImage, price.OutputCostPerImage)
	assert.Equal(t, 0.0, price.InputCostPerToken)
	assert.Equal(t, 0.0, price.OutputCostPerToken)
}

func TestConvertPricingToModelPrice_InputImageOnly(t *testing.T) {
	inputImage := 0.088113

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		InputCostPerImage: &inputImage,
	})

	require.NotNil(t, price)
	assert.Equal(t, inputImage, price.InputCostPerImage)
	assert.Equal(t, 0.0, price.InputCostPerToken)
	assert.Equal(t, 0.0, price.OutputCostPerToken)
}

func TestConvertPricingToModelPrice_AllFields(t *testing.T) {
	input := 0.01
	output := 0.02
	inputAbove200k := 0.03
	outputAbove200k := 0.04
	inputAbove272k := 0.045
	outputAbove272k := 0.047
	inputAudio := 0.05
	outputAudio := 0.06
	outputReasoning := 0.07
	cacheRead := 0.08
	cacheCreation := 0.085
	cacheReadAbove272k := 0.086
	cacheCreationAbove272k := 0.087
	inputImage := 0.088
	outputImage := 0.09
	outputImageToken := 0.10

	price := convertPricingToModelPrice(&queries.CustomPricingLiteLLMParams{
		InputCostPerToken:                          &input,
		OutputCostPerToken:                         &output,
		InputCostPerTokenAbove200kTokens:           &inputAbove200k,
		OutputCostPerTokenAbove200kTokens:          &outputAbove200k,
		InputCostPerTokenAbove272kTokens:           &inputAbove272k,
		OutputCostPerTokenAbove272kTokens:          &outputAbove272k,
		InputCostPerAudioToken:                     &inputAudio,
		OutputCostPerAudioToken:                    &outputAudio,
		OutputCostPerReasoningToken:                &outputReasoning,
		CacheReadInputTokenCost:                    &cacheRead,
		CacheCreationInputTokenCost:                &cacheCreation,
		CacheReadInputTokenCostAbove272kTokens:     &cacheReadAbove272k,
		CacheCreationInputTokenCostAbove272kTokens: &cacheCreationAbove272k,
		InputCostPerImage:                          &inputImage,
		OutputCostPerImage:                         &outputImage,
		OutputCostPerImageToken:                    &outputImageToken,
	})

	assert.NotNil(t, price)
	assert.Equal(t, input, price.InputCostPerToken)
	assert.Equal(t, output, price.OutputCostPerToken)
	assert.Equal(t, inputAbove200k, price.InputCostPerTokenAbove200k)
	assert.Equal(t, outputAbove200k, price.OutputCostPerTokenAbove200k)
	assert.Equal(t, inputAbove272k, price.InputCostPerTokenAbove272k)
	assert.Equal(t, outputAbove272k, price.OutputCostPerTokenAbove272k)
	assert.Equal(t, inputAudio, price.InputCostPerAudioToken)
	assert.Equal(t, outputAudio, price.OutputCostPerAudioToken)
	assert.Equal(t, outputReasoning, price.OutputCostPerReasoningToken)
	assert.Equal(t, cacheRead, price.InputCostPerCachedToken)
	assert.Equal(t, cacheCreation, price.CacheCreationInputTokenCost)
	assert.Equal(t, cacheReadAbove272k, price.CacheReadInputTokenCostAbove272k)
	assert.Equal(t, cacheCreationAbove272k, price.CacheCreationInputTokenCostAbove272k)
	assert.Equal(t, inputImage, price.InputCostPerImage)
	assert.Equal(t, outputImage, price.OutputCostPerImage)
	assert.Equal(t, outputImageToken, price.OutputCostPerImageToken)
}
