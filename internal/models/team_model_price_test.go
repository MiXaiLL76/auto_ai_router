package models

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecodeTeamModelPrices(t *testing.T) {
	prices, err := DecodeTeamModelPrices(json.RawMessage(`{"m1":{"input_cost_per_token":2e-6,"output_cost_per_token":8e-6},"m2":{"output_cost_per_image":0.04}}`))
	require.NoError(t, err)
	require.Len(t, prices, 2)
	assert.Equal(t, 2e-6, prices["m1"].InputCostPerToken)
	assert.Equal(t, 8e-6, prices["m1"].OutputCostPerToken)
	assert.Equal(t, 0.04, prices["m2"].OutputCostPerImage)

	prices, err = DecodeTeamModelPrices(json.RawMessage(`{}`))
	require.NoError(t, err)
	assert.Empty(t, prices)

	// An image tier condition may hold any request value.
	prices, err = DecodeTeamModelPrices(json.RawMessage(`{"m1":{"output_cost_per_image_tiers":[{"when":{"seed":-1},"output_cost_per_image":0.02},{"output_cost_per_image":0.04}]}}`))
	require.NoError(t, err)
	assert.Len(t, prices["m1"].OutputCostPerImageTiers, 2)
}

// A field is parsed once: requests carrying it share the result, an error
// included.
func TestDecodeTeamModelPrices_ParsesAFieldOnce(t *testing.T) {
	raw := `{"shared-m1":{"input_cost_per_token":1}}`
	first, err := DecodeTeamModelPrices(json.RawMessage(raw))
	require.NoError(t, err)
	second, err := DecodeTeamModelPrices(json.RawMessage(raw))
	require.NoError(t, err)
	assert.Same(t, first["shared-m1"], second["shared-m1"])

	_, firstErr := DecodeTeamModelPrices(json.RawMessage(`{"shared-m1":null}`))
	_, secondErr := DecodeTeamModelPrices(json.RawMessage(`{"shared-m1":null}`))
	assert.Same(t, firstErr, secondErr)
}

func TestDecodeTeamModelPrices_RejectsTheWholeField(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{"null field", `null`, "not null"},
		{"not an object", `[{"input_cost_per_token":1}]`, "must be an object"},
		{"empty model name", `{" ":{"input_cost_per_token":1}}`, "empty model name"},
		{"null row", `{"m1":{"input_cost_per_token":1},"m2":null}`, `air_model_prices "m2": null price row`},
		{"row not an object", `{"m1":1}`, "price row must be an object"},
		{"empty row", `{"m1":{}}`, "empty price row"},
		{"unknown field", `{"m1":{"input_cost_per_token":1,"discount":0.1}}`, "unknown price field"},
		{"not a price field", `{"m1":{"input_cost_per_token":1,"litellm_provider":"openai"}}`, `"litellm_provider" is not a price field`},
		{"null price", `{"m1":{"input_cost_per_token":1,"output_cost_per_token":null}}`, "null value"},
		{"negative price", `{"m1":{"input_cost_per_token":-1}}`, "negative amount"},
		{"negative nested price", `{"m1":{"search_context_cost_per_query":{"search_context_size_low":-0.01}}}`, "negative amount"},
		{"negative image tier price", `{"m1":{"output_cost_per_image_tiers":[{"output_cost_per_image":-0.02}]}}`, "negative output_cost_per_image"},
		{"null image tier size", `{"m1":{"output_cost_per_image_tiers":[{"max_pixels":null,"output_cost_per_image":0}]}}`, "null value"},
		{"null image tier operation", `{"m1":{"output_cost_per_image_tiers":[{"operation":null,"output_cost_per_image":0}]}}`, "null value"},
		{"price float64 cannot hold", `{"m1":{"input_cost_per_token":1e309}}`, "air_model_prices"},
		{"web search size the calculator never looks up", `{"m1":{"search_context_cost_per_query":{"medium":0.01}}}`, `unknown key "medium"`},
		{"router's own tool price", `{"m1":{"tool_cost_per_call":{"moonshot_search":0.01}}}`, "unknown tool"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prices, err := DecodeTeamModelPrices(json.RawMessage(tt.raw))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
			assert.Nil(t, prices)
		})
	}
}

func TestTeamModelPrice(t *testing.T) {
	general := &ModelPrice{
		InputCostPerToken:         1.25e-6,
		OutputCostPerToken:        10e-6,
		CacheReadInputTokenCost:   0.125e-6,
		InputCostPerImageToken:    5e-6,
		OutputCostPerImage:        0.04,
		LiteLLMProvider:           "openai",
		ReasoningTokensAdditive:   true,
		InputImagesFreePerRequest: 1,
		ImageGenerationToolModel:  "image-model",
		SearchContextCostPerQuery: map[string]float64{"search_context_size_low": 0.01},
		ToolCostPerCall: map[string]float64{
			ToolCodeExecution:     0.05,
			ToolMoonshotSearch:    0.013,
			ToolMoonshotSearchPro: 0.02,
			ToolMoonshotFetch:     0.001,
		},
	}
	team := &ModelPrice{
		InputCostPerToken:  2e-6,
		OutputCostPerToken: 8e-6,
		ToolCostPerCall:    map[string]float64{ToolWebFetch: 0.002},
	}

	price := TeamModelPrice(general, team)

	assert.Equal(t, &ModelPrice{
		InputCostPerToken:  2e-6,
		OutputCostPerToken: 8e-6,
		// how the model is billed stays the general row's
		LiteLLMProvider:           "openai",
		ReasoningTokensAdditive:   true,
		InputImagesFreePerRequest: 1,
		ImageGenerationToolModel:  "image-model",
		ToolCostPerCall: map[string]float64{
			ToolWebFetch:          0.002,
			ToolMoonshotSearch:    0.013,
			ToolMoonshotSearchPro: 0.02,
			ToolMoonshotFetch:     0.001,
		},
	}, price)
	assert.Equal(t, map[string]float64{ToolWebFetch: 0.002}, team.ToolCostPerCall)
	assert.Equal(t, 1.25e-6, general.InputCostPerToken)
	assert.Len(t, general.ToolCostPerCall, 4)

	assert.Equal(t, &ModelPrice{InputCostPerToken: 2}, TeamModelPrice(&ModelPrice{InputCostPerToken: 1}, &ModelPrice{InputCostPerToken: 2}))
}

// A team row lists only input and output: the calculator's rules for a row
// without the other prices apply, as for any price row.
func TestTeamModelPrice_MissingPricesFollowTheCalculator(t *testing.T) {
	general := &ModelPrice{
		InputCostPerToken:         1e-6,
		OutputCostPerToken:        4e-6,
		CacheReadInputTokenCost:   0.1e-6,
		InputCostPerImageToken:    9e-6,
		OutputCostPerImageToken:   9e-6,
		SearchContextCostPerQuery: map[string]float64{"search_context_size_medium": 0.01},
	}
	price := TeamModelPrice(general, &ModelPrice{InputCostPerToken: 2e-6, OutputCostPerToken: 8e-6})

	costs := price.CalculateCosts(&converter.TokenUsage{
		PromptTokens:      1000,
		CompletionTokens:  100,
		CachedInputTokens: 400,
		ImageTokens:       100,
		OutputImageTokens: 10,
		WebSearchRequests: 1,
	})
	require.NotNil(t, costs)
	// cached input at the input price, image tokens at the input and output
	// prices, web search free
	assert.InDelta(t, 1000*2e-6+100*8e-6, costs.TotalCost, 1e-12)
	assert.Zero(t, costs.WebSearchCost)
}
