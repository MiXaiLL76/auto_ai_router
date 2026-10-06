package modeltable

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Prices kept in LiteLLM_ProxyModelTable must carry the xAI tool prices and
// billing modes through to the router exactly like the price file does.
func TestConvertPricingToModelPrice_XAIToolAndModeFields(t *testing.T) {
	var params queries.CustomPricingLiteLLMParams
	require.NoError(t, json.Unmarshal([]byte(`{
		"input_cost_per_token": 0.0000026,
		"output_cost_per_token": 0.0000078,
		"tool_cost_per_call": {"code_execution": 0.0065, "attachment_search": 0.013, "collections_search": 0.00325},
		"x_search_cost_per_post": 0.0065,
		"x_search_cost_per_profile": 0.013,
		"image_generation_tool_model": "grok-imagine-image-2.0",
		"long_context_pricing_mode": "full_request_200k_inclusive",
		"reasoning_tokens_accounting": "auto"
	}`), &params))

	price := convertPricingToModelPrice(&params)

	require.NotNil(t, price)
	assert.Equal(t, map[string]float64{"code_execution": 0.0065, "attachment_search": 0.013, "collections_search": 0.00325}, price.ToolCostPerCall)
	assert.Equal(t, 0.0065, price.XSearchCostPerPost)
	assert.Equal(t, 0.013, price.XSearchCostPerProfile)
	assert.Equal(t, "grok-imagine-image-2.0", price.ImageGenerationToolModel)
	assert.Equal(t, "full_request_200k_inclusive", price.LongContextPricingMode)
	assert.Equal(t, "auto", price.ReasoningTokensAccounting)
}
