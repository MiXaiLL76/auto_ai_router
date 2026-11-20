package models_test

import (
	"encoding/json"
	"strconv"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// geminiImagePrice is a Gemini image model price row in round numbers: text and
// image input share one rate, as in Google's pricing for these models.
const (
	geminiInputRate       = 0.000001
	geminiOutputRate      = 0.000005
	geminiOutputImageRate = 0.00003
	geminiSearchQueryCost = 0.014
	geminiImagePrice      = `{
		"input_cost_per_token": 0.000001,
		"input_cost_per_image_token": 0.000001,
		"output_cost_per_token": 0.000005,
		"output_cost_per_image_token": 0.00003,
		"cache_read_input_token_cost": 0.0000001,
		"supports_web_search": true,
		"search_context_cost_per_query": {
			"search_context_size_low": 0.014,
			"search_context_size_medium": 0.014,
			"search_context_size_high": 0.014
		},
		"web_search_billing_unit": "per_query"
	}`
)

func geminiCosts(t *testing.T, model string, imageEndpoint bool, providerBody string) (*converter.TokenUsage, *converter.TokenCosts) {
	t.Helper()
	conv := converter.New(config.ProviderTypeVertexAI, converter.RequestMode{
		IsImageGeneration: imageEndpoint,
		ModelID:           model,
	})
	convertedBody, err := conv.ResponseTo([]byte(providerBody))
	require.NoError(t, err)
	usage := conv.UsageFromResponse(convertedBody)
	require.NotNil(t, usage)
	var price models.ModelPrice
	require.NoError(t, json.Unmarshal([]byte(geminiImagePrice), &price))
	costs := price.CalculateCosts(usage)
	require.NotNil(t, costs)
	return usage, costs
}

// TestGeminiImageEndpointBilling bills the images endpoint by the image tokens
// Gemini reports, and its thinking once, at the text output rate — never as image
// output. Every Gemini image model thinks this way, not only the profiled one.
func TestGeminiImageEndpointBilling(t *testing.T) {
	for _, model := range []string{"gemini-nano-banana-2.1", "gemini-3-pro-image-preview"} {
		t.Run(model, func(t *testing.T) {
			body := `{
				"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]},"finishReason":"STOP"}],
				"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":1120,"thoughtsTokenCount":400,
					"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1120}]}
			}`
			usage, costs := geminiCosts(t, model, true, body)
			assert.Equal(t, 1120, usage.OutputImageTokens)
			assert.Equal(t, 400, usage.ReasoningTokens)

			assert.InDelta(t, 12*geminiInputRate, costs.InputCost, 1e-12)
			assert.Zero(t, costs.OutputCost, "no text output")
			assert.InDelta(t, 400*geminiOutputRate, costs.ReasoningCost, 1e-12)
			assert.InDelta(t, 1120*geminiOutputImageRate, costs.ImageCost, 1e-12)
			assert.InDelta(t, 12*geminiInputRate+400*geminiOutputRate+1120*geminiOutputImageRate, costs.TotalCost, 1e-12)
		})
	}
}

// TestGeminiSearchBilling bills every Google Web and Image Search query the
// grounding metadata confirms, and nothing without a search. The context the
// search retrieved is billed as input except on a model priced without it
// (gemini-nano-banana-2.1).
func TestGeminiSearchBilling(t *testing.T) {
	response := func(grounding string) string {
		return `{
			"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"` + grounding + `}],
			"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":900,"candidatesTokenCount":50,"totalTokenCount":1050}
		}`
	}
	tests := []struct {
		name      string
		model     string
		grounding string
		queries   int
		input     int
	}{
		{name: "web search", model: "gemini-nano-banana-2.1",
			grounding: `,"groundingMetadata":{"webSearchQueries":["a","b"]}`, queries: 2, input: 100},
		{name: "image search", model: "gemini-nano-banana-2.1",
			grounding: `,"groundingMetadata":{"imageSearchQueries":["a"]}`, queries: 1, input: 100},
		{name: "web and image search", model: "gemini-nano-banana-2.1",
			grounding: `,"groundingMetadata":{"webSearchQueries":["a","b"],"imageSearchQueries":["a"]}`, queries: 3, input: 100},
		{name: "search on a model that bills its context", model: "gemini-2.5-flash",
			grounding: `,"groundingMetadata":{"webSearchQueries":["a"]}`, queries: 1, input: 1000},
		// Tool-use context without a confirmed search stays billed.
		{name: "no search", model: "gemini-nano-banana-2.1", grounding: "", queries: 0, input: 1000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage, costs := geminiCosts(t, tt.model, false, response(tt.grounding))
			assert.Equal(t, tt.queries, usage.WebSearchRequests)
			assert.InDelta(t, float64(tt.queries)*geminiSearchQueryCost, costs.WebSearchCost, 1e-12)
			assert.InDelta(t, float64(tt.input)*geminiInputRate, costs.InputCost, 1e-12, "input tokens: "+strconv.Itoa(tt.input))
		})
	}
}
