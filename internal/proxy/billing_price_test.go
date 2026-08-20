package proxy

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
)

func TestLookupBillingModelPrice_TwoPassLookup(t *testing.T) {
	registry := models.NewModelPriceRegistry()
	registry.Update(map[string]*models.ModelPrice{
		"gpt-5-mini":    {InputCostPerToken: 2.25e-07, OutputCostPerToken: 1.8e-06},
		"gpt-5-mini-or": {InputCostPerToken: 3.25e-07, OutputCostPerToken: 2.6e-06},
	})

	tests := []struct {
		name          string
		publicModelID string
		modelID       string
		realModelID   string
		wantModelID   string
		wantInputCost float64
	}{
		{
			name:          "openrouter/gpt-5-mini: raw miss on publicModelID, raw hit on modelID",
			publicModelID: "openrouter/gpt-5-mini",
			modelID:       "gpt-5-mini-or",
			realModelID:   "gpt-5-mini",
			wantModelID:   "gpt-5-mini-or",
			wantInputCost: 3.25e-07,
		},
		{
			name:          "openai/gpt-5-mini: raw miss, normalised hit on publicModelID",
			publicModelID: "openai/gpt-5-mini",
			modelID:       "gpt-5-mini",
			realModelID:   "gpt-5-mini",
			wantModelID:   "gpt-5-mini",
			wantInputCost: 2.25e-07,
		},
		{
			name:          "plain modelID when no publicModelID",
			publicModelID: "",
			modelID:       "gpt-5-mini-or",
			realModelID:   "",
			wantModelID:   "gpt-5-mini-or",
			wantInputCost: 3.25e-07,
		},
		{
			name:          "realModelID used as fallback",
			publicModelID: "",
			modelID:       "unknown-model",
			realModelID:   "gpt-5-mini",
			wantModelID:   "gpt-5-mini",
			wantInputCost: 2.25e-07,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotModelID, gotPrice := lookupBillingModelPrice(registry, tt.publicModelID, tt.modelID, tt.realModelID)
			assert.Equal(t, tt.wantModelID, gotModelID)
			assert.NotNil(t, gotPrice)
			assert.Equal(t, tt.wantInputCost, gotPrice.InputCostPerToken)
		})
	}
}

func TestLookupBillingModelPrice_RawEntryBeatsNormalised(t *testing.T) {
	registry := models.NewModelPriceRegistry()
	registry.Update(map[string]*models.ModelPrice{
		"gemini-3-flash-preview":                   {InputCostPerToken: 4.5e-07, OutputCostPerToken: 2.7e-06},
		"google/gemini-3-flash-preview-highlimits": {InputCostPerToken: 9e-07, OutputCostPerToken: 5.4e-06},
	})

	gotModelID, gotPrice := lookupBillingModelPrice(
		registry,
		"google/gemini-3-flash-preview-highlimits",
		"gemini-3-flash-preview",
		"",
	)
	// publicModelID raw key matches first — highlimits price wins.
	assert.Equal(t, "google/gemini-3-flash-preview-highlimits", gotModelID)
	assert.NotNil(t, gotPrice)
	assert.Equal(t, 9e-07, gotPrice.InputCostPerToken)
}

func TestLookupBillingModelPrice_NilRegistry(t *testing.T) {
	gotModelID, gotPrice := lookupBillingModelPrice(nil, "openrouter/gpt-5-mini", "gpt-5-mini-or", "")
	assert.Equal(t, "gpt-5-mini-or", gotModelID)
	assert.Nil(t, gotPrice)
}

func TestLookupBillingModelPrice_DeduplicatesModelIDAndRealModelID(t *testing.T) {
	registry := models.NewModelPriceRegistry()
	registry.Update(map[string]*models.ModelPrice{
		"gpt-5-mini": {InputCostPerToken: 2.25e-07},
	})

	gotModelID, gotPrice := lookupBillingModelPrice(registry, "", "gpt-5-mini", "gpt-5-mini")
	assert.Equal(t, "gpt-5-mini", gotModelID)
	assert.NotNil(t, gotPrice)
}
