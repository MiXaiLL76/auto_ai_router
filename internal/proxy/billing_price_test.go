package proxy

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
			// publicModelID's normalised match (higher priority) wins over
			// modelID's raw match (lower priority) — this is the priority
			// inversion GetPriceAny used to have: trying every candidate's
			// raw key before any candidate's normalised key let modelID's
			// coincidental raw hit shadow the correct publicModelID match.
			name:          "publicModelID's normalised match beats modelID's raw match",
			publicModelID: "openrouter/gpt-5-mini",
			modelID:       "gpt-5-mini-or",
			realModelID:   "gpt-5-mini",
			wantModelID:   "openrouter/gpt-5-mini",
			wantInputCost: 2.25e-07,
		},
		{
			name:          "openai/gpt-5-mini: raw miss, normalised hit on publicModelID",
			publicModelID: "openai/gpt-5-mini",
			modelID:       "gpt-5-mini",
			realModelID:   "gpt-5-mini",
			wantModelID:   "openai/gpt-5-mini",
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

// MergeDB must update the exact key it names (and that key's own raw form)
// but must NOT leak into an unrelated raw-keyed entry just because it
// shares a normalized form — see TestModelPriceRegistry_MergeDB_ScopedToExactKey
// in internal/models for the direct regression test (a DB override for
// "gpt-5.1" clobbering the independently-priced "yandex/gpt-5.1" alias was
// caught here during review of an earlier version of this fix that swept
// the whole registry by normalized form).
func TestLookupBillingModelPrice_MergeDBUpdatesExactKeyOnly(t *testing.T) {
	registry := models.NewModelPriceRegistry()

	registry.Update(map[string]*models.ModelPrice{
		"gpt-5-mini":            {InputCostPerToken: 2.25e-07},
		"openrouter/gpt-5-mini": {InputCostPerToken: 9.99e-07}, // deliberately distinct alias price
	})

	registry.MergeDB(map[string]*models.ModelPrice{
		"gpt-5-mini": {InputCostPerToken: 5.0e-07},
	})

	// The plain model picks up the DB override...
	_, plainPrice := lookupBillingModelPrice(registry, "", "gpt-5-mini", "")
	require.NotNil(t, plainPrice)
	assert.Equal(t, 5.0e-07, plainPrice.InputCostPerToken)

	// ...but the deliberately distinct alias price is untouched, since the
	// DB override didn't name it.
	matched, aliasPrice := lookupBillingModelPrice(registry, "openrouter/gpt-5-mini", "gpt-5-mini-or", "")
	require.NotNil(t, aliasPrice)
	assert.Equal(t, "openrouter/gpt-5-mini", matched)
	assert.Equal(t, 9.99e-07, aliasPrice.InputCostPerToken)
}

// P1: LoadModelPrices must prefer bare keys over prefixed keys in collisions.
func TestLookupBillingModelPrice_BareKeyWinsOverPrefixed(t *testing.T) {
	// Simulate what LoadModelPrices returns when both "gpt-4" and
	// "openai/gpt-4" exist in the price file: the bare key must win.
	registry := models.NewModelPriceRegistry()
	registry.Update(map[string]*models.ModelPrice{
		"gpt-4":        {InputCostPerToken: 1.0e-06},
		"openai/gpt-4": {InputCostPerToken: 2.0e-06},
	})

	// Bare key must be returned, not the prefixed one
	gotModelID, gotPrice := lookupBillingModelPrice(registry, "", "gpt-4", "")
	assert.Equal(t, "gpt-4", gotModelID)
	assert.NotNil(t, gotPrice)
	assert.Equal(t, 1.0e-06, gotPrice.InputCostPerToken)
}
