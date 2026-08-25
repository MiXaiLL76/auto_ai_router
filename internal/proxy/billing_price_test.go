package proxy

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// billingTestInstant is the request-start timestamp for fixtures without a
// time-of-day schedule.
var billingTestInstant = time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)

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
			gotModelID, gotPrice := lookupBillingModelPrice(registry, billingTestInstant, tt.publicModelID, tt.modelID, tt.realModelID)
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
		billingTestInstant,
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
	gotModelID, gotPrice := lookupBillingModelPrice(nil, billingTestInstant, "openrouter/gpt-5-mini", "gpt-5-mini-or", "")
	assert.Equal(t, "gpt-5-mini-or", gotModelID)
	assert.Nil(t, gotPrice)
}

func TestLookupBillingModelPrice_DeduplicatesModelIDAndRealModelID(t *testing.T) {
	registry := models.NewModelPriceRegistry()
	registry.Update(map[string]*models.ModelPrice{
		"gpt-5-mini": {InputCostPerToken: 2.25e-07},
	})

	gotModelID, gotPrice := lookupBillingModelPrice(registry, billingTestInstant, "", "gpt-5-mini", "gpt-5-mini")
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
	_, plainPrice := lookupBillingModelPrice(registry, billingTestInstant, "", "gpt-5-mini", "")
	require.NotNil(t, plainPrice)
	assert.Equal(t, 5.0e-07, plainPrice.InputCostPerToken)

	// ...but the deliberately distinct alias price is untouched, since the
	// DB override didn't name it.
	matched, aliasPrice := lookupBillingModelPrice(registry, billingTestInstant, "openrouter/gpt-5-mini", "gpt-5-mini-or", "")
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
	gotModelID, gotPrice := lookupBillingModelPrice(registry, billingTestInstant, "", "gpt-4", "")
	assert.Equal(t, "gpt-4", gotModelID)
	assert.NotNil(t, gotPrice)
	assert.Equal(t, 1.0e-06, gotPrice.InputCostPerToken)
}

// scheduledPriceRegistry goes through the real loader: schedules only ever
// come from JSON.
func scheduledPriceRegistry(t *testing.T) *models.ModelPriceRegistry {
	t.Helper()
	filePath := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(filePath, []byte(`{
		"scheduled-model": {
			"input_cost_per_token": 2e-06,
			"output_cost_per_token": 6e-06,
			"pricing_schedule": [
				{"name": "peak", "start_utc": "00:00", "end_utc": "14:00", "input_cost_per_token": 2e-06},
				{"name": "off_peak", "start_utc": "14:00", "end_utc": "00:00", "input_cost_per_token": 1e-06}
			]
		}
	}`), 0o600))

	prices, err := models.LoadModelPrices(filePath)
	require.NoError(t, err)
	registry := models.NewModelPriceRegistry()
	registry.Update(prices)
	return registry
}

// The tariff follows the instant the request started, not the instant the
// billing code runs.
func TestResolveBillingPrice_TariffFollowsRequestStart(t *testing.T) {
	registry := scheduledPriceRegistry(t)
	proxy := &Proxy{priceRegistry: registry}

	tests := []struct {
		name       string
		startTime  time.Time
		wantWindow string
		wantInput  float64
	}{
		{
			name:       "request started one second before the boundary is Peak",
			startTime:  time.Date(2026, 8, 25, 13, 59, 59, 0, time.UTC),
			wantWindow: "peak",
			wantInput:  2e-06,
		},
		{
			name:       "request started on the boundary is Off-Peak",
			startTime:  time.Date(2026, 8, 25, 14, 0, 0, 0, time.UTC),
			wantWindow: "off_peak",
			wantInput:  1e-06,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logCtx := &RequestLogContext{StartTime: tt.startTime}
			_, price := proxy.resolveBillingPrice(logCtx, "scheduled-model", "scheduled-model", "")
			require.NotNil(t, price)
			assert.Equal(t, tt.wantWindow, price.PricingWindowName())
			assert.InDelta(t, tt.wantInput, price.InputCostPerToken, 1e-15)
		})
	}
}

// Budget reservation and spend logging must bill the same window even when the
// request spans the boundary.
func TestResolveBillingPrice_WindowIsPinnedForTheWholeRequest(t *testing.T) {
	registry := scheduledPriceRegistry(t)
	proxy := &Proxy{priceRegistry: registry}

	logCtx := &RequestLogContext{StartTime: time.Date(2026, 8, 25, 13, 59, 59, 0, time.UTC)}

	_, reserved := proxy.resolveBillingPrice(logCtx, "scheduled-model", "scheduled-model", "")
	require.NotNil(t, reserved)
	require.Equal(t, "peak", reserved.PricingWindowName())

	// Second resolution, as logSpendToLiteLLMDB does after the response.
	_, logged := proxy.resolveBillingPrice(logCtx, "scheduled-model", "scheduled-model", "")
	assert.Same(t, reserved, logged, "a request must not switch tariff mid-flight")
}

func TestBillingInstant(t *testing.T) {
	started := time.Date(2026, 8, 25, 13, 59, 59, 0, time.UTC)
	assert.Equal(t, started, billingInstant(&RequestLogContext{StartTime: started}))

	// Contexts without a start time fall back to "now".
	assert.WithinDuration(t, time.Now().UTC(), billingInstant(&RequestLogContext{}), time.Minute)
	assert.WithinDuration(t, time.Now().UTC(), billingInstant(nil), time.Minute)
}
