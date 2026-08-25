package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Rates of a real catalogue entry, kept verbatim.
const (
	peakInputCost      = 0.0000016536
	peakOutputCost     = 0.0000049608
	peakCacheReadCost  = 0.0000001651
	offPeakInputCost   = 0.0000008268
	offPeakOutputCost  = 0.0000024804
	offPeakCacheCost   = 0.0000000832
	priceCompareDelta  = 1e-15
	scheduledModelName = "model-with-schedule"
)

// scheduledPricesJSON is that entry as it appears in a price file.
const scheduledPricesJSON = `{
		"model-with-schedule": {
			"input_cost_per_token": 0.0000016536,
			"output_cost_per_token": 0.0000049608,
			"cache_read_input_token_cost": 0.0000001651,
			"pricing_schedule": [
				{
					"name": "peak",
					"start_utc": "00:00",
					"end_utc": "14:00",
					"input_cost_per_token": 0.0000016536,
					"output_cost_per_token": 0.0000049608,
					"cache_read_input_token_cost": 0.0000001651
				},
				{
					"name": "off_peak",
					"start_utc": "14:00",
					"end_utc": "00:00",
					"input_cost_per_token": 0.0000008268,
					"output_cost_per_token": 0.0000024804,
					"cache_read_input_token_cost": 0.0000000832
				}
			]
		}
	}`

// loadPricesFromJSON goes through the real loader, so every test also covers
// the schedule being compiled at load time.
func loadPricesFromJSON(t *testing.T, pricesJSON string) map[string]*ModelPrice {
	t.Helper()
	filePath := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(filePath, []byte(pricesJSON), 0o600))
	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	return prices
}

func TestEffectiveAt_PeakOffPeakBoundaries(t *testing.T) {
	price := loadPricesFromJSON(t, scheduledPricesJSON)[scheduledModelName]
	require.NotNil(t, price)

	tests := []struct {
		name          string
		at            time.Time
		wantWindow    string
		wantInput     float64
		wantOutput    float64
		wantCacheRead float64
	}{
		{
			name:          "13:59:59 UTC is still Peak",
			at:            time.Date(2026, 8, 24, 13, 59, 59, 0, time.UTC),
			wantWindow:    "peak",
			wantInput:     peakInputCost,
			wantOutput:    peakOutputCost,
			wantCacheRead: peakCacheReadCost,
		},
		{
			name:          "14:00:00 UTC switches to Off-Peak",
			at:            time.Date(2026, 8, 24, 14, 0, 0, 0, time.UTC),
			wantWindow:    "off_peak",
			wantInput:     offPeakInputCost,
			wantOutput:    offPeakOutputCost,
			wantCacheRead: offPeakCacheCost,
		},
		{
			name:          "23:59:59 UTC is Off-Peak",
			at:            time.Date(2026, 8, 24, 23, 59, 59, 0, time.UTC),
			wantWindow:    "off_peak",
			wantInput:     offPeakInputCost,
			wantOutput:    offPeakOutputCost,
			wantCacheRead: offPeakCacheCost,
		},
		{
			name:          "00:00:00 UTC switches back to Peak",
			at:            time.Date(2026, 8, 25, 0, 0, 0, 0, time.UTC),
			wantWindow:    "peak",
			wantInput:     peakInputCost,
			wantOutput:    peakOutputCost,
			wantCacheRead: peakCacheReadCost,
		},
		{
			name:          "sub-second past the boundary is already Off-Peak",
			at:            time.Date(2026, 8, 24, 14, 0, 0, 500_000_000, time.UTC),
			wantWindow:    "off_peak",
			wantInput:     offPeakInputCost,
			wantOutput:    offPeakOutputCost,
			wantCacheRead: offPeakCacheCost,
		},
		{
			name:          "non-UTC timestamps are converted before matching",
			at:            time.Date(2026, 8, 24, 16, 59, 59, 0, time.FixedZone("MSK", 3*60*60)),
			wantWindow:    "peak",
			wantInput:     peakInputCost,
			wantOutput:    peakOutputCost,
			wantCacheRead: peakCacheReadCost,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			effective := price.EffectiveAt(tt.at)
			require.NotNil(t, effective)
			assert.Equal(t, tt.wantWindow, effective.PricingWindowName())
			assert.InDelta(t, tt.wantInput, effective.InputCostPerToken, priceCompareDelta)
			assert.InDelta(t, tt.wantOutput, effective.OutputCostPerToken, priceCompareDelta)
			assert.InDelta(t, tt.wantCacheRead, effective.CacheReadInputTokenCost, priceCompareDelta)
		})
	}
}

func TestEffectiveAt_CostsFollowTheActiveWindow(t *testing.T) {
	price := loadPricesFromJSON(t, scheduledPricesJSON)[scheduledModelName]
	require.NotNil(t, price)

	newUsage := func() *converter.TokenUsage {
		return &converter.TokenUsage{
			PromptTokens:      1000,
			CompletionTokens:  500,
			CachedInputTokens: 200,
		}
	}

	peakCosts := CalculateTokenCosts(newUsage(), price.EffectiveAt(
		time.Date(2026, 8, 24, 13, 59, 59, 0, time.UTC)))
	require.NotNil(t, peakCosts)
	assert.InDelta(t, 800*peakInputCost, peakCosts.InputCost, priceCompareDelta)
	assert.InDelta(t, 500*peakOutputCost, peakCosts.OutputCost, priceCompareDelta)
	assert.InDelta(t, 200*peakCacheReadCost, peakCosts.CachedInputCost, priceCompareDelta)
	assert.InDelta(t,
		800*peakInputCost+500*peakOutputCost+200*peakCacheReadCost,
		peakCosts.TotalCost, priceCompareDelta)

	offPeakCosts := CalculateTokenCosts(newUsage(), price.EffectiveAt(
		time.Date(2026, 8, 24, 14, 0, 0, 0, time.UTC)))
	require.NotNil(t, offPeakCosts)
	assert.InDelta(t, 800*offPeakInputCost, offPeakCosts.InputCost, priceCompareDelta)
	assert.InDelta(t, 500*offPeakOutputCost, offPeakCosts.OutputCost, priceCompareDelta)
	assert.InDelta(t, 200*offPeakCacheCost, offPeakCosts.CachedInputCost, priceCompareDelta)
	assert.Less(t, offPeakCosts.TotalCost, peakCosts.TotalCost,
		"Off-Peak must be cheaper than Peak for identical usage")
}

func TestEffectiveAt_ModelWithoutScheduleIsUnchanged(t *testing.T) {
	price := loadPricesFromJSON(t, `{
		"gpt-5.5": {
			"input_cost_per_token": 0.0000045,
			"output_cost_per_token": 0.000027
		}
	}`)["gpt-5.5"]
	require.NotNil(t, price)

	assert.Same(t, price, price.EffectiveAt(time.Date(2026, 8, 24, 3, 0, 0, 0, time.UTC)))
	assert.Same(t, price, price.EffectiveAt(time.Date(2026, 8, 24, 20, 0, 0, 0, time.UTC)))
	assert.Empty(t, price.PricingWindowName())
}

func TestEffectiveAt_NilPrice(t *testing.T) {
	var price *ModelPrice
	assert.Nil(t, price.EffectiveAt(time.Now()))
	assert.Empty(t, price.PricingWindowName())
}

func TestEffectiveAt_WindowInheritsUndeclaredBaseFields(t *testing.T) {
	price := loadPricesFromJSON(t, `{
		"scheduled-with-extras": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"cache_read_input_token_cost": 0.0000002,
			"input_cost_per_token_above_128k_tokens": 0.000004,
			"litellm_provider": "openai",
			"web_search_billing_unit": "per_query",
			"search_context_cost_per_query": {
				"search_context_size_low": 0.013,
				"search_context_size_medium": 0.013,
				"search_context_size_high": 0.013
			},
			"pricing_schedule": [
				{
					"name": "night",
					"start_utc": "22:00",
					"end_utc": "06:00",
					"input_cost_per_token": 0.000001,
					"output_cost_per_token": 0.000003
				}
			]
		}
	}`)["scheduled-with-extras"]
	require.NotNil(t, price)

	night := price.EffectiveAt(time.Date(2026, 8, 24, 23, 30, 0, 0, time.UTC))
	require.NotNil(t, night)
	assert.Equal(t, "night", night.PricingWindowName())
	assert.InDelta(t, 0.000001, night.InputCostPerToken, priceCompareDelta)
	assert.InDelta(t, 0.000003, night.OutputCostPerToken, priceCompareDelta)
	// Inherited from the model.
	assert.InDelta(t, 0.0000002, night.CacheReadInputTokenCost, priceCompareDelta)
	assert.InDelta(t, 0.000004, night.InputCostPerTokenAbove128k, priceCompareDelta)
	assert.Equal(t, "openai", night.LiteLLMProvider)
	assert.Equal(t, "per_query", night.WebSearchBillingUnit)
	assert.Equal(t, 0.013, night.SearchContextCostPerQuery["search_context_size_high"])

	day := price.EffectiveAt(time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC))
	require.NotNil(t, day)
	assert.Empty(t, day.PricingWindowName())
	assert.InDelta(t, 0.000002, day.InputCostPerToken, priceCompareDelta)
}

// A window is a copy of the model's fields, so it must be taken after the
// loader infers litellm_provider — billing reads the provider off that copy.
func TestEffectiveAt_WindowInheritsProviderInferredByLoader(t *testing.T) {
	price := loadPricesFromJSON(t, `{
		"vertex_ai/gemini-scheduled": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"input_cost_per_token_above_200k_tokens": 0.000004,
			"pricing_schedule": [
				{
					"name": "night",
					"start_utc": "22:00",
					"end_utc": "06:00",
					"input_cost_per_token": 0.000001
				}
			]
		}
	}`)["gemini-scheduled"]
	require.NotNil(t, price)
	require.Equal(t, "vertex_ai", price.LiteLLMProvider)

	night := price.EffectiveAt(time.Date(2026, 8, 24, 23, 0, 0, 0, time.UTC))
	require.NotNil(t, night)
	assert.Equal(t, "vertex_ai", night.LiteLLMProvider,
		"window must inherit the provider the loader inferred, not an empty one")
	assert.Equal(t, "per_prompt", night.webSearchBillingUnit(),
		"provider-dependent Web Search billing must survive into the window")

	// With the provider carried over, the 200k tier applies to the whole request.
	costs := CalculateTokenCosts(&converter.TokenUsage{PromptTokens: 300_000}, night)
	require.NotNil(t, costs)
	assert.InDelta(t, 300_000*0.000004, costs.InputCost, priceCompareDelta)
}

func TestEffectiveAt_WindowMapOverrideDoesNotLeakIntoBase(t *testing.T) {
	price := loadPricesFromJSON(t, `{
		"searchy": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"search_context_cost_per_query": {
				"search_context_size_medium": 0.013
			},
			"pricing_schedule": [
				{
					"name": "night",
					"start_utc": "22:00",
					"end_utc": "06:00",
					"search_context_cost_per_query": {
						"search_context_size_medium": 0.005
					}
				}
			]
		}
	}`)["searchy"]
	require.NotNil(t, price)

	night := price.EffectiveAt(time.Date(2026, 8, 24, 23, 0, 0, 0, time.UTC))
	require.NotNil(t, night)
	assert.Equal(t, 0.005, night.SearchContextCostPerQuery["search_context_size_medium"])
	assert.Equal(t, 0.013, price.SearchContextCostPerQuery["search_context_size_medium"],
		"a window's map must not be decoded into the model's own map")
}

func TestEffectiveAt_WholeDayAndEndOfDayWindows(t *testing.T) {
	prices := loadPricesFromJSON(t, `{
		"end-of-day": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"pricing_schedule": [
				{"name": "early", "start_utc": "00:00", "end_utc": "14:00", "input_cost_per_token": 0.000001},
				{"name": "late", "start_utc": "14:00", "end_utc": "24:00", "input_cost_per_token": 0.000003}
			]
		},
		"always-on": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"pricing_schedule": [
				{"name": "flat", "start_utc": "00:00", "end_utc": "00:00", "input_cost_per_token": 0.000009}
			]
		},
		"partial-day": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"pricing_schedule": [
				{"name": "morning", "start_utc": "08:00", "end_utc": "09:00", "input_cost_per_token": 0.000001}
			]
		}
	}`)

	endOfDay := prices["end-of-day"]
	require.NotNil(t, endOfDay)
	assert.Equal(t, "late", endOfDay.EffectiveAt(
		time.Date(2026, 8, 24, 23, 59, 59, 0, time.UTC)).PricingWindowName())
	assert.Equal(t, "early", endOfDay.EffectiveAt(
		time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)).PricingWindowName())

	alwaysOn := prices["always-on"]
	require.NotNil(t, alwaysOn)
	for _, hour := range []int{0, 7, 13, 14, 23} {
		effective := alwaysOn.EffectiveAt(time.Date(2026, 8, 24, hour, 0, 0, 0, time.UTC))
		assert.Equal(t, "flat", effective.PricingWindowName())
		assert.InDelta(t, 0.000009, effective.InputCostPerToken, priceCompareDelta)
	}

	partial := prices["partial-day"]
	require.NotNil(t, partial)
	assert.Equal(t, "morning", partial.EffectiveAt(
		time.Date(2026, 8, 24, 8, 30, 0, 0, time.UTC)).PricingWindowName())
	uncovered := partial.EffectiveAt(time.Date(2026, 8, 24, 10, 0, 0, 0, time.UTC))
	assert.Empty(t, uncovered.PricingWindowName())
	assert.InDelta(t, 0.000002, uncovered.InputCostPerToken, priceCompareDelta)
}

func TestEffectiveAt_OverlappingWindowsResolveToTheFirstMatch(t *testing.T) {
	price := loadPricesFromJSON(t, `{
		"overlapping": {
			"input_cost_per_token": 0.000002,
			"output_cost_per_token": 0.000006,
			"pricing_schedule": [
				{"name": "first", "start_utc": "10:00", "end_utc": "14:00", "input_cost_per_token": 0.000001},
				{"name": "second", "start_utc": "12:00", "end_utc": "16:00", "input_cost_per_token": 0.000003}
			]
		}
	}`)["overlapping"]
	require.NotNil(t, price)

	assert.Equal(t, "first", price.EffectiveAt(
		time.Date(2026, 8, 24, 13, 0, 0, 0, time.UTC)).PricingWindowName())
	assert.Equal(t, "second", price.EffectiveAt(
		time.Date(2026, 8, 24, 15, 0, 0, 0, time.UTC)).PricingWindowName())
}

func TestLoadModelPrices_DropsOnlyTheModelWithABrokenSchedule(t *testing.T) {
	prices := loadPricesFromJSON(t, `{
		"healthy-model": {
			"input_cost_per_token": 0.000001,
			"output_cost_per_token": 0.000002
		},
		"bad-time-model": {
			"input_cost_per_token": 0.000001,
			"output_cost_per_token": 0.000002,
			"pricing_schedule": [
				{"name": "peak", "start_utc": "0000", "end_utc": "14:00"}
			]
		},
		"null-window-model": {
			"input_cost_per_token": 0.000001,
			"output_cost_per_token": 0.000002,
			"pricing_schedule": [null]
		}
	}`)

	assert.NotNil(t, prices["healthy-model"], "a valid entry must survive a broken sibling")
	assert.Nil(t, prices["bad-time-model"],
		"an unparseable window must drop the model instead of billing it at base rates")
	assert.Nil(t, prices["null-window-model"])
}

func TestCompileSchedule_RejectsInvalidWindows(t *testing.T) {
	tests := []struct {
		name   string
		window *PricingWindow
	}{
		{name: "empty start", window: &PricingWindow{Name: "x", StartUTC: "", EndUTC: "14:00"}},
		{name: "empty end", window: &PricingWindow{Name: "x", StartUTC: "00:00", EndUTC: ""}},
		{name: "hour out of range", window: &PricingWindow{Name: "x", StartUTC: "25:00", EndUTC: "14:00"}},
		{name: "not a time", window: &PricingWindow{Name: "x", StartUTC: "morning", EndUTC: "14:00"}},
		{name: "too many fields", window: &PricingWindow{Name: "x", StartUTC: "00:00:00:00", EndUTC: "14:00"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			price := &ModelPrice{
				InputCostPerToken:  0.000001,
				OutputCostPerToken: 0.000002,
				PricingSchedule:    []*PricingWindow{tt.window},
			}
			require.Error(t, price.compileSchedule())
		})
	}
}

func TestParseWindowTime(t *testing.T) {
	valid := map[string]int{
		"00:00":    0,
		"00:00:00": 0,
		"08:30":    8*3600 + 30*60,
		"14:00":    14 * 3600,
		"23:59:59": 23*3600 + 59*60 + 59,
		"24:00":    0,
		"24:00:00": 0,
		" 14:00 ":  14 * 3600,
	}
	for value, want := range valid {
		got, err := parseWindowTime(value)
		require.NoError(t, err, value)
		assert.Equal(t, want, got, value)
	}

	for _, value := range []string{"", "14", "24:01", "25:00", "14:60", "1a:00", "-1:00"} {
		_, err := parseWindowTime(value)
		require.Error(t, err, value)
	}
}

func TestPricingWindow_MarshalKeepsWindowRates(t *testing.T) {
	price := loadPricesFromJSON(t, scheduledPricesJSON)[scheduledModelName]
	require.NotNil(t, price)

	encoded, err := json.Marshal(price)
	require.NoError(t, err)

	var decoded ModelPrice
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	require.NoError(t, decoded.compileSchedule())

	offPeak := decoded.EffectiveAt(time.Date(2026, 8, 24, 18, 0, 0, 0, time.UTC))
	require.NotNil(t, offPeak)
	assert.Equal(t, "off_peak", offPeak.PricingWindowName())
	assert.InDelta(t, offPeakInputCost, offPeak.InputCostPerToken, priceCompareDelta)
}
