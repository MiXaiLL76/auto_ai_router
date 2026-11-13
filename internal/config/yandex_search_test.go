package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func validYandexSearchYAML() string {
	return `
enabled: true
api_key: yandex-secret
folder_id: b1gfolder0000000000
rpm: "60"
pricing:
  exchange_rate_rub_per_usd: "85"
  rate: "1.3"
  enabled_operations: ["web_sync"]
  pricing_timezone: "Europe/Moscow"
  night_start: "00:00"
  night_end: "08:00"
  cost_per_query:
    web_sync:
      day: "0.007463529412"
      night: "0.005597647059"
`
}

// fullYandexSearchCatalogYAML exercises the whole price catalog, matching
// the documented reference JSON: web_sync enabled, every other operation's
// price present but NOT enabled (future catalog, no adapter yet).
func fullYandexSearchCatalogYAML() string {
	return `
enabled: true
api_key: yandex-secret
folder_id: b1gfolder0000000000
rpm: "60"
pricing:
  exchange_rate_rub_per_usd: "85"
  rate: "1.3"
  enabled_operations: ["web_sync"]
  pricing_timezone: "Europe/Moscow"
  night_start: "00:00"
  night_end: "08:00"
  cost_per_query:
    web_sync:
      day: "0.007463529412"
      night: "0.005597647059"
    web_async:
      day: "0.000466470588"
      night: "0.000388623529"
    image_search: "0.013994117647"
    smart_snippets_ru: "0.022941176471"
    generative_search_sync: "0.077694117647"
    wordstat_get_top: "0.000305882353"
    wordstat_get_dynamics: "0.000305882353"
    wordstat_get_regions_distribution: "0.000764705882"
    wordstat_get_regions_tree: "0"
`
}

func TestYandexSearchConfigDefaultsAndEnvironment(t *testing.T) {
	t.Setenv("YANDEX_SEARCH_KEY", "yandex-secret")

	var cfg YandexSearchConfig
	err := yaml.Unmarshal([]byte(`
enabled: true
api_key: os.environ/YANDEX_SEARCH_KEY
folder_id: b1gfolder0000000000
rpm: "60"
pricing:
  exchange_rate_rub_per_usd: "85"
  rate: "1.3"
  enabled_operations: ["web_sync"]
  pricing_timezone: "Europe/Moscow"
  night_start: "00:00"
  night_end: "08:00"
  cost_per_query:
    web_sync:
      day: "0.007463529412"
      night: "0.005597647059"
`), &cfg)

	require.NoError(t, err)
	require.NoError(t, cfg.Validate())
	assert.Equal(t, "yandex-secret", cfg.APIKey)
	assert.Equal(t, "b1gfolder0000000000", cfg.FolderID)
	assert.Equal(t, defaultYandexSearchBaseURL, cfg.BaseURL)
	assert.Equal(t, 60, cfg.RPM)
	assert.Equal(t, defaultYandexSearchRPS, cfg.RPS)
	assert.Equal(t, defaultYandexSearchHourlyLimit, cfg.HourlyLimit)
	assert.Equal(t, defaultYandexSearchRedisKeyPrefix, cfg.RedisKeyPrefix)
	assert.Equal(t, defaultYandexSearchRequestTimeout, cfg.RequestTimeout)
	assert.Equal(t, defaultYandexSearchMaxBodyBytes, cfg.MaxBodyBytes)
	assert.Equal(t, 85.0, cfg.Pricing.ExchangeRateRUBPerUSD)
	assert.Equal(t, 1.3, cfg.Pricing.Markup)
	assert.Equal(t, []string{"web_sync"}, cfg.Pricing.EnabledOperations)
	require.NotNil(t, cfg.Pricing.CostPerQuery.WebSync)
	assert.InDelta(t, 0.007463529412, cfg.Pricing.CostPerQuery.WebSync.Day, 1e-12)
	assert.InDelta(t, 0.005597647059, cfg.Pricing.CostPerQuery.WebSync.Night, 1e-12)
	assert.True(t, cfg.Pricing.IsOperationEnabled(OperationWebSync))
	assert.False(t, cfg.Pricing.IsOperationEnabled(OperationImageSearch))
}

func TestYandexSearchConfigFullCatalogParsesAndValidates(t *testing.T) {
	var cfg YandexSearchConfig
	require.NoError(t, yaml.Unmarshal([]byte(fullYandexSearchCatalogYAML()), &cfg))
	require.NoError(t, cfg.Validate())

	c := cfg.Pricing.CostPerQuery
	require.NotNil(t, c.WebAsync)
	assert.InDelta(t, 0.000466470588, c.WebAsync.Day, 1e-12)
	assert.InDelta(t, 0.000388623529, c.WebAsync.Night, 1e-12)
	require.NotNil(t, c.ImageSearch)
	assert.InDelta(t, 0.013994117647, *c.ImageSearch, 1e-12)
	require.NotNil(t, c.SmartSnippetsRU)
	assert.InDelta(t, 0.022941176471, *c.SmartSnippetsRU, 1e-12)
	require.NotNil(t, c.GenerativeSearchSync)
	assert.InDelta(t, 0.077694117647, *c.GenerativeSearchSync, 1e-12)
	require.NotNil(t, c.WordstatGetTop)
	assert.InDelta(t, 0.000305882353, *c.WordstatGetTop, 1e-12)
	require.NotNil(t, c.WordstatGetDynamics)
	require.NotNil(t, c.WordstatGetRegionsDistribution)
	require.NotNil(t, c.WordstatGetRegionsTree)
	assert.Equal(t, 0.0, *c.WordstatGetRegionsTree, "wordstat_get_regions_tree is genuinely free")

	// None of these other operations are enabled, regardless of having a price.
	assert.False(t, cfg.Pricing.IsOperationEnabled(OperationWebAsync))
	assert.False(t, cfg.Pricing.IsOperationEnabled(OperationImageSearch))
	assert.False(t, cfg.Pricing.IsOperationEnabled(OperationWordstatGetRegionsTree))
	assert.True(t, cfg.Pricing.IsOperationEnabled(OperationWebSync))
}

func TestYandexSearchConfigDisabledSkipsValidation(t *testing.T) {
	var cfg YandexSearchConfig
	require.NoError(t, yaml.Unmarshal([]byte(`enabled: false`), &cfg))
	require.NoError(t, cfg.Validate())
}

func validYandexSearchConfig(t *testing.T) YandexSearchConfig {
	t.Helper()
	var cfg YandexSearchConfig
	require.NoError(t, yaml.Unmarshal([]byte(validYandexSearchYAML()), &cfg))
	return cfg
}

func TestYandexSearchConfigRejectsMissingDependencies(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	cfg.APIKey = ""
	require.ErrorContains(t, cfg.Validate(), "api_key")

	cfg = validYandexSearchConfig(t)
	cfg.FolderID = ""
	require.ErrorContains(t, cfg.Validate(), "folder_id")

	cfg = validYandexSearchConfig(t)
	cfg.BaseURL = "http://insecure.example.com"
	require.ErrorContains(t, cfg.Validate(), "https")

	cfg = validYandexSearchConfig(t)
	cfg.RPM = 0
	require.ErrorContains(t, cfg.Validate(), "rpm")

	cfg = validYandexSearchConfig(t)
	cfg.RPS = 0
	require.ErrorContains(t, cfg.Validate(), "rps")

	cfg = validYandexSearchConfig(t)
	cfg.HourlyLimit = 0
	require.ErrorContains(t, cfg.Validate(), "hourly_limit")

	cfg = validYandexSearchConfig(t)
	cfg.RequestTimeout = 0
	require.ErrorContains(t, cfg.Validate(), "request_timeout")

	cfg = validYandexSearchConfig(t)
	cfg.MaxBodyBytes = 0
	require.ErrorContains(t, cfg.Validate(), "max_body_bytes")

	cfg = validYandexSearchConfig(t)
	cfg.ProxyURL = "not a url"
	require.ErrorContains(t, cfg.Validate(), "proxy_url")
}

func TestYandexSearchConfigRejectsBadPricing(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	cfg.Pricing.ExchangeRateRUBPerUSD = 0
	require.ErrorContains(t, cfg.Validate(), "exchange_rate_rub_per_usd")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.Markup = 0
	require.ErrorContains(t, cfg.Validate(), "rate")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.EnabledOperations = nil
	require.ErrorContains(t, cfg.Validate(), "enabled_operations")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.EnabledOperations = []string{"not_a_real_operation"}
	require.ErrorContains(t, cfg.Validate(), "unknown operation")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.WebSync = nil
	require.ErrorContains(t, cfg.Validate(), "web_sync")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.WebSync.Day = 0
	require.ErrorContains(t, cfg.Validate(), "web_sync")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.WebSync.Night = 0
	require.ErrorContains(t, cfg.Validate(), "web_sync")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.Timezone = "Not/A_Zone"
	require.ErrorContains(t, cfg.Validate(), "pricing_timezone")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.NightStart = "25:00"
	require.ErrorContains(t, cfg.Validate(), "night_start")
}

func TestYandexSearchConfigAllowsWebSyncUnpricedWhenNotEnabled(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	cfg.Pricing.EnabledOperations = []string{OperationWordstatGetTop}
	cfg.Pricing.CostPerQuery.WebSync = nil
	cfg.Pricing.CostPerQuery.WordstatGetTop = float64Ptr(0.001)
	require.NoError(t, cfg.Validate(), "web_sync's price is only required when web_sync is enabled")
}

func TestYandexSearchConfigRejectsNegativeCatalogEntries(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.ImageSearch = float64Ptr(-1)
	require.ErrorContains(t, cfg.Validate(), "image_search")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.WordstatGetRegionsTree = float64Ptr(-1)
	require.ErrorContains(t, cfg.Validate(), "wordstat_get_regions_tree")

	cfg = validYandexSearchConfig(t)
	cfg.Pricing.CostPerQuery.WordstatGetRegionsTree = float64Ptr(0)
	require.NoError(t, cfg.Validate(), "zero is a legitimate (free) price for this one operation")
}

func TestYandexSearchConfigRejectsOneSidedDayNightRate(t *testing.T) {
	var cfg YandexSearchConfig
	err := yaml.Unmarshal([]byte(`
enabled: true
api_key: key
folder_id: folder
rpm: "60"
pricing:
  exchange_rate_rub_per_usd: "85"
  rate: "1.3"
  enabled_operations: ["web_sync"]
  pricing_timezone: "Europe/Moscow"
  cost_per_query:
    web_sync:
      day: "0.005"
`), &cfg)
	require.ErrorContains(t, err, "requires both day and night")
}

func float64Ptr(v float64) *float64 { return &v }

func TestYandexSearchPricingCostForTime(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	require.NoError(t, cfg.Validate())

	moscow, err := time.LoadLocation("Europe/Moscow")
	require.NoError(t, err)

	cases := []struct {
		name   string
		hour   int
		minute int
		want   string
	}{
		{"start of night window", 0, 0, "night"},
		{"just before night ends", 7, 59, "night"},
		{"night ends exactly at boundary", 8, 0, "day"},
		{"mid-day", 14, 30, "day"},
		{"just before midnight", 23, 59, "day"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts := time.Date(2026, 10, 23, tc.hour, tc.minute, 0, 0, moscow)
			cost, period := cfg.Pricing.CostForTime(ts)
			assert.Equal(t, tc.want, period)
			if tc.want == "day" {
				assert.Equal(t, cfg.Pricing.CostPerQuery.WebSync.Day, cost)
			} else {
				assert.Equal(t, cfg.Pricing.CostPerQuery.WebSync.Night, cost)
			}
		})
	}
}

func TestYandexSearchPricingCostForTimeConvertsTimezone(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	require.NoError(t, cfg.Validate())

	// 01:30 UTC == 04:30 Moscow (UTC+3) -> night.
	utc := time.Date(2026, 10, 23, 1, 30, 0, 0, time.UTC)
	_, period := cfg.Pricing.CostForTime(utc)
	assert.Equal(t, "night", period)
}

func TestYandexSearchPricingCostForTimeUnresolvedFailsSafe(t *testing.T) {
	cfg := validYandexSearchConfig(t)
	// Deliberately skip Validate(): location stays nil.
	cost, period := cfg.Pricing.CostForTime(time.Now())
	assert.Equal(t, cfg.Pricing.CostPerQuery.WebSync.Day, cost, "unresolved pricing must fail safe by overcharging, not by charging nothing")
	assert.Equal(t, "day", period)
}

func TestYandexSearchPricingCostForTimeNoWebSyncPriceReturnsZero(t *testing.T) {
	var p YandexSearchPricing
	cost, period := p.CostForTime(time.Now())
	assert.Equal(t, 0.0, cost)
	assert.Equal(t, "day", period)
}

func TestParseClockMinutes(t *testing.T) {
	m, err := parseClockMinutes("08:00")
	require.NoError(t, err)
	assert.Equal(t, 480, m)

	_, err = parseClockMinutes("not-a-time")
	require.Error(t, err)
}

func TestInNightWindowWraparound(t *testing.T) {
	// A window like 22:00-06:00 wraps past midnight.
	assert.True(t, inNightWindow(23*60, 22*60, 6*60))
	assert.True(t, inNightWindow(1*60, 22*60, 6*60))
	assert.False(t, inNightWindow(12*60, 22*60, 6*60))
	assert.False(t, inNightWindow(0, 0, 0)) // zero window never matches
}
