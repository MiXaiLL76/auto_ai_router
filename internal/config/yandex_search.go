package config

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	defaultYandexSearchBaseURL        = "https://searchapi.api.cloud.yandex.net"
	defaultYandexSearchRequestTimeout = 10 * time.Second
	defaultYandexSearchMaxBodyBytes   = int64(16 * 1024)
	defaultYandexSearchRedisKeyPrefix = "yssq:"
	defaultYandexSearchRPS            = 1
	defaultYandexSearchHourlyLimit    = 10000
)

// Yandex Search API operation names, as used in yandex_search.pricing's
// enabled_operations and cost_per_query. These are NEVER client-supplied:
// the adapter that handles a request determines its own operation type
// from which endpoint and verified parameters it is — there is currently
// exactly one adapter, OperationWebSync, for POST /v1/search/yandex's
// synchronous WebSearch.Search. The catalog documents prices for
// operations this router does not implement yet (async search, image
// search, smart snippets, generative search, Wordstat) so pricing can be
// reviewed and landed ahead of the code that would use it — listing a
// price here never enables an operation; only IsOperationEnabled does,
// and only an adapter that actually checks it can act on that.
const (
	OperationWebSync                        = "web_sync"
	OperationWebAsync                       = "web_async"
	OperationImageSearch                    = "image_search"
	OperationSmartSnippetsRU                = "smart_snippets_ru"
	OperationGenerativeSearchSync           = "generative_search_sync"
	OperationWordstatGetTop                 = "wordstat_get_top"
	OperationWordstatGetDynamics            = "wordstat_get_dynamics"
	OperationWordstatGetRegionsDistribution = "wordstat_get_regions_distribution"
	OperationWordstatGetRegionsTree         = "wordstat_get_regions_tree"
)

var knownYandexSearchOperations = map[string]bool{
	OperationWebSync:                        true,
	OperationWebAsync:                       true,
	OperationImageSearch:                    true,
	OperationSmartSnippetsRU:                true,
	OperationGenerativeSearchSync:           true,
	OperationWordstatGetTop:                 true,
	OperationWordstatGetDynamics:            true,
	OperationWordstatGetRegionsDistribution: true,
	OperationWordstatGetRegionsTree:         true,
}

// YandexSearchConfig configures the /v1/search/yandex passthrough: the
// router forwards the client's request body to Yandex Cloud's Search API,
// injecting its own API key and Folder ID, so Yandex only ever sees the
// router's egress IP. Disabled by default; even when enabled, individual
// client keys/teams/organizations must additionally be opted in via
// metadata.air_scopes: ["yandex_search"] (see internal/scope) — enabling
// this section alone does not expose the endpoint to every client.
//
// RPS/HourlyLimit are Yandex's quota for the whole folder_id, shared across
// every router replica — verify the catalog's actual numbers before launch
// rather than trusting these defaults.
type YandexSearchConfig struct {
	Enabled        bool                `yaml:"enabled"`
	APIKey         string              `yaml:"api_key"`
	FolderID       string              `yaml:"folder_id"`
	BaseURL        string              `yaml:"base_url"`
	ProxyURL       string              `yaml:"proxy_url,omitempty"`
	RPM            int                 `yaml:"rpm"`
	RPS            int                 `yaml:"rps"`
	HourlyLimit    int                 `yaml:"hourly_limit"`
	RedisKeyPrefix string              `yaml:"redis_key_prefix,omitempty"`
	RequestTimeout time.Duration       `yaml:"request_timeout"`
	MaxBodyBytes   int64               `yaml:"max_body_bytes"`
	Pricing        YandexSearchPricing `yaml:"pricing"`
}

// DayNightRate is a USD price that differs by time of day (see
// YandexSearchPricing's night window).
type DayNightRate struct {
	Day   float64
	Night float64
}

// YandexSearchCostCatalog is Yandex's full operation price list — already
// marked up (YandexSearchPricing.Markup), already converted to USD. Only
// entries for operations actually in use (see enabled_operations) need to
// be filled in; the rest document future pricing ahead of the adapters
// that would use them. Every field is optional except WebSync, which is
// required when "web_sync" is in enabled_operations (it's the only
// operation this router implements).
type YandexSearchCostCatalog struct {
	WebSync                        *DayNightRate
	WebAsync                       *DayNightRate
	ImageSearch                    *float64
	SmartSnippetsRU                *float64
	GenerativeSearchSync           *float64
	WordstatGetTop                 *float64
	WordstatGetDynamics            *float64
	WordstatGetRegionsDistribution *float64
	// WordstatGetRegionsTree is Yandex's only free operation — 0 is a
	// valid, intentional price here, unlike every other field.
	WordstatGetRegionsTree *float64
}

// YandexSearchPricing is Yandex's day/night-aware price catalog: cost
// figures are already marked up (Markup) and converted to USD — do not
// multiply by Markup again. The day/night rate is selected by the
// wall-clock time (in Timezone) at which the router sends the request
// upstream.
type YandexSearchPricing struct {
	ExchangeRateRUBPerUSD float64 `yaml:"exchange_rate_rub_per_usd"`
	Markup                float64 `yaml:"rate"`
	// EnabledOperations is the actual gate on which operations this
	// deployment may serve — a price existing in CostPerQuery does NOT by
	// itself enable the operation it prices (e.g. shipping the
	// smart_snippets_ru rate ahead of that adapter existing must not turn
	// smart snippets on).
	EnabledOperations []string `yaml:"enabled_operations"`
	Timezone          string   `yaml:"pricing_timezone"`
	NightStart        string   `yaml:"night_start"`
	NightEnd          string   `yaml:"night_end"`
	CostPerQuery      YandexSearchCostCatalog

	location      *time.Location
	nightStartMin int
	nightEndMin   int
	enabledOps    map[string]bool
}

func (c *YandexSearchConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawDayNightRate struct {
		Day   string `yaml:"day"`
		Night string `yaml:"night"`
	}
	type rawCostPerQuery struct {
		WebSync                        *rawDayNightRate `yaml:"web_sync"`
		WebAsync                       *rawDayNightRate `yaml:"web_async"`
		ImageSearch                    string           `yaml:"image_search"`
		SmartSnippetsRU                string           `yaml:"smart_snippets_ru"`
		GenerativeSearchSync           string           `yaml:"generative_search_sync"`
		WordstatGetTop                 string           `yaml:"wordstat_get_top"`
		WordstatGetDynamics            string           `yaml:"wordstat_get_dynamics"`
		WordstatGetRegionsDistribution string           `yaml:"wordstat_get_regions_distribution"`
		WordstatGetRegionsTree         string           `yaml:"wordstat_get_regions_tree"`
	}
	type rawPricing struct {
		ExchangeRateRUBPerUSD string          `yaml:"exchange_rate_rub_per_usd"`
		Rate                  string          `yaml:"rate"`
		EnabledOperations     []string        `yaml:"enabled_operations"`
		PricingTimezone       string          `yaml:"pricing_timezone"`
		NightStart            string          `yaml:"night_start"`
		NightEnd              string          `yaml:"night_end"`
		CostPerQuery          rawCostPerQuery `yaml:"cost_per_query"`
	}
	type rawYandexSearchConfig struct {
		Enabled        string     `yaml:"enabled"`
		APIKey         string     `yaml:"api_key"`
		FolderID       string     `yaml:"folder_id"`
		BaseURL        string     `yaml:"base_url"`
		ProxyURL       string     `yaml:"proxy_url,omitempty"`
		RPM            string     `yaml:"rpm"`
		RPS            string     `yaml:"rps"`
		HourlyLimit    string     `yaml:"hourly_limit"`
		RedisKeyPrefix string     `yaml:"redis_key_prefix,omitempty"`
		RequestTimeout string     `yaml:"request_timeout"`
		MaxBodyBytes   string     `yaml:"max_body_bytes"`
		Pricing        rawPricing `yaml:"pricing"`
	}

	var raw rawYandexSearchConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}

	var err error
	if c.Enabled, err = parseField(raw.Enabled, false, strconv.ParseBool, "yandex_search.enabled"); err != nil {
		return err
	}
	if c.RPM, err = parseField(raw.RPM, 0, strconv.Atoi, "yandex_search.rpm"); err != nil {
		return err
	}
	if c.RPS, err = parseField(raw.RPS, defaultYandexSearchRPS, strconv.Atoi, "yandex_search.rps"); err != nil {
		return err
	}
	if c.HourlyLimit, err = parseField(raw.HourlyLimit, defaultYandexSearchHourlyLimit, strconv.Atoi, "yandex_search.hourly_limit"); err != nil {
		return err
	}
	if c.RequestTimeout, err = parseField(raw.RequestTimeout, defaultYandexSearchRequestTimeout, time.ParseDuration, "yandex_search.request_timeout"); err != nil {
		return err
	}
	if c.MaxBodyBytes, err = parseField(raw.MaxBodyBytes, defaultYandexSearchMaxBodyBytes, func(v string) (int64, error) {
		return strconv.ParseInt(v, 10, 64)
	}, "yandex_search.max_body_bytes"); err != nil {
		return err
	}

	c.APIKey = resolveEnvString(raw.APIKey)
	c.FolderID = resolveEnvString(raw.FolderID)
	c.BaseURL = valueOrDefault(resolveEnvString(raw.BaseURL), defaultYandexSearchBaseURL)
	c.ProxyURL = resolveEnvString(raw.ProxyURL)
	c.RedisKeyPrefix = valueOrDefault(resolveEnvString(raw.RedisKeyPrefix), defaultYandexSearchRedisKeyPrefix)

	if c.Pricing.ExchangeRateRUBPerUSD, err = parseField(raw.Pricing.ExchangeRateRUBPerUSD, 0, func(v string) (float64, error) {
		return strconv.ParseFloat(v, 64)
	}, "yandex_search.pricing.exchange_rate_rub_per_usd"); err != nil {
		return err
	}
	if c.Pricing.Markup, err = parseField(raw.Pricing.Rate, 0, func(v string) (float64, error) {
		return strconv.ParseFloat(v, 64)
	}, "yandex_search.pricing.rate"); err != nil {
		return err
	}
	c.Pricing.EnabledOperations = make([]string, 0, len(raw.Pricing.EnabledOperations))
	for _, op := range raw.Pricing.EnabledOperations {
		if resolved := strings.TrimSpace(resolveEnvString(op)); resolved != "" {
			c.Pricing.EnabledOperations = append(c.Pricing.EnabledOperations, resolved)
		}
	}
	c.Pricing.Timezone = valueOrDefault(resolveEnvString(raw.Pricing.PricingTimezone), "UTC")
	c.Pricing.NightStart = valueOrDefault(resolveEnvString(raw.Pricing.NightStart), "00:00")
	c.Pricing.NightEnd = valueOrDefault(resolveEnvString(raw.Pricing.NightEnd), "00:00")

	webSyncDay, webSyncNight := "", ""
	if raw.Pricing.CostPerQuery.WebSync != nil {
		webSyncDay, webSyncNight = raw.Pricing.CostPerQuery.WebSync.Day, raw.Pricing.CostPerQuery.WebSync.Night
	}
	if c.Pricing.CostPerQuery.WebSync, err = parseOptionalDayNightRate(webSyncDay, webSyncNight, "yandex_search.pricing.cost_per_query.web_sync"); err != nil {
		return err
	}
	webAsyncDay, webAsyncNight := "", ""
	if raw.Pricing.CostPerQuery.WebAsync != nil {
		webAsyncDay, webAsyncNight = raw.Pricing.CostPerQuery.WebAsync.Day, raw.Pricing.CostPerQuery.WebAsync.Night
	}
	if c.Pricing.CostPerQuery.WebAsync, err = parseOptionalDayNightRate(webAsyncDay, webAsyncNight, "yandex_search.pricing.cost_per_query.web_async"); err != nil {
		return err
	}
	for _, field := range []struct {
		raw    string
		path   string
		target **float64
	}{
		{raw.Pricing.CostPerQuery.ImageSearch, "yandex_search.pricing.cost_per_query.image_search", &c.Pricing.CostPerQuery.ImageSearch},
		{raw.Pricing.CostPerQuery.SmartSnippetsRU, "yandex_search.pricing.cost_per_query.smart_snippets_ru", &c.Pricing.CostPerQuery.SmartSnippetsRU},
		{raw.Pricing.CostPerQuery.GenerativeSearchSync, "yandex_search.pricing.cost_per_query.generative_search_sync", &c.Pricing.CostPerQuery.GenerativeSearchSync},
		{raw.Pricing.CostPerQuery.WordstatGetTop, "yandex_search.pricing.cost_per_query.wordstat_get_top", &c.Pricing.CostPerQuery.WordstatGetTop},
		{raw.Pricing.CostPerQuery.WordstatGetDynamics, "yandex_search.pricing.cost_per_query.wordstat_get_dynamics", &c.Pricing.CostPerQuery.WordstatGetDynamics},
		{raw.Pricing.CostPerQuery.WordstatGetRegionsDistribution, "yandex_search.pricing.cost_per_query.wordstat_get_regions_distribution", &c.Pricing.CostPerQuery.WordstatGetRegionsDistribution},
		{raw.Pricing.CostPerQuery.WordstatGetRegionsTree, "yandex_search.pricing.cost_per_query.wordstat_get_regions_tree", &c.Pricing.CostPerQuery.WordstatGetRegionsTree},
	} {
		v, err := parseOptionalFloatPtr(field.raw, field.path)
		if err != nil {
			return err
		}
		*field.target = v
	}
	return nil
}

// parseOptionalFloatPtr resolves and parses an optional numeric field: a
// blank (or unset/env-unresolved) value yields nil rather than a default,
// so callers can distinguish "not configured" from "configured as zero"
// (WordstatGetRegionsTree is legitimately priced at 0).
func parseOptionalFloatPtr(raw, fieldPath string) (*float64, error) {
	resolved := resolveEnvString(raw)
	if resolved == "" {
		return nil, nil
	}
	v, err := strconv.ParseFloat(resolved, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid %s: %w", fieldPath, err)
	}
	return &v, nil
}

// parseOptionalDayNightRate parses a day/night pair where both fields must
// be either both absent (nil result, nothing configured) or both present
// (a day without a night, or vice versa, is a config error).
func parseOptionalDayNightRate(day, night, fieldPath string) (*DayNightRate, error) {
	dayVal, err := parseOptionalFloatPtr(day, fieldPath+".day")
	if err != nil {
		return nil, err
	}
	nightVal, err := parseOptionalFloatPtr(night, fieldPath+".night")
	if err != nil {
		return nil, err
	}
	if dayVal == nil && nightVal == nil {
		return nil, nil
	}
	if dayVal == nil || nightVal == nil {
		return nil, fmt.Errorf("%s requires both day and night", fieldPath)
	}
	return &DayNightRate{Day: *dayVal, Night: *nightVal}, nil
}

func (c *YandexSearchConfig) Validate() error {
	if !c.Enabled {
		return nil
	}
	required := map[string]string{
		"api_key":   c.APIKey,
		"folder_id": c.FolderID,
	}
	for field, value := range required {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("yandex_search.%s is required when yandex_search is enabled", field)
		}
	}
	if err := validateBaseURL("yandex_search.base_url", c.BaseURL); err != nil {
		return err
	}
	if parsed, err := url.Parse(c.BaseURL); err != nil || parsed.Scheme != "https" {
		return fmt.Errorf("yandex_search.base_url must use https")
	}
	if c.ProxyURL != "" {
		if parsed, err := url.Parse(c.ProxyURL); err != nil || parsed.Host == "" {
			return fmt.Errorf("yandex_search.proxy_url is invalid")
		}
	}
	if c.RPM <= 0 {
		return fmt.Errorf("yandex_search.rpm must be positive")
	}
	if c.RPS <= 0 {
		return fmt.Errorf("yandex_search.rps must be positive")
	}
	if c.HourlyLimit <= 0 {
		return fmt.Errorf("yandex_search.hourly_limit must be positive")
	}
	if c.RequestTimeout <= 0 {
		return fmt.Errorf("yandex_search.request_timeout must be positive")
	}
	if c.MaxBodyBytes <= 0 {
		return fmt.Errorf("yandex_search.max_body_bytes must be positive")
	}
	if err := c.Pricing.resolve(); err != nil {
		return err
	}
	return nil
}

// resolve parses and caches the pricing table's timezone and night-window
// boundaries, validates enabled_operations against the known operation
// list, and validates every cost_per_query entry that's actually present.
// Must be called (via Validate) before CostForTime/IsOperationEnabled are
// used.
func (p *YandexSearchPricing) resolve() error {
	if p.ExchangeRateRUBPerUSD <= 0 {
		return fmt.Errorf("yandex_search.pricing.exchange_rate_rub_per_usd must be positive")
	}
	if p.Markup <= 0 {
		return fmt.Errorf("yandex_search.pricing.rate must be positive")
	}
	if len(p.EnabledOperations) == 0 {
		return fmt.Errorf("yandex_search.pricing.enabled_operations must list at least one operation")
	}
	enabled := make(map[string]bool, len(p.EnabledOperations))
	for _, op := range p.EnabledOperations {
		if !knownYandexSearchOperations[op] {
			return fmt.Errorf("yandex_search.pricing.enabled_operations: unknown operation %q", op)
		}
		enabled[op] = true
	}

	loc, err := time.LoadLocation(p.Timezone)
	if err != nil {
		return fmt.Errorf("yandex_search.pricing.pricing_timezone: %w", err)
	}
	startMin, err := parseClockMinutes(p.NightStart)
	if err != nil {
		return fmt.Errorf("yandex_search.pricing.night_start: %w", err)
	}
	endMin, err := parseClockMinutes(p.NightEnd)
	if err != nil {
		return fmt.Errorf("yandex_search.pricing.night_end: %w", err)
	}

	// web_sync is the only operation this router implements — its price
	// is required (both day and night) whenever it's enabled, since the
	// handler cannot bill without it.
	if enabled[OperationWebSync] {
		r := p.CostPerQuery.WebSync
		if r == nil || r.Day <= 0 || r.Night <= 0 {
			return fmt.Errorf("yandex_search.pricing.cost_per_query.web_sync (day and night, both positive) is required when %q is in enabled_operations", OperationWebSync)
		}
	}
	// Every other catalog entry is optional (future operations), but if
	// present must be sane — except WordstatGetRegionsTree, which is
	// genuinely free.
	if r := p.CostPerQuery.WebAsync; r != nil {
		if r.Day < 0 || r.Night < 0 {
			return fmt.Errorf("yandex_search.pricing.cost_per_query.web_async must not be negative")
		}
	}
	for name, v := range map[string]*float64{
		OperationImageSearch:                    p.CostPerQuery.ImageSearch,
		OperationSmartSnippetsRU:                p.CostPerQuery.SmartSnippetsRU,
		OperationGenerativeSearchSync:           p.CostPerQuery.GenerativeSearchSync,
		OperationWordstatGetTop:                 p.CostPerQuery.WordstatGetTop,
		OperationWordstatGetDynamics:            p.CostPerQuery.WordstatGetDynamics,
		OperationWordstatGetRegionsDistribution: p.CostPerQuery.WordstatGetRegionsDistribution,
	} {
		if v != nil && *v < 0 {
			return fmt.Errorf("yandex_search.pricing.cost_per_query.%s must not be negative", name)
		}
	}
	if v := p.CostPerQuery.WordstatGetRegionsTree; v != nil && *v < 0 {
		return fmt.Errorf("yandex_search.pricing.cost_per_query.wordstat_get_regions_tree must not be negative")
	}

	p.location = loc
	p.nightStartMin = startMin
	p.nightEndMin = endMin
	p.enabledOps = enabled
	return nil
}

// parseClockMinutes parses "HH:MM" (24h) into minutes since midnight.
func parseClockMinutes(value string) (int, error) {
	t, err := time.Parse("15:04", value)
	if err != nil {
		return 0, fmt.Errorf("invalid HH:MM value %q: %w", value, err)
	}
	return t.Hour()*60 + t.Minute(), nil
}

// IsOperationEnabled reports whether operation is listed in
// enabled_operations — the actual runtime gate; a price existing in
// CostPerQuery does not by itself enable anything. Must be called after
// Validate (resolve) has run; an unresolved Pricing reports every
// operation as disabled.
func (p YandexSearchPricing) IsOperationEnabled(operation string) bool {
	return p.enabledOps[operation]
}

// CostForTime returns web_sync's marked-up USD price — the only operation
// this router's adapter calls — for a request sent at t, and which rate
// period ("day" or "night") applied. Validate (via resolve) must have run
// first; an unresolved Pricing (location nil) treats every time as "day"
// rather than panicking, so a config that bypassed validation fails safe
// (over-estimates cost) instead of crashing or charging nothing.
func (p YandexSearchPricing) CostForTime(t time.Time) (usd float64, period string) {
	if p.CostPerQuery.WebSync == nil {
		return 0, "day"
	}
	if p.location == nil {
		return p.CostPerQuery.WebSync.Day, "day"
	}
	local := t.In(p.location)
	minute := local.Hour()*60 + local.Minute()
	if inNightWindow(minute, p.nightStartMin, p.nightEndMin) {
		return p.CostPerQuery.WebSync.Night, "night"
	}
	return p.CostPerQuery.WebSync.Day, "day"
}

// inNightWindow reports whether minute falls in [start, end), handling a
// window that wraps past midnight (start > end).
func inNightWindow(minute, start, end int) bool {
	if start == end {
		return false
	}
	if start < end {
		return minute >= start && minute < end
	}
	return minute >= start || minute < end
}
