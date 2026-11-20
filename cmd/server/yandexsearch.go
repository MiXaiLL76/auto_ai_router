package main

import (
	"log/slog"
	"net/http"
	"os"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/yandexsearch"
)

// initializeYandexSearchOrExit wires the /v1/search/yandex passthrough
// handler when yandex_search.enabled is set.
//
// It fails fast at startup (os.Exit) when the LiteLLM DB isn't usable for
// synchronous spend accounting — same requirement video generation has,
// for the same reason: billing the client depends on it, and every
// environment that runs this router is expected to have one.
//
// It does NOT fail fast when Redis is unavailable — it logs loudly and
// leaves the endpoint unwired (404) instead, the same graceful-disable
// path as enabled_operations below. Redis coordinates the shared Yandex
// folder_id quota (1 RPS + hourly) across every replica, and running it
// in-process-only would silently stop protecting that quota — but some
// legitimate environments (e.g. this org's own ephemeral PR stages) run
// the router without Redis at all for unrelated reasons; crashing the
// entire router over one optional feature's missing dependency would be a
// worse outcome than that one feature staying off. Not wiring the handler
// gives the same safety guarantee (the quota is never enforced
// uncoordinated, because the feature simply never runs) with a much
// smaller blast radius.
//
// budgetReserver may be nil (budget reservation disabled system-wide, or
// Redis just connected for other reasons) — yandexsearch.Handler degrades
// to always-allow/no-op settle in that case, same as the LLM request path.
//
// kafkaLog is the analytics write-path (kafkalog.Manager, feeding ClickHouse's
// air.logs). Passing a NoopManager when Kafka is disabled keeps the handler's
// publishing a no-op, exactly as the LLM path does; unlike Postgres it is
// never required for startup.
func initializeYandexSearchOrExit(cfg *config.Config, prx *proxy.Proxy, db litellmdb.Manager, kafkaLog kafkalog.Manager, redisBackend *ratelimit.RedisBackend, budgetReserver *budget.Reserver, logger *slog.Logger) http.Handler {
	if cfg == nil || !cfg.YandexSearch.Enabled {
		return nil
	}
	// enabled_operations is the real gate on which Yandex Search operation
	// this deployment may serve: a price existing in the catalog (pricing
	// ships ahead of the adapters that would use it — async search, image
	// search, smart snippets, ...) never by itself enables anything. This
	// router implements exactly one operation, web_sync; if it's not
	// enabled, the endpoint stays unwired (404) rather than failing
	// startup — an operator may legitimately turn yandex_search.enabled on
	// before flipping web_sync on in enabled_operations.
	if !cfg.YandexSearch.Pricing.IsOperationEnabled(config.OperationWebSync) {
		logger.Warn("Yandex search: web_sync is not in pricing.enabled_operations, endpoint not wired",
			"path", "/v1/search/yandex")
		return nil
	}
	if db == nil || !db.IsEnabled() {
		logger.Error("Yandex search requires an enabled LiteLLM database")
		os.Exit(1)
	}
	spendStatus, ok := db.(interface{ SpendLoggingEnabled() bool })
	if !ok || !spendStatus.SpendLoggingEnabled() {
		logger.Error("Yandex search requires synchronous SpendLogs accounting")
		os.Exit(1)
	}
	committer, ok := db.(litellmdb.SpendCommitter)
	if !ok {
		logger.Error("Yandex search requires synchronous spend settlement")
		os.Exit(1)
	}
	if redisBackend == nil {
		logger.Error("Yandex search: Redis is unavailable, endpoint not wired — the shared folder_id quota (1 RPS + hourly) cannot be coordinated across replicas without it",
			"path", "/v1/search/yandex")
		return nil
	}

	client := yandexsearch.NewClient(yandexsearch.ClientConfig{
		BaseURL:  cfg.YandexSearch.BaseURL,
		APIKey:   cfg.YandexSearch.APIKey,
		FolderID: cfg.YandexSearch.FolderID,
		Timeout:  cfg.YandexSearch.RequestTimeout,
		ProxyURL: cfg.YandexSearch.ProxyURL,
	})

	quota := yandexsearch.NewFolderQuota(redisBackend.Client(), cfg.YandexSearch.RedisKeyPrefix,
		cfg.YandexSearch.RPS, cfg.YandexSearch.HourlyLimit)

	return yandexsearch.NewHandler(yandexsearch.HandlerConfig{
		Client:         client,
		Auth:           prx, // *proxy.Proxy satisfies yandexsearch.Authenticator structurally
		Spend:          committer,
		Kafka:          kafkaLog, // analytics projection ClickHouse's air.logs ingests; NoopManager when Kafka is off
		Identity:       prx,      // *proxy.Proxy satisfies yandexsearch.RouterIdentity structurally
		Quota:          quota,
		Pricing:        cfg.YandexSearch.Pricing, // config.YandexSearchPricing satisfies yandexsearch.PricingTable structurally
		Budget:         budgetReserver,
		RPM:            cfg.YandexSearch.RPM,
		MaxBodyBytes:   cfg.YandexSearch.MaxBodyBytes,
		RequestTimeout: cfg.YandexSearch.RequestTimeout,
		Logger:         logger,
	})
}
