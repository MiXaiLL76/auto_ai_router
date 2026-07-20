package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/health"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/modelupdate"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/router"
	"github.com/mixaill76/auto_ai_router/internal/startup"
	"github.com/mixaill76/auto_ai_router/internal/telemetry"

	// Register native Responses API converters for Vertex AI, Anthropic, and Bedrock.
	_ "github.com/mixaill76/auto_ai_router/internal/converter/anthropic/responses"
	_ "github.com/mixaill76/auto_ai_router/internal/converter/bedrock/responses"
	_ "github.com/mixaill76/auto_ai_router/internal/converter/vertex/responses"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/propagation"
)

var (
	Version = "dev"
	Commit  = "unknown"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()

	// ==================== Load Configuration ====================
	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	httputil.SetProxyFetchTimeout(cfg.Server.ProxyHealthTimeout)

	// ==================== Initialize OpenTelemetry ====================
	// Must happen before logger creation so OTLP log export covers all startup logs.
	// Export diagnostics go to a stdout-only logger: routing them through the
	// OTEL pipeline would generate a record per export batch and loop forever.
	otelDiagLog := logger.New(cfg.Server.LoggingLevel)
	otelSDK, otelErr := telemetry.Setup(context.Background(), &cfg.OTEL, Version, Commit, otelDiagLog)

	stdoutLogs := cfg.Server.StdoutLogsEnabled
	if !stdoutLogs && otelSDK.LogHandler() == nil {
		// Refusing to run a server with no log destination at all:
		// stdout was disabled but OTEL log export is not active either.
		slog.Warn("stdout_logs_enabled=false but OTEL log export is not active, keeping stdout logs")
		stdoutLogs = true
	}
	log := logger.NewMulti(cfg.Server.LoggingLevel, stdoutLogs, otelSDK.LogHandler())
	if otelErr != nil {
		// OTEL is observability, not core functionality — degrade instead of failing startup.
		log.Error("Failed to initialize OpenTelemetry, continuing without it", "error", otelErr)
	} else if otelSDK != nil {
		log.Info("OpenTelemetry initialized",
			"endpoint", cfg.OTEL.Endpoint,
			"protocol", cfg.OTEL.Protocol,
			"logs_enabled", cfg.OTEL.LogsEnabled,
			"traces_enabled", cfg.OTEL.TracesEnabled,
			"metrics_enabled", otelSDK.MetricsEnabled(),
		)
	}

	config.PrintConfig(log, cfg)

	log.Info("Starting auto_ai_router",
		"version", Version,
		"commit", Commit,
		"logging_level", cfg.Server.LoggingLevel,
		"port", cfg.Server.Port,
	)

	logCredentials(log, cfg.Credentials)

	// ==================== Startup Validation ====================
	startup.ValidateProxyCredentialsAtStartup(cfg, log)

	// ==================== Initialize Core Components ====================

	// Create a shared Redis/Valkey backend if enabled.
	// The same underlying client is reused by both the rate limiter and the response store.
	var redisBackend *ratelimit.RedisBackend
	if cfg.Redis.Enabled {
		rb, err := ratelimit.NewRedisBackend(cfg.Redis)
		if err != nil {
			log.Error("Failed to connect to Redis, falling back to local backends", "error", err)
		} else {
			// Verify Redis is responsive with a health check ping.
			// Use explicit cancel (not defer) so the context is released immediately.
			pingCtx, pingCancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := rb.Ping(pingCtx)
			pingCancel()
			if pingErr != nil {
				log.Warn("Redis health check failed, falling back to local backends", "error", pingErr)
				rb.Close()
			} else {
				log.Info("Connected to Redis/Valkey", "addresses", cfg.Redis.InitAddresses)
				redisBackend = rb
				defer rb.Close()
			}
		}
	}

	litellmDBManager := initializeLiteLLMDB(cfg, log)
	kafkaLogManager := initializeKafkaLog(cfg, log, litellmDBManager)

	// ==================== Budget reservation & key-level RPM/TPM ====================
	// Both are Redis-backed and reuse the shared valkey client with isolated key
	// namespaces. When Redis is disabled they stay nil and the proxy falls back to
	// snapshot-only budget checks (see todo_auth_billing.md P1.4).
	var budgetReserver *budget.Reserver
	var keyRateLimiter *ratelimit.RPMLimiter
	if redisBackend != nil && cfg.LiteLLMDB.Enabled {
		if cfg.LiteLLMDB.EnforceBudgetReservation {
			budgetReserver = budget.New(redisBackend.Client(), cfg.Redis.KeyPrefix+"litellmbudget:", cfg.LiteLLMDB.BudgetReservationTTL, log)
			log.Info("Budget reservation: enabled (Redis-backed, atomic overspend protection)")
		}
		if cfg.LiteLLMDB.EnforceKeyRateLimits {
			authRedisBackend := ratelimit.NewRedisBackendFromClient(redisBackend.Client(), cfg.Redis.KeyPrefix+"litellmauth:")
			if cfg.Redis.Hybrid {
				hybridAuthBackend := ratelimit.NewHybridBackend(authRedisBackend, cfg.Redis.SyncInterval)
				defer hybridAuthBackend.Close()
				keyRateLimiter = ratelimit.NewWithHybrid(hybridAuthBackend)
			} else {
				keyRateLimiter = ratelimit.NewWithRedis(authRedisBackend)
			}
			log.Info("Key-level TPM/RPM enforcement: enabled (Redis-backed, per key/user/team/org)")
		}
	} else if cfg.LiteLLMDB.Enabled {
		log.Warn("Budget reservation and key-level rate limits disabled: Redis is not enabled (redis.enabled=false). Falling back to snapshot-only budget checks (see todo_auth_billing.md P1.4 for the overspend race this leaves open).")
	}

	// ==================== Initialize Balancer & Model Manager (YAML-only) ====================
	// IMPORTANT: Do NOT modify cfg.Credentials or cfg.Models here.
	// The balancer and model manager snapshot YAML-only data as their immutable
	// "static" baseline. DB data is applied AFTER construction via UpdateDBCredentials /
	// UpdateDBModels so that the sync loop can correctly add/remove DB-sourced entries.

	priceRegistry := models.NewModelPriceRegistry()
	_, rateLimiter, bal := initializeBalancer(cfg, log, redisBackend)
	modelManager := initializeModelManager(log, cfg, rateLimiter, bal)

	// ==================== Apply Initial DB Model Table ====================
	// Fetch DB data and apply it through the same code-path used by the sync loop,
	// so that staticCreds / staticModelLimits stay YAML-only.
	// staticCreds is the YAML-only snapshot used by the sync loop to differentiate
	// static vs. DB-sourced credentials.
	staticCreds := append([]config.CredentialConfig(nil), cfg.Credentials...)
	if litellmDBManager.IsEnabled() {
		applyInitialDBModelTable(context.Background(), litellmDBManager, staticCreds, bal, modelManager, rateLimiter, priceRegistry, cfg, log)
	}
	tokenManager := auth.NewVertexTokenManager(log)
	defer tokenManager.Stop()

	// Record metrics whenever any sink consumes them — the pull /metrics
	// endpoint (prometheus_enabled) and/or OTLP push (otel.enabled). The pull
	// endpoint and the push pipeline are wired up separately below.
	metrics := monitoring.New(cfg.MetricsCollectionEnabled())

	// ==================== Initialize Model Pricing ====================
	if cfg.Server.ModelPricesLink != "" {
		log.Info("Using model prices from", "link", cfg.Server.ModelPricesLink)
	} else {
		log.Debug("Model prices not configured (model_prices_link empty)")
	}

	// ==================== Create Health Checker ====================
	healthChecker := health.NewDBHealthChecker()
	if litellmDBManager.IsEnabled() && !litellmDBManager.IsHealthy() {
		healthChecker.SetHealthy(false)
		log.Warn("LiteLLM DB initial health check failed (marked unhealthy)")
	} else if litellmDBManager.IsEnabled() {
		log.Info("LiteLLM DB initial health check passed (marked healthy)")
	}

	// ==================== Create Response Store ====================
	var respStore responsestore.Store
	if redisBackend != nil {
		respStore = responsestore.NewRedis(redisBackend.Client(), cfg.Redis.KeyPrefix)
		log.Info("Response store: using Redis backend")
	} else {
		var storeErr error
		respStore, storeErr = responsestore.New()
		if storeErr != nil {
			log.Warn("Failed to initialize response store (Responses API store/previous_response_id will be disabled)",
				"error", storeErr)
			respStore = nil
		} else {
			log.Info("Response store initialized (bbolt)")
			defer func() {
				if err := respStore.Close(); err != nil {
					log.Error("Failed to close response store", "error", err)
				}
			}()
		}
	}

	// ==================== Create Proxy ====================
	prx := proxy.New(&proxy.Config{
		Balancer:                   bal,
		Logger:                     log,
		MaxBodySizeMB:              cfg.Server.MaxBodySizeMB,
		ResponseBodyMultiplier:     cfg.Server.ResponseBodyMultiplier,
		RequestTimeout:             cfg.Server.RequestTimeout,
		MaxIdleConns:               cfg.Server.MaxIdleConns,
		MaxIdleConnsPerHost:        cfg.Server.MaxIdleConnsPerHost,
		IdleConnTimeout:            cfg.Server.IdleConnTimeout,
		Metrics:                    metrics,
		MasterKey:                  cfg.Server.MasterKey,
		RateLimiter:                rateLimiter,
		TokenManager:               tokenManager,
		ModelManager:               modelManager,
		Version:                    Version,
		Commit:                     Commit,
		LiteLLMDB:                  litellmDBManager,
		KafkaLog:                   kafkaLogManager,
		HealthChecker:              healthChecker,
		PriceRegistry:              priceRegistry,
		MaxProviderRetries:         cfg.Server.MaxProviderRetries,
		MaxFallbackAttempts:        cfg.Server.MaxFallbackAttempts,
		ResponseStore:              respStore,
		SessionStickyEnabled:       cfg.Server.SessionStickyEnabled,
		SessionStickyAutoCacheCtrl: cfg.Server.SessionStickyAutoCacheCtrl,
		SessionStoreTTL:            time.Duration(cfg.Server.SessionStickyTTL) * time.Minute,
		DrainUpstreamOnAbort:       cfg.Server.DrainUpstreamOnAbort,

		BudgetReserver:                   budgetReserver,
		KeyRateLimiter:                   keyRateLimiter,
		BudgetReservationEnabled:         cfg.LiteLLMDB.EnforceBudgetReservation || cfg.LiteLLMDB.EnforceKeyRateLimits,
		DefaultEstimatedCompletionTokens: cfg.LiteLLMDB.DefaultEstimatedCompletionTokens,
	})

	// ==================== Background Goroutines ====================
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()

	prx.Start(bgCtx)

	var wg sync.WaitGroup
	var updateMutex sync.Mutex

	startMetricsUpdater(cfg, log, bgCtx, bal, rateLimiter, metrics, &wg, &updateMutex)
	startProxyStatsUpdater(log, bgCtx, bal, rateLimiter, modelManager, &wg, &updateMutex)
	if kafkaLogManager.IsEnabled() {
		startKafkaMetricsUpdater(cfg, log, bgCtx, kafkaLogManager, metrics, &wg)
	}

	if respStore != nil {
		startResponseStoreCleanup(log, bgCtx, respStore, &wg)
	}

	if litellmDBManager.IsEnabled() {
		startDBHealthMonitor(log, bgCtx, litellmDBManager, healthChecker, &wg)
		if err := litellmDBManager.FetchMasterKey(bgCtx, cfg.Server.MasterKey); err != nil {
			log.Warn("Failed to fetch master key from LiteLLM DB.", "error", err)
		}
		if cfg.LiteLLMDB.LoadLitellmDBModels {
			startDBModelTableSyncLoop(log, bgCtx, litellmDBManager, staticCreds,
				bal, modelManager, rateLimiter, priceRegistry, cfg, cfg.LiteLLMDB.LitellmDBSyncInterval, &wg)
		}
	}

	// Start model price sync loop (only if configured)
	if cfg.Server.ModelPricesLink != "" {
		startPriceSyncLoop(cfg.Server.ModelPricesLink, priceRegistry, log, bgCtx, &wg)
	}

	// ==================== HTTP Server Setup ====================
	rtr := router.New(prx, modelManager, &cfg.Monitoring, log, cfg)
	mux := http.NewServeMux()
	mux.Handle("/", rtr)

	if cfg.Monitoring.PrometheusEnabled {
		mux.Handle("/metrics", promhttp.Handler())
		log.Info("Prometheus metrics enabled", "path", "/metrics")
	}

	var rootHandler http.Handler = mux
	if otelSDK.TracesEnabled() {
		// Server spans for every API request; health/readiness probes and
		// metrics scrapes are excluded to avoid trace noise.
		otelOpts := []otelhttp.Option{
			otelhttp.WithFilter(func(r *http.Request) bool {
				switch r.URL.Path {
				case cfg.Monitoring.HealthCheckPath, "/health/readiness", "/metrics":
					return false
				}
				return true
			}),
			otelhttp.WithSpanNameFormatter(func(operation string, r *http.Request) string {
				return r.Method + " " + r.URL.Path
			}),
		}
		// When the incoming traceparent is not trusted (no trusted hop such as a
		// LiteLLM proxy in front), override the handler's propagator with a no-op
		// extractor so client-supplied trace context is ignored and every request
		// starts a fresh root span. Outgoing propagation to upstreams still uses
		// the global propagator via the HTTP client transport and is unaffected.
		if !cfg.OTEL.TrustIncomingTraceparent {
			otelOpts = append(otelOpts, otelhttp.WithPropagators(propagation.NewCompositeTextMapPropagator()))
		}
		rootHandler = otelhttp.NewHandler(mux, "auto_ai_router", otelOpts...)
	}

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      rootHandler,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Bind port explicitly so readiness is set only after the socket is open.
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Server.Port))
	if err != nil {
		log.Error("Failed to bind server port", "error", err, "port", cfg.Server.Port)
		os.Exit(1)
	}

	// Mark ready — TCP listener is bound, pod can accept traffic.
	rtr.SetReady(true)
	log.Info("Server ready", "port", cfg.Server.Port)

	go func() {
		if err := server.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Error("Server failed", "error", err)
			os.Exit(1)
		}
	}()

	// ==================== Signal Handling & Graceful Shutdown ====================
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Info("Shutting down server...")

	// Mark not ready — stop Kubernetes from sending new traffic before draining.
	rtr.SetReady(false)

	// Wait for the load balancer / readiness probe to observe the 503 and stop
	// routing new traffic to this pod before we close the listener.
	if cfg.Server.ShutdownDelay > 0 {
		log.Info("Waiting for load balancer drain", "delay", cfg.Server.ShutdownDelay)
		time.Sleep(cfg.Server.ShutdownDelay)
	}

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Error("Server forced to shutdown", "error", err)
		os.Exit(1)
	}

	// Stop background goroutines
	log.Info("Stopping background goroutines...")
	bgCancel()

	// Wait for completion
	doneChan := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneChan)
	}()

	select {
	case <-doneChan:
		log.Info("All background goroutines stopped gracefully")
	case <-time.After(60 * time.Second):
		log.Warn("Background goroutines did not stop within 60 seconds timeout")
	}

	// Shutdown LiteLLM DB
	if litellmDBManager.IsEnabled() {
		log.Info("Shutting down LiteLLM DB...")
		dbShutdownCtx, dbShutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer dbShutdownCancel()
		if err := litellmDBManager.Shutdown(dbShutdownCtx); err != nil {
			log.Error("LiteLLM DB shutdown error", "error", err)
		}
	}

	// Shutdown Kafka spend-log publisher
	if kafkaLogManager.IsEnabled() {
		log.Info("Shutting down Kafka spend-log publisher...")
		kafkaShutdownCtx, kafkaShutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer kafkaShutdownCancel()
		if err := kafkaLogManager.Shutdown(kafkaShutdownCtx); err != nil {
			log.Error("Kafka spend-log publisher shutdown error", "error", err)
		}
	}

	if err := router.CloseErrorLogFiles(); err != nil {
		log.Error("Failed to close error log files", "error", err)
	}

	log.Info("Server shutdown complete")

	// Flush pending OTEL spans and log records last so the shutdown logs above
	// are exported too.
	if err := otelSDK.Shutdown(context.Background()); err != nil {
		slog.Error("OpenTelemetry shutdown error", "error", err)
	}
}

// ==================== Helper Functions ====================

func logCredentials(log *slog.Logger, credentials []config.CredentialConfig) {
	log.Info("Loaded credentials", "count", len(credentials))
	for i, cred := range credentials {
		log.Info("Credential configured",
			"index", i+1,
			"name", cred.Name,
			"type", cred.Type,
			"base_url", cred.BaseURL,
			"rpm", cred.RPM,
		)
	}
}

func initializeBalancer(
	cfg *config.Config,
	log *slog.Logger,
	redisBackend *ratelimit.RedisBackend,
) (*fail2ban.Fail2Ban, *ratelimit.RPMLimiter, *balancer.RoundRobin) {
	rules := convertFailBanRules(cfg.Fail2Ban.ErrorCodeRules, cfg.Fail2Ban.BanDuration, log)
	f2b := fail2ban.NewWithRules(cfg.Fail2Ban.MaxAttempts, cfg.Fail2Ban.BanDuration,
		cfg.Fail2Ban.ErrorCodes, rules)
	f2b.SetLogger(log)

	var rateLimiter *ratelimit.RPMLimiter
	if redisBackend != nil {
		if cfg.Redis.Hybrid {
			log.Info("Rate limiter: using hybrid backend (local decisions, async Redis sync)",
				"sync_interval", cfg.Redis.SyncInterval)
			hybridBackend := ratelimit.NewHybridBackend(redisBackend, cfg.Redis.SyncInterval)
			defer hybridBackend.Close()
			rateLimiter = ratelimit.NewWithHybrid(hybridBackend)
		} else {
			log.Info("Rate limiter: using Redis backend")
			rateLimiter = ratelimit.NewWithRedis(redisBackend)
		}
	} else {
		rateLimiter = ratelimit.New()
	}

	bal := balancer.New(cfg.Credentials, f2b, rateLimiter)
	bal.SetLogger(log)

	return f2b, rateLimiter, bal
}

func convertFailBanRules(
	rules []config.ErrorCodeRuleConfig,
	defaultBanDuration time.Duration,
	log *slog.Logger,
) []fail2ban.ErrorCodeRule {
	converted := make([]fail2ban.ErrorCodeRule, 0, len(rules))
	for _, rule := range rules {
		banDuration := defaultBanDuration
		if rule.BanDuration == "permanent" {
			banDuration = 0 // 0 = permanent ban in fail2ban
		} else if rule.BanDuration != "" {
			if dur, err := time.ParseDuration(rule.BanDuration); err == nil {
				banDuration = dur
			} else {
				log.Error("Invalid ban_duration in error_code_rules",
					"error_code", rule.Code, "error", err)
			}
		}

		converted = append(converted, fail2ban.ErrorCodeRule{
			Code:        rule.Code,
			MaxAttempts: rule.MaxAttempts,
			BanDuration: banDuration,
		})
	}
	return converted
}

func initializeModelManager(
	log *slog.Logger,
	cfg *config.Config,
	rateLimiter *ratelimit.RPMLimiter,
	bal *balancer.RoundRobin,
) *models.Manager {
	modelManager := models.New(log, cfg.Server.DefaultModelsRPM, cfg.Models)
	modelManager.LoadModelsFromConfig(cfg.Credentials)
	modelManager.SetCredentials(cfg.Credentials)
	if len(cfg.ModelAlias) > 0 {
		modelManager.SetModelAliases(cfg.ModelAlias)
	}

	// Initialize rate limiters for each model
	modelsResp := modelManager.GetAllModels()
	for _, cred := range cfg.Credentials {
		for _, model := range modelsResp.Data {
			if modelManager.HasModel(cred.Name, model.ID) {
				rpm := modelManager.GetModelRPMForCredential(model.ID, cred.Name)
				tpm := modelManager.GetModelTPMForCredential(model.ID, cred.Name)
				rateLimiter.AddModelWithTPM(cred.Name, model.ID, rpm, tpm)
				weight := balancer.EffectiveWeight(modelManager.GetModelWeightForCredential(model.ID, cred.Name), cred.Weight)
				log.Debug("Initialized model rate limiters",
					"credential", cred.Name,
					"model", model.ID,
					"rpm", rpm,
					"tpm", tpm,
					"weight", weight,
				)
				if weight != 1 {
					log.Info("Weighted routing configured",
						"credential", cred.Name,
						"model", model.ID,
						"weight", weight,
					)
				}
			}
		}
	}

	bal.SetModelChecker(modelManager)
	return modelManager
}

// startDBModelTableSyncLoop starts a background goroutine that periodically reloads
// credentials and models from the LiteLLM DB and applies a diff to the live router.
// - New DB credentials are added to the balancer and rate limiter.
// - Removed DB credentials are dropped from the balancer (rate limiter entries left stale).
// - New/changed model limits are reflected immediately in the model manager.
// - Static (YAML) credentials and models are never modified.
func startDBModelTableSyncLoop(
	log *slog.Logger,
	bgCtx context.Context,
	dbManager litellmdb.Manager,
	staticCreds []config.CredentialConfig,
	bal *balancer.RoundRobin,
	modelManager *models.Manager,
	rateLimiter *ratelimit.RPMLimiter,
	priceRegistry *models.ModelPriceRegistry,
	cfg *config.Config,
	interval time.Duration,
	wg *sync.WaitGroup,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-bgCtx.Done():
				log.Debug("DB model table sync loop stopped")
				return
			case <-ticker.C:
				syncDBModelTable(bgCtx, log, dbManager, staticCreds, bal, modelManager, rateLimiter, priceRegistry, cfg)
			}
		}
	}()

	log.Info("DB model table sync loop started", "interval", interval)
}

// syncDBModelTable performs a single sync cycle: fetches fresh DB data and applies diffs.
func syncDBModelTable(
	ctx context.Context,
	log *slog.Logger,
	dbManager litellmdb.Manager,
	staticCreds []config.CredentialConfig,
	bal *balancer.RoundRobin,
	modelManager *models.Manager,
	rateLimiter *ratelimit.RPMLimiter,
	priceRegistry *models.ModelPriceRegistry,
	cfg *config.Config,
) {
	defer func() {
		if r := recover(); r != nil {
			log.Error("Panic in sync loop", "panic", r)
		}
	}()

	fetchCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	dbCreds, dbModelCfgs, dbPrices, err := dbManager.FetchModelsForAIR(fetchCtx, cfg.Server.MasterKey)
	if err != nil {
		log.Warn("DB model table sync: fetch failed", "error", err)
		return
	}

	// Apply DB credentials to balancer (diff is computed inside UpdateDBCredentials).
	bal.UpdateDBCredentials(dbCreds)

	// Build the current complete credential list (static + new DB) for model mapping.
	allCreds := append(append([]config.CredentialConfig(nil), staticCreds...), dbCreds...)

	// Apply DB models to model manager and update its credential list so that
	// DB-sourced proxy credentials participate in GetAllModels remote fetches.
	modelManager.UpdateDBModels(dbModelCfgs, staticCreds, allCreds)
	modelManager.SetCredentials(allCreds)

	// Upsert rate limiter entries for all DB credential+model pairs.
	// For models with no specific credential, register only static (YAML) creds —
	// synthetic DB credentials (db-model-*) must not be cross-mapped to other models.
	for _, dm := range dbModelCfgs {
		if dm.Credential != "" {
			rateLimiter.AddModelWithTPM(dm.Credential, dm.Name, dm.RPM, dm.TPM)
		} else {
			credTargets := staticCreds
			if len(credTargets) == 0 {
				// DB-only setup: map global models to non-synthetic DB creds.
				credTargets = dbCreds
			}
			for _, cred := range credTargets {
				if strings.HasPrefix(cred.Name, "db-model-") {
					continue
				}
				rateLimiter.AddModelWithTPM(cred.Name, dm.Name, dm.RPM, dm.TPM)
			}
		}
	}

	// Merge DB prices into the price registry (does not replace file-loaded prices for
	// models that are absent from the DB).
	if len(dbPrices) > 0 {
		priceRegistry.MergeDB(dbPrices)
	}

	log.Debug("DB model table sync completed",
		"credentials", len(dbCreds),
		"models", len(dbModelCfgs),
		"prices", len(dbPrices),
	)
}

// applyInitialDBModelTable fetches DB data and applies it through UpdateDBCredentials /
// UpdateDBModels (same path as the sync loop). This guarantees that staticCreds and
// staticModelLimits inside the balancer and model manager are YAML-only, so subsequent
// sync cycles can correctly add/remove DB-sourced entries.
func applyInitialDBModelTable(
	ctx context.Context,
	dbManager litellmdb.Manager,
	staticCreds []config.CredentialConfig,
	bal *balancer.RoundRobin,
	modelManager *models.Manager,
	rateLimiter *ratelimit.RPMLimiter,
	priceRegistry *models.ModelPriceRegistry,
	cfg *config.Config,
	log *slog.Logger,
) {
	fetchCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	dbCreds, dbModelCfgs, dbPrices, err := dbManager.FetchModelsForAIR(fetchCtx, cfg.Server.MasterKey)
	if err != nil {
		log.Warn("Failed to load initial model table from LiteLLM DB (continuing without DB models)",
			"error", err,
		)
		return
	}

	bal.UpdateDBCredentials(dbCreds)

	allCreds := append(append([]config.CredentialConfig(nil), staticCreds...), dbCreds...)
	modelManager.UpdateDBModels(dbModelCfgs, staticCreds, allCreds)
	// Let the model manager know about all credentials (including DB proxy creds)
	// so that GetAllModels can fetch remote model lists from DB-sourced proxy credentials.
	modelManager.SetCredentials(allCreds)

	// For models with no specific credential, register only static (YAML) creds.
	// Synthetic DB credentials (db-model-*) are model-specific and must not be
	// cross-mapped to unrelated models.
	for _, dm := range dbModelCfgs {
		if dm.Credential != "" {
			rateLimiter.AddModelWithTPM(dm.Credential, dm.Name, dm.RPM, dm.TPM)
		} else {
			credTargets := staticCreds
			if len(credTargets) == 0 {
				// DB-only setup: map global models to non-synthetic DB creds.
				credTargets = dbCreds
			}
			for _, cred := range credTargets {
				if strings.HasPrefix(cred.Name, "db-model-") {
					continue
				}
				rateLimiter.AddModelWithTPM(cred.Name, dm.Name, dm.RPM, dm.TPM)
			}
		}
	}

	if len(dbPrices) > 0 {
		priceRegistry.MergeDB(dbPrices)
	}

	log.Info("Applied initial DB model table",
		"credentials", len(dbCreds),
		"models", len(dbModelCfgs),
		"prices", len(dbPrices),
	)
}

func initializeLiteLLMDB(cfg *config.Config, log *slog.Logger) litellmdb.Manager {
	if !cfg.LiteLLMDB.Enabled {
		log.Info("LiteLLM DB integration disabled - using NoopManager (no security checks)")
		return litellmdb.NewNoopManager()
	}

	log.Info("Initializing LiteLLM DB integration...", "is_required", cfg.LiteLLMDB.IsRequired)

	litellmCfg := &litellmdb.Config{
		DatabaseURL:           cfg.LiteLLMDB.DatabaseURL,
		MaxConns:              int32(cfg.LiteLLMDB.MaxConns),
		MinConns:              int32(cfg.LiteLLMDB.MinConns),
		HealthCheckInterval:   cfg.LiteLLMDB.HealthCheckInterval,
		ConnectTimeout:        cfg.LiteLLMDB.ConnectTimeout,
		AuthCacheTTL:          cfg.LiteLLMDB.AuthCacheTTL,
		AuthCacheSize:         cfg.LiteLLMDB.AuthCacheSize,
		LogQueueSize:          cfg.LiteLLMDB.LogQueueSize,
		LogBatchSize:          cfg.LiteLLMDB.LogBatchSize,
		LogFlushInterval:      cfg.LiteLLMDB.LogFlushInterval,
		DisableSpendLogsWrite: cfg.LiteLLMDB.DisableSpendLogsWrite,
		Logger:                log,
	}

	manager, err := litellmdb.New(litellmCfg)
	if err != nil {
		if cfg.LiteLLMDB.IsRequired {
			log.Error("CRITICAL: Failed to initialize required LiteLLM DB integration",
				"error", err,
				"reason", "LiteLLM DB is configured as required (is_required=true)",
				"action", "Fix database connectivity or set is_required=false",
			)
			os.Exit(1)
		}

		log.Warn("Failed to initialize optional LiteLLM DB, degrading to NoopManager",
			"error", err,
			"impact", "Budget checks, rate limits, and token auth validation will be disabled",
		)
		return litellmdb.NewNoopManager()
	}
	log.Info("LiteLLM DB integration initialized successfully")
	return manager
}

// initializeKafkaLog sets up the Kafka spend-log publisher (internal/kafkalog),
// an independent analytics write-path alongside (not instead of) LiteLLM
// Postgres. There is no "is_required" flag here: broker unavailability never
// blocks startup or request processing on its own — it only keeps
// kafkalog.Manager.IsHealthy() false until connectivity is established (see
// auto_ai_router_kafka_spend_log_tz.md section 6). The one exception is
// Kafka-only mode (litellm_db.disable_spend_logs_write=true): there, Kafka is
// the *only* spend-log write-path, so degrading to NoopManager would silently
// drop every spend event with no write-path left at all — that case is fatal.
func initializeKafkaLog(cfg *config.Config, log *slog.Logger, litellmDBManager litellmdb.Manager) kafkalog.Manager {
	if !cfg.Kafka.Enabled {
		log.Info("Kafka spend-log publishing disabled - using NoopManager")
		return kafkalog.NewNoopManager()
	}

	log.Info("Initializing Kafka spend-log publisher...", "brokers", cfg.Kafka.Brokers, "topic", cfg.Kafka.Topic)

	kafkaCfg := &kafkalog.Config{
		Brokers:          cfg.Kafka.Brokers,
		Topic:            cfg.Kafka.Topic,
		ClientID:         cfg.Kafka.ClientID,
		LogQueueSize:     cfg.Kafka.LogQueueSize,
		LogBatchSize:     cfg.Kafka.LogBatchSize,
		LogFlushInterval: cfg.Kafka.LogFlushInterval,
		TLSEnabled:       cfg.Kafka.TLSEnabled,
		SASLMechanism:    cfg.Kafka.SASLMechanism,
		SASLUsername:     cfg.Kafka.SASLUsername,
		SASLPassword:     cfg.Kafka.SASLPassword,
		Logger:           log,
		// Flags a batch's underlying LiteLLM_SpendLogs rows for later re-send
		// when the batch is dropped from the in-memory DLQ after a sustained
		// Kafka outage (see kafkalog.Config.FallbackNotifier doc comment).
		FallbackNotifier: litellmDBManager.MarkSpendLogKafkaFallback,
	}

	manager, err := kafkalog.New(kafkaCfg)
	if err != nil {
		if cfg.LiteLLMDB.DisableSpendLogsWrite {
			log.Error("CRITICAL: Failed to initialize Kafka spend-log publisher in Kafka-only mode",
				"error", err,
				"reason", "litellm_db.disable_spend_logs_write=true leaves Kafka as the only spend-log write-path",
				"action", "Fix Kafka connectivity/credentials or re-enable litellm_db spend log writes",
			)
			os.Exit(1)
		}

		log.Warn("Failed to initialize Kafka spend-log publisher, degrading to NoopManager",
			"error", err,
			"impact", "Spend events will not be published to Kafka/ClickHouse; Postgres logging is unaffected",
		)
		return kafkalog.NewNoopManager()
	}
	log.Info("Kafka spend-log publisher initialized successfully")
	return manager
}

// loadAndUpdateModelPrices loads model prices and updates the registry
func loadAndUpdateModelPrices(
	link string,
	registry *models.ModelPriceRegistry,
	log *slog.Logger,
	context string, // "startup" or "update" for logging
) error {
	prices, err := models.LoadModelPrices(link)
	if err != nil {
		logMessage := "Failed to load model prices"
		if context != "" {
			logMessage += " during " + context
		}
		log.Warn(logMessage, "error", err)
		return err
	}
	registry.Update(prices)
	if context == "startup" {
		log.Info("Model prices loaded on startup", "count", len(prices), "link", link)
	} else {
		log.Debug("Model prices updated", "count", len(prices))
	}
	return nil
}

// startPriceSyncLoop starts a background goroutine that periodically syncs model prices
func startPriceSyncLoop(
	modelPricesLink string,
	registry *models.ModelPriceRegistry,
	log *slog.Logger,
	bgCtx context.Context,
	wg *sync.WaitGroup,
) {
	if modelPricesLink == "" {
		return
	}

	wg.Add(1)
	go func() {
		defer wg.Done()

		// Load prices immediately on startup
		_ = loadAndUpdateModelPrices(modelPricesLink, registry, log, "startup")

		// Periodic update loop (every 5 minutes)
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-bgCtx.Done():
				log.Debug("Model prices sync loop stopped")
				return
			case <-ticker.C:
				_ = loadAndUpdateModelPrices(modelPricesLink, registry, log, "update")
			}
		}
	}()

	log.Debug("Model price sync loop started", "interval", "5 minutes", "link", modelPricesLink)
}

func startMetricsUpdater(
	cfg *config.Config,
	log *slog.Logger,
	bgCtx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	metrics *monitoring.Metrics,
	wg *sync.WaitGroup,
	updateMutex *sync.Mutex,
) {
	if !cfg.MetricsCollectionEnabled() {
		return
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-bgCtx.Done():
				return
			case <-ticker.C:
				updateMutex.Lock()
				updateMetrics(bal, rateLimiter, metrics)
				updateMutex.Unlock()
			}
		}
	}()

	log.Info("Metrics updater started (updates every 10 seconds)")
}

// startKafkaMetricsUpdater periodically publishes internal/kafkalog producer
// statistics (queue/DLQ counters, broker health) as Prometheus metrics.
// Without this, kafkalog.Manager.Stats()/IsHealthy() are only reachable from
// tests — this is what makes them observable in production.
func startKafkaMetricsUpdater(
	cfg *config.Config,
	log *slog.Logger,
	bgCtx context.Context,
	kafkaLogManager kafkalog.Manager,
	metrics *monitoring.Metrics,
	wg *sync.WaitGroup,
) {
	if !cfg.MetricsCollectionEnabled() {
		return
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-bgCtx.Done():
				return
			case <-ticker.C:
				stats := kafkaLogManager.Stats()
				metrics.UpdateKafkaSpendLoggerStats(stats.Queued, stats.Produced, stats.Dropped, stats.Errors, stats.DLQSize, stats.Healthy)
			}
		}
	}()

	log.Info("Kafka spend logger metrics updater started (updates every 10 seconds)")
}

func updateMetrics(
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	metrics *monitoring.Metrics,
) {
	credentials := bal.GetCredentialsSnapshot()

	credNames := make([]string, 0, len(credentials))
	for _, cred := range credentials {
		if !bal.IsProxyCredential(cred.Name) {
			credNames = append(credNames, cred.Name)
		}
	}

	modelPairs := rateLimiter.GetAllModelPairs()
	filteredPairs := make([]ratelimit.ModelPair, 0, len(modelPairs))
	for _, p := range modelPairs {
		if !bal.IsProxyCredential(p.Credential) {
			filteredPairs = append(filteredPairs, p)
		}
	}

	credStats, modelStats := rateLimiter.BatchCurrentStats(context.Background(), credNames, filteredPairs)

	for _, name := range credNames {
		cs := credStats[name]
		metrics.UpdateCredentialRPM(name, cs.RPM)
		metrics.UpdateCredentialTPM(name, cs.TPM)
	}

	for _, p := range filteredPairs {
		ms := modelStats[p.Credential+":"+p.Model]
		metrics.UpdateModelRPM(p.Credential, p.Model, ms.RPM)
		metrics.UpdateModelTPM(p.Credential, p.Model, ms.TPM)
	}
}

func startProxyStatsUpdater(
	log *slog.Logger,
	bgCtx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	modelManager *models.Manager,
	wg *sync.WaitGroup,
	updateMutex *sync.Mutex,
) {
	// Run initial update synchronously so proxy model lists are populated before
	// the HTTP server starts accepting requests. Without this, the first few requests
	// arrive before credentialModels is populated, causing HasModel to fall through
	// to the permissive fallback and routing requests to the wrong proxy.
	modelupdate.UpdateAllProxyCredentials(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)

	wg.Add(1)
	go func() {
		defer wg.Done()

		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-bgCtx.Done():
				return
			case <-ticker.C:
				modelupdate.UpdateAllProxyCredentials(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)
			}
		}
	}()

	log.Info("Proxy stats updater started (updates every 30 seconds)")
}

func startDBHealthMonitor(
	log *slog.Logger,
	bgCtx context.Context,
	dbManager litellmdb.Manager,
	healthChecker *health.DBHealthChecker,
	wg *sync.WaitGroup,
) {
	monitorCfg := &health.MonitorConfig{
		CheckInterval:    30 * time.Second,
		FailureThreshold: 3,
		Logger:           log,
	}

	monitor := health.NewMonitor(monitorCfg, healthChecker, dbManager)

	wg.Add(1)
	go func() {
		defer wg.Done()
		monitor.Start(bgCtx)
	}()

	log.Info("LiteLLM DB health monitor started (checks every 30 seconds)")
}

func startResponseStoreCleanup(
	log *slog.Logger,
	bgCtx context.Context,
	store responsestore.Store,
	wg *sync.WaitGroup,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for {
			select {
			case <-bgCtx.Done():
				return
			case <-ticker.C:
				if err := store.CleanupExpired(bgCtx); err != nil {
					log.Warn("Response store cleanup error", "error", err)
				} else {
					log.Debug("Response store cleanup completed")
				}
			}
		}
	}()
	log.Info("Response store cleanup worker started (runs every 1 hour)")
}
