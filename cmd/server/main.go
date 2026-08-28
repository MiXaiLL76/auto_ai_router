// Package main is the entrypoint for the auto_ai_router proxy server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"runtime"
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
	"github.com/mixaill76/auto_ai_router/internal/requestid"
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

	// Record metrics whenever any sink consumes them — the pull /metrics
	// endpoint (prometheus_enabled) and/or OTLP push (otel.enabled). The pull
	// endpoint and the push pipeline are wired up separately below. Created
	// early so the Redis connection attempt below can record fallback/error
	// metrics.
	metrics := monitoring.New(cfg.MetricsCollectionEnabled())

	// Create a shared Redis/Valkey backend if enabled.
	// The same underlying client is reused by both the rate limiter and the response store.
	var redisBackend *ratelimit.RedisBackend
	if cfg.Redis.Enabled {
		redisBackend = connectRedisWithRetry(cfg.Redis, log, metrics)
		if redisBackend != nil {
			defer redisBackend.Close()
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
				hybridAuthBackend := ratelimit.NewHybridBackend(authRedisBackend, cfg.Redis.SyncInterval, log, metrics)
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
	_, rateLimiter, bal, hybridBackend := initializeBalancer(cfg, log, redisBackend, metrics)
	if hybridBackend != nil {
		defer hybridBackend.Close()
	}
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
	organizationPolicies := loadOrganizationPoliciesOrExit(log, cfg, modelManager)
	if !organizationPolicies.Empty() {
		log.Info("Organization policies loaded", "count", len(cfg.OrganizationPolicies))
	}
	tokenManager := auth.NewVertexTokenManager(log)
	defer tokenManager.Stop()

	// ==================== Initialize Model Pricing ====================
	if cfg.Server.ModelPricesLink != "" {
		log.Info("Using model prices from", "link", cfg.Server.ModelPricesLink, "sync_interval", cfg.Server.ModelPricesSyncInterval.String())
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
		OrganizationPolicies:       organizationPolicies,
		MaxProviderRetries:         cfg.Server.MaxProviderRetries,
		MaxFallbackAttempts:        cfg.Server.MaxFallbackAttempts,
		ResponseStore:              respStore,
		SessionStickyEnabled:       cfg.Server.SessionStickyEnabled,
		SessionStickyAutoCacheCtrl: cfg.Server.SessionStickyAutoCacheCtrl,
		SessionStoreTTL:            time.Duration(cfg.Server.SessionStickyTTL) * time.Minute,
		DrainUpstreamOnAbort:       cfg.Server.DrainUpstreamOnAbort,
		ResponseCompatibility:      cfg.Server.ResponseCompatibility,
		TiktokenEnabled:            cfg.Server.TiktokenEnabled,
		StrictAllTeamModelsACL:     cfg.Server.StrictAllTeamModelsACL,
		ResponseHeaderMode:         cfg.Server.ResponseHeaders.Mode,
		CredentialNameAsTeamID:     cfg.Server.CredentialNameAsTeamID,

		BudgetReserver:                   budgetReserver,
		KeyRateLimiter:                   keyRateLimiter,
		BudgetReservationEnabled:         cfg.LiteLLMDB.EnforceBudgetReservation,
		KeyRateLimitsEnabled:             cfg.LiteLLMDB.EnforceKeyRateLimits,
		DefaultEstimatedCompletionTokens: cfg.LiteLLMDB.DefaultEstimatedCompletionTokens,
	})

	// ==================== Background Goroutines ====================
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()

	prx.Start(bgCtx)

	var wg sync.WaitGroup
	var updateMutex sync.Mutex

	startMetricsUpdater(bgCtx, cfg, log, bal, rateLimiter, metrics, &wg, &updateMutex)
	startProxyStatsUpdater(bgCtx, log, bal, rateLimiter, modelManager, &wg, &updateMutex)
	if kafkaLogManager.IsEnabled() {
		startKafkaMetricsUpdater(bgCtx, cfg, log, kafkaLogManager, metrics, &wg)
	}

	if respStore != nil {
		startResponseStoreCleanup(bgCtx, log, respStore, &wg)
	}

	if litellmDBManager.IsEnabled() {
		startDBHealthMonitor(bgCtx, log, litellmDBManager, healthChecker, &wg)
		if err := litellmDBManager.FetchMasterKey(bgCtx, cfg.Server.MasterKey); err != nil {
			log.Warn("Failed to fetch master key from LiteLLM DB.", "error", err)
		}
		if cfg.LiteLLMDB.LoadLitellmDBModels {
			startDBModelTableSyncLoop(bgCtx, log, litellmDBManager, staticCreds,
				bal, modelManager, rateLimiter, priceRegistry, cfg, cfg.LiteLLMDB.LitellmDBSyncInterval, &wg)
		}
	}

	// Start model price sync loop (only if configured)
	if cfg.Server.ModelPricesLink != "" {
		startPriceSyncLoop(bgCtx, cfg.Server.ModelPricesLink, cfg.Server.ModelPricesSyncInterval, priceRegistry, log, &wg)
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
			otelhttp.WithSpanNameFormatter(func(_ string, r *http.Request) string {
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
	// Must wrap (run before) the otelhttp handler above, not be wrapped by it:
	// the request ID needs to already be in context by the time otelhttp starts
	// the server span, so the telemetry package's IDGenerator can reuse it as
	// the span's trace_id. Also echoes the ID to the client via X-Request-Id,
	// independently of whether tracing is enabled.
	//
	// trustIncomingTraceparent mirrors the otelhttp propagator override above:
	// only trust an inbound traceparent's trace ID as our request ID when
	// otelhttp will actually honor it (tracing enabled AND the config trusts
	// it) — otherwise otelhttp starts a fresh root span and adopting the
	// header's ID here would just be a second, uncorrelated fork.
	trustIncomingTraceparent := otelSDK.TracesEnabled() && cfg.OTEL.TrustIncomingTraceparent
	rootHandler = requestid.Middleware(trustIncomingTraceparent)(rootHandler)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      rootHandler,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Bind port explicitly so readiness is set only after the socket is open.
	ln := bindOrExit(log, "server", cfg.Server.Port)

	// Mark ready — TCP listener is bound, pod can accept traffic.
	rtr.SetReady(true)
	log.Info("Server ready", "port", cfg.Server.Port)

	go func() {
		if err := server.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Error("Server failed", "error", err)
			os.Exit(1)
		}
	}()

	// ==================== pprof (internal only, opt-in) ====================
	// Deliberately on its own listener/mux rather than the public router: pprof
	// exposes heap contents, goroutine stacks, and lets any caller trigger a
	// blocking 30s CPU profile. The Service/Ingress in front of this pod must
	// not route to PprofPort — only reachable via kubectl port-forward or
	// from inside the cluster network.
	var pprofServer *http.Server
	if cfg.Monitoring.PprofEnabled {
		// Off by default (rate 0): only sampled while pprof is explicitly
		// enabled, so this cost is opt-in like the rest of the endpoint.
		// Fraction=1 reports every contended mutex acquisition — cheap because
		// it only fires on actual contention, not on every Lock()/Unlock().
		// This is what lets /debug/pprof/mutex show contention on locks like
		// the balancer's global RWMutex (internal/balancer/roundrobin.go).
		runtime.SetMutexProfileFraction(1)

		// Sample block-profile events at a 10us granularity: fine enough to
		// catch second-scale waits that actually matter (io.Pipe backpressure
		// in streaming, the 5s spend-logger queue timeout, lock waits) without
		// recording every microsecond-scale channel op in normal I/O loops.
		// rate=1 (every event) is too heavy here — unlike mutex contention,
		// almost every goroutine spends most of its life blocked on I/O.
		runtime.SetBlockProfileRate(10000)

		pprofMux := http.NewServeMux()
		pprofMux.HandleFunc("/debug/pprof/", pprof.Index)
		pprofMux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		pprofMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		pprofMux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		pprofMux.HandleFunc("/debug/pprof/trace", pprof.Trace)

		pprofServer = &http.Server{ //nolint:gosec // G112: internal-only debug listener, never exposed via Service/Ingress (see warning log below)
			Addr:    fmt.Sprintf(":%d", cfg.Monitoring.PprofPort),
			Handler: pprofMux,
		}
		pprofLn := bindOrExit(log, "pprof", cfg.Monitoring.PprofPort)
		log.Warn("pprof enabled on internal listener — must not be exposed via Service/Ingress",
			"port", cfg.Monitoring.PprofPort)
		go func() {
			if err := pprofServer.Serve(pprofLn); err != nil && err != http.ErrServerClosed {
				log.Error("pprof server failed", "error", err)
			}
		}()
	}

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
	shutdownOrExit(log, server)

	if pprofServer != nil {
		pprofCtx, pprofCancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := pprofServer.Shutdown(pprofCtx); err != nil {
			log.Error("pprof server forced to shutdown", "error", err)
		}
		pprofCancel()
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

const (
	redisConnectMaxAttempts = 5
	redisConnectBaseDelay   = 2 * time.Second
	redisConnectMaxDelay    = 5 * time.Second
	// redisConnectOverallBound caps the total wall-clock time
	// connectRedisWithRetry can spend before giving up and falling back to
	// local backends. Without this, 5 attempts x up to (5s connect timeout +
	// 5s ping timeout) plus backoff delays could block the HTTP listener for
	// ~75s on a genuine Redis outage (not just a cold-start blip) — long
	// enough to blow past most k8s startup/liveness probe budgets and cause a
	// probe-driven restart loop, the opposite of this function's
	// self-healing intent.
	redisConnectOverallBound = 15 * time.Second
)

// connectRedisWithRetry attempts to establish and health-check a Redis/Valkey
// connection, retrying with exponential backoff to survive transient
// cold-start races (container up before DNS/CNI/Redis is actually reachable).
// Without this, a 1-2s blip at startup silently degrades the process to
// local-only rate limiting/response storage for its entire lifetime, with no
// self-healing until the next restart. Bounded to redisConnectOverallBound
// total wall-clock time (see its doc comment) — returns nil (and falls back
// to local backends) once either that bound or redisConnectMaxAttempts is
// reached, whichever comes first.
// bindOrExit binds the main HTTP port or terminates the process. Deliberately
// its own function, not inlined in main: main has a `defer bgCancel()` for
// graceful shutdown, and os.Exit skips deferred calls — gocritic's
// exitAfterDefer flags any os.Exit in a function that also defers. Here that
// defer hasn't bought anything to skip yet (no background work has produced
// anything worth draining before the listener is even bound), so moving the
// exit into its own defer-free function is the correct fix, not a workaround.
// loadOrganizationPoliciesOrExit is its own defer-free function for the same
// reason as bindOrExit: by this point main holds `defer hybridBackend.Close()`,
// and gocritic's exitAfterDefer flags any os.Exit that would skip it. A failed
// policy load is a fatal startup misconfiguration, before any background work
// worth draining exists.
func loadOrganizationPoliciesOrExit(log *slog.Logger, cfg *config.Config, modelManager *models.Manager) *models.OrganizationPolicyRegistry {
	policies, err := models.LoadOrganizationPolicies(cfg.OrganizationPolicies, modelManager, models.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled:      cfg.LiteLLMDB.Enabled,
		LiteLLMDBRequired:     cfg.LiteLLMDB.IsRequired,
		DisableSpendLogsWrite: cfg.LiteLLMDB.DisableSpendLogsWrite,
	})
	if err != nil {
		log.Error("Failed to load organization policies", "error", err)
		os.Exit(1)
	}
	return policies
}

// poolConns narrows a configured pool size to the int32 pgxpool expects,
// saturating rather than wrapping on a nonsensical value.
func poolConns(v int) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < 0 {
		return 0
	}
	return int32(v)
}

func bindOrExit(log *slog.Logger, what string, port int) net.Listener {
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Error("Failed to bind "+what+" port", "error", err, "port", port)
		os.Exit(1)
	}
	return ln
}

// shutdownOrExit is its own function for the same reason as bindOrExit
// above, not just to bundle the ctx/cancel pair: cancel() is called
// explicitly on both branches instead of deferred, so this function has no
// defer of its own either — a local `defer cancel()` right next to os.Exit
// would trip gocritic's exitAfterDefer exactly as before.
func shutdownOrExit(log *slog.Logger, server *http.Server) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	err := server.Shutdown(ctx)
	cancel()
	if err != nil {
		log.Error("Server forced to shutdown", "error", err)
		os.Exit(1)
	}
}

func connectRedisWithRetry(cfg config.RedisConfig, log *slog.Logger, metrics *monitoring.Metrics) *ratelimit.RedisBackend {
	deadline := time.Now().Add(redisConnectOverallBound)
	delay := redisConnectBaseDelay
	for attempt := 1; attempt <= redisConnectMaxAttempts; attempt++ {
		// Always let attempt 1 run regardless of the deadline — this check
		// only short-circuits *subsequent* attempts once we're already over
		// budget, it never skips the first try.
		if attempt > 1 && time.Now().After(deadline) {
			log.Warn("Redis connect retry budget exhausted, falling back to local backends", "attempt", attempt, "max_attempts", redisConnectMaxAttempts)
			break
		}
		rb, err := ratelimit.NewRedisBackend(cfg)
		if err != nil {
			metrics.RecordRedisConnectionError("connect")
			log.Warn("Failed to connect to Redis, will retry", "error", err, "attempt", attempt, "max_attempts", redisConnectMaxAttempts)
		} else {
			// Verify Redis is responsive with a health check ping.
			// Use explicit cancel (not defer) so the context is released immediately.
			pingCtx, pingCancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := rb.Ping(pingCtx)
			pingCancel()
			if pingErr == nil {
				log.Info("Connected to Redis/Valkey", "addresses", cfg.InitAddresses, "attempt", attempt)
				return rb
			}
			metrics.RecordRedisConnectionError("ping")
			log.Warn("Redis health check failed, will retry", "error", pingErr, "attempt", attempt, "max_attempts", redisConnectMaxAttempts)
			rb.Close()
		}

		if attempt < redisConnectMaxAttempts {
			sleepFor := delay
			if remaining := time.Until(deadline); remaining < sleepFor {
				sleepFor = remaining // don't oversleep past the overall bound
			}
			if sleepFor > 0 {
				time.Sleep(sleepFor)
			}
			delay = min(delay*2, redisConnectMaxDelay)
		}
	}

	log.Error("Failed to connect to Redis after retries, falling back to local backends", "attempts", redisConnectMaxAttempts)
	metrics.RecordRedisFallback()
	return nil
}

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

// initializeBalancer builds the fail2ban tracker, rate limiter, and balancer.
// The returned *ratelimit.HybridBackend (nil unless cfg.Redis.Hybrid is set
// with a non-nil redisBackend) must be Close()'d by the caller at process
// shutdown — NOT from a defer inside this function, which would close it
// (and kill its background writeWorker/syncWorker) within microseconds of
// construction, before the server ever serves a request.
func initializeBalancer(
	cfg *config.Config,
	log *slog.Logger,
	redisBackend *ratelimit.RedisBackend,
	metrics *monitoring.Metrics,
) (*fail2ban.Fail2Ban, *ratelimit.RPMLimiter, *balancer.RoundRobin, *ratelimit.HybridBackend) {
	rules := convertFailBanRules(cfg.Fail2Ban.ErrorCodeRules, cfg.Fail2Ban.BanDuration, log)
	f2b := fail2ban.NewWithRules(cfg.Fail2Ban.MaxAttempts, cfg.Fail2Ban.BanDuration,
		cfg.Fail2Ban.ErrorCodes, rules)
	f2b.SetLogger(log)
	if len(cfg.Fail2Ban.CredentialOverrides) > 0 {
		overrides := make(map[string]fail2ban.CredentialOverride, len(cfg.Fail2Ban.CredentialOverrides))
		for name, override := range cfg.Fail2Ban.CredentialOverrides {
			overrides[name] = fail2ban.CredentialOverride{
				ErrorCodes:     override.ErrorCodes,
				ErrorCodeRules: convertFailBanRules(override.ErrorCodeRules, cfg.Fail2Ban.BanDuration, log),
			}
		}
		f2b.SetCredentialOverrides(overrides)
	}

	var rateLimiter *ratelimit.RPMLimiter
	var hybridBackend *ratelimit.HybridBackend
	if redisBackend != nil {
		if cfg.Redis.Hybrid {
			log.Info("Rate limiter: using hybrid backend (local decisions, async Redis sync)",
				"sync_interval", cfg.Redis.SyncInterval)
			hybridBackend = ratelimit.NewHybridBackend(redisBackend, cfg.Redis.SyncInterval, log, metrics)
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

	return f2b, rateLimiter, bal, hybridBackend
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
	// nil preserves legacy discovery; an explicit empty list denies all client IDs.
	if cfg.ClientModelIDs != nil {
		if len(cfg.ClientModelIDs) == 0 {
			log.Warn("client_model_ids is explicitly empty: client model surface is deny-all")
		}
		modelManager.SetClientModelIDs(cfg.ClientModelIDs)
	}
	if len(cfg.PublicModelAlias) > 0 {
		modelManager.SetPublicModelAliases(cfg.PublicModelAlias)
	}
	if len(cfg.AcceptedModelAlias) > 0 {
		modelManager.SetAcceptedModelAliases(cfg.AcceptedModelAlias)
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
	bgCtx context.Context,
	log *slog.Logger,
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
		DatabaseURL:                 cfg.LiteLLMDB.DatabaseURL,
		MaxConns:                    poolConns(cfg.LiteLLMDB.MaxConns),
		MinConns:                    poolConns(cfg.LiteLLMDB.MinConns),
		HealthCheckInterval:         cfg.LiteLLMDB.HealthCheckInterval,
		ConnectTimeout:              cfg.LiteLLMDB.ConnectTimeout,
		AuthCacheTTL:                cfg.LiteLLMDB.AuthCacheTTL,
		AuthCacheSize:               cfg.LiteLLMDB.AuthCacheSize,
		LogQueueSize:                cfg.LiteLLMDB.LogQueueSize,
		LogBatchSize:                cfg.LiteLLMDB.LogBatchSize,
		LogFlushInterval:            cfg.LiteLLMDB.LogFlushInterval,
		LogWorkers:                  cfg.LiteLLMDB.LogWorkers,
		DisableSpendLogsWrite:       cfg.LiteLLMDB.DisableSpendLogsWrite,
		IncludeTeamSpendInUserSpend: &cfg.LiteLLMDB.IncludeTeamSpendInUserSpend,
		Logger:                      log,
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
		LogWorkers:       cfg.Kafka.LogWorkers,
		TLSEnabled:       cfg.Kafka.TLSEnabled,
		SASLMechanism:    cfg.Kafka.SASLMechanism,
		SASLUsername:     cfg.Kafka.SASLUsername,
		SASLPassword:     cfg.Kafka.SASLPassword,
		TLSCACert:        cfg.Kafka.TLSCACert,
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
	bgCtx context.Context,
	modelPricesLink string,
	syncInterval time.Duration,
	registry *models.ModelPriceRegistry,
	log *slog.Logger,
	wg *sync.WaitGroup,
) {
	if modelPricesLink == "" {
		return
	}
	if syncInterval <= 0 {
		syncInterval = config.DefaultModelPricesSyncInterval
	}

	wg.Add(1)
	go func() {
		defer wg.Done()

		// Load prices immediately on startup
		_ = loadAndUpdateModelPrices(modelPricesLink, registry, log, "startup")

		// Periodic update loop (every server.model_prices_sync_interval)
		ticker := time.NewTicker(syncInterval)
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

	log.Debug("Model price sync loop started", "interval", syncInterval.String(), "link", modelPricesLink)
}

func startMetricsUpdater(
	bgCtx context.Context,
	cfg *config.Config,
	log *slog.Logger,
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
	bgCtx context.Context,
	cfg *config.Config,
	log *slog.Logger,
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
		metrics.UpdateCredentialBanStatus(name, bal.HasAnyBan(name))
	}

	for _, p := range filteredPairs {
		ms := modelStats[p.Credential+":"+p.Model]
		metrics.UpdateModelRPM(p.Credential, p.Model, ms.RPM)
		metrics.UpdateModelTPM(p.Credential, p.Model, ms.TPM)
	}
}

func startProxyStatsUpdater(
	bgCtx context.Context,
	log *slog.Logger,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	modelManager *models.Manager,
	wg *sync.WaitGroup,
	updateMutex *sync.Mutex,
) {
	var startupWG sync.WaitGroup
	startupWG.Add(2)
	go func() {
		defer startupWG.Done()
		modelupdate.UpdateAllProxyCredentials(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)
	}()
	go func() {
		defer startupWG.Done()
		updateFromRemoteHealth(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)
	}()
	startupWG.Wait()

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
				updateFromRemoteHealth(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)
				// Unrelated to the two calls above beyond sharing this tick: piggybacked
				// here rather than a dedicated ticker since a proxy/AIR credential's
				// health-learned priority (updated just above) is exactly what makes
				// r.swrr's SWRR-cycle-per-(priority,membership) keys churn over time — see
				// PruneStaleSWRRState's doc comment for why this map is otherwise unbounded.
				bal.PruneStaleSWRRState(10 * time.Minute)
			}
		}
	}()

	log.Info("Proxy stats updater started (updates every 30 seconds)")
}

func updateFromRemoteHealth(
	ctx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	log *slog.Logger,
	modelManager *models.Manager,
	updateMutex *sync.Mutex,
) {
	proxy.UpdateAllFromRemoteHealth(ctx, bal, rateLimiter, log, modelManager, updateMutex)
}

func startDBHealthMonitor(
	bgCtx context.Context,
	log *slog.Logger,
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
	bgCtx context.Context,
	log *slog.Logger,
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
