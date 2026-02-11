package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/health"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/modelupdate"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/router"
	"github.com/mixaill76/auto_ai_router/internal/startup"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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

	log := logger.New(cfg.Server.LoggingLevel)

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
	_, rateLimiter, bal := initializeBalancer(cfg, log)
	modelManager := initializeModelManager(log, cfg, rateLimiter, bal)
	tokenManager := auth.NewVertexTokenManager(log)
	defer tokenManager.Stop()

	litellmDBManager := initializeLiteLLMDB(cfg, log)
	metrics := monitoring.New(cfg.Monitoring.PrometheusEnabled)

	// ==================== Initialize Model Pricing ====================
	priceRegistry := models.NewModelPriceRegistry()
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

	// ==================== Create Proxy ====================
	prx := proxy.New(&proxy.Config{
		Balancer:               bal,
		Logger:                 log,
		MaxBodySizeMB:          cfg.Server.MaxBodySizeMB,
		ResponseBodyMultiplier: cfg.Server.ResponseBodyMultiplier,
		RequestTimeout:         cfg.Server.RequestTimeout,
		MaxIdleConns:           cfg.Server.MaxIdleConns,
		MaxIdleConnsPerHost:    cfg.Server.MaxIdleConnsPerHost,
		IdleConnTimeout:        cfg.Server.IdleConnTimeout,
		Metrics:                metrics,
		MasterKey:              cfg.Server.MasterKey,
		RateLimiter:            rateLimiter,
		TokenManager:           tokenManager,
		ModelManager:           modelManager,
		Version:                Version,
		Commit:                 Commit,
		LiteLLMDB:              litellmDBManager,
		HealthChecker:          healthChecker,
		PriceRegistry:          priceRegistry,
	})

	// ==================== Background Goroutines ====================
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()

	var wg sync.WaitGroup
	var updateMutex sync.Mutex

	startMetricsUpdater(cfg, log, bgCtx, bal, rateLimiter, metrics, &wg, &updateMutex)
	startProxyStatsUpdater(log, bgCtx, bal, rateLimiter, modelManager, &wg, &updateMutex)

	if litellmDBManager.IsEnabled() {
		startDBHealthMonitor(log, bgCtx, litellmDBManager, healthChecker, &wg)
	}

	// Start model price sync loop (only if configured)
	if cfg.Server.ModelPricesLink != "" {
		startPriceSyncLoop(cfg.Server.ModelPricesLink, priceRegistry, log, bgCtx, &wg)
	}

	// ==================== HTTP Server Setup ====================
	rtr := router.New(prx, modelManager, &cfg.Monitoring, log)
	mux := http.NewServeMux()
	mux.Handle("/", rtr)

	if cfg.Monitoring.PrometheusEnabled {
		mux.Handle("/metrics", promhttp.Handler())
		log.Info("Prometheus metrics enabled", "path", "/metrics")
	}

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      mux,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Start server in background
	go func() {
		log.Info("Server starting", "port", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("Server failed", "error", err)
			os.Exit(1)
		}
	}()

	// ==================== Signal Handling & Graceful Shutdown ====================
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Info("Shutting down server...")

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

	if err := router.CloseErrorLogFiles(); err != nil {
		log.Error("Failed to close error log files", "error", err)
	}

	log.Info("Server shutdown complete")
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
) (*fail2ban.Fail2Ban, *ratelimit.RPMLimiter, *balancer.RoundRobin) {
	rules := convertFailBanRules(cfg.Fail2Ban.ErrorCodeRules, cfg.Fail2Ban.BanDuration, log)
	f2b := fail2ban.NewWithRules(cfg.Fail2Ban.MaxAttempts, cfg.Fail2Ban.BanDuration,
		cfg.Fail2Ban.ErrorCodes, rules)

	rateLimiter := ratelimit.New()
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
		if rule.BanDuration != "" && rule.BanDuration != "permanent" {
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

	// Initialize rate limiters for each model
	modelsResp := modelManager.GetAllModels()
	for _, cred := range cfg.Credentials {
		for _, model := range modelsResp.Data {
			if modelManager.HasModel(cred.Name, model.ID) {
				rpm := modelManager.GetModelRPMForCredential(model.ID, cred.Name)
				tpm := modelManager.GetModelTPMForCredential(model.ID, cred.Name)
				rateLimiter.AddModelWithTPM(cred.Name, model.ID, rpm, tpm)
				log.Debug("Initialized model rate limiters",
					"credential", cred.Name,
					"model", model.ID,
					"rpm", rpm,
					"tpm", tpm,
				)
			}
		}
	}

	bal.SetModelChecker(modelManager)
	return modelManager
}

func initializeLiteLLMDB(cfg *config.Config, log *slog.Logger) litellmdb.Manager {
	if !cfg.LiteLLMDB.Enabled {
		log.Info("LiteLLM DB integration disabled - using NoopManager (no security checks)")
		return litellmdb.NewNoopManager()
	}

	log.Info("Initializing LiteLLM DB integration...", "is_required", cfg.LiteLLMDB.IsRequired)

	litellmCfg := &litellmdb.Config{
		DatabaseURL:         cfg.LiteLLMDB.DatabaseURL,
		MaxConns:            int32(cfg.LiteLLMDB.MaxConns),
		MinConns:            int32(cfg.LiteLLMDB.MinConns),
		HealthCheckInterval: cfg.LiteLLMDB.HealthCheckInterval,
		ConnectTimeout:      cfg.LiteLLMDB.ConnectTimeout,
		AuthCacheTTL:        cfg.LiteLLMDB.AuthCacheTTL,
		AuthCacheSize:       cfg.LiteLLMDB.AuthCacheSize,
		LogQueueSize:        cfg.LiteLLMDB.LogQueueSize,
		LogBatchSize:        cfg.LiteLLMDB.LogBatchSize,
		LogFlushInterval:    cfg.LiteLLMDB.LogFlushInterval,
		LogRetryAttempts:    cfg.LiteLLMDB.LogRetryAttempts,
		LogRetryDelay:       cfg.LiteLLMDB.LogRetryDelay,
		Logger:              log,
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
	if !cfg.Monitoring.PrometheusEnabled {
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

func updateMetrics(
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	metrics *monitoring.Metrics,
) {
	credentials := bal.GetCredentialsSnapshot()

	for _, cred := range credentials {
		if bal.IsProxyCredential(cred.Name) {
			continue
		}

		metrics.UpdateCredentialRPM(cred.Name, rateLimiter.GetCurrentRPM(cred.Name))
		metrics.UpdateCredentialTPM(cred.Name, rateLimiter.GetCurrentTPM(cred.Name))
	}

	// Update model metrics
	for _, key := range rateLimiter.GetAllModels() {
		parts := modelupdate.SplitCredentialModel(key)
		if len(parts) != 2 || bal.IsProxyCredential(parts[0]) {
			continue
		}

		metrics.UpdateModelRPM(parts[0], parts[1], rateLimiter.GetCurrentModelRPM(parts[0], parts[1]))
		metrics.UpdateModelTPM(parts[0], parts[1], rateLimiter.GetCurrentModelTPM(parts[0], parts[1]))
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
	wg.Add(1)
	go func() {
		defer wg.Done()

		// Update immediately on startup
		modelupdate.UpdateAllProxyCredentials(bgCtx, bal, rateLimiter, log, modelManager, updateMutex)

		// Then update periodically
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
