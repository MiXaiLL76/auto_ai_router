package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
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
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/router"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	Version = "dev"
	Commit  = "unknown"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	log := logger.New(cfg.Server.LoggingLevel)

	// Startup info (INFO level)
	log.Info("Starting auto_ai_router",
		"version", Version,
		"commit", Commit,
		"logging_level", cfg.Server.LoggingLevel,
		"port", cfg.Server.Port,
	)

	// Log loaded credentials (INFO level)
	log.Info("Loaded credentials", "count", len(cfg.Credentials))
	for i, cred := range cfg.Credentials {
		log.Info("Credential configured",
			"index", i+1,
			"name", cred.Name,
			"type", cred.Type,
			"base_url", cred.BaseURL,
			"rpm", cred.RPM,
		)
	}

	// Convert config rules to fail2ban rules
	var rules []fail2ban.ErrorCodeRule
	for _, rule := range cfg.Fail2Ban.ErrorCodeRules {
		// Parse ban_duration from string
		var banDuration time.Duration
		if rule.BanDuration == "permanent" || rule.BanDuration == "" {
			banDuration = 0 // permanent
		} else {
			var err error
			banDuration, err = time.ParseDuration(rule.BanDuration)
			if err != nil {
				log.Error("Invalid ban_duration in error_code_rules", "error_code", rule.Code, "error", err)
				banDuration = cfg.Fail2Ban.BanDuration
			}
		}

		rules = append(rules, fail2ban.ErrorCodeRule{
			Code:        rule.Code,
			MaxAttempts: rule.MaxAttempts,
			BanDuration: banDuration,
		})
	}

	f2b := fail2ban.NewWithRules(cfg.Fail2Ban.MaxAttempts, cfg.Fail2Ban.BanDuration, cfg.Fail2Ban.ErrorCodes, rules)
	rateLimiter := ratelimit.New()
	bal := balancer.New(cfg.Credentials, f2b, rateLimiter)
	bal.SetLogger(log)

	// Initialize model manager with static models from config.yaml
	modelManager := models.New(log, cfg.Server.DefaultModelsRPM, cfg.Models)

	// Load credential-specific models from config
	modelManager.LoadModelsFromConfig(cfg.Credentials)

	// Set credentials for fetching remote models from proxies
	modelManager.SetCredentials(cfg.Credentials)

	// Initialize model RPM and TPM limiters for each (credential, model) pair
	modelsResp := modelManager.GetAllModels()
	if len(modelsResp.Data) > 0 {
		for _, cred := range cfg.Credentials {
			for _, model := range modelsResp.Data {
				// Only add model if it's available for this credential
				hasModel := modelManager.HasModel(cred.Name, model.ID)
				log.Debug("Checking model availability",
					"credential", cred.Name,
					"model", model.ID,
					"has_model", hasModel,
				)
				if hasModel {
					modelRPM := modelManager.GetModelRPMForCredential(model.ID, cred.Name)
					modelTPM := modelManager.GetModelTPMForCredential(model.ID, cred.Name)
					rateLimiter.AddModelWithTPM(cred.Name, model.ID, modelRPM, modelTPM)
					log.Debug("Initialized model rate limiters",
						"credential", cred.Name,
						"model", model.ID,
						"rpm", modelRPM,
						"tpm", modelTPM,
					)
				}
			}
		}
	}

	// Set model checker in balancer for model-aware routing
	bal.SetModelChecker(modelManager)

	// Create Vertex AI token manager
	tokenManager := auth.NewVertexTokenManager(log)
	log.Info("Vertex AI token manager initialized")
	defer tokenManager.Stop()

	metrics := monitoring.New(cfg.Monitoring.PrometheusEnabled)
	prx := proxy.New(&proxy.Config{
		Balancer:       bal,
		Logger:         log,
		MaxBodySizeMB:  cfg.Server.MaxBodySizeMB,
		RequestTimeout: cfg.Server.RequestTimeout,
		Metrics:        metrics,
		MasterKey:      cfg.Server.MasterKey,
		RateLimiter:    rateLimiter,
		TokenManager:   tokenManager,
		ModelManager:   modelManager,
		Version:        Version,
		Commit:         Commit,
	})

	// Start background metrics updater
	var metricsCancel context.CancelFunc
	var updateMutex sync.Mutex // Synchronize metrics and proxy stats updates
	if cfg.Monitoring.PrometheusEnabled {
		metricsCtx, cancel := context.WithCancel(context.Background())
		metricsCancel = cancel
		go func() {
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-metricsCtx.Done():
					return
				case <-ticker.C:
					updateMutex.Lock()
					credentials := bal.GetCredentialsSnapshot()

					// Update credential metrics (exclude proxy type credentials)
					// Proxy credentials are internal forwarding nodes, not real providers
					for _, cred := range credentials {
						// Skip proxy credentials - they are monitored via remote /health endpoint
						if bal.IsProxyCredential(cred.Name) {
							continue
						}

						rpm := rateLimiter.GetCurrentRPM(cred.Name)
						metrics.UpdateCredentialRPM(cred.Name, rpm)

						tpm := rateLimiter.GetCurrentTPM(cred.Name)
						metrics.UpdateCredentialTPM(cred.Name, tpm)

						banned := f2b.IsBanned(cred.Name)
						metrics.UpdateCredentialBanStatus(cred.Name, banned)
					}

					// Update model RPM/TPM metrics (exclude proxy credentials)
					// GetAllModels() returns keys in format "credential:model"
					for _, key := range rateLimiter.GetAllModels() {
						// Parse credential:model format
						parts := splitCredentialModel(key)
						if len(parts) == 2 {
							credentialName := parts[0]
							modelName := parts[1]

							// Skip models for proxy credentials
							if bal.IsProxyCredential(credentialName) {
								continue
							}

							modelRPM := rateLimiter.GetCurrentModelRPM(credentialName, modelName)
							metrics.UpdateModelRPM(credentialName, modelName, modelRPM)

							modelTPM := rateLimiter.GetCurrentModelTPM(credentialName, modelName)
							metrics.UpdateModelTPM(credentialName, modelName, modelTPM)
						}
					}
					updateMutex.Unlock()
				}
			}
		}()
		log.Info("Metrics updater started (updates every 10 seconds)")
	}

	// Start background proxy stats updater
	// Fetches RPM/TPM limits from remote proxy /health endpoint
	go func() {
		// Update immediately on startup
		updateAllProxyCredentials(bal, rateLimiter, log, modelManager, &updateMutex)

		// Then update periodically with staggered timing
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			updateAllProxyCredentials(bal, rateLimiter, log, modelManager, &updateMutex)
		}
	}()
	log.Info("Proxy stats updater started (updates every 30 seconds)")

	rtr := router.New(prx, modelManager, &cfg.Monitoring)

	mux := http.NewServeMux()
	mux.Handle("/", rtr)

	if cfg.Monitoring.PrometheusEnabled {
		mux.Handle("/metrics", promhttp.Handler())
		log.Info("Prometheus metrics enabled", "path", "/metrics")
	}

	// Calculate server timeouts based on request timeout
	var readTimeout, writeTimeout, idleTimeout time.Duration

	if cfg.Server.RequestTimeout > 0 {
		// ReadTimeout: time to read request from client (usually fast)
		readTimeout = 60 * time.Second
		// WriteTimeout: request_timeout + 50% buffer for response writing
		writeTimeout = time.Duration(float64(cfg.Server.RequestTimeout) * 1.5)
		// IdleTimeout: 2x WriteTimeout for keep-alive connections
		idleTimeout = writeTimeout * 2
	} else {
		// If request timeout is disabled (-1), use reasonable defaults
		readTimeout = 60 * time.Second
		writeTimeout = 10 * time.Minute
		idleTimeout = 20 * time.Minute
	}

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      mux,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
		IdleTimeout:  idleTimeout,
	}

	go func() {
		log.Info("Server starting", "port", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("Server failed", "error", err)
			os.Exit(1)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Info("Shutting down server...")

	// Cancel metrics updater goroutine
	if metricsCancel != nil {
		metricsCancel()
		log.Info("Metrics updater stopped")
	}

	// Create context with timeout for graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Attempt graceful shutdown
	if err := server.Shutdown(ctx); err != nil {
		log.Error("Server forced to shutdown", "error", err)
		os.Exit(1)
	}

	if err := router.CloseErrorLogFiles(); err != nil {
		log.Error("Failed to close error log files", "error", err)
	}

	log.Info("Server shutdown complete")
}

// updateAllProxyCredentials updates all proxy credentials with staggered timing
// to avoid thundering herd problem when multiple proxies are updated simultaneously
func updateAllProxyCredentials(bal *balancer.RoundRobin, rateLimiter *ratelimit.RPMLimiter, log *slog.Logger, modelManager *models.Manager, updateMutex *sync.Mutex) {
	credentials := bal.GetCredentialsSnapshot()
	proxyCredentials := make([]config.CredentialConfig, 0, len(credentials))
	for _, cred := range credentials {
		if cred.Type == config.ProviderTypeProxy {
			proxyCredentials = append(proxyCredentials, cred)
		}
	}

	if len(proxyCredentials) == 0 {
		return
	}

	// Stagger updates: distribute them evenly across the update interval
	// For each proxy, calculate a small delay to spread requests
	staggerDelay := 100 * time.Millisecond // Small delay between each proxy update
	updateTimeout := 5 * time.Second
	for index, cred := range proxyCredentials {
		// Create a copy of credential for closure (avoid loop variable capture)
		credCopy := cred

		if index > 0 {
			time.Sleep(staggerDelay * time.Duration(index))
		}

		ctx, cancel := context.WithTimeout(context.Background(), updateTimeout)
		health, err := proxy.FetchHealthFromRemoteProxy(ctx, &credCopy, log)
		cancel()
		if err != nil {
			continue
		}

		updateMutex.Lock()
		proxy.UpdateStatsFromHealth(health, &credCopy, rateLimiter, log, modelManager)
		updateMutex.Unlock()
	}
}

// splitCredentialModel splits "credential:model" format into [credential, model]
func splitCredentialModel(key string) []string {
	return strings.SplitN(key, ":", 2)
}
