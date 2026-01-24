package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
		"logging_level", cfg.Server.LoggingLevel,
		"port", cfg.Server.Port,
		"replace_v1_models", cfg.Server.ReplaceV1Models,
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

	f2b := fail2ban.New(cfg.Fail2Ban.MaxAttempts, cfg.Fail2Ban.BanDuration, cfg.Fail2Ban.ErrorCodes)
	rateLimiter := ratelimit.New()
	bal := balancer.New(cfg.Credentials, f2b, rateLimiter)

	// Initialize model manager and fetch models from credentials
	modelManager := models.New(log, cfg.Server.ReplaceV1Models)
	if cfg.Server.ReplaceV1Models {
		modelManager.FetchModels(cfg.Credentials, cfg.Server.RequestTimeout)
	}

	// Set model checker in balancer for model-aware routing
	bal.SetModelChecker(modelManager)

	metrics := monitoring.New(cfg.Monitoring.PrometheusEnabled)
	prx := proxy.New(bal, log, cfg.Server.MaxBodySizeMB, cfg.Server.RequestTimeout, metrics, cfg.Server.MasterKey)

	// Start background metrics updater
	if cfg.Monitoring.PrometheusEnabled {
		go func() {
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				for _, cred := range bal.GetCredentials() {
					rpm := rateLimiter.GetCurrentRPM(cred.Name)
					metrics.UpdateCredentialRPM(cred.Name, rpm)

					banned := f2b.IsBanned(cred.Name)
					metrics.UpdateCredentialBanStatus(cred.Name, banned)
				}
			}
		}()
		log.Info("Metrics updater started (updates every 10 seconds)")
	}

	rtr := router.New(prx, cfg.Monitoring.HealthCheckPath, modelManager)

	mux := http.NewServeMux()
	mux.Handle("/", rtr)

	if cfg.Monitoring.PrometheusEnabled {
		mux.Handle("/metrics", promhttp.Handler())
		log.Info("Prometheus metrics enabled", "path", "/metrics")
	}

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Server.Port),
		Handler: mux,
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

	// Create context with timeout for graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Attempt graceful shutdown
	if err := server.Shutdown(ctx); err != nil {
		log.Error("Server forced to shutdown", "error", err)
		os.Exit(1)
	}

	log.Info("Server shutdown complete")
}
