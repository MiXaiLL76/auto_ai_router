package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/video"
)

type videoRuntime struct {
	handler http.Handler
	workers []*video.Worker
	cleaner *video.Cleaner
	logger  *slog.Logger
}

func initializeVideoOrExit(cfg *config.Config, prx *proxy.Proxy, db litellmdb.Manager, logger *slog.Logger) *videoRuntime {
	if cfg == nil || !cfg.Video.Enabled {
		return nil
	}
	if db == nil || !db.IsEnabled() || db.GetPool() == nil {
		logger.Error("Video requires an enabled LiteLLM database")
		os.Exit(1)
	}
	spendStatus, ok := db.(interface{ SpendLoggingEnabled() bool })
	if !ok || !spendStatus.SpendLoggingEnabled() {
		logger.Error("Video requires synchronous SpendLogs accounting")
		os.Exit(1)
	}
	committer, ok := db.(litellmdb.SpendCommitter)
	if !ok {
		logger.Error("Video requires synchronous spend settlement")
		os.Exit(1)
	}

	store := video.NewPostgresStore(db.GetPool())
	schemaCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	err := store.CheckSchema(schemaCtx)
	cancel()
	if err != nil {
		logger.Error("Video database schema is unavailable or incompatible", "error", err)
		os.Exit(1)
	}

	objects, err := video.NewS3Store(video.S3Config{
		Endpoint:         cfg.Video.S3Endpoint,
		Region:           cfg.Video.S3Region,
		Bucket:           cfg.Video.S3Bucket,
		Prefix:           cfg.Video.S3Prefix,
		AccessKey:        cfg.Video.S3AccessKey,
		SecretKey:        cfg.Video.S3SecretKey,
		MaxArtifactBytes: cfg.Video.MaxArtifactBytes,
		ArtifactProxyURL: cfg.Video.ArtifactProxyURL,
	})
	if err != nil {
		logger.Error("Video object storage initialization failed", "error", err)
		os.Exit(1)
	}

	models := make(map[string]string, len(cfg.Video.Models))
	for _, model := range cfg.Video.Models {
		models[model.Name] = model.ProviderModel
	}
	provider, err := video.NewRunwayClient(video.RunwayConfig{
		BaseURL:    cfg.Video.RunwayBaseURL,
		APIVersion: cfg.Video.RunwayAPIVersion,
		APIKey:     cfg.Video.RunwayAPIKey,
		Models:     models,
	})
	if err != nil {
		logger.Error("Runway initialization failed", "error", err)
		os.Exit(1)
	}

	billing := proxy.NewVideoBilling(committer)
	service, err := video.NewService(video.ServiceConfig{
		Store:     store,
		Objects:   objects,
		Billing:   billing,
		UploadTTL: cfg.Video.UploadTTL,
	})
	if err != nil {
		logger.Error("Video service initialization failed", "error", err)
		os.Exit(1)
	}
	handler, err := video.NewHandler(video.HandlerConfig{
		Service:          service,
		Resolver:         proxy.NewVideoPrincipalResolver(prx),
		UploadSigningKey: []byte(cfg.Video.UploadSigningKey),
	})
	if err != nil {
		logger.Error("Video handler initialization failed", "error", err)
		os.Exit(1)
	}

	hostname, _ := os.Hostname()
	workers := make([]*video.Worker, 0, cfg.Video.WorkerConcurrency)
	for index := 0; index < cfg.Video.WorkerConcurrency; index++ {
		worker, workerErr := video.NewWorker(video.WorkerConfig{
			Store:        store,
			Provider:     provider,
			Objects:      objects,
			Billing:      billing,
			ID:           fmt.Sprintf("%s-%d", hostname, index),
			LeaseTTL:     cfg.Video.LeaseTTL,
			PollInterval: cfg.Video.PollInterval,
		})
		if workerErr != nil {
			logger.Error("Video worker initialization failed", "error", workerErr)
			os.Exit(1)
		}
		workers = append(workers, worker)
	}

	return &videoRuntime{
		handler: handler,
		workers: workers,
		cleaner: video.NewCleaner(store, objects, time.Minute, func(err error) {
			logger.Warn("Video upload cleanup failed", "error", err)
		}),
		logger: logger,
	}
}

func (runtime *videoRuntime) start(ctx context.Context, wg *sync.WaitGroup) {
	if runtime == nil {
		return
	}
	for _, worker := range runtime.workers {
		current := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := current.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
				runtime.logger.Error("Video worker stopped", "error", err)
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := runtime.cleaner.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			runtime.logger.Error("Video upload cleaner stopped", "error", err)
		}
	}()
}
