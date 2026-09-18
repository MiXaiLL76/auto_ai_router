// Command migrate applies the video database migrations.
package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/video"
)

func main() {
	if err := run(); err != nil {
		slog.Error("Video database migration failed", "error", err)
		os.Exit(1)
	}
	slog.Info("Video database migrations applied")
}

func run() error {
	databaseURL := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	if databaseURL == "" {
		return errors.New("DATABASE_URL is required")
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	return video.MigrateDatabase(ctx, databaseURL)
}
