package testhelpers

import (
	"io"
	"log/slog"
)

// NewTestLogger creates a logger that discards all output for testing.
// This is used across multiple test files to avoid duplication.
func NewTestLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{
		Level: slog.LevelError,
	}))
}
