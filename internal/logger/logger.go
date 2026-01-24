package logger

import (
	"io"
	"log/slog"
	"os"
)

// New creates a new slog.Logger instance
// If debug is true, logs are output to stdout with DEBUG level
// If debug is false, logs are discarded
func New(debug bool) *slog.Logger {
	if debug {
		handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
		return slog.New(handler)
	}

	// Discard logs in non-debug mode
	handler := slog.NewTextHandler(io.Discard, nil)
	return slog.New(handler)
}

// NewJSON creates a new slog.Logger with JSON output
func NewJSON(debug bool) *slog.Logger {
	if debug {
		handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
		return slog.New(handler)
	}

	handler := slog.NewJSONHandler(io.Discard, nil)
	return slog.New(handler)
}
