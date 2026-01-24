package logger

import (
	"log/slog"
	"os"
	"strings"
)

// New creates a new slog.Logger instance with the specified logging level
// level can be: "info", "debug", "error"
// Default is "info"
func New(level string) *slog.Logger {
	slogLevel := parseLevel(level)

	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slogLevel,
	})
	return slog.New(handler)
}

// NewJSON creates a new slog.Logger with JSON output
func NewJSON(level string) *slog.Logger {
	slogLevel := parseLevel(level)

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slogLevel,
	})
	return slog.New(handler)
}

// parseLevel converts string level to slog.Level
func parseLevel(level string) slog.Level {
	switch strings.ToLower(level) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo // Default to info
	}
}
