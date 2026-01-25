package logger

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_Debug(t *testing.T) {
	logger := New("debug")

	assert.NotNil(t, logger)
	// Logger should be created successfully
	// We can't easily test the level directly, but we can verify it doesn't panic
}

func TestNew_Info(t *testing.T) {
	logger := New("info")

	assert.NotNil(t, logger)
}

func TestNew_Error(t *testing.T) {
	logger := New("error")

	assert.NotNil(t, logger)
}

func TestNew_Default(t *testing.T) {
	// Invalid level should default to info
	logger := New("invalid_level")

	assert.NotNil(t, logger)
}

func TestNew_Empty(t *testing.T) {
	// Empty level should default to info
	logger := New("")

	assert.NotNil(t, logger)
}

func TestNewJSON(t *testing.T) {
	logger := NewJSON("info")

	assert.NotNil(t, logger)
}

func TestNewJSON_Debug(t *testing.T) {
	logger := NewJSON("debug")

	assert.NotNil(t, logger)
}

func TestParseLevel(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected slog.Level
	}{
		{"debug lowercase", "debug", slog.LevelDebug},
		{"debug uppercase", "DEBUG", slog.LevelDebug},
		{"debug mixed", "Debug", slog.LevelDebug},
		{"info lowercase", "info", slog.LevelInfo},
		{"info uppercase", "INFO", slog.LevelInfo},
		{"error lowercase", "error", slog.LevelError},
		{"error uppercase", "ERROR", slog.LevelError},
		{"invalid defaults to info", "warning", slog.LevelInfo},
		{"empty defaults to info", "", slog.LevelInfo},
		{"random defaults to info", "xyz", slog.LevelInfo},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			level := parseLevel(tt.input)
			assert.Equal(t, tt.expected, level)
		})
	}
}

func TestLogger_Usage(t *testing.T) {
	// Test that logger can be used without panics
	logger := New("debug")

	// These should not panic
	logger.Info("test info message")
	logger.Debug("test debug message")
	logger.Error("test error message", "key", "value")

	// Test with different levels
	loggerInfo := New("info")
	loggerInfo.Info("info message")
	loggerInfo.Debug("debug message") // Should not appear but shouldn't panic

	loggerError := New("error")
	loggerError.Error("error message")
	loggerError.Info("info message") // Should not appear but shouldn't panic
}

func TestNewJSON_Usage(t *testing.T) {
	// Test that JSON logger can be used without panics
	logger := NewJSON("info")

	logger.Info("json test message", "key", "value")
	logger.Error("json error message", "error", "test error")
}
