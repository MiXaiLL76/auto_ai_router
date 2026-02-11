package logger

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
)

// New creates a new slog.Logger instance with the specified logging level
// Uses a custom pretty formatter with colors
// level can be: "info", "debug", "error"
// Default is "info"
func New(level string) *slog.Logger {
	slogLevel := parseLevel(level)

	handler := &PrettyHandler{
		opts: &slog.HandlerOptions{
			Level: slogLevel,
		},
	}
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

// ANSI color codes
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m" // Error
	colorYellow = "\033[33m" // Warn
	colorGreen  = "\033[32m" // Info
	colorCyan   = "\033[36m" // Debug
	colorGray   = "\033[90m" // Time
	colorBold   = "\033[1m"  // Bold
)

// PrettyHandler is a custom slog handler that formats logs nicely with colors
type PrettyHandler struct {
	opts *slog.HandlerOptions
}

// Handle implements the slog.Handler interface
func (h *PrettyHandler) Handle(ctx context.Context, record slog.Record) error {
	// Get level with color
	levelColor := getLevelColor(record.Level)
	levelStr := strings.ToUpper(record.Level.String())

	// Format time
	timeStr := record.Time.Format("02.01.06 15:04:05")

	// Start building the log line
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s%s%s %s[%s]%s » %s",
		colorGray, timeStr, colorReset,
		levelColor, levelStr, colorReset,
		record.Message,
	))

	// Add attributes
	record.Attrs(func(attr slog.Attr) bool {
		sb.WriteString(fmt.Sprintf(" %s=%v", attr.Key, attr.Value.Any()))
		return true
	})

	sb.WriteString("\n")
	fmt.Fprint(os.Stdout, sb.String())

	return nil
}

// WithAttrs returns a new handler with the given attributes attached
func (h *PrettyHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return h
}

// WithGroup returns a new handler with the given group name
func (h *PrettyHandler) WithGroup(name string) slog.Handler {
	return h
}

// Enabled reports whether the handler handles records at the given level
func (h *PrettyHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= h.opts.Level.Level()
}

// getLevelColor returns the appropriate ANSI color code for a log level
func getLevelColor(level slog.Level) string {
	switch level {
	case slog.LevelError:
		return colorRed + colorBold
	case slog.LevelWarn:
		return colorYellow + colorBold
	case slog.LevelInfo:
		return colorGreen
	case slog.LevelDebug:
		return colorCyan
	default:
		return colorReset
	}
}

// TruncateLongFields truncates long fields in JSON for logging purposes
// This prevents extremely long base64 strings, embeddings, etc. from cluttering logs
func TruncateLongFields(body string, maxFieldLength int) string {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return body // Return as-is if not valid JSON
	}

	truncateValue(data, maxFieldLength)

	truncated, err := json.Marshal(data)
	if err != nil {
		return body // Return original if marshaling fails
	}

	return string(truncated)
}

// truncateValue recursively truncates long string values in a map or slice
func truncateValue(v interface{}, maxLength int) {
	switch val := v.(type) {
	case map[string]interface{}:
		for key, value := range val {
			switch key {
			case "embedding", "b64_json", "content":
				// Truncate known long fields more aggressively
				if str, ok := value.(string); ok && len(str) > 50 {
					val[key] = fmt.Sprintf("%s... [truncated %d chars]", str[:50], len(str)-50)
				}
			case "messages":
				// For messages array, truncate each message content
				if arr, ok := value.([]interface{}); ok {
					for i := range arr {
						truncateValue(arr[i], maxLength)
					}
				}
			default:
				// For other fields, use standard truncation or recurse
				if str, ok := value.(string); ok && len(str) > maxLength {
					val[key] = str[:maxLength] + "... [truncated]"
				} else {
					truncateValue(value, maxLength)
				}
			}
		}
	case []interface{}:
		for _, item := range val {
			truncateValue(item, maxLength)
		}
	}
}
