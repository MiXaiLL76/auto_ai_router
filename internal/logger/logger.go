package logger

import (
	"encoding/json"
	"fmt"
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
