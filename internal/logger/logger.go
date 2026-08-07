package logger

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
)

type callIDContextKey struct{}

const callIDAttribute = "litellm_call_id"

// WithCallID makes the LiteLLM correlation ID available to every contextual
// slog record without requiring each call site to repeat the attribute.
func WithCallID(ctx context.Context, callID string) context.Context {
	if callID == "" {
		return ctx
	}
	return context.WithValue(ctx, callIDContextKey{}, callID)
}

func CallIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	callID, _ := ctx.Value(callIDContextKey{}).(string)
	return callID
}

type correlationHandler struct {
	next slog.Handler
}

func (h *correlationHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.next.Enabled(ctx, level)
}

func (h *correlationHandler) Handle(ctx context.Context, record slog.Record) error {
	if callID := CallIDFromContext(ctx); callID != "" && !recordHasAttribute(record, callIDAttribute) {
		record.AddAttrs(slog.String(callIDAttribute, callID))
	}
	return h.next.Handle(ctx, record)
}

func (h *correlationHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &correlationHandler{next: h.next.WithAttrs(attrs)}
}

func (h *correlationHandler) WithGroup(name string) slog.Handler {
	return &correlationHandler{next: h.next.WithGroup(name)}
}

func recordHasAttribute(record slog.Record, key string) bool {
	found := false
	record.Attrs(func(attr slog.Attr) bool {
		if attr.Key == key {
			found = true
			return false
		}
		return true
	})
	return found
}

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
	return slog.New(&correlationHandler{next: handler})
}

// NewMulti creates a slog.Logger that fans out every record to stdout (pretty
// colored output, when stdout is true) and to the given extra handlers
// (e.g. an OTLP/OTEL bridge handler). The configured level is applied to all
// destinations. Nil extra handlers are skipped.
// With stdout=false and no extra handlers the logger discards everything.
func NewMulti(level string, stdout bool, extra ...slog.Handler) *slog.Logger {
	slogLevel := parseLevel(level)

	handlers := make([]slog.Handler, 0, len(extra)+1)
	if stdout {
		handlers = append(handlers, &PrettyHandler{
			opts: &slog.HandlerOptions{
				Level: slogLevel,
			},
		})
	}
	for _, h := range extra {
		if h != nil {
			handlers = append(handlers, h)
		}
	}

	if len(handlers) == 0 {
		return slog.New(slog.DiscardHandler)
	}
	// Always go through MultiHandler so the level filter applies even to
	// destinations that accept all levels (like the OTEL bridge).
	return slog.New(&correlationHandler{next: &MultiHandler{handlers: handlers, level: slogLevel}})
}

// MultiHandler fans out log records to multiple slog handlers.
// The level filter is applied centrally so destinations that accept all
// levels (like the OTEL bridge) still respect the configured logging_level.
type MultiHandler struct {
	handlers []slog.Handler
	level    slog.Level
}

// Enabled reports whether at least one destination handles the level.
func (m *MultiHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= m.level
}

// Handle dispatches the record to every destination handler.
// Errors are joined so one failing destination doesn't hide the others.
func (m *MultiHandler) Handle(ctx context.Context, record slog.Record) error {
	var errs []error
	for _, h := range m.handlers {
		if h.Enabled(ctx, record.Level) {
			if err := h.Handle(ctx, record.Clone()); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errors.Join(errs...)
}

// WithAttrs returns a new MultiHandler with attributes applied to all destinations.
func (m *MultiHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	handlers := make([]slog.Handler, len(m.handlers))
	for i, h := range m.handlers {
		handlers[i] = h.WithAttrs(attrs)
	}
	return &MultiHandler{handlers: handlers, level: m.level}
}

// WithGroup returns a new MultiHandler with the group applied to all destinations.
func (m *MultiHandler) WithGroup(name string) slog.Handler {
	handlers := make([]slog.Handler, len(m.handlers))
	for i, h := range m.handlers {
		handlers[i] = h.WithGroup(name)
	}
	return &MultiHandler{handlers: handlers, level: m.level}
}

// NewJSON creates a new slog.Logger with JSON output
func NewJSON(level string) *slog.Logger {
	slogLevel := parseLevel(level)

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slogLevel,
	})
	return slog.New(&correlationHandler{next: handler})
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
	opts   *slog.HandlerOptions
	attrs  []slog.Attr // pre-attached attributes from Logger.With(...)
	groups []string    // open groups from Logger.WithGroup(...)
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
	fmt.Fprintf(&sb, "%s%s%s %s[%s]%s » %s",
		colorGray, timeStr, colorReset,
		levelColor, levelStr, colorReset,
		record.Message,
	)

	// Add pre-attached attributes (from Logger.With), already group-prefixed
	for _, attr := range h.attrs {
		fmt.Fprintf(&sb, " %s=%v", attr.Key, attr.Value.Any())
	}

	// Add record attributes, prefixed with any open groups
	prefix := h.groupPrefix()
	record.Attrs(func(attr slog.Attr) bool {
		fmt.Fprintf(&sb, " %s%s=%v", prefix, attr.Key, attr.Value.Any())
		return true
	})

	sb.WriteString("\n")
	_, err := fmt.Fprint(os.Stdout, sb.String())
	return err
}

// groupPrefix returns the dotted prefix for attribute keys ("g1.g2." or "")
func (h *PrettyHandler) groupPrefix() string {
	if len(h.groups) == 0 {
		return ""
	}
	return strings.Join(h.groups, ".") + "."
}

// WithAttrs returns a new handler with the given attributes attached
func (h *PrettyHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	if len(attrs) == 0 {
		return h
	}
	h2 := *h
	prefix := h.groupPrefix()
	h2.attrs = make([]slog.Attr, 0, len(h.attrs)+len(attrs))
	h2.attrs = append(h2.attrs, h.attrs...)
	for _, a := range attrs {
		a.Key = prefix + a.Key
		h2.attrs = append(h2.attrs, a)
	}
	return &h2
}

// WithGroup returns a new handler with the given group name
func (h *PrettyHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	h2 := *h
	h2.groups = make([]string, 0, len(h.groups)+1)
	h2.groups = append(h2.groups, h.groups...)
	h2.groups = append(h2.groups, name)
	return &h2
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
			case "embedding", "b64_json", "content", "values":
				// Truncate known long fields more aggressively.
				if str, ok := value.(string); ok && len(str) > 50 {
					val[key] = fmt.Sprintf("%s... [truncated %d chars]", str[:50], len(str)-50)
				} else if arr, ok := value.([]interface{}); ok && len(arr) > 3 {
					// Arrays (e.g. embedding vectors): show first, ..., last
					val[key] = []interface{}{arr[0], fmt.Sprintf("... [%d more]", len(arr)-2), arr[len(arr)-1]}
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
