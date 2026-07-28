package logger

import (
	"context"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type captureHandler struct {
	records []slog.Record
}

func (h *captureHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *captureHandler) Handle(_ context.Context, record slog.Record) error {
	h.records = append(h.records, record.Clone())
	return nil
}
func (h *captureHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *captureHandler) WithGroup(string) slog.Handler      { return h }

func TestCorrelationHandlerAddsCallIDFromContext(t *testing.T) {
	capture := &captureHandler{}
	log := slog.New(&correlationHandler{next: capture})

	log.InfoContext(WithCallID(context.Background(), "call-123"), "request")

	require.Len(t, capture.records, 1)
	attrs := map[string]string{}
	capture.records[0].Attrs(func(attr slog.Attr) bool {
		attrs[attr.Key] = attr.Value.String()
		return true
	})
	assert.Equal(t, "call-123", attrs[callIDAttribute])
}
