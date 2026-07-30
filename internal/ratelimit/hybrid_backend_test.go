package ratelimit

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// levelCapturingHandler records the level of every log record it receives,
// so a test can assert on log severity without depending on message text.
type levelCapturingHandler struct {
	levels []slog.Level
}

func (h *levelCapturingHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *levelCapturingHandler) Handle(_ context.Context, r slog.Record) error {
	h.levels = append(h.levels, r.Level)
	return nil
}
func (h *levelCapturingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *levelCapturingHandler) WithGroup(string) slog.Handler      { return h }

// TestRecordRedisError_LogsAtWarnNotError is a regression test: background
// write/sync failures against Redis are an expected, self-healing degraded
// state for this backend (local counting keeps working) — not an
// operator-actionable emergency — so they must log at Warn, not Error. The
// metric (unaffected by log level) is the real alerting signal.
func TestRecordRedisError_LogsAtWarnNotError(t *testing.T) {
	handler := &levelCapturingHandler{}
	h := &HybridBackend{log: slog.New(handler)}

	h.recordRedisError("hybrid_sync", errors.New("boom"))

	require.Len(t, handler.levels, 1)
	assert.Equal(t, slog.LevelWarn, handler.levels[0])
	assert.NotEqual(t, slog.LevelError, handler.levels[0])
}

// TestRecordRedisError_NilMetricsDoesNotPanic confirms recordRedisError stays
// safe when constructed without a *monitoring.Metrics (as some tests do).
func TestRecordRedisError_NilMetricsDoesNotPanic(t *testing.T) {
	h := &HybridBackend{log: slog.New(&levelCapturingHandler{})}
	assert.NotPanics(t, func() {
		h.recordRedisError("hybrid_write", errors.New("boom"))
	})
}
