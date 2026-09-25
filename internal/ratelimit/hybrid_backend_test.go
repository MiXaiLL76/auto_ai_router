package ratelimit

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

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

func newSyncTestBackend(keys ...string) (*HybridBackend, *levelCapturingHandler) {
	handler := &levelCapturingHandler{}
	h := &HybridBackend{
		log:         slog.New(handler),
		tracked:     make(map[string]struct{}),
		remoteStats: make(map[string][2]int),
		remoteSync:  make(map[string]*remoteSyncState),
	}
	for _, k := range keys {
		h.tracked[k] = struct{}{}
	}
	return h, handler
}

func okRound(key string, total, local [2]int) syncResult {
	return syncResult{total: map[string][2]int{key: total}, local: map[string][2]int{key: local}}
}

// A failed read comes back as zero; writing it made every instance treat the
// others as idle and admit up to the full limit on its own.
func TestApplySync_KeepsEstimateWhenKeyReadFails(t *testing.T) {
	key := "m:grant:claude-opus-4.6"
	h, _ := newSyncTestBackend(key)
	t0 := time.Now()

	h.applySync([]string{key}, okRound(key, [2]int{7, 700}, [2]int{2, 200}), t0)
	rpm, tpm := h.remoteFor(key)
	require.Equal(t, 5, rpm)
	require.Equal(t, 500, tpm)

	h.applySync([]string{key}, syncResult{
		total:  map[string][2]int{key: {0, 0}},
		local:  map[string][2]int{key: {2, 200}},
		failed: map[string]bool{key: true},
		err:    errors.New("redis: connection refused"),
	}, t0.Add(10*time.Second))
	rpm, _ = h.remoteFor(key)
	assert.Equal(t, 5, rpm, "a failed read must keep the last good estimate")
	assert.Equal(t, 2, h.effectiveRPMLimit(key, 7))

	h.applySync([]string{key}, syncResult{timedOut: true}, t0.Add(20*time.Second))
	rpm, _ = h.remoteFor(key)
	assert.Equal(t, 5, rpm, "a timed-out round must keep the last good estimate")
}

// One failing key must not freeze or later wipe the estimates of healthy keys.
func TestApplySync_PartialFailureUpdatesHealthyKeys(t *testing.T) {
	bad, good := "m:a:x", "m:b:y"
	h, _ := newSyncTestBackend(bad, good)
	t0 := time.Now()
	keys := []string{bad, good}

	h.applySync(keys, syncResult{
		total: map[string][2]int{bad: {6, 0}, good: {4, 0}},
		local: map[string][2]int{bad: {1, 0}, good: {1, 0}},
	}, t0)

	for i := 1; i <= 3; i++ {
		now := t0.Add(time.Duration(i) * 30 * time.Second)
		h.applySync(keys, syncResult{
			total:  map[string][2]int{bad: {0, 0}, good: {9, 0}},
			local:  map[string][2]int{bad: {1, 0}, good: {1, 0}},
			failed: map[string]bool{bad: true},
			err:    errors.New("ERR script error"),
		}, now)
	}
	rpm, _ := h.remoteFor(good)
	assert.Equal(t, 8, rpm, "healthy keys keep refreshing while another key fails")
	rpm, _ = h.remoteFor(bad)
	assert.Equal(t, 0, rpm, "the failing key's estimate is dropped after one window")
}

// Writes dropped during an outage are never re-sent, so the first totals after
// recovery underestimate the others; the last good estimate is a floor until it
// is one window old.
func TestApplySync_FloorAfterRecovery(t *testing.T) {
	key := "m:grant:claude-opus-4.6"
	h, _ := newSyncTestBackend(key)
	t0 := time.Now()

	h.applySync([]string{key}, okRound(key, [2]int{7, 0}, [2]int{2, 0}), t0)
	h.applySync([]string{key}, syncResult{timedOut: true}, t0.Add(10*time.Second))

	h.applySync([]string{key}, okRound(key, [2]int{2, 0}, [2]int{2, 0}), t0.Add(20*time.Second))
	rpm, _ := h.remoteFor(key)
	assert.Equal(t, 5, rpm, "first totals after recovery must not drop below the last good estimate")

	h.applySync([]string{key}, okRound(key, [2]int{3, 0}, [2]int{2, 0}), t0.Add(rpmWindow+time.Second))
	rpm, _ = h.remoteFor(key)
	assert.Equal(t, 1, rpm, "the floor expires one window after the estimate was measured")
}

func TestApplySync_DropsStaleEstimateAndWarns(t *testing.T) {
	key := "m:grant:claude-opus-4.6"
	h, handler := newSyncTestBackend(key)
	t0 := time.Now()

	h.applySync([]string{key}, okRound(key, [2]int{7, 700}, [2]int{2, 200}), t0)
	h.applySync([]string{key}, syncResult{timedOut: true}, t0.Add(rpmWindow+time.Second))

	rpm, tpm := h.remoteFor(key)
	assert.Equal(t, 0, rpm)
	assert.Equal(t, 0, tpm)
	assert.Contains(t, handler.levels, slog.LevelWarn, "dropping the estimate must be visible in logs")
}

// A round that started before deleteKey must not bring the key's estimate back.
func TestApplySync_SkipsKeysDeletedDuringRound(t *testing.T) {
	key := "m:proxy:model-x"
	h, _ := newSyncTestBackend() // key already removed from tracked by deleteKey
	h.applySync([]string{key}, okRound(key, [2]int{40, 0}, [2]int{0, 0}), time.Now())

	h.remoteMu.RLock()
	_, ok := h.remoteStats[key]
	h.remoteMu.RUnlock()
	assert.False(t, ok)
}
