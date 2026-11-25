package spendlog

import (
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fullQueueLogger has a queue of one, already full, and no worker.
func fullQueueLogger(t *testing.T, timeout *time.Duration) *Logger {
	t.Helper()
	cfg := &models.Config{DatabaseURL: "postgresql://localhost/test", LogQueueSize: 1, LogQueueFullTimeout: timeout}
	cfg.ApplyDefaults()
	sl := &Logger{config: cfg, logger: cfg.Logger, queue: make(chan *models.SpendLogEntry, 1), stopChan: make(chan struct{})}
	require.NoError(t, sl.Log(&models.SpendLogEntry{RequestID: "fills-the-queue"}))
	return sl
}

func droppedFor(reason string) float64 {
	return testutil.ToFloat64(monitoring.SpendDroppedTotal.WithLabelValues(reason))
}

func TestLogDropsAtOnceWhenTheQueueFullTimeoutIsZero(t *testing.T) {
	zero := time.Duration(0)
	sl := fullQueueLogger(t, &zero)
	noWait, timedOut := droppedFor("queue_full_no_wait"), droppedFor("queue_full_timeout")

	start := time.Now()
	err := sl.Log(&models.SpendLogEntry{RequestID: "dropped"})

	assert.ErrorIs(t, err, models.ErrQueueFull)
	assert.Less(t, time.Since(start), 100*time.Millisecond)
	stats := sl.Stats()
	assert.Equal(t, uint64(1), stats.Queued)
	assert.Equal(t, uint64(1), stats.Dropped)
	assert.Equal(t, uint64(1), stats.QueueFullCount)
	assert.Equal(t, 1, stats.PendingEntries, "only the queued entry is pending")
	assert.Equal(t, noWait+1, droppedFor("queue_full_no_wait"))
	assert.Equal(t, timedOut, droppedFor("queue_full_timeout"))
}

func TestLogWaitsForTheQueueFullTimeout(t *testing.T) {
	short := 200 * time.Millisecond
	sl := fullQueueLogger(t, &short)
	noWait, timedOut := droppedFor("queue_full_no_wait"), droppedFor("queue_full_timeout")

	start := time.Now()
	err := sl.Log(&models.SpendLogEntry{RequestID: "dropped"})
	elapsed := time.Since(start)

	assert.ErrorIs(t, err, models.ErrQueueFull)
	assert.GreaterOrEqual(t, elapsed, short)
	assert.Less(t, elapsed, 2*time.Second)
	assert.Equal(t, uint64(1), sl.Stats().Dropped)
	assert.Equal(t, timedOut+1, droppedFor("queue_full_timeout"))
	assert.Equal(t, noWait, droppedFor("queue_full_no_wait"))
}

func TestLogQueuesWhenRoomFreesWithinTheQueueFullTimeout(t *testing.T) {
	long := 5 * time.Second
	sl := fullQueueLogger(t, &long)
	go func() {
		time.Sleep(50 * time.Millisecond)
		<-sl.queue
	}()

	require.NoError(t, sl.Log(&models.SpendLogEntry{RequestID: "queued-after-wait"}))
	assert.Zero(t, sl.Stats().Dropped)
	assert.Equal(t, "queued-after-wait", (<-sl.queue).RequestID)
}

func TestQueueFullTimeoutDefaultsToFiveSeconds(t *testing.T) {
	assert.Equal(t, 5*time.Second, (&models.Config{}).QueueFullTimeout())
	assert.Equal(t, 5*time.Second, (*models.Config)(nil).QueueFullTimeout())
	zero := time.Duration(0)
	assert.Zero(t, (&models.Config{LogQueueFullTimeout: &zero}).QueueFullTimeout())
}
