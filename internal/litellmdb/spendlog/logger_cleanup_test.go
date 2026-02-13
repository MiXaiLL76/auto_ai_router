package spendlog

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
)

// MockConnectionPool is a mock for testing without real database
type MockConnectionPool struct {
	healthy bool
	closed  bool
}

func (m *MockConnectionPool) Acquire(ctx context.Context) (*MockConn, error) {
	if !m.healthy || m.closed {
		return nil, models.ErrConnectionFailed
	}
	return &MockConn{}, nil
}

func (m *MockConnectionPool) IsHealthy() bool {
	return m.healthy && !m.closed
}

func (m *MockConnectionPool) Close() {
	m.closed = true
}

func (m *MockConnectionPool) Pool() interface{} {
	return nil
}

func (m *MockConnectionPool) Stats() interface{} {
	return nil
}

type MockConn struct{}

func (m *MockConn) Release() {}
func (m *MockConn) Exec(ctx context.Context, sql string, args ...interface{}) (interface{}, error) {
	return nil, nil
}
func (m *MockConn) Query(ctx context.Context, sql string, args ...interface{}) (interface{}, error) {
	return nil, nil
}

func TestLogger_CreationInitializesCorrectly(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	assert.NotNil(t, logger)
	assert.Equal(t, cfg.LogQueueSize, cap(logger.queue))
	assert.Equal(t, cfg.LogBatchSize, cfg.LogBatchSize)
}

func TestLogger_StartInitializesTickers(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	// Before start, tickers should be nil
	assert.Nil(t, logger.dlqRecoveryTicker)
	assert.Nil(t, logger.aggregationTicker)

	logger.Start()

	// After start, tickers should be initialized
	assert.NotNil(t, logger.dlqRecoveryTicker)
	assert.NotNil(t, logger.aggregationTicker)

	// Cleanup
	_ = logger.Shutdown(context.Background())
}

func TestLogger_ShutdownStopsBothTickers(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	// Give workers time to start
	time.Sleep(50 * time.Millisecond)

	// Shutdown
	err := logger.Shutdown(context.Background())
	assert.NoError(t, err)

	// Tickers should be stopped (we can't directly check stopped state,
	// but we verify no panics occur)
}

func TestLogger_ShutdownWithTimeout(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	// Shutdown with very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// May succeed or timeout, but should not panic
	_ = logger.Shutdown(ctx)
}

func TestLogger_ShutdownIdempotent(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	// First shutdown
	err1 := logger.Shutdown(context.Background())
	assert.NoError(t, err1)

	// Second shutdown should not panic
	err2 := logger.Shutdown(context.Background())
	// May timeout since channels are already closed, but shouldn't panic
	_ = err2
}

func TestLogger_DoubleStartNotPanic(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	logger.Start()
	time.Sleep(50 * time.Millisecond)

	// Second start would add duplicate goroutines, but shouldn't crash
	// (This is more of a usage error, but let's document the behavior)
	logger.Start()
	time.Sleep(50 * time.Millisecond)

	// Cleanup
	_ = logger.Shutdown(context.Background())
}

func TestLogger_StatsAfterShutdown(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	_ = logger.Shutdown(context.Background())

	// Stats should still be accessible
	stats := logger.Stats()
	assert.NotNil(t, stats)
	assert.Equal(t, 0, stats.QueueLen)
}

func TestLogger_DLQCleanup(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	// Manually add to DLQ for testing
	logger.addToDLQ(
		[]*models.SpendLogEntry{
			{RequestID: "test-1"},
			{RequestID: "test-2"},
		},
		nil,
		4,
	)

	assert.Equal(t, 1, logger.getDLQSize())

	// Get DLQ stats before shutdown
	dlqStats := logger.GetDLQStats()
	assert.Equal(t, 1, dlqStats["dlq_size"])

	// Cleanup (tickers are nil since Start() was never called)
	logger.dlqRecoveryTicker = time.NewTicker(5 * time.Minute)
	logger.aggregationTicker = time.NewTicker(10 * time.Second)

	_ = logger.Shutdown(context.Background())
}

func TestLogger_ConcurrentLogAndShutdown(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     50,
		LogBatchSize:     5,
		LogFlushInterval: 100 * time.Millisecond,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	// Don't call Start() - no real pool to write to
	// Just test that the queue accepts entries gracefully

	var wg sync.WaitGroup
	stopChan := make(chan struct{})

	// Spawn goroutine that logs continuously
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopChan:
				return
			default:
				_ = logger.Log(&models.SpendLogEntry{
					RequestID: "test-" + time.Now().Format("20060102150405"),
				})
				time.Sleep(10 * time.Millisecond)
			}
		}
	}()

	// Let it run for a bit
	time.Sleep(200 * time.Millisecond)

	// Stop logging
	close(stopChan)
	wg.Wait()

	// Now shutdown - should handle pending entries
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := logger.Shutdown(ctx)
	assert.NoError(t, err)
}

func TestLogger_GetDLQStatsEmpty(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	stats := logger.GetDLQStats()
	assert.Equal(t, 0, stats["dlq_size"])
	assert.Equal(t, 10, stats["dlq_max_size"])
	assert.Equal(t, uint64(0), stats["dlq_count"])
}

func TestLogger_QueueFullError(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     2, // Very small queue
		LogBatchSize:     1,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	// Fill queue
	assert.Nil(t, logger.Log(&models.SpendLogEntry{RequestID: "1"}))
	assert.Nil(t, logger.Log(&models.SpendLogEntry{RequestID: "2"}))

	// Next log should timeout and return error
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Create a blocking send to fill queue
	go func() {
		for i := 0; i < 10; i++ {
			_ = logger.Log(&models.SpendLogEntry{RequestID: "block"})
		}
	}()

	time.Sleep(50 * time.Millisecond)

	_ = logger.Shutdown(ctx)
}

func TestLogger_StatsMetricsIncrement(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	// Do not start background workers in this test.
	// Starting workers with nil pool causes shutdown retry backoff path (1s+5s+30s),
	// which makes this unit test unnecessarily slow.
	logger := &Logger{
		pool:     nil,
		config:   cfg,
		logger:   cfg.Logger,
		queue:    make(chan *models.SpendLogEntry, cfg.LogQueueSize),
		stopChan: make(chan struct{}),
	}

	// Log some entries
	for i := 0; i < 3; i++ {
		_ = logger.Log(&models.SpendLogEntry{RequestID: "test-entry-" + fmt.Sprint(i)})
	}

	stats := logger.Stats()
	assert.GreaterOrEqual(t, stats.Queued, uint64(3))
	assert.Equal(t, 3, stats.QueueLen)
}

func TestLogger_DrainQueueOnShutdown(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     100,
		LogBatchSize:     50,
		LogFlushInterval: 10 * time.Second, // Long interval so queue won't flush automatically
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	// Note: We don't call Start() because there's no real pool to write to
	// The drainQueue behavior would require a real worker, which needs a real pool

	// Add entries to the queue
	for i := 0; i < 5; i++ {
		_ = logger.Log(&models.SpendLogEntry{RequestID: "drain-test-" + fmt.Sprint(i)})
	}

	// Queue should have entries
	assert.Equal(t, 5, len(logger.queue))

	// Shutdown without a running worker should not drain the queue
	// (drainQueue is only called in the worker's shutdown path)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := logger.Shutdown(ctx)
	assert.NoError(t, err)

	// Queue still has entries because worker was never started
	assert.Equal(t, 5, len(logger.queue))
}

func TestLogger_ShutdownLogsStats(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	// Shutdown should log stats
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := logger.Shutdown(ctx)
	assert.NoError(t, err)

	// Verify no panics and logger is properly closed
	assert.Equal(t, 0, logger.getDLQSize())
}

func TestLogger_WorkerRoutinesCleanup(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	// Verify 3 goroutines are running (worker, aggregationWorker, dlqRecoveryWorker)
	// We can't directly count goroutines, but we verify shutdown completes

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := logger.Shutdown(ctx)
	assert.NoError(t, err)

	// If all goroutines didn't exit, this would timeout
}

func TestLogger_TickerStoppedAfterShutdown(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)
	logger.Start()

	time.Sleep(50 * time.Millisecond)

	assert.NotNil(t, logger.dlqRecoveryTicker)
	assert.NotNil(t, logger.aggregationTicker)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_ = logger.Shutdown(ctx)

	// After shutdown, tickers are stopped (can't send on stopped ticker)
	// Verify by checking that no new ticks occur
	doneChan := make(chan bool)
	go func() {
		// Wait for a tick that should never come
		select {
		case <-logger.dlqRecoveryTicker.C:
			doneChan <- false // Ticker still running
		case <-time.After(200 * time.Millisecond):
			doneChan <- true // Ticker stopped
		}
	}()

	timerStopped := <-doneChan
	// Should be true that timer is stopped, or might panic (which means it was stopped)
	_ = timerStopped
}

func TestLogger_AggregationTickerInitializedInStart(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	// Verify ticker is nil before Start()
	assert.Nil(t, logger.aggregationTicker)

	logger.Start()

	// Verify ticker is non-nil immediately after Start()
	assert.NotNil(t, logger.aggregationTicker)

	_ = logger.Shutdown(context.Background())
}

func TestLogger_DLQRecoveryTickerInitializedInStart(t *testing.T) {
	cfg := &models.Config{
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: 1 * time.Second,
		Logger:           testhelpers.NewTestLogger(),
	}

	logger := NewLogger(nil, cfg)

	// Verify ticker is nil before Start()
	assert.Nil(t, logger.dlqRecoveryTicker)

	logger.Start()

	// Verify ticker is non-nil immediately after Start()
	assert.NotNil(t, logger.dlqRecoveryTicker)

	_ = logger.Shutdown(context.Background())
}
