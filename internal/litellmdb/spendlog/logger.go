package spendlog

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// deadLetterBatch represents a batch that failed to insert after all retries
type deadLetterBatch struct {
	batch     []*models.SpendLogEntry
	failedAt  time.Time
	lastError error
	attempts  int
}

var (
	ErrDrainIncomplete = errors.New("spend logger drain incomplete")
	ErrLoggerStopped   = errors.New("spend logger is stopping")
)

// Logger is an asynchronous logger for LiteLLM_SpendLogs table
//
// Features:
// - Non-blocking: Log() returns immediately
// - Batching: collects entries and does batch INSERT
// - Graceful shutdown: waits for all logs to be written
// - Retry: retries on database errors with exponential backoff
// - Dead Letter Queue: retains failed batches and fail-open queue overflow in a bounded in-memory queue
// - DLQ Recovery: periodically retries failed batches from DLQ
// - Backpressure: blocking producers report terminal timeout; fail-open producers spill to the DLQ
// - Daily aggregation: aggregates logs into LiteLLM_DailyUserSpend
type Logger struct {
	pool   *connection.ConnectionPool
	logger *slog.Logger
	config *models.Config

	// Queue
	queue chan *models.SpendLogEntry

	// Lifecycle
	stopChan     chan struct{}
	workerDone   chan struct{}
	shutdownDone chan struct{}
	wg           sync.WaitGroup
	lifecycleMu  sync.Mutex
	started      bool
	stopping     bool
	doneOnce     sync.Once
	startOnce    sync.Once // Ensure Start() is called only once
	acceptStop   chan struct{}
	enqueueWG    sync.WaitGroup
	writeCtx     context.Context
	writeCancel  context.CancelFunc
	syncCtx      context.Context
	syncCancel   context.CancelFunc

	// Metrics
	queued            uint64 // Total queued
	written           uint64 // Successfully written
	dropped           uint64 // Dropped (queue full - timeout reached)
	errors            uint64 // Write errors
	batchesOK         uint64 // Successful batches
	queueFullCount    uint64 // Queue full events (timeouts)
	dlqCount          uint64 // Batches sent to DLQ
	dlqRecovered      uint64 // Batches recovered from DLQ
	dlqOverflow       uint64 // Batches dropped due to DLQ full
	aggregationCount  uint64 // Aggregations completed
	aggregationErrors uint64 // Aggregation errors
	duplicates        uint64 // Rows ignored by ON CONFLICT

	comparisonEligible   uint64 // Inserted rows eligible for full comparison
	comparisonIneligible uint64 // Inserted rows excluded from full comparison

	// Accepted input remains pending while it is buffered locally, being written
	// atomically, or retained in the DLQ. There is no separate daily queue: raw,
	// counters, and daily aggregates share one transaction.
	pendingEntries int64

	// Dead Letter Queue (in-memory circular buffer)
	dlqMu               sync.Mutex
	dlq                 []*deadLetterBatch // Max 10 failed batches
	dlqRecoveryTicker   *time.Ticker       // Periodic DLQ recovery (5 minutes)
	lastDLQRecoveryTime time.Time

	mu                  sync.RWMutex
	lastAggregationTime time.Time
}

// NewLogger creates a new asynchronous logger
func NewLogger(pool *connection.ConnectionPool, cfg *models.Config) *Logger {
	cfg.ApplyDefaults()
	writeCtx, writeCancel := context.WithCancel(context.Background())
	syncCtx, syncCancel := context.WithCancel(context.Background())

	sl := &Logger{
		pool:         pool,
		config:       cfg,
		logger:       cfg.Logger,
		queue:        make(chan *models.SpendLogEntry, cfg.LogQueueSize),
		stopChan:     make(chan struct{}),
		workerDone:   make(chan struct{}),
		shutdownDone: make(chan struct{}),
		acceptStop:   make(chan struct{}),
		writeCtx:     writeCtx,
		writeCancel:  writeCancel,
		syncCtx:      syncCtx,
		syncCancel:   syncCancel,
	}
	sl.publishSnapshot()

	return sl
}

// Start starts the background writer and DLQ recovery worker.
// Must be called once after creation. Safe to call multiple times (idempotent).
func (sl *Logger) Start() {
	sl.startOnce.Do(func() {
		sl.lifecycleMu.Lock()
		defer sl.lifecycleMu.Unlock()
		if sl.stopping {
			return
		}

		// Initialize tickers BEFORE starting goroutines to prevent nil dereference race
		sl.dlqRecoveryTicker = time.NewTicker(5 * time.Minute)

		sl.wg.Add(2)
		sl.started = true
		go sl.worker()
		go sl.dlqRecoveryWorker()
		go func() {
			sl.wg.Wait()
			sl.doneOnce.Do(func() { close(sl.shutdownDone) })
		}()
		sl.logger.Info("[DB] SpendLogger started",
			"queue_size", sl.config.LogQueueSize,
			"batch_size", sl.config.LogBatchSize,
			"flush_interval", sl.config.LogFlushInterval,
			"dlq_max_size", 10,
			"dlq_recovery_interval", "5m",
		)
	})
}

// Log adds an entry to the queue with backpressure handling
// BLOCKING: Waits up to 5 seconds for queue space if full
// Returns ErrQueueFull if timeout reached (entry not queued)
// This preserves all spend entries with slight latency impact on API calls
// If queue has space, returns immediately
func (sl *Logger) Log(entry *models.SpendLogEntry) error {
	if entry == nil {
		return nil
	}
	if !sl.beginEnqueue() {
		return ErrLoggerStopped
	}
	defer sl.enqueueWG.Done()

	if sl.tryEnqueueNow(entry) {
		return nil
	}

	// Queue was full, now attempt blocking send with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	atomic.AddInt64(&sl.pendingEntries, 1)

	select {
	case sl.queue <- entry:
		atomic.AddUint64(&sl.queued, 1)
		sl.publishSnapshot()
		sl.logger.Debug("[DB] SpendLog entry queued after backpressure",
			"request_id", entry.RequestID,
			"queue_len", len(sl.queue),
		)
		return nil

	case <-ctx.Done():
		atomic.AddInt64(&sl.pendingEntries, -1)
		// Timeout reached - queue still full after 5 seconds
		sl.recordQueueDrop(entry, "queue_full_timeout")
		return models.ErrQueueFull

	case <-sl.acceptStop:
		atomic.AddInt64(&sl.pendingEntries, -1)
		return ErrLoggerStopped
	}
}

// TryLog performs the fail-open spend enqueue. It never waits for queue space;
// callers can return the provider response without spend DB backpressure. A
// full writer queue is not itself data loss: the exact entry is retained in the
// bounded DLQ and recovered by the same idempotent atomic writer. Only a later
// DLQ overflow is a terminal loss and invalidates the comparison window.
func (sl *Logger) TryLog(entry *models.SpendLogEntry) error {
	if entry == nil {
		return nil
	}
	if !sl.beginEnqueue() {
		return ErrLoggerStopped
	}
	defer sl.enqueueWG.Done()

	if sl.tryEnqueueNow(entry) {
		return nil
	}

	// tryEnqueueNow rolled back its pending reservation when the channel was
	// full. Reserve the entry again before exposing it through the DLQ so a
	// concurrent recovery cannot resolve an entry that was never counted.
	atomic.AddInt64(&sl.pendingEntries, 1)
	atomic.AddUint64(&sl.queued, 1)
	atomic.AddUint64(&sl.queueFullCount, 1)
	sl.addToDLQ([]*models.SpendLogEntry{entry}, models.ErrQueueFull, 0)
	sl.logger.Warn("[DB] SpendLog writer queue full; entry retained in DLQ",
		"request_id", entry.RequestID,
		"queue_len", len(sl.queue),
		"queue_cap", cap(sl.queue),
	)
	return nil
}

// beginEnqueue gives an asynchronous producer a lifecycle ticket.
func (sl *Logger) beginEnqueue() bool {
	return sl.beginOperation()
}

// beginOperation admits a database or queue operation while the logger is
// running. Shutdown closes this barrier under the same mutex and waits for all
// admitted operations before the worker drains and the owning sink closes its
// connection pool.
func (sl *Logger) beginOperation() bool {
	sl.lifecycleMu.Lock()
	defer sl.lifecycleMu.Unlock()
	if sl.stopping {
		return false
	}
	sl.enqueueWG.Add(1)
	return true
}

func (sl *Logger) tryEnqueueNow(entry *models.SpendLogEntry) bool {
	// Reserve before publishing to the channel so a fast worker cannot resolve
	// the entry before pendingEntries observes it.
	atomic.AddInt64(&sl.pendingEntries, 1)
	select {
	case sl.queue <- entry:
		atomic.AddUint64(&sl.queued, 1)
		sl.publishSnapshot()
		return true
	default:
		atomic.AddInt64(&sl.pendingEntries, -1)
		return false
	}
}

func (sl *Logger) recordQueueDrop(entry *models.SpendLogEntry, reason string) {
	atomic.AddUint64(&sl.dropped, 1)
	atomic.AddUint64(&sl.queueFullCount, 1)
	monitoring.RecordSpendDropped(1)
	sl.publishSnapshot()
	sl.logger.Error("[DB] SpendLog entry dropped: queue full",
		"request_id", entry.RequestID,
		"queue_len", len(sl.queue),
		"queue_cap", cap(sl.queue),
		"reason", reason,
	)
}

// Shutdown stops the logger and waits for all logs to be written
// Idempotent: safe to call multiple times
func (sl *Logger) Shutdown(ctx context.Context) error {
	sl.lifecycleMu.Lock()
	if !sl.stopping {
		sl.stopping = true
		// Synchronous response-path reads/commits must release their lifecycle
		// ticket as soon as shutdown starts. The asynchronous writer has a
		// separate context so already accepted queue entries can still drain.
		if sl.syncCancel != nil {
			sl.syncCancel()
		}
		sl.logger.Info("[DB] SpendLogger shutting down...",
			"pending", len(sl.queue),
		)

		if sl.dlqRecoveryTicker != nil {
			sl.dlqRecoveryTicker.Stop()
			select {
			case <-sl.dlqRecoveryTicker.C:
			default:
			}
		}
		close(sl.acceptStop)
		started := sl.started
		go sl.finishShutdown(started)
	}
	done := sl.shutdownDone
	sl.lifecycleMu.Unlock()

	select {
	case <-done:
	case <-ctx.Done():
		if sl.writeCancel != nil {
			sl.writeCancel()
		}
		// Retain shutdownDone for a later caller. The owning PostgresSink must not
		// close the pool until a subsequent call observes terminal completion.
		return ctx.Err()
	}
	if sl.writeCancel != nil {
		sl.writeCancel()
	}

	drainErr := sl.drainError()
	sl.logger.Info("[DB] SpendLogger shutdown complete",
		"written", atomic.LoadUint64(&sl.written),
		"dropped", atomic.LoadUint64(&sl.dropped),
		"errors", atomic.LoadUint64(&sl.errors),
		"dlq_size", sl.getDLQSize(),
		"dlq_recovered", atomic.LoadUint64(&sl.dlqRecovered),
		"drained", drainErr == nil,
	)
	return drainErr
}

func (sl *Logger) finishShutdown(started bool) {
	// No new Add can race this Wait: Shutdown set stopping while holding the
	// same lifecycle mutex used by beginOperation before starting this goroutine.
	sl.enqueueWG.Wait()
	close(sl.stopChan)
	if !started {
		sl.doneOnce.Do(func() { close(sl.shutdownDone) })
	}
}

func (sl *Logger) drainError() error {
	stats := sl.snapshot()
	if stats.QueueLen == 0 && stats.PendingEntries == 0 && stats.DLQSize == 0 {
		return nil
	}
	return fmt.Errorf("%w: queue=%d pending=%d dlq=%d", ErrDrainIncomplete, stats.QueueLen, stats.PendingEntries, stats.DLQSize)
}

func (sl *Logger) snapshot() models.SpendLoggerStats {
	sl.mu.RLock()
	lastAgg := sl.lastAggregationTime
	sl.mu.RUnlock()

	sl.dlqMu.Lock()
	dlqSize := len(sl.dlq)
	sl.dlqMu.Unlock()

	stats := models.SpendLoggerStats{
		QueueLen:                   len(sl.queue),
		QueueCap:                   cap(sl.queue),
		PendingEntries:             int(atomic.LoadInt64(&sl.pendingEntries)),
		PendingAggregation:         0,
		DLQSize:                    dlqSize,
		Queued:                     atomic.LoadUint64(&sl.queued),
		Written:                    atomic.LoadUint64(&sl.written),
		Dropped:                    atomic.LoadUint64(&sl.dropped),
		Errors:                     atomic.LoadUint64(&sl.errors),
		BatchesOK:                  atomic.LoadUint64(&sl.batchesOK),
		QueueFullCount:             atomic.LoadUint64(&sl.queueFullCount),
		DLQCount:                   atomic.LoadUint64(&sl.dlqCount),
		DLQRecovered:               atomic.LoadUint64(&sl.dlqRecovered),
		DLQOverflow:                atomic.LoadUint64(&sl.dlqOverflow),
		Duplicates:                 atomic.LoadUint64(&sl.duplicates),
		AggregationCount:           atomic.LoadUint64(&sl.aggregationCount),
		AggregationErrors:          atomic.LoadUint64(&sl.aggregationErrors),
		PendingAggregationOverflow: 0,
		ComparisonEligible:         atomic.LoadUint64(&sl.comparisonEligible),
		ComparisonIneligible:       atomic.LoadUint64(&sl.comparisonIneligible),
		LastAggregationTime:        lastAgg,
		AggregationLag:             0,
	}
	stats.ComparisonWindowValid = stats.QueueLen == 0 &&
		stats.PendingEntries == 0 &&
		stats.PendingAggregation == 0 &&
		stats.DLQSize == 0 &&
		stats.Dropped == 0 &&
		stats.DLQOverflow == 0 &&
		stats.AggregationErrors == 0 &&
		stats.PendingAggregationOverflow == 0
	return stats
}

func (sl *Logger) publishSnapshot() {
	observeSnapshot(sl.snapshot())
}

func observeSnapshot(stats models.SpendLoggerStats) {
	monitoring.ObserveSpendSnapshot(monitoring.SpendSnapshot{
		QueueDepth:            stats.QueueLen,
		PendingEntries:        stats.PendingEntries,
		PendingAggregation:    stats.PendingAggregation,
		DLQSize:               stats.DLQSize,
		AggregationLag:        stats.AggregationLag,
		ComparisonWindowValid: stats.ComparisonWindowValid,
	})
}

// Stats returns logger statistics and refreshes instantaneous Prometheus gauges.
func (sl *Logger) Stats() models.SpendLoggerStats {
	stats := sl.snapshot()
	observeSnapshot(stats)
	return stats
}

// GetDLQStats returns dead letter queue statistics
func (sl *Logger) GetDLQStats() map[string]interface{} {
	sl.dlqMu.Lock()
	dlqSize := len(sl.dlq)
	dlqData := make([]map[string]interface{}, 0, dlqSize)
	for _, dlb := range sl.dlq {
		errorMsg := ""
		if dlb.lastError != nil {
			errorMsg = dlb.lastError.Error()
		}
		dlqData = append(dlqData, map[string]interface{}{
			"batch_size": len(dlb.batch),
			"failed_at":  dlb.failedAt,
			"attempts":   dlb.attempts,
			"last_error": errorMsg,
		})
	}
	sl.dlqMu.Unlock()

	sl.mu.RLock()
	lastRecovery := sl.lastDLQRecoveryTime
	sl.mu.RUnlock()

	return map[string]interface{}{
		"dlq_size":      dlqSize,
		"dlq_max_size":  10,
		"dlq_count":     atomic.LoadUint64(&sl.dlqCount),
		"dlq_recovered": atomic.LoadUint64(&sl.dlqRecovered),
		"dlq_overflow":  atomic.LoadUint64(&sl.dlqOverflow),
		"dlq_entries":   dlqData,
		"last_recovery": lastRecovery,
	}
}

// worker is the background goroutine that processes the queue
func (sl *Logger) worker() {
	defer sl.wg.Done()
	defer close(sl.workerDone)

	batch := make([]*models.SpendLogEntry, 0, sl.config.LogBatchSize)
	ticker := time.NewTicker(sl.config.LogFlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-sl.stopChan:
			// Shutdown: write remaining entries
			sl.drainQueue(&batch)
			if len(batch) > 0 {
				sl.flushBatch(batch)
			}
			return

		case entry := <-sl.queue:
			batch = append(batch, entry)
			sl.publishSnapshot()
			// Check batch size
			if len(batch) >= sl.config.LogBatchSize {
				sl.flushBatch(batch)
				batch = batch[:0] // Reset slice, keep capacity
			}

		case <-ticker.C:
			// Timer: write accumulated entries
			if len(batch) > 0 {
				sl.flushBatch(batch)
				batch = batch[:0]
			}
		}
	}
}

// drainQueue reads all remaining entries from the queue
func (sl *Logger) drainQueue(batch *[]*models.SpendLogEntry) {
	for {
		select {
		case entry := <-sl.queue:
			*batch = append(*batch, entry)
		default:
			return
		}
	}
}

// flushBatch writes a batch to the database with retry and DLQ fallback
// Retry strategy with exponential backoff:
// - Attempt 1: Immediate (0s)
// - Attempt 2: After 1s backoff
// - Attempt 3: After 5s backoff
// - Attempt 4: After 30s backoff
// - If all attempts fail: Move to Dead Letter Queue
func (sl *Logger) flushBatch(batch []*models.SpendLogEntry) {
	if len(batch) == 0 {
		return
	}

	const maxAttempts = 4
	backoffDurations := []time.Duration{0, 1 * time.Second, 5 * time.Second, 30 * time.Second}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Apply exponential backoff before attempt (except first)
		if attempt > 0 {
			// Jitter (up to +50%): without it, every pod retries on the exact
			// same fixed schedule, so two batches that just deadlocked on the
			// same rows (see sortedKeys in spend_updater.go) retry in lockstep
			// and can deadlock again. litellm's db_spend_update_writer.py hits
			// the same problem and randomizes retry sleep for this reason.
			backoff := backoffDurations[attempt]
			// Desynchronize replicas after a transient lock conflict. Fixed retry
			// intervals can make the same batches collide again in lockstep.
			backoff += time.Duration(rand.Int64N(int64(backoff)/2 + 1))
			sl.logger.Debug("[DB] SpendLog batch retry backoff",
				"attempt", attempt+1,
				"backoff_ms", backoff.Milliseconds(),
				"batch_size", len(batch),
			)
			// Use select so shutdown can interrupt retry sleep
			select {
			case <-time.After(backoff):
				// Backoff elapsed, proceed with retry
			case <-sl.stopChan:
				// Shutdown during backoff: attempt one immediate final write before giving up.
				// This handles schema-detection retries (e.g. missing column detected on attempt 1,
				// flag already updated — the next write would use the correct query).
				sl.logger.Debug("[DB] SpendLog shutdown during retry backoff, attempting final flush",
					"attempt", attempt+1,
					"batch_size", len(batch),
				)
				if finalErr := sl.flushBatchWithSpendUpdate(batch); finalErr == nil {
					atomic.AddUint64(&sl.batchesOK, 1)
					return
				} else {
					lastErr = finalErr
				}
				sl.logger.Warn("[DB] SpendLog batch retry interrupted by shutdown",
					"attempt", attempt+1,
					"batch_size", len(batch),
					"error", lastErr,
				)
				atomic.AddUint64(&sl.errors, uint64(len(batch)))
				sl.addToDLQ(batch, lastErr, attempt)
				return
			}
		}

		err := sl.flushBatchWithSpendUpdate(batch)
		if err == nil {
			atomic.AddUint64(&sl.batchesOK, 1)
			sl.logger.Debug("[DB] SpendLog batch written with spend updates",
				"count", len(batch),
				"attempt", attempt+1,
			)
			return
		}

		lastErr = err
		sl.logger.Warn("[DB] SpendLog batch insert and spend update failed",
			"attempt", attempt+1,
			"max_attempts", maxAttempts,
			"batch_size", len(batch),
			"error", err,
		)
	}

	// All attempts exhausted: send to Dead Letter Queue
	atomic.AddUint64(&sl.errors, uint64(len(batch)))
	sl.addToDLQ(batch, lastErr, maxAttempts)
}

// flushBatchWithSpendUpdate writes the complete LiteLLM accounting projection
// atomically: raw SpendLogs, entity counters, and all five daily tables. A retry
// can therefore safely replay the whole batch without leaving partial state.
func (sl *Logger) flushBatchWithSpendUpdate(batch []*models.SpendLogEntry) error {
	// Skip if pool is not initialized (e.g., in tests)
	if sl.pool == nil {
		return models.ErrConnectionFailed
	}

	if !sl.pool.IsHealthy() {
		return models.ErrConnectionFailed
	}

	writeCtx := sl.writeCtx
	if writeCtx == nil {
		writeCtx = context.Background()
	}
	ctx, cancel := context.WithTimeout(writeCtx, 3*time.Minute)
	defer cancel()

	conn, err := sl.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire connection: %w", err)
	}
	defer conn.Release()

	// Collision recovery claims a conflicting row's ownership in a statement
	// after the preferred-ID INSERT. Pin READ COMMITTED so that statement can
	// observe a concurrent transaction whose ON CONFLICT row just won and
	// committed.
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}

	insertedIDs, err := sl.commitBatchTransaction(ctx, tx, batch)
	if err != nil {
		return err
	}

	sl.recordCommittedBatch(batch, insertedIDs)
	if len(insertedIDs) > 0 {
		sl.recordAggregationSuccess()
	}
	sl.publishSnapshot()
	return nil
}

const transactionRollbackTimeout = 250 * time.Millisecond

// rollbackTransaction uses an independent bounded context. In particular, a
// response deadline must not prevent PostgreSQL from cleaning up a failed
// pre-commit attempt. pgx treats Rollback as a no-op after a successful commit.
func rollbackTransaction(parent context.Context, tx pgx.Tx) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), transactionRollbackTimeout)
	defer cancel()
	_ = tx.Rollback(ctx)
}

// commitBatchTransaction owns the transaction lifecycle. Rollback is always
// attempted; pgx treats it as a no-op after a successful commit.
func (sl *Logger) commitBatchTransaction(ctx context.Context, tx pgx.Tx, batch []*models.SpendLogEntry) ([]string, error) {
	defer func() {
		rollbackTransaction(ctx, tx)
	}()

	insertedIDs, err := sl.writeBatchInTransaction(ctx, tx, batch)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, sl.handleCommitError(err)
	}
	return insertedIDs, nil
}

// writeBatchInTransaction performs every accounting mutation on tx. Only rows
// produced by INSERT ... RETURNING are eligible for counters or daily tables;
// conflicts are a completed no-op and cannot double-charge on replay.
func (sl *Logger) writeBatchInTransaction(ctx context.Context, tx pgx.Tx, batch []*models.SpendLogEntry) ([]string, error) {
	insertedIDs, err := insertSpendRowsCollisionSafe(ctx, tx, batch)
	if err != nil {
		return nil, err
	}

	if len(insertedIDs) == 0 {
		return nil, nil
	}

	filteredBatch := filterBatchByInsertedIDs(batch, insertedIDs)
	if len(filteredBatch) != len(insertedIDs) {
		return nil, fmt.Errorf("map inserted rows to batch: expected %d, mapped %d", len(insertedIDs), len(filteredBatch))
	}
	spendUpdates := aggregateSpendUpdates(insertedEntries(filteredBatch))
	if err := executeSpendUpdates(ctx, tx, spendUpdates); err != nil {
		return nil, fmt.Errorf("spend updates: %w", err)
	}

	// Daily aggregates are built from the in-memory entries. Re-reading the
	// just-inserted rows from LiteLLM_SpendLogs is a heavy query on the LiteLLM
	// schema, and everything it returned is already known to the writer.
	records, err := buildSpendLogRecords(filteredBatch, sl.logger, "atomic")
	if err != nil {
		return nil, fmt.Errorf("build daily aggregation records: %w", err)
	}
	if err := sl.runAggregators(ctx, tx, "atomic", records); err != nil {
		return nil, err
	}

	return insertedIDs, nil
}

// addToDLQ adds a failed batch to the dead letter queue
// Max queue size: 10 failed batches (~100KB max memory)
// If DLQ is full, drops the oldest batch and logs error
func (sl *Logger) addToDLQ(batch []*models.SpendLogEntry, lastErr error, attempts int) {
	sl.dlqMu.Lock()

	dlb := &deadLetterBatch{
		// worker() reuses its batch backing array after flushBatch returns.
		// Retain an owned slice so later queue traffic cannot rewrite the DLQ.
		batch:     append([]*models.SpendLogEntry(nil), batch...),
		failedAt:  utils.NowUTC(),
		lastError: lastErr,
		attempts:  attempts,
	}

	// DLQ is a circular buffer with max 10 batches
	if len(sl.dlq) >= 10 {
		// DLQ overflow: drop oldest batch
		dropped := sl.dlq[0]
		sl.dlq = sl.dlq[1:]
		atomic.AddUint64(&sl.dlqOverflow, 1)
		monitoring.RecordSpendDLQOverflow(1)
		sl.resolvePendingEntries(len(dropped.batch))

		sl.logger.Error("[DB] SpendLog DLQ overflow - batch dropped (billing data loss)",
			"dropped_records", len(dropped.batch),
			"dropped_at", dropped.failedAt,
			"dlq_size", len(sl.dlq),
			"total_dropped_batches", atomic.LoadUint64(&sl.dlqOverflow),
			"sample_request_ids", getSampleRequestIDs(dropped.batch, 3),
			"reason", "dlq_full",
		)
	}

	sl.dlq = append(sl.dlq, dlb)
	atomic.AddUint64(&sl.dlqCount, 1)
	dlqSize := len(sl.dlq)
	sl.dlqMu.Unlock()
	sl.publishSnapshot()

	// Log batch details
	sl.logger.Error("[DB] SpendLog batch sent to Dead Letter Queue",
		"batch_size", len(batch),
		"dlq_size", dlqSize,
		"failed_at", dlb.failedAt,
		"last_error", lastErr,
		"attempts", attempts,
		"sample_request_ids", getSampleRequestIDs(batch, 3),
	)
}

// getDLQSize returns the current size of the dead letter queue
func (sl *Logger) getDLQSize() int {
	sl.dlqMu.Lock()
	defer sl.dlqMu.Unlock()
	return len(sl.dlq)
}

// dlqRecoveryWorker periodically retries failed batches from the DLQ
// Runs every 5 minutes, uses same retry logic as normal batches
func (sl *Logger) dlqRecoveryWorker() {
	defer sl.wg.Done()

	for {
		select {
		case <-sl.stopChan:
			sl.finalizeDLQ()
			return

		case <-sl.dlqRecoveryTicker.C:
			// Priority check: if stopChan is also ready, prefer shutdown
			select {
			case <-sl.stopChan:
				sl.finalizeDLQ()
				return
			default:
			}
			sl.flushDLQ()
		}
	}
}

func (sl *Logger) finalizeDLQ() {
	// The writer is the only producer of new DLQ batches. Wait for its final
	// queue drain before making the terminal recovery attempt.
	<-sl.workerDone
	sl.flushDLQ()
}

// flushDLQ attempts to recover batches from the dead letter queue
// Tries to insert each batch, removes successful ones
// If DLQ grows beyond 5 entries, logs warning alert
func (sl *Logger) flushDLQ() {
	sl.dlqMu.Lock()
	if len(sl.dlq) == 0 {
		sl.dlqMu.Unlock()
		return
	}

	// Alert if DLQ is growing too large
	if len(sl.dlq) >= 5 {
		sl.logger.Error("[DB] SpendLog DLQ size alert",
			"dlq_size", len(sl.dlq),
			"dlq_max_size", 10,
			"total_batches_at_risk", countEntriesInDLQ(sl.dlq),
		)
	}

	// Process batches (retry in order)
	recovered := 0
	failed := 0
	failedBatches := make([]*deadLetterBatch, 0, len(sl.dlq))

	// Copy DLQ under lock and clear original to avoid race with addToDLQ
	dlqCopy := make([]*deadLetterBatch, len(sl.dlq))
	copy(dlqCopy, sl.dlq)
	sl.dlq = sl.dlq[:0] // Clear original; failed batches will be re-added below
	sl.dlqMu.Unlock()

	// Try to insert each batch
	for _, dlb := range dlqCopy {
		err := sl.flushBatchWithSpendUpdate(dlb.batch)
		if err == nil {
			// Batch recovered successfully
			atomic.AddUint64(&sl.batchesOK, 1)
			atomic.AddUint64(&sl.dlqRecovered, 1)
			recovered++

			sl.logger.Warn("[DB] SpendLog batch recovered from DLQ",
				"batch_size", len(dlb.batch),
				"originally_failed_at", dlb.failedAt,
				"time_in_dlq", time.Since(dlb.failedAt).String(),
				"original_attempts", dlb.attempts,
			)
		} else {
			failed++
			sl.logger.Debug("[DB] SpendLog batch DLQ retry failed",
				"batch_size", len(dlb.batch),
				"in_dlq_since", dlb.failedAt,
				"error", err,
			)

			failedBatches = append(failedBatches, dlb)
		}
	}
	sl.restoreFailedDLQBatches(failedBatches)

	// Update recovery time
	sl.mu.Lock()
	sl.lastDLQRecoveryTime = utils.NowUTC()
	sl.mu.Unlock()

	if recovered > 0 || failed > 0 {
		sl.logger.Info("[DB] SpendLog DLQ recovery attempt completed",
			"recovered", recovered,
			"failed", failed,
			"dlq_size", sl.getDLQSize(),
		)
	}
	sl.publishSnapshot()
}

// restoreFailedDLQBatches merges failed recovery candidates ahead of batches
// accepted concurrently during recovery. The oldest batches are discarded if
// the bounded DLQ would otherwise exceed ten entries.
func (sl *Logger) restoreFailedDLQBatches(failed []*deadLetterBatch) {
	if len(failed) == 0 {
		return
	}

	sl.dlqMu.Lock()
	merged := make([]*deadLetterBatch, 0, len(failed)+len(sl.dlq))
	merged = append(merged, failed...)
	merged = append(merged, sl.dlq...)
	overflow := len(merged) - 10
	if overflow < 0 {
		overflow = 0
	}
	dropped := append([]*deadLetterBatch(nil), merged[:overflow]...)
	sl.dlq = merged[overflow:]
	sl.dlqMu.Unlock()

	for _, batch := range dropped {
		atomic.AddUint64(&sl.dlqOverflow, 1)
		monitoring.RecordSpendDLQOverflow(1)
		sl.resolvePendingEntries(len(batch.batch))
		sl.logger.Error("[DB] SpendLog DLQ overflow while restoring failed recovery batch",
			"dropped_batch_size", len(batch.batch),
			"dropped_at", batch.failedAt,
			"reason", "dlq_full_during_recovery",
		)
	}
	sl.publishSnapshot()
}

// countEntriesInDLQ counts total number of spend log entries in all DLQ batches
func countEntriesInDLQ(dlq []*deadLetterBatch) int {
	count := 0
	for _, dlb := range dlq {
		count += len(dlb.batch)
	}
	return count
}

// getSampleRequestIDs extracts sample request IDs from a batch
func getSampleRequestIDs(batch []*models.SpendLogEntry, count int) []string {
	if count > len(batch) {
		count = len(batch)
	}
	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = batch[i].RequestID
	}
	return result
}

// insertedSpendEntry pairs a batch entry with the effective request_id that
// INSERT ... RETURNING produced for it: the preferred provider ID or the AIR
// event ID fallback.
type insertedSpendEntry struct {
	entry     *models.SpendLogEntry
	requestID string
}

// filterBatchByInsertedIDs returns only entries whose preferred provider ID or
// unique AIR event fallback was produced by INSERT ... RETURNING. Each returned
// ID is consumed once so same-event replays in one batch cannot feed accounting.
func filterBatchByInsertedIDs(batch []*models.SpendLogEntry, insertedIDs []string) []insertedSpendEntry {
	if len(insertedIDs) == 0 {
		return nil
	}
	idSet := make(map[string]struct{}, len(insertedIDs))
	for _, id := range insertedIDs {
		idSet[id] = struct{}{}
	}
	result := make([]insertedSpendEntry, 0, len(insertedIDs))
	for _, entry := range batch {
		if entry == nil {
			continue
		}
		if _, ok := idSet[entry.RequestID]; ok {
			result = append(result, insertedSpendEntry{entry: entry, requestID: entry.RequestID})
			delete(idSet, entry.RequestID)
			continue
		}
		if entry.AirEventID != "" {
			if _, ok := idSet[entry.AirEventID]; ok {
				result = append(result, insertedSpendEntry{entry: entry, requestID: entry.AirEventID})
				delete(idSet, entry.AirEventID)
			}
		}
	}
	return result
}

// insertedEntries flattens inserted spend pairs back to their batch entries.
func insertedEntries(inserted []insertedSpendEntry) []*models.SpendLogEntry {
	entries := make([]*models.SpendLogEntry, 0, len(inserted))
	for _, item := range inserted {
		entries = append(entries, item.entry)
	}
	return entries
}

func (sl *Logger) resolvePendingEntries(count int) {
	if count <= 0 {
		return
	}
	for {
		current := atomic.LoadInt64(&sl.pendingEntries)
		next := current - int64(count)
		if next < 0 {
			next = 0
		}
		if atomic.CompareAndSwapInt64(&sl.pendingEntries, current, next) {
			return
		}
	}
}

func (sl *Logger) recordCommittedBatch(batch []*models.SpendLogEntry, insertedIDs []string) {
	inserted := filterBatchByInsertedIDs(batch, insertedIDs)
	duplicateCount := len(batch) - len(inserted)
	if duplicateCount < 0 {
		duplicateCount = 0
	}

	eligible, ineligible := uint64(0), uint64(0)
	for _, item := range inserted {
		if item.entry.ComparisonEligible {
			eligible++
		} else {
			ineligible++
		}
	}

	atomic.AddUint64(&sl.written, uint64(len(inserted)))
	atomic.AddUint64(&sl.duplicates, uint64(duplicateCount))
	atomic.AddUint64(&sl.comparisonEligible, eligible)
	atomic.AddUint64(&sl.comparisonIneligible, ineligible)
	sl.resolvePendingEntries(len(batch))

	monitoring.RecordSpendDuplicates(uint64(duplicateCount))
	monitoring.RecordSpendComparisonRows(true, eligible)
	monitoring.RecordSpendComparisonRows(false, ineligible)
}

func (sl *Logger) recordAggregationError() {
	atomic.AddUint64(&sl.aggregationErrors, 1)
	monitoring.RecordSpendAggregationErrors(1)
	sl.publishSnapshot()
}

func (sl *Logger) recordAggregationSuccess() {
	atomic.AddUint64(&sl.aggregationCount, 1)
	sl.mu.Lock()
	sl.lastAggregationTime = utils.NowUTC()
	sl.mu.Unlock()
}

func (sl *Logger) handleCommitError(err error) error {
	// A commit acknowledgement can be outcome-ambiguous. The accounting unit is
	// nevertheless safe to retry: a committed raw row conflicts and feeds no
	// counters/daily updates, while a rolled-back row is inserted and projected
	// in full on the next attempt. Keep the comparison window conservative
	// because this process can no longer attribute the committed rows to its
	// terminal observability counters with certainty. Pre-commit failures are
	// different: the transaction is rolled back and the still-pending exact
	// batch remains owned by retry/DLQ until it is durably replayed.
	sl.recordAggregationError()
	return fmt.Errorf("commit transaction: %w", err)
}
