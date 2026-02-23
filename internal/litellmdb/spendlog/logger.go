package spendlog

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/connection"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// deadLetterBatch represents a batch that failed to insert after all retries
type deadLetterBatch struct {
	batch     []*models.SpendLogEntry
	failedAt  time.Time
	lastError error
	attempts  int
}

// Logger is an asynchronous logger for LiteLLM_SpendLogs table
//
// Features:
// - Non-blocking: Log() returns immediately
// - Batching: collects entries and does batch INSERT
// - Graceful shutdown: waits for all logs to be written
// - Retry: retries on database errors with exponential backoff
// - Dead Letter Queue: persists batches that fail after all retries
// - DLQ Recovery: periodically retries failed batches from DLQ
// - Backpressure: drops entries when queue is full
// - Daily aggregation: aggregates logs into LiteLLM_DailyUserSpend
type Logger struct {
	pool   *connection.ConnectionPool
	logger *slog.Logger
	config *models.Config

	// Queue
	queue chan *models.SpendLogEntry

	// Lifecycle
	stopChan  chan struct{}
	wg        sync.WaitGroup
	shutdown  atomic.Bool // Track if shutdown has been called
	startOnce sync.Once   // Ensure Start() is called only once

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

	// Dead Letter Queue (in-memory circular buffer)
	dlqMu               sync.Mutex
	dlq                 []*deadLetterBatch // Max 10 failed batches
	dlqRecoveryTicker   *time.Ticker       // Periodic DLQ recovery (5 minutes)
	lastDLQRecoveryTime time.Time

	mu                  sync.RWMutex
	lastAggregationTime time.Time
	aggregationTicker   *time.Ticker

	// pendingAggregation receives insertedIDs from flushBatchWithSpendUpdate for immediate aggregation.
	// Buffered: on overflow, safety-net (QuerySelectUnprocessedRequestIDs) will pick up.
	pendingAggregation chan []string
}

// NewLogger creates a new asynchronous logger
func NewLogger(pool *connection.ConnectionPool, cfg *models.Config) *Logger {
	cfg.ApplyDefaults()

	sl := &Logger{
		pool:               pool,
		config:             cfg,
		logger:             cfg.Logger,
		queue:              make(chan *models.SpendLogEntry, cfg.LogQueueSize),
		stopChan:           make(chan struct{}),
		pendingAggregation: make(chan []string, 500),
	}

	return sl
}

// Start starts the background worker and aggregation ticker
// Must be called once after creation. Safe to call multiple times (idempotent).
func (sl *Logger) Start() {
	sl.startOnce.Do(func() {
		// Initialize tickers BEFORE starting goroutines to prevent nil dereference race
		sl.dlqRecoveryTicker = time.NewTicker(5 * time.Minute)
		sl.aggregationTicker = time.NewTicker(5 * time.Minute)

		sl.wg.Add(3)
		go sl.worker()
		go sl.aggregationWorker()
		go sl.dlqRecoveryWorker()
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

	// Try non-blocking send first (fast path)
	select {
	case sl.queue <- entry:
		atomic.AddUint64(&sl.queued, 1)
		return nil
	default:
		// Queue is full, use blocking send with 5 second timeout
	}

	// Queue was full, now attempt blocking send with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	select {
	case sl.queue <- entry:
		atomic.AddUint64(&sl.queued, 1)
		sl.logger.Debug("[DB] SpendLog entry queued after backpressure",
			"request_id", entry.RequestID,
			"queue_len", len(sl.queue),
		)
		return nil

	case <-ctx.Done():
		// Timeout reached - queue still full after 5 seconds
		atomic.AddUint64(&sl.dropped, 1)
		atomic.AddUint64(&sl.queueFullCount, 1)
		sl.logger.Error("[DB] SpendLog entry dropped: queue full timeout",
			"request_id", entry.RequestID,
			"queue_len", len(sl.queue),
			"queue_cap", cap(sl.queue),
			"timeout_sec", 5,
		)
		return models.ErrQueueFull
	}
}

// Shutdown stops the logger and waits for all logs to be written
// Idempotent: safe to call multiple times
func (sl *Logger) Shutdown(ctx context.Context) error {
	// Check if already shut down
	if !sl.shutdown.CompareAndSwap(false, true) {
		return nil // Already shut down
	}

	sl.logger.Info("[DB] SpendLogger shutting down...",
		"pending", len(sl.queue),
	)

	// Stop tickers and drain channels to prevent spurious post-shutdown fires
	if sl.dlqRecoveryTicker != nil {
		sl.dlqRecoveryTicker.Stop()
		// Drain ticker channel (Stop doesn't drain per Go docs)
		select {
		case <-sl.dlqRecoveryTicker.C:
		default:
		}
	}
	if sl.aggregationTicker != nil {
		sl.aggregationTicker.Stop()
		// Drain ticker channel (Stop doesn't drain per Go docs)
		select {
		case <-sl.aggregationTicker.C:
		default:
		}
	}

	// Signal worker to stop
	close(sl.stopChan)

	// Wait for completion with timeout
	done := make(chan struct{})
	go func() {
		sl.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		dlqSize := sl.getDLQSize()
		sl.logger.Info("[DB] SpendLogger shutdown complete",
			"written", atomic.LoadUint64(&sl.written),
			"dropped", atomic.LoadUint64(&sl.dropped),
			"errors", atomic.LoadUint64(&sl.errors),
			"dlq_size", dlqSize,
			"dlq_recovered", atomic.LoadUint64(&sl.dlqRecovered),
		)
		return nil
	case <-ctx.Done():
		sl.logger.Warn("[DB] SpendLogger shutdown timeout",
			"pending", len(sl.queue),
		)
		return ctx.Err()
	}
}

// Stats returns logger statistics
func (sl *Logger) Stats() models.SpendLoggerStats {
	sl.mu.RLock()
	lastAgg := sl.lastAggregationTime
	sl.mu.RUnlock()

	return models.SpendLoggerStats{
		QueueLen:            len(sl.queue),
		QueueCap:            cap(sl.queue),
		Queued:              atomic.LoadUint64(&sl.queued),
		Written:             atomic.LoadUint64(&sl.written),
		Dropped:             atomic.LoadUint64(&sl.dropped),
		Errors:              atomic.LoadUint64(&sl.errors),
		BatchesOK:           atomic.LoadUint64(&sl.batchesOK),
		QueueFullCount:      atomic.LoadUint64(&sl.queueFullCount),
		AggregationCount:    atomic.LoadUint64(&sl.aggregationCount),
		AggregationErrors:   atomic.LoadUint64(&sl.aggregationErrors),
		LastAggregationTime: lastAgg,
	}
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
			backoff := backoffDurations[attempt]
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
				// Shutdown requested during backoff - send to DLQ and return
				sl.logger.Warn("[DB] SpendLog batch retry interrupted by shutdown",
					"attempt", attempt+1,
					"batch_size", len(batch),
				)
				atomic.AddUint64(&sl.errors, uint64(len(batch)))
				sl.addToDLQ(batch, lastErr, attempt)
				return
			}
		}

		err := sl.flushBatchWithSpendUpdate(batch)
		if err == nil {
			atomic.AddUint64(&sl.written, uint64(len(batch)))
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

// flushBatchWithSpendUpdate executes batch INSERT and spend updates atomically
// 1. INSERT batch into SpendLogs
// 2. Aggregate and UPDATE spend for Token, User, Team, Org, TeamMember, OrgMember
// All operations executed in a single transaction for atomicity
func (sl *Logger) flushBatchWithSpendUpdate(batch []*models.SpendLogEntry) error {
	// Skip if pool is not initialized (e.g., in tests)
	if sl.pool == nil {
		return models.ErrConnectionFailed
	}

	if !sl.pool.IsHealthy() {
		return models.ErrConnectionFailed
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	conn, err := sl.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire connection: %w", err)
	}
	defer conn.Release()

	// Begin transaction
	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}

	defer func() {
		// Rollback is a no-op if tx is already committed
		_ = tx.Rollback(ctx)
	}()

	// 1. INSERT batch into SpendLogs with RETURNING request_id
	query := queries.BuildBatchInsertQuery(len(batch))
	params := GetBatchParams(batch)
	insertRows, err := tx.Query(ctx, query, params...)
	if err != nil {
		return fmt.Errorf("batch insert: %w", err)
	}

	var insertedIDs []string
	for insertRows.Next() {
		var id string
		if err := insertRows.Scan(&id); err != nil {
			insertRows.Close()
			return fmt.Errorf("scan returning request_id: %w", err)
		}
		insertedIDs = append(insertedIDs, id)
	}
	insertRows.Close()
	if err := insertRows.Err(); err != nil {
		return fmt.Errorf("iterate returning rows: %w", err)
	}

	// 2. Aggregate and UPDATE spend (only for actually inserted entries)
	filteredBatch := filterBatchByInsertedIDs(batch, insertedIDs)
	spendUpdates := aggregateSpendUpdates(filteredBatch)
	err = executeSpendUpdates(ctx, tx, spendUpdates)
	if err != nil {
		return fmt.Errorf("spend updates: %w", err)
	}

	// Commit transaction
	err = tx.Commit(ctx)
	if err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}

	// Send insertedIDs to aggregationWorker (non-blocking)
	if len(insertedIDs) > 0 {
		select {
		case sl.pendingAggregation <- insertedIDs:
			// sent
		default:
			sl.logger.Warn("[DB] pendingAggregation channel full, safety-net will handle",
				"ids_count", len(insertedIDs),
			)
		}
	}

	return nil
}

// addToDLQ adds a failed batch to the dead letter queue
// Max queue size: 10 failed batches (~100KB max memory)
// If DLQ is full, drops the oldest batch and logs error
func (sl *Logger) addToDLQ(batch []*models.SpendLogEntry, lastErr error, attempts int) {
	sl.dlqMu.Lock()
	defer sl.dlqMu.Unlock()

	dlb := &deadLetterBatch{
		batch:     batch,
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

		sl.logger.Error("[DB] SpendLog DLQ overflow - batch dropped",
			"dropped_batch_size", len(dropped.batch),
			"dropped_at", dropped.failedAt,
			"dlq_size", len(sl.dlq),
			"reason", "dlq_full",
		)
	}

	sl.dlq = append(sl.dlq, dlb)
	atomic.AddUint64(&sl.dlqCount, 1)

	// Log batch details
	sl.logger.Error("[DB] SpendLog batch sent to Dead Letter Queue",
		"batch_size", len(batch),
		"dlq_size", len(sl.dlq),
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
			// Shutdown: attempt final DLQ recovery
			sl.flushDLQ()
			return

		case <-sl.dlqRecoveryTicker.C:
			// Priority check: if stopChan is also ready, prefer shutdown
			select {
			case <-sl.stopChan:
				sl.flushDLQ()
				return
			default:
			}
			sl.flushDLQ()
		}
	}
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
			atomic.AddUint64(&sl.written, uint64(len(dlb.batch)))
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

			// Re-add failed batch back to DLQ
			sl.dlqMu.Lock()
			sl.dlq = append(sl.dlq, dlb)
			sl.dlqMu.Unlock()
		}
	}

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

// filterBatchByInsertedIDs returns only entries whose RequestID is in insertedIDs.
// Used after INSERT ... RETURNING to exclude conflicting (already existing) entries.
func filterBatchByInsertedIDs(batch []*models.SpendLogEntry, insertedIDs []string) []*models.SpendLogEntry {
	if len(insertedIDs) == 0 {
		return nil
	}
	idSet := make(map[string]struct{}, len(insertedIDs))
	for _, id := range insertedIDs {
		idSet[id] = struct{}{}
	}
	result := make([]*models.SpendLogEntry, 0, len(insertedIDs))
	for _, entry := range batch {
		if _, ok := idSet[entry.RequestID]; ok {
			result = append(result, entry)
		}
	}
	return result
}

// aggregationWorker processes push-path aggregation and periodic safety-net
// Push path: aggregates only IDs inserted by this router (no distributed lock needed)
// Safety-net: periodic catch-all with pg_try_advisory_lock for crash recovery
func (sl *Logger) aggregationWorker() {
	defer sl.wg.Done()

	for {
		select {
		case <-sl.stopChan:
			// Shutdown: drain pending channel and run final safety-net
			sl.drainPendingAggregation()
			sl.aggregateSpendLogs()
			return

		case ids := <-sl.pendingAggregation:
			// Main path: aggregate only what this router inserted
			sl.aggregateByIDs(ids)

		case <-sl.aggregationTicker.C:
			// Priority check: if stopChan is also ready, prefer shutdown
			select {
			case <-sl.stopChan:
				sl.drainPendingAggregation()
				sl.aggregateSpendLogs()
				return
			default:
			}
			// Safety-net: catch "lost" logs (with advisory lock)
			sl.aggregateSpendLogs()
		}
	}
}

// drainPendingAggregation processes all accumulated IDs during shutdown
func (sl *Logger) drainPendingAggregation() {
	for {
		select {
		case ids := <-sl.pendingAggregation:
			sl.aggregateByIDs(ids)
		default:
			return
		}
	}
}

// aggregationLockID is a unique constant for pg_advisory_lock (safety-net only)
const aggregationLockID = int64(7_463_951_234)

// aggregateSpendLogs is the safety-net aggregator for crash recovery.
// Runs every 5 minutes with pg_try_advisory_lock to prevent multiple routers
// from aggregating the same logs simultaneously.
// Flow:
// 1. Acquire advisory lock (skip if another router holds it)
// 2. Fetch unprocessed request_ids (snapshot at aggregation start)
// 3. Call aggregators for: User, Team, Organization, EndUser, Agent, Tag
// 4. Mark all request_ids as processed
func (sl *Logger) aggregateSpendLogs() {
	// Skip if pool is not initialized (e.g., in tests)
	if sl.pool == nil {
		return
	}

	if !sl.pool.IsHealthy() {
		atomic.AddUint64(&sl.aggregationErrors, 1)
		sl.logger.Warn("[DB] Cannot aggregate: database not healthy")
		return
	}

	// Use a generous top-level context for the entire aggregation flow (3 minutes).
	// Each individual aggregator will get its own 30-second timeout below.
	aggCtx, aggCancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer aggCancel()

	conn, err := sl.pool.Acquire(aggCtx)
	if err != nil {
		atomic.AddUint64(&sl.aggregationErrors, 1)
		sl.logger.Error("[DB] Aggregation: failed to acquire connection", "error", err)
		return
	}

	// Try to acquire distributed lock (one router aggregates at a time)
	var lockAcquired bool
	if err := conn.QueryRow(aggCtx, "SELECT pg_try_advisory_lock($1)", aggregationLockID).Scan(&lockAcquired); err != nil || !lockAcquired {
		conn.Release()
		sl.logger.Debug("[DB] Safety-net aggregation skipped: another router holds lock")
		return
	}

	// Guaranteed order: unlock → release
	defer func() {
		unlockCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = conn.Exec(unlockCtx, "SELECT pg_advisory_unlock($1)", aggregationLockID)
		conn.Release()
	}()

	// 1. Fetch unprocessed request_ids (snapshot)
	fetchCtx, fetchCancel := context.WithTimeout(aggCtx, 30*time.Second)

	rows, err := conn.Query(fetchCtx, queries.QuerySelectUnprocessedRequestIDs)
	if err != nil {
		fetchCancel()
		atomic.AddUint64(&sl.aggregationErrors, 1)
		sl.logger.Error("[DB] Aggregation: failed to fetch request_ids", "error", err)
		return
	}

	var requestIDs []string
	for rows.Next() {
		var requestID string
		if err := rows.Scan(&requestID); err != nil {
			rows.Close()
			fetchCancel()
			sl.logger.Error("[DB] Aggregation: failed to scan request_id", "error", err)
			atomic.AddUint64(&sl.aggregationErrors, 1)
			return
		}
		requestIDs = append(requestIDs, requestID)
	}

	if err := rows.Err(); err != nil {
		rows.Close()
		fetchCancel()
		atomic.AddUint64(&sl.aggregationErrors, 1)
		sl.logger.Error("[DB] Aggregation: failed to iterate request_ids", "error", err)
		return
	}

	rows.Close()
	fetchCancel()

	// No unprocessed logs
	if len(requestIDs) == 0 {
		return
	}

	sl.logger.Debug("[DB] Aggregation: starting aggregators",
		"request_ids_count", len(requestIDs),
	)

	// 2. Load spend log records ONCE and pass to all aggregators
	loadCtx, loadCancel := context.WithTimeout(aggCtx, 30*time.Second)
	records, err := loadUnprocessedSpendLogRecords(loadCtx, conn, sl.logger, "Aggregation", requestIDs)
	loadCancel()
	if err != nil {
		atomic.AddUint64(&sl.aggregationErrors, 1)
		sl.logger.Error("[DB] Aggregation: failed to load spend log records", "error", err)
		return
	}

	if len(records) == 0 {
		return
	}

	// 3. Run all aggregators and mark as processed
	if ok := sl.runAggregators(aggCtx, conn, "Aggregation", records, requestIDs); ok {
		sl.logger.Debug("[DB] All aggregations completed",
			"request_ids_count", len(requestIDs),
		)
	} else {
		sl.logger.Warn("[DB] Aggregation completed with errors, request_ids NOT marked as processed",
			"request_ids_count", len(requestIDs),
		)
	}
}
