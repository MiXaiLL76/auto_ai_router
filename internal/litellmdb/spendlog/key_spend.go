package spendlog

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// CommitResult describes an acknowledged single-entry accounting transaction.
// EffectiveRequestID is blank when the entry was an idempotent replay/no-op.
// KeySpend is safe to expose only when KeySpendKnown is true and CommitSpend
// returned a nil error.
type CommitResult struct {
	Inserted           bool
	EffectiveRequestID string
	KeySpend           float64
	KeySpendKnown      bool
	// ReplayRetained reports that a failed synchronous attempt handed the exact
	// event to the asynchronous writer while it still owned its lifecycle
	// admission. This closes the shutdown race between CommitSpend returning and
	// a caller trying to enqueue the replay as a separate operation.
	ReplayRetained bool
}

type keySpendQuerier interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

// ReadKeySpend returns a committed statement snapshot of a virtual key's
// scalar spend. It deliberately reports missing keys and NULL spend as
// unknown; callers must not substitute a cached or calculated value.
func (sl *Logger) ReadKeySpend(ctx context.Context, apiKeyHash string) (value float64, known bool, err error) {
	if apiKeyHash == "" {
		return 0, false, nil
	}
	if !sl.beginOperation() {
		return 0, false, ErrLoggerStopped
	}
	defer sl.enqueueWG.Done()
	ctx, cancel := sl.synchronousOperationContext(ctx)
	defer cancel()

	if sl.pool == nil || !sl.pool.IsHealthy() {
		return 0, false, models.ErrConnectionFailed
	}
	conn, err := sl.pool.Acquire(ctx)
	if err != nil {
		return 0, false, fmt.Errorf("read key spend: acquire connection: %w", err)
	}
	defer conn.Release()

	value, known, err = readKeySpend(ctx, conn, queries.QuerySelectKeySpend, apiKeyHash)
	if err != nil {
		return 0, false, fmt.Errorf("read key spend: %w", err)
	}
	return value, known, nil
}

// CommitSpend synchronously commits one spend entry and returns the scalar key
// spend observed under the same transaction's row lock. It performs one
// database attempt only: response-path callers can fall back to the existing
// asynchronous replay API using the exact same entry and idempotency IDs.
func (sl *Logger) CommitSpend(ctx context.Context, entry *models.SpendLogEntry) (CommitResult, error) {
	if entry == nil {
		return CommitResult{}, nil
	}
	if !sl.beginOperation() {
		return CommitResult{}, ErrLoggerStopped
	}
	defer sl.enqueueWG.Done()
	ctx, cancel := sl.synchronousOperationContext(ctx)
	defer cancel()

	// Synchronous work participates in the same terminal pending accounting as
	// queued work, but is counted as accepted only after commit succeeds. On an
	// error, a later asynchronous replay therefore contributes exactly one
	// accepted/queued event instead of double-counting the failed first attempt.
	atomic.AddInt64(&sl.pendingEntries, 1)
	sl.publishSnapshot()

	result, insertedIDs, err := sl.commitSpendOnce(ctx, entry)
	if err != nil {
		// Keep the existing pending reservation and hand the exact entry to the
		// async writer before releasing this operation's lifecycle ticket. This
		// remains safe even when shutdown has already closed new admissions.
		sl.retainReservedSpend(entry, err)
		sl.publishSnapshot()
		return CommitResult{ReplayRetained: true}, err
	}

	atomic.AddUint64(&sl.queued, 1)
	atomic.AddUint64(&sl.batchesOK, 1)
	sl.recordCommittedBatch([]*models.SpendLogEntry{entry}, insertedIDs)
	if len(insertedIDs) > 0 {
		sl.recordAggregationSuccess()
	}
	sl.publishSnapshot()
	return result, nil
}

// retainReservedSpend transfers the pending reservation created by
// CommitSpend to the asynchronous writer. It deliberately does not call
// beginOperation: the synchronous commit still owns an admitted lifecycle
// ticket, and shutdown cannot stop the worker until that ticket is released.
func (sl *Logger) retainReservedSpend(entry *models.SpendLogEntry, cause error) {
	atomic.AddUint64(&sl.queued, 1)
	select {
	case sl.queue <- entry:
		return
	default:
		atomic.AddUint64(&sl.queueFullCount, 1)
		sl.addToDLQ([]*models.SpendLogEntry{entry}, cause, 1)
	}
}

// synchronousOperationContext is canceled by either the caller or logger
// shutdown. context.AfterFunc avoids one watcher goroutine per request and the
// returned cancel function unregisters the callback on normal completion.
func (sl *Logger) synchronousOperationContext(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	if sl == nil || sl.syncCtx == nil {
		return ctx, cancel
	}
	stop := context.AfterFunc(sl.syncCtx, cancel)
	return ctx, func() {
		stop()
		cancel()
	}
}

func (sl *Logger) commitSpendOnce(
	ctx context.Context,
	entry *models.SpendLogEntry,
) (CommitResult, []string, error) {
	if sl.pool == nil || !sl.pool.IsHealthy() {
		return CommitResult{}, nil, models.ErrConnectionFailed
	}
	conn, err := sl.pool.Acquire(ctx)
	if err != nil {
		return CommitResult{}, nil, fmt.Errorf("commit spend: acquire connection: %w", err)
	}
	defer conn.Release()

	// The collision ownership claim relies on a statement snapshot after
	// INSERT's conflict wait, so retain the asynchronous writer's READ COMMITTED
	// level.
	tx, err := conn.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return CommitResult{}, nil, fmt.Errorf("commit spend: begin transaction: %w", err)
	}
	return sl.commitSpendTransaction(ctx, tx, entry)
}

// commitSpendTransaction is the testable transaction core. It reuses the full
// atomic writer projection and only adds the locked scalar-spend read before
// commit. No value read here escapes when commit acknowledgement is ambiguous.
func (sl *Logger) commitSpendTransaction(
	ctx context.Context,
	tx pgx.Tx,
	entry *models.SpendLogEntry,
) (CommitResult, []string, error) {
	defer func() {
		rollbackTransaction(ctx, tx)
	}()

	insertedIDs, err := sl.writeBatchInTransaction(ctx, tx, []*models.SpendLogEntry{entry})
	if err != nil {
		return CommitResult{}, nil, err
	}
	if len(insertedIDs) > 1 {
		return CommitResult{}, nil, fmt.Errorf("commit spend: single entry produced %d inserted rows", len(insertedIDs))
	}

	keySpend, keySpendKnown, err := readKeySpend(
		ctx,
		tx,
		queries.QuerySelectKeySpendForUpdate,
		entry.APIKey,
	)
	if err != nil {
		return CommitResult{}, nil, fmt.Errorf("commit spend: read locked key spend: %w", err)
	}

	result := CommitResult{
		Inserted:      len(insertedIDs) == 1,
		KeySpend:      keySpend,
		KeySpendKnown: keySpendKnown,
	}
	if result.Inserted {
		result.EffectiveRequestID = insertedIDs[0]
	}

	if err := tx.Commit(ctx); err != nil {
		// A value read before an unacknowledged commit is never returned. The
		// exact same entry remains safe to replay through INSERT ... ON CONFLICT.
		return CommitResult{}, nil, sl.handleCommitError(err)
	}
	return result, insertedIDs, nil
}

func readKeySpend(
	ctx context.Context,
	querier keySpendQuerier,
	query string,
	apiKeyHash string,
) (float64, bool, error) {
	if apiKeyHash == "" {
		return 0, false, nil
	}

	var spend float64
	err := querier.QueryRow(ctx, query, apiKeyHash).Scan(&spend)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return spend, true, nil
}
