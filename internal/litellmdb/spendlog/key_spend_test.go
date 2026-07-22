package spendlog

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCommitSpendTransactionReturnsLockedInclusiveSpendAfterCommit(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-sync")
	tx := &keySpendTestTx{
		atomicTestTx: &atomicTestTx{
			insertedIDs: []string{entry.RequestID},
		},
		keySpend:      2.75,
		keySpendKnown: true,
	}

	result, insertedIDs, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.NoError(t, err)
	assert.Equal(t, []string{entry.RequestID}, insertedIDs)
	assert.Equal(t, CommitResult{
		Inserted:           true,
		EffectiveRequestID: entry.RequestID,
		KeySpend:           2.75,
		KeySpendKnown:      true,
	}, result)
	assert.True(t, tx.committed)
	assert.False(t, tx.rolledBack)
	require.Equal(t, []string{queries.QuerySelectKeySpendForUpdate}, tx.rowQueries)
	require.Len(t, tx.rowArgs, 1)
	assert.Equal(t, []any{entry.APIKey}, tx.rowArgs[0])
}

func TestCommitSpendTransactionReplayReturnsCurrentSpendAndBlankEffectiveID(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-replay")
	tx := &keySpendTestTx{
		atomicTestTx:  &atomicTestTx{},
		keySpend:      4.5,
		keySpendKnown: true,
	}

	result, insertedIDs, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.NoError(t, err)
	assert.Empty(t, insertedIDs)
	assert.False(t, result.Inserted)
	assert.Empty(t, result.EffectiveRequestID, "replay/no-op must not claim a new row ID")
	assert.Equal(t, 4.5, result.KeySpend)
	assert.True(t, result.KeySpendKnown)
	assert.True(t, tx.committed)
	assert.Empty(t, tx.attemptedSQL, "replay must not update any accounting projection")
}

func TestCommitSpendTransactionReturnsCollisionFallbackID(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicProviderIDEntry("chatcmpl-shared", "air-event-new")
	tx := &keySpendTestTx{
		atomicTestTx: &atomicTestTx{
			insertResults: [][]string{{}, {entry.AirEventID}},
		},
		keySpend:      1.25,
		keySpendKnown: true,
	}

	result, insertedIDs, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.NoError(t, err)
	assert.Equal(t, []string{entry.AirEventID}, insertedIDs)
	assert.True(t, result.Inserted)
	assert.Equal(t, entry.AirEventID, result.EffectiveRequestID)
}

func TestCommitSpendTransactionKeepsMissingOrNullKeySpendUnknown(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-unknown-key-spend")
	tx := &keySpendTestTx{
		atomicTestTx: &atomicTestTx{
			insertedIDs: []string{entry.RequestID},
		},
		keySpendKnown: false,
	}

	result, _, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.NoError(t, err)
	assert.True(t, result.Inserted)
	assert.False(t, result.KeySpendKnown)
	assert.Zero(t, result.KeySpend)
}

func TestCommitSpendTransactionReadFailureRollsBackAndReturnsNoValue(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-read-failure")
	tx := &keySpendTestTx{
		atomicTestTx: &atomicTestTx{
			insertedIDs: []string{entry.RequestID},
		},
		queryRowErr: errors.New("injected key spend read failure"),
	}

	result, insertedIDs, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "read locked key spend")
	assert.Equal(t, CommitResult{}, result)
	assert.Empty(t, insertedIDs)
	assert.False(t, tx.committed)
	assert.True(t, tx.rolledBack)
	assert.Empty(t, tx.committedSQL)
}

func TestCommitSpendTransactionAmbiguousCommitReturnsNoPreCommitValue(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-commit-ambiguous")
	tx := &keySpendTestTx{
		atomicTestTx: &atomicTestTx{
			insertedIDs: []string{entry.RequestID},
			commitErr:   errors.New("injected commit acknowledgement loss"),
		},
		keySpend:      9.75,
		keySpendKnown: true,
	}

	result, insertedIDs, err := logger.commitSpendTransaction(context.Background(), tx, entry)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "commit transaction")
	assert.Equal(t, CommitResult{}, result)
	assert.Empty(t, insertedIDs)
	assert.True(t, tx.rolledBack)
	assert.Equal(t, uint64(1), logger.Stats().AggregationErrors)
}

func TestReadKeySpendHelperDistinguishesZeroUnknownAndFailure(t *testing.T) {
	ctx := context.Background()

	zero := &keySpendTestTx{atomicTestTx: &atomicTestTx{}, keySpendKnown: true}
	value, known, err := readKeySpend(ctx, zero, queries.QuerySelectKeySpend, "key-zero")
	require.NoError(t, err)
	assert.True(t, known)
	assert.Zero(t, value)

	unknown := &keySpendTestTx{atomicTestTx: &atomicTestTx{}}
	value, known, err = readKeySpend(ctx, unknown, queries.QuerySelectKeySpend, "key-null")
	require.NoError(t, err)
	assert.False(t, known)
	assert.Zero(t, value)

	failing := &keySpendTestTx{atomicTestTx: &atomicTestTx{}, queryRowErr: assert.AnError}
	value, known, err = readKeySpend(ctx, failing, queries.QuerySelectKeySpend, "key-error")
	assert.ErrorIs(t, err, assert.AnError)
	assert.False(t, known)
	assert.Zero(t, value)

	empty := &keySpendTestTx{atomicTestTx: &atomicTestTx{}, queryRowErr: assert.AnError}
	value, known, err = readKeySpend(ctx, empty, queries.QuerySelectKeySpend, "")
	require.NoError(t, err)
	assert.False(t, known)
	assert.Zero(t, value)
	assert.Empty(t, empty.rowQueries, "an absent key cannot produce a trustworthy DB value")
}

func TestCommitSpendPoolFailureReleasesLifecycleAccounting(t *testing.T) {
	cfg := models.DefaultConfig()
	cfg.Logger = testhelpers.NewTestLogger()
	logger := NewLogger(nil, cfg)

	result, err := logger.CommitSpend(context.Background(), atomicTestEntry("req-no-pool"))

	assert.ErrorIs(t, err, models.ErrConnectionFailed)
	assert.True(t, result.ReplayRetained)
	stats := logger.Stats()
	assert.Equal(t, 1, stats.PendingEntries)
	assert.Equal(t, uint64(1), stats.Queued)
	assert.Zero(t, stats.Written)
	assert.False(t, stats.ComparisonWindowValid)
	require.Len(t, logger.queue, 1)

	shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	assert.ErrorIs(t, logger.Shutdown(shutdownCtx), ErrDrainIncomplete)

	_, err = logger.CommitSpend(context.Background(), atomicTestEntry("req-after-stop"))
	assert.ErrorIs(t, err, ErrLoggerStopped)
}

func TestFailedCommitRetentionStillWorksAfterShutdownClosesAdmissions(t *testing.T) {
	cfg := models.DefaultConfig()
	cfg.Logger = testhelpers.NewTestLogger()
	logger := NewLogger(nil, cfg)
	entry := atomicTestEntry("req-shutdown-race")

	// Model a CommitSpend call that was admitted before shutdown. Its pending
	// reservation and lifecycle ticket already exist when shutdown closes the
	// public enqueue barrier.
	require.True(t, logger.beginOperation())
	atomic.AddInt64(&logger.pendingEntries, 1)
	logger.lifecycleMu.Lock()
	logger.stopping = true
	logger.lifecycleMu.Unlock()

	logger.retainReservedSpend(entry, context.Canceled)
	logger.enqueueWG.Done()

	stats := logger.Stats()
	assert.Equal(t, uint64(1), stats.Queued)
	assert.Equal(t, 1, stats.PendingEntries)
	require.Len(t, logger.queue, 1)
	assert.Same(t, entry, <-logger.queue)
}

type keySpendTestTx struct {
	*atomicTestTx
	keySpend      float64
	keySpendKnown bool
	queryRowErr   error
	rowQueries    []string
	rowArgs       [][]any
}

func (tx *keySpendTestTx) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	if strings.Contains(sql, `UPDATE "LiteLLM_SpendLogs"`) {
		return tx.atomicTestTx.QueryRow(ctx, sql, args...)
	}
	tx.rowQueries = append(tx.rowQueries, sql)
	tx.rowArgs = append(tx.rowArgs, append([]any(nil), args...))
	return keySpendTestRow{
		value: tx.keySpend,
		known: tx.keySpendKnown,
		err:   tx.queryRowErr,
	}
}

type keySpendTestRow struct {
	value float64
	known bool
	err   error
}

func (row keySpendTestRow) Scan(dest ...any) error {
	if row.err != nil {
		return row.err
	}
	if !row.known {
		return pgx.ErrNoRows
	}
	if len(dest) != 1 {
		return errors.New("key spend row expects one scan destination")
	}
	spend, ok := dest[0].(*float64)
	if !ok || spend == nil {
		return errors.New("key spend destination must be *float64")
	}
	*spend = row.value
	return nil
}
