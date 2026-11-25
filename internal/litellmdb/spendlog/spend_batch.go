package spendlog

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
)

// spendBatch queues a transaction's statements and sends them to Postgres in one round trip.
type spendBatch struct {
	tx    pgx.Tx
	batch pgx.Batch
}

// Exec queues the statement. Nothing has run yet, so the returned tag is empty and the statement's
// error is reported by flush.
func (b *spendBatch) Exec(_ context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	b.batch.Queue(sql, args...)
	return pgconn.CommandTag{}, nil
}

// flush runs the queued statements in their order and stops at the first error.
func (b *spendBatch) flush(ctx context.Context) error {
	statements := b.batch.QueuedQueries
	if len(statements) == 0 {
		return nil
	}
	b.batch = pgx.Batch{}
	results := b.tx.SendBatch(ctx, &pgx.Batch{QueuedQueries: statements})
	for i, statement := range statements {
		if _, err := results.Exec(); err != nil {
			// A statement that fails to prepare fails the whole batch before any runs.
			sql := statement.SQL
			var prepare pgx.ErrPreprocessingBatch
			if errors.As(err, &prepare) {
				sql = prepare.SQL()
			}
			if flag := schemaFallbackFlag(sql); flag != nil && isColumnNotExist(err) {
				flag.Store(false)
			}
			_ = results.Close()
			return fmt.Errorf("statement %d of %d: %w", i+1, len(statements), err)
		}
	}
	return results.Close()
}

// flushSpendUpdates sends the statements exec has queued. An executor that runs each statement
// at once, such as a pgx.Tx, has nothing to flush.
func flushSpendUpdates(ctx context.Context, exec spendUpdateExecer) error {
	queued, ok := exec.(interface{ flush(context.Context) error })
	if !ok {
		return nil
	}
	if err := queued.flush(ctx); err != nil {
		return fmt.Errorf("spend updates batch: %w", err)
	}
	return nil
}

// schemaFallbackFlag is the schema flag an undefined-column error of sql clears, as the one-by-one updates do.
func schemaFallbackFlag(sql string) *atomic.Bool {
	switch sql {
	case queries.QueryUpdateTokenSpendWithLastActive, queries.QueryUpdateTokenModelSpendWithLastActive:
		return &schemaTokenHasLastActive
	case queries.QueryUpdateTeamMemberSpendWithTotal:
		return &schemaTeamMemberHasTotalSpend
	}
	return nil
}
