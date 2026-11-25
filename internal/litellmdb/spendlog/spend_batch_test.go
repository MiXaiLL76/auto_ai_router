package spendlog

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// batchTestEntries repeats keys, users, teams and models across a batch.
func batchTestEntries() []*models.SpendLogEntry {
	entries := make([]*models.SpendLogEntry, 0, 12)
	for i := range 12 {
		entry := atomicTestEntry("req-batch-" + string(rune('a'+i)))
		entry.APIKey = []string{"key-b", "key-a", "key-c"}[i%3]
		entry.Model = []string{"model-2", "model-1"}[i%2]
		entry.UserID = []string{"user-2", "user-1", ""}[i%3]
		entry.TeamID = []string{"team-1", "team-2", "", "team-1"}[i%4]
		entry.BillingTeamID = entry.TeamID
		entry.OrganizationID = []string{"org-1", ""}[i%2]
		entry.EndUser = []string{"", "end-1", "end-2"}[i%3]
		entry.Spend = 0.25 * float64(i+1)
		entries = append(entries, entry)
	}
	return entries
}

func batchTestIDs(entries []*models.SpendLogEntry) []string {
	ids := make([]string, 0, len(entries))
	for _, entry := range entries {
		ids = append(ids, entry.RequestID)
	}
	return ids
}

func restoreSchemaFlags(t *testing.T) {
	t.Helper()
	schemaTokenHasLastActive.Store(true)
	schemaTeamMemberHasTotalSpend.Store(true)
	schemaHasKeyBudgetWindows.Store(true)
	t.Cleanup(func() {
		schemaTokenHasLastActive.Store(true)
		schemaTeamMemberHasTotalSpend.Store(true)
		schemaHasKeyBudgetWindows.Store(true)
	})
}

func TestSpendUpdatesSingleRoundTripSendsTheSameStatementsInOneBatch(t *testing.T) {
	for _, windows := range []bool{false, true} {
		restoreSchemaFlags(t)
		entries := batchTestEntries()

		oneByOne := &atomicTestTx{insertedIDs: batchTestIDs(entries)}
		logger := newAtomicTestLogger()
		logger.config.EnforceKeyBudgetWindows = windows
		_, err := logger.commitBatchTransaction(context.Background(), oneByOne, entries)
		require.NoError(t, err)

		batched := &atomicTestTx{insertedIDs: batchTestIDs(entries)}
		logger.config.SpendUpdatesSingleRoundTrip = true
		_, err = logger.commitBatchTransaction(context.Background(), batched, entries)
		require.NoError(t, err)

		require.NotEmpty(t, oneByOne.attemptedSQL)
		assert.Equal(t, oneByOne.attemptedSQL, batched.attemptedSQL, "same statements in the same lock order")
		assert.Equal(t, oneByOne.attemptedArgs, batched.attemptedArgs)
		assert.Equal(t, batched.attemptedSQL, batched.batchedSQL, "no statement after the insert is sent on its own")
		assert.Zero(t, oneByOne.batches)
		// The budget window read needs the token rows locked, so the updates before it go first.
		wantBatches := 1
		if windows {
			wantBatches = 2
		}
		assert.Equal(t, wantBatches, batched.batches, "windows=%v", windows)
		assert.True(t, batched.committed)
		assert.Equal(t, oneByOne.committedSQL, batched.committedSQL)
	}
}

func TestSpendUpdatesSingleRoundTripRollsBackOnAFailedStatement(t *testing.T) {
	restoreSchemaFlags(t)
	logger := newAtomicTestLogger()
	logger.config.SpendUpdatesSingleRoundTrip = true
	entry := atomicTestEntry("req-batch-fail")
	tx := &atomicTestTx{
		insertedIDs:       []string{entry.RequestID},
		failExecSubstring: `INSERT INTO "LiteLLM_DailyEndUserSpend"`,
	}

	_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "spend updates batch")
	assert.False(t, tx.committed)
	assert.True(t, tx.rolledBack)
	assert.Empty(t, tx.committedSQL)
	assert.Equal(t, 1, tx.batches)
	assert.Zero(t, logger.Stats().AggregationErrors)
}

func TestSpendUpdatesSingleRoundTripKeepsSchemaFallbacks(t *testing.T) {
	restoreSchemaFlags(t)
	logger := newAtomicTestLogger()
	logger.config.SpendUpdatesSingleRoundTrip = true
	entry := atomicTestEntry("req-batch-schema")
	entry.BillingTeamID = "team-1"
	undefinedColumn := &pgconn.PgError{Code: "42703"}

	first := &atomicTestTx{insertedIDs: []string{entry.RequestID}, failExecSubstring: "last_active", failExecErr: undefinedColumn}
	_, err := logger.commitBatchTransaction(context.Background(), first, []*models.SpendLogEntry{entry})
	require.Error(t, err)
	assert.False(t, schemaTokenHasLastActive.Load())
	assert.True(t, schemaTeamMemberHasTotalSpend.Load())

	second := &atomicTestTx{insertedIDs: []string{entry.RequestID}, failExecSubstring: "total_spend", failExecErr: undefinedColumn}
	_, err = logger.commitBatchTransaction(context.Background(), second, []*models.SpendLogEntry{entry})
	require.Error(t, err)
	assert.False(t, schemaTeamMemberHasTotalSpend.Load())
	assert.Contains(t, second.attemptedSQL, queries.QueryUpdateTokenModelSpend)

	third := &atomicTestTx{insertedIDs: []string{entry.RequestID}, failExecSubstring: "total_spend"}
	_, err = logger.commitBatchTransaction(context.Background(), third, []*models.SpendLogEntry{entry})
	require.NoError(t, err)
	assert.Contains(t, third.committedSQL, queries.QueryUpdateTeamMemberSpend)
}

func TestSpendBatchIgnoresUndefinedColumnsOfOtherStatements(t *testing.T) {
	restoreSchemaFlags(t)
	tx := &atomicTestTx{failExecSubstring: `"LiteLLM_UserTable"`, failExecErr: &pgconn.PgError{Code: "42703"}}
	batch := &spendBatch{tx: tx}
	_, _ = batch.Exec(context.Background(), queries.QueryUpdateTokenSpendWithLastActive, 1.0, "key")
	_, _ = batch.Exec(context.Background(), `UPDATE "LiteLLM_UserTable" SET spend = $1`, 1.0)
	_, _ = batch.Exec(context.Background(), queries.QueryUpdateTeamMemberSpendWithTotal, 1.0, "team", "user")

	err := batch.flush(context.Background())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "statement 2 of 3")
	assert.True(t, schemaTokenHasLastActive.Load())
	assert.True(t, schemaTeamMemberHasTotalSpend.Load())
	assert.Equal(t, 2, len(tx.attemptedSQL), "nothing runs after the failed statement")
	require.NoError(t, batch.flush(context.Background()), "a sent batch is empty")
}
