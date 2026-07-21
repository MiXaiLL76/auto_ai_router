package spendlog

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAtomicWriterRollsBackAllProjectionsOnPartialDailyFailure(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-partial")
	tx := &atomicTestTx{
		insertedIDs:       []string{entry.RequestID},
		failExecSubstring: `INSERT INTO "LiteLLM_DailyTagSpend"`,
	}

	_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "Tag daily aggregation")
	assert.False(t, tx.committed)
	assert.True(t, tx.rolledBack)
	assert.Empty(t, tx.committedSQL, "raw, counters, and the first four daily tables must roll back together")
	assert.Equal(t, 5, countSQLContaining(tx.attemptedSQL, `INSERT INTO "LiteLLM_Daily`),
		"the injected tag failure happens after the other four daily projections")
	assert.Zero(t, logger.Stats().AggregationErrors,
		"a rolled-back projection attempt is recoverable through the exact retained batch")
}

func TestAtomicWriterRollbackIgnoresExpiredRequestContext(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-deadline")
	tx := &atomicTestTx{
		insertedIDs:       []string{entry.RequestID},
		failExecSubstring: `INSERT INTO "LiteLLM_DailyTagSpend"`,
		failExecErr:       context.DeadlineExceeded,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := logger.commitBatchTransaction(ctx, tx, []*models.SpendLogEntry{entry})

	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.True(t, tx.rolledBack)
	assert.NoError(t, tx.rollbackContextErr,
		"rollback must not inherit the expired response context")
	assert.True(t, tx.rollbackContextHadDeadline,
		"rollback cleanup must remain bounded")
	assert.Zero(t, logger.Stats().AggregationErrors)
}

func TestAtomicWriterRetryAndReplayCannotDoubleCharge(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-retry")
	batch := []*models.SpendLogEntry{entry}

	failed := &atomicTestTx{
		insertedIDs:       []string{entry.RequestID},
		failExecSubstring: `INSERT INTO "LiteLLM_DailyTagSpend"`,
	}
	_, err := logger.commitBatchTransaction(context.Background(), failed, batch)
	require.Error(t, err)
	assert.Empty(t, failed.committedSQL)

	retry := &atomicTestTx{
		insertedIDs: []string{entry.RequestID},
	}
	inserted, err := logger.commitBatchTransaction(context.Background(), retry, batch)
	require.NoError(t, err)
	assert.Equal(t, []string{entry.RequestID}, inserted)
	assert.True(t, retry.committed)
	assert.Equal(t, 5, countSQLContaining(retry.committedSQL, `INSERT INTO "LiteLLM_Daily`))
	require.NotEmpty(t, retry.committedSQL, "the successful retry applies counters and daily projections once")

	replay := &atomicTestTx{}
	inserted, err = logger.commitBatchTransaction(context.Background(), replay, batch)
	require.NoError(t, err)
	assert.Empty(t, inserted)
	assert.True(t, replay.committed)
	assert.Empty(t, replay.attemptedSQL,
		"a raw-row conflict returns no IDs, so replay cannot update counters or daily tables")
	assert.Len(t, replay.queries, 1, "a conflict-only replay must not reload or aggregate existing rows")
}

func TestAtomicWriterUsesProviderResponseIDForNormalInsert(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicProviderIDEntry("chatcmpl-provider", "air-event-normal")
	tx := &atomicTestTx{
		insertResults: [][]string{{entry.RequestID}},
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.NoError(t, err)
	assert.Equal(t, []string{"chatcmpl-provider"}, inserted)
	require.Len(t, tx.queryArgs, 1)
	assert.Equal(t, "chatcmpl-provider", tx.queryArgs[0][0])
	assert.Equal(t, 1, countSQLContaining(tx.committedSQL, `UPDATE "LiteLLM_VerificationToken"`))
	assert.Equal(t, 5, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
}

func TestAtomicWriterKeepsLegacyFailureRawCallTypeEmptyButProjectsOriginalRoute(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-failure")
	entry.CallType = ""
	entry.Status = "failure"
	entry.Spend = 0
	entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"acompletion"}}`
	tx := &atomicTestTx{
		insertedIDs: []string{entry.RequestID},
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.NoError(t, err)
	assert.Equal(t, []string{entry.RequestID}, inserted)
	require.NotEmpty(t, tx.queryArgs)
	assert.Equal(t, "", tx.queryArgs[0][1], "the writer must not mutate a legacy raw failure row")
	assert.Equal(t, 5, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`),
		"the preserved original route must feed every daily aggregate")
	assert.Equal(t, 1, countSQLContaining(tx.committedSQL, `UPDATE "LiteLLM_TagTable" SET spend = spend + $1, updated_at = NOW()`))
}

func TestLegacyFailureOriginalRouteEnablesDailyAggregationWithoutPopulatingEndpoint(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-failure-dimensions")
	entry.CallType = ""
	entry.Status = "failure"
	entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"acompletion"}}`

	records, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.False(t, records[0].SkipDaily)
	assert.Empty(t, records[0].Endpoint, "legacy empty raw call_type retains its historical daily key")
}

func TestAtomicWriterSortsPreferredIDsBeforeUniqueIndexLocks(t *testing.T) {
	logger := newAtomicTestLogger()
	secondID := atomicProviderIDEntry("chatcmpl-q", "air-event-q")
	firstID := atomicProviderIDEntry("chatcmpl-p", "air-event-p")
	tx := &atomicTestTx{
		insertResults: [][]string{{"chatcmpl-p", "chatcmpl-q"}},
	}

	_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{secondID, firstID})

	require.NoError(t, err)
	require.NotEmpty(t, tx.queryArgs)
	assert.Equal(t, "chatcmpl-p", tx.queryArgs[0][0])
	assert.Equal(t, "chatcmpl-q", tx.queryArgs[0][queries.SpendLogParamCount])
}

func TestAtomicWriterSameBatchProviderIDCollisionUsesEventFallback(t *testing.T) {
	logger := newAtomicTestLogger()
	first := atomicProviderIDEntry("chatcmpl-shared", "air-event-1")
	second := atomicProviderIDEntry("chatcmpl-shared", "air-event-2")
	tx := &atomicTestTx{
		insertResults: [][]string{{"chatcmpl-shared"}, {"air-event-2"}},
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{first, second})

	require.NoError(t, err)
	assert.Equal(t, []string{"chatcmpl-shared", "air-event-2"}, inserted)
	require.Len(t, tx.queryArgs, 2)
	assert.Len(t, tx.queryArgs[0], queries.SpendLogParamCount, "one representative claims the provider ID")
	assert.Equal(t, "chatcmpl-shared", tx.queryArgs[0][0])
	assert.Len(t, tx.queryArgs[1], queries.SpendLogParamCount, "the colliding logical effect uses its event ID")
	assert.Equal(t, "air-event-2", tx.queryArgs[1][0])
	assert.Equal(t, 5, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
}

func TestAtomicWriterConcurrentProviderIDWinnerFallsBackToEventID(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicProviderIDEntry("chatcmpl-concurrent", "air-event-loser")
	tx := &atomicTestTx{
		insertResults: [][]string{{}, {"air-event-loser"}},
		// The preferred row is owned by a different event ("air-event-winner"),
		// so the ownership claim misses and the event falls back to its own ID.
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.NoError(t, err)
	assert.Equal(t, []string{"air-event-loser"}, inserted)
	require.Len(t, tx.queries, 3)
	assert.Equal(t, queries.QueryClaimSpendLogEventOwner, tx.queries[1],
		"the ownership claim must run in a statement after the conflicting INSERT")
	assert.Equal(t, []any{"chatcmpl-concurrent", "air-event-loser"}, tx.queryArgs[1])
	assert.Equal(t, "air-event-loser", tx.queryArgs[2][0])
	assert.Equal(t, 5, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
}

func TestAtomicWriterProviderIDReplayDoesNotFeedAnyAccounting(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicProviderIDEntry("chatcmpl-replay", "air-event-replay")
	tx := &atomicTestTx{
		insertResults: [][]string{{}},
		claimHits:     map[[2]string]bool{{"chatcmpl-replay", "air-event-replay"}: true},
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.NoError(t, err)
	assert.Empty(t, inserted)
	assert.Len(t, tx.queries, 2, "replay needs only preferred INSERT and the ownership claim")
	assert.Empty(t, tx.attemptedSQL, "the claim matches our event, so counters/daily must not run")
}

func TestAtomicWriterExistingOwnerCanBeLaterEntryInSameBatch(t *testing.T) {
	logger := newAtomicTestLogger()
	newEffect := atomicProviderIDEntry("chatcmpl-reordered", "air-event-new")
	replayedOwner := atomicProviderIDEntry("chatcmpl-reordered", "air-event-owner")
	tx := &atomicTestTx{
		insertResults: [][]string{{}, {"air-event-new"}},
		claimHits:     map[[2]string]bool{{"chatcmpl-reordered", "air-event-owner"}: true},
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{newEffect, replayedOwner})

	require.NoError(t, err)
	assert.Equal(t, []string{"air-event-new"}, inserted)
	require.Len(t, tx.queryArgs, 4)
	assert.Len(t, tx.queryArgs[3], queries.SpendLogParamCount, "only the non-owner effect gets a fallback row")
	assert.Equal(t, "air-event-new", tx.queryArgs[3][0])
}

func TestAtomicWriterEventFallbackReplayDoesNotFeedAccounting(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicProviderIDEntry("chatcmpl-collision-replay", "air-event-already-stored")
	tx := &atomicTestTx{
		insertResults: [][]string{{}, {}},
		// The preferred row belongs to "air-event-original-owner", so the claim
		// misses; the fallback row already exists and its INSERT conflicts.
	}

	inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})

	require.NoError(t, err)
	assert.Empty(t, inserted)
	assert.Len(t, tx.queries, 3, "fallback conflict is a terminal replay")
	assert.Empty(t, tx.attemptedSQL, "only INSERT RETURNING rows may feed accounting")
}

func newAtomicTestLogger() *Logger {
	return &Logger{
		logger: testhelpers.NewTestLogger(),
		queue:  make(chan *models.SpendLogEntry),
	}
}

func atomicTestEntry(requestID string) *models.SpendLogEntry {
	return &models.SpendLogEntry{
		RequestID:          requestID,
		StartTime:          time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC),
		EndTime:            time.Date(2026, 7, 15, 12, 0, 1, 0, time.UTC),
		CallType:           "acompletion",
		APIKey:             "key-1",
		Spend:              1.25,
		PromptTokens:       10,
		CompletionTokens:   5,
		TotalTokens:        15,
		Model:              "model-1",
		ModelGroup:         "group-1",
		CustomLLMProvider:  "openai",
		UserID:             "user-1",
		TeamID:             "team-1",
		OrganizationID:     "org-1",
		EndUser:            "end-user-1",
		RequestTags:        `["tag-1"]`,
		Status:             "success",
		ComparisonEligible: true,
	}
}

func atomicProviderIDEntry(providerID, eventID string) *models.SpendLogEntry {
	entry := atomicTestEntry(providerID)
	entry.AirEventID = eventID
	entry.Metadata = fmt.Sprintf(`{"spend_logs_metadata":{"air_event_id":%q,"provider_response_id":%q}}`, eventID, providerID)
	return entry
}

func countSQLContaining(statements []string, fragment string) int {
	count := 0
	for _, statement := range statements {
		if strings.Contains(statement, fragment) {
			count++
		}
	}
	return count
}

type atomicTestTx struct {
	insertedIDs                []string
	insertResults              [][]string
	insertQueryIndex           int
	claimHits                  map[[2]string]bool
	failExecSubstring          string
	failExecErr                error
	commitErr                  error
	queries                    []string
	queryArgs                  [][]any
	attemptedSQL               []string
	stagedSQL                  []string
	committedSQL               []string
	committed                  bool
	rolledBack                 bool
	rollbackContextErr         error
	rollbackContextHadDeadline bool
}

func (tx *atomicTestTx) Begin(context.Context) (pgx.Tx, error) { return tx, nil }

func (tx *atomicTestTx) Commit(context.Context) error {
	if tx.commitErr != nil {
		return tx.commitErr
	}
	tx.committed = true
	tx.committedSQL = append(tx.committedSQL, tx.stagedSQL...)
	return nil
}

func (tx *atomicTestTx) Rollback(ctx context.Context) error {
	tx.rollbackContextErr = ctx.Err()
	_, tx.rollbackContextHadDeadline = ctx.Deadline()
	if tx.committed {
		return pgx.ErrTxClosed
	}
	tx.rolledBack = true
	tx.stagedSQL = nil
	return nil
}

func (tx *atomicTestTx) CopyFrom(context.Context, pgx.Identifier, []string, pgx.CopyFromSource) (int64, error) {
	return 0, errors.New("unexpected CopyFrom")
}

func (tx *atomicTestTx) SendBatch(context.Context, *pgx.Batch) pgx.BatchResults { return nil }
func (tx *atomicTestTx) LargeObjects() pgx.LargeObjects                         { return pgx.LargeObjects{} }

func (tx *atomicTestTx) Prepare(context.Context, string, string) (*pgconn.StatementDescription, error) {
	return nil, errors.New("unexpected Prepare")
}

func (tx *atomicTestTx) Exec(_ context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
	tx.attemptedSQL = append(tx.attemptedSQL, sql)
	if tx.failExecSubstring != "" && strings.Contains(sql, tx.failExecSubstring) {
		if tx.failExecErr != nil {
			return pgconn.CommandTag{}, tx.failExecErr
		}
		return pgconn.CommandTag{}, errors.New("injected exec failure")
	}
	tx.stagedSQL = append(tx.stagedSQL, sql)
	return pgconn.CommandTag{}, nil
}

func (tx *atomicTestTx) Query(_ context.Context, sql string, args ...any) (pgx.Rows, error) {
	tx.queries = append(tx.queries, sql)
	tx.queryArgs = append(tx.queryArgs, append([]any(nil), args...))
	switch {
	case strings.Contains(sql, `INSERT INTO "LiteLLM_SpendLogs"`):
		insertedIDs := tx.insertedIDs
		if tx.insertResults != nil {
			if tx.insertQueryIndex >= len(tx.insertResults) {
				return nil, errors.New("unexpected extra spend log insert")
			}
			insertedIDs = tx.insertResults[tx.insertQueryIndex]
		}
		tx.insertQueryIndex++
		rows := make([][]any, 0, len(insertedIDs))
		for _, id := range insertedIDs {
			rows = append(rows, []any{id})
		}
		if len(rows) > 0 {
			tx.stagedSQL = append(tx.stagedSQL, sql)
		}
		return &atomicTestRows{rows: rows}, nil
	default:
		return nil, errors.New("unexpected query")
	}
}

func (tx *atomicTestTx) QueryRow(_ context.Context, sql string, args ...any) pgx.Row {
	tx.queries = append(tx.queries, sql)
	tx.queryArgs = append(tx.queryArgs, append([]any(nil), args...))
	if strings.Contains(sql, `UPDATE "LiteLLM_SpendLogs"`) {
		if tx.claimHits[[2]string{args[0].(string), args[1].(string)}] {
			return atomicTestRow{value: args[0].(string)}
		}
		return atomicTestRow{err: pgx.ErrNoRows}
	}
	return atomicTestRow{err: errors.New("unexpected QueryRow")}
}

func (tx *atomicTestTx) Conn() *pgx.Conn { return nil }

type atomicTestRow struct {
	err   error
	value string
}

func (row atomicTestRow) Scan(dest ...any) error {
	if row.err != nil {
		return row.err
	}
	if len(dest) != 1 {
		return errors.New("unexpected scan destination count")
	}
	value, ok := dest[0].(*string)
	if !ok {
		return errors.New("unexpected scan destination type")
	}
	*value = row.value
	return nil
}

type atomicTestRows struct {
	rows    [][]any
	index   int
	current []any
	closed  bool
	err     error
}

func (rows *atomicTestRows) Close() { rows.closed = true }
func (rows *atomicTestRows) Err() error {
	return rows.err
}
func (rows *atomicTestRows) CommandTag() pgconn.CommandTag { return pgconn.CommandTag{} }
func (rows *atomicTestRows) FieldDescriptions() []pgconn.FieldDescription {
	return nil
}
func (rows *atomicTestRows) Next() bool {
	if rows.index >= len(rows.rows) {
		rows.closed = true
		return false
	}
	rows.current = rows.rows[rows.index]
	rows.index++
	return true
}
func (rows *atomicTestRows) Scan(dest ...any) error {
	if len(dest) != len(rows.current) {
		return errors.New("scan destination count mismatch")
	}
	for i := range dest {
		if err := assignAtomicTestValue(dest[i], rows.current[i]); err != nil {
			return err
		}
	}
	return nil
}
func (rows *atomicTestRows) Values() ([]any, error) {
	return append([]any(nil), rows.current...), nil
}
func (rows *atomicTestRows) RawValues() [][]byte { return nil }
func (rows *atomicTestRows) Conn() *pgx.Conn     { return nil }

func assignAtomicTestValue(destination, source any) error {
	dest := reflect.ValueOf(destination)
	if dest.Kind() != reflect.Pointer || dest.IsNil() {
		return errors.New("scan destination is not a pointer")
	}
	dest = dest.Elem()
	if source == nil {
		dest.SetZero()
		return nil
	}

	src := reflect.ValueOf(source)
	if src.Type().AssignableTo(dest.Type()) {
		dest.Set(src)
		return nil
	}
	if dest.Kind() == reflect.Pointer && src.Type().AssignableTo(dest.Type().Elem()) {
		value := reflect.New(dest.Type().Elem())
		value.Elem().Set(src)
		dest.Set(value)
		return nil
	}
	return errors.New("incompatible scan value")
}
