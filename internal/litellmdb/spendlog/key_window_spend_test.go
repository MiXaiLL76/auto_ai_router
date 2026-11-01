package spendlog

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// noWindowTableTx is a transaction on a schema from before LiteLLM added
// LiteLLM_BudgetWindowSpend.
type noWindowTableTx struct {
	*atomicTestTx
}

func (tx noWindowTableTx) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	if sql == queries.QuerySelectKeysBudgetWindows {
		tx.queries = append(tx.queries, sql)
		return nil, &pgconn.PgError{Code: "42P01"}
	}
	return tx.atomicTestTx.Query(ctx, sql, args...)
}

func TestUpdateKeyBudgetWindows_SchemaWithoutWindowTable(t *testing.T) {
	t.Cleanup(func() { schemaHasKeyBudgetWindows.Store(true) })
	logger := newAtomicTestLogger()
	tx := noWindowTableTx{&atomicTestTx{}}
	entries := []*models.SpendLogEntry{atomicTestEntry("req-old-schema")}

	err := logger.updateKeyBudgetWindows(context.Background(), tx, entries)
	require.Error(t, err, "the aborted transaction is retried")
	assert.False(t, schemaHasKeyBudgetWindows.Load())

	// The retry writes the batch without the window projection.
	tx.queries = nil
	require.NoError(t, logger.updateKeyBudgetWindows(context.Background(), tx, entries))
	assert.Empty(t, tx.queries)
}

// windowTestPool connects to AIR_LITELLM_TEST_DATABASE_URL with a schema of
// its own holding the LiteLLM tables the window projection reads and writes
// (columns as in LiteLLM v1.103.3's schema.prisma).
func windowTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("AIR_LITELLM_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("AIR_LITELLM_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	schema := "window_test_" + uuid.NewString()
	_, err = admin.Exec(t.Context(), "CREATE SCHEMA "+pgx.Identifier{schema}.Sanitize())
	require.NoError(t, err)
	t.Cleanup(func() {
		_, cleanupErr := admin.Exec(context.Background(), "DROP SCHEMA "+pgx.Identifier{schema}.Sanitize()+" CASCADE")
		require.NoError(t, cleanupErr)
		admin.Close()
	})

	cfg, err := pgxpool.ParseConfig(databaseURL)
	require.NoError(t, err)
	cfg.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(t.Context(), cfg)
	require.NoError(t, err)
	t.Cleanup(pool.Close)

	_, err = pool.Exec(t.Context(), `
CREATE TABLE "LiteLLM_VerificationToken" (
  token TEXT PRIMARY KEY,
  spend DOUBLE PRECISION NOT NULL DEFAULT 0,
  budget_limits JSONB
);
CREATE TABLE "LiteLLM_SpendLogs" (
  request_id TEXT PRIMARY KEY,
  api_key TEXT NOT NULL DEFAULT '',
  spend DOUBLE PRECISION NOT NULL DEFAULT 0,
  "startTime" TIMESTAMP(3) NOT NULL,
  metadata JSONB DEFAULT '{}'
);
CREATE TABLE "LiteLLM_BudgetWindowSpend" (
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  window_duration TEXT NOT NULL,
  window_start TIMESTAMP(3) NOT NULL,
  spend DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP(3) NOT NULL,
  PRIMARY KEY (entity_type, entity_id, window_duration)
);`)
	require.NoError(t, err)
	return pool
}

type windowTestDB struct {
	t    *testing.T
	pool *pgxpool.Pool
}

func (db windowTestDB) exec(sql string, args ...any) {
	db.t.Helper()
	_, err := db.pool.Exec(db.t.Context(), sql, args...)
	require.NoError(db.t, err)
}

// spendLog stores a spend log row as the writer would have before, or as
// another writer did.
func (db windowTestDB) spendLog(q interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}, key string, spend float64, startTime time.Time, metadata string) {
	db.t.Helper()
	_, err := q.Exec(db.t.Context(),
		`INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, spend, "startTime", metadata) VALUES ($1, $2, $3, $4, $5)`,
		uuid.NewString(), key, spend, startTime, metadata)
	require.NoError(db.t, err)
}

// row returns the key's window row of duration.
func (db windowTestDB) row(key, duration string) (time.Time, float64, bool) {
	db.t.Helper()
	var start time.Time
	var spend float64
	err := db.pool.QueryRow(db.t.Context(),
		`SELECT window_start, spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = 'key' AND entity_id = $1 AND window_duration = $2`,
		key, duration).Scan(&start, &spend)
	if err == pgx.ErrNoRows {
		return time.Time{}, 0, false
	}
	require.NoError(db.t, err)
	return start, spend, true
}

// write runs the projection the way writeBatchInTransaction does: the batch's
// own spend log rows are inserted and its keys row-locked first.
func (db windowTestDB) write(logger *Logger, entries ...*models.SpendLogEntry) {
	db.t.Helper()
	tx, err := db.pool.Begin(db.t.Context())
	require.NoError(db.t, err)
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, entry := range entries {
		db.spendLog(tx, entry.APIKey, entry.Spend, entry.StartTime, "{}")
		_, err := tx.Exec(db.t.Context(), `UPDATE "LiteLLM_VerificationToken" SET spend = spend + $1 WHERE token = $2`, entry.Spend, entry.APIKey)
		require.NoError(db.t, err)
	}
	require.NoError(db.t, logger.updateKeyBudgetWindows(db.t.Context(), tx, entries))
	require.NoError(db.t, tx.Commit(db.t.Context()))
}

func windowTestEntry(key string, spend float64, startTime time.Time) *models.SpendLogEntry {
	return &models.SpendLogEntry{RequestID: uuid.NewString(), APIKey: key, Spend: spend, StartTime: startTime}
}

func TestUpdateKeyBudgetWindows_Postgres(t *testing.T) {
	pool := windowTestPool(t)
	db := windowTestDB{t: t, pool: pool}
	moscow, err := time.LoadLocation("Europe/Moscow")
	require.NoError(t, err)
	logger := newAtomicTestLogger()
	logger.config.BudgetTimezone = moscow

	// 11 March 2026 in Moscow runs from 10 March 21:00 UTC.
	day := time.Date(2026, 3, 10, 21, 0, 0, 0, time.UTC)
	nextDay := day.AddDate(0, 0, 1)

	t.Run("first write seeds the window from the accounted spend logs", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('seed', '[{"budget_duration":"1d","max_budget":10}]')`)
		db.spendLog(pool, "seed", 2, day.Add(time.Hour), "{}")
		db.spendLog(pool, "seed", 5, day.Add(-time.Minute), "{}")                                                   // previous day
		db.spendLog(pool, "seed", 7, day.Add(2*time.Hour), `{"spend_logs_metadata":{"accounting_eligible":false}}`) // never accounted
		db.spendLog(pool, "seed", 0.5, day.Add(3*time.Hour), `{"spend_logs_metadata":{"accounting_eligible":true}}`)

		db.write(logger, windowTestEntry("seed", 1.25, day.Add(4*time.Hour)))

		start, spend, ok := db.row("seed", "1d")
		require.True(t, ok)
		assert.Equal(t, day, start)
		assert.InDelta(t, 3.75, spend, 1e-9, "2 + 0.5 already logged + the batch's own 1.25, counted once")

		db.write(logger, windowTestEntry("seed", 1, day.Add(5*time.Hour)))
		_, spend, _ = db.row("seed", "1d")
		assert.InDelta(t, 4.75, spend, 1e-9, "a row on the window gets the batch added")
	})

	t.Run("a new window starts from its own spend", func(t *testing.T) {
		db.write(logger, windowTestEntry("seed", 0.25, nextDay.Add(time.Minute)))
		start, spend, _ := db.row("seed", "1d")
		assert.Equal(t, nextDay, start)
		assert.InDelta(t, 0.25, spend, 1e-9)
	})

	t.Run("a late entry of an earlier window leaves the row alone", func(t *testing.T) {
		db.write(logger, windowTestEntry("seed", 9, nextDay.Add(-time.Second)))
		start, spend, _ := db.row("seed", "1d")
		assert.Equal(t, nextDay, start)
		assert.InDelta(t, 0.25, spend, 1e-9)
	})

	t.Run("a batch across the boundary", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('boundary', '[{"budget_duration":"1d","max_budget":10}]')`)
		db.write(logger, windowTestEntry("boundary", 1, day.Add(time.Hour)))

		db.write(logger,
			windowTestEntry("boundary", 2, nextDay.Add(-time.Minute)),
			windowTestEntry("boundary", 3, nextDay.Add(time.Minute)),
		)
		start, spend, _ := db.row("boundary", "1d")
		assert.Equal(t, nextDay, start)
		assert.InDelta(t, 3, spend, 1e-9)
	})

	t.Run("a row keeps counting while its limit is off", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('toggle', '[{"budget_duration":"1d","max_budget":10}]')`)
		db.write(logger, windowTestEntry("toggle", 1, day.Add(time.Hour)))

		db.exec(`UPDATE "LiteLLM_VerificationToken" SET budget_limits = NULL WHERE token = 'toggle'`)
		db.write(logger, windowTestEntry("toggle", 6, day.Add(2*time.Hour)))

		_, spend, _ := db.row("toggle", "1d")
		assert.InDelta(t, 7, spend, 1e-9, "putting the limit back finds the spend made without it")
	})

	t.Run("a window row is named by the duration as stored", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('spellings', '[{"budget_duration":"daily","max_budget":10},{"budget_duration":"1h","max_budget":10}]')`)
		db.write(logger, windowTestEntry("spellings", 1, day.Add(30*time.Minute)))
		start, spend, ok := db.row("spellings", "daily")
		require.True(t, ok)
		assert.Equal(t, day, start, "daily is LiteLLM's 24h: from Moscow midnight")
		assert.InDelta(t, 1, spend, 1e-9)
		start, _, ok = db.row("spellings", "1h")
		require.True(t, ok)
		assert.Equal(t, day, start)
	})

	t.Run("keys without windows get no rows", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token) VALUES ('plain')`)
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('two-days', '[{"budget_duration":"2d","max_budget":10}]')`)
		db.write(logger, windowTestEntry("plain", 1, day), windowTestEntry("two-days", 1, day))
		_, _, ok := db.row("plain", "1d")
		assert.False(t, ok)
		_, _, ok = db.row("two-days", "2d")
		assert.False(t, ok, "durations AIR cannot place on a calendar are not written")
	})

	t.Run("two writers of one key count both batches", func(t *testing.T) {
		db.exec(`INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ('race', '[{"budget_duration":"1d","max_budget":10}]')`)
		first, err := pool.Begin(t.Context())
		require.NoError(t, err)
		defer func() { _ = first.Rollback(context.Background()) }()
		second, err := pool.Begin(t.Context())
		require.NoError(t, err)
		defer func() { _ = second.Rollback(context.Background()) }()

		firstEntry := windowTestEntry("race", 1, day.Add(time.Hour))
		secondEntry := windowTestEntry("race", 2, day.Add(time.Hour))
		// Both writers have their spend log rows in before either takes the
		// key's lock, so neither can see the other's in its seed.
		db.spendLog(first, "race", firstEntry.Spend, firstEntry.StartTime, "{}")
		db.spendLog(second, "race", secondEntry.Spend, secondEntry.StartTime, "{}")

		_, err = first.Exec(t.Context(), `UPDATE "LiteLLM_VerificationToken" SET spend = spend + 1 WHERE token = 'race'`)
		require.NoError(t, err)
		require.NoError(t, logger.updateKeyBudgetWindows(t.Context(), first, []*models.SpendLogEntry{firstEntry}))

		done := make(chan error, 1)
		go func() {
			if _, err := second.Exec(context.Background(), `UPDATE "LiteLLM_VerificationToken" SET spend = spend + 2 WHERE token = 'race'`); err != nil {
				done <- err
				return
			}
			if err := logger.updateKeyBudgetWindows(context.Background(), second, []*models.SpendLogEntry{secondEntry}); err != nil {
				done <- err
				return
			}
			done <- second.Commit(context.Background())
		}()
		select {
		case err := <-done:
			t.Fatalf("the second writer did not wait for the first one's key lock: %v", err)
		case <-time.After(200 * time.Millisecond):
		}
		require.NoError(t, first.Commit(t.Context()))
		require.NoError(t, <-done)

		_, spend, _ := db.row("race", "1d")
		assert.InDelta(t, 3, spend, 1e-9)
	})
}
