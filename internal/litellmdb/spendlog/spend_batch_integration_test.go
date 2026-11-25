package spendlog

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// accountingTables are the tables the writer's transaction changes, in dump order.
var accountingTables = []string{
	"LiteLLM_SpendLogs", "LiteLLM_VerificationToken", "LiteLLM_UserTable", "LiteLLM_TeamTable",
	"LiteLLM_OrganizationTable", "LiteLLM_TeamMembership", "LiteLLM_OrganizationMembership",
	"LiteLLM_EndUserTable", "LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend",
	"LiteLLM_DailyOrganizationSpend", "LiteLLM_DailyEndUserSpend", "LiteLLM_BudgetWindowSpend",
}

// accountingTestPool connects to AIR_LITELLM_TEST_DATABASE_URL with a schema of its own holding
// every table the writer changes; legacy leaves out last_active and total_spend.
func accountingTestPool(t *testing.T, legacy bool) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("AIR_LITELLM_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("AIR_LITELLM_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	schema := "accounting_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
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

	lastActive, totalSpend := ", last_active TIMESTAMP(3)", ", total_spend DOUBLE PRECISION NOT NULL DEFAULT 0"
	if legacy {
		lastActive, totalSpend = "", ""
	}
	daily := func(table, entity string) string {
		return fmt.Sprintf(`CREATE TABLE %q (
  id TEXT PRIMARY KEY, %s TEXT, date TEXT NOT NULL, api_key TEXT NOT NULL, model TEXT, model_group TEXT,
  custom_llm_provider TEXT, mcp_namespaced_tool_name TEXT, endpoint TEXT,
  prompt_tokens BIGINT NOT NULL DEFAULT 0, completion_tokens BIGINT NOT NULL DEFAULT 0,
  cache_read_input_tokens BIGINT NOT NULL DEFAULT 0, cache_creation_input_tokens BIGINT NOT NULL DEFAULT 0,
  spend DOUBLE PRECISION NOT NULL DEFAULT 0, api_requests BIGINT NOT NULL DEFAULT 0,
  successful_requests BIGINT NOT NULL DEFAULT 0, failed_requests BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP(3) NOT NULL,
  UNIQUE (%s, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
);`, table, entity, entity)
	}
	entity := func(table, id string) string {
		return fmt.Sprintf(`CREATE TABLE %q (%s TEXT PRIMARY KEY, spend DOUBLE PRECISION NOT NULL DEFAULT 0,
  model_spend JSONB NOT NULL DEFAULT '{}', updated_at TIMESTAMP(3));`, table, id)
	}
	_, err = pool.Exec(t.Context(), `
CREATE TABLE "LiteLLM_SpendLogs" (
  request_id TEXT PRIMARY KEY, call_type TEXT NOT NULL, api_key TEXT NOT NULL DEFAULT '',
  spend DOUBLE PRECISION NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
  "startTime" TIMESTAMP(3) NOT NULL, "endTime" TIMESTAMP(3) NOT NULL, request_duration_ms INTEGER,
  "completionStartTime" TIMESTAMP(3), model TEXT NOT NULL DEFAULT '', model_id TEXT, model_group TEXT,
  custom_llm_provider TEXT, api_base TEXT, "user" TEXT, metadata JSONB DEFAULT '{}', cache_hit TEXT,
  cache_key TEXT, team_id TEXT, organization_id TEXT, end_user TEXT, requester_ip_address TEXT,
  session_id TEXT, status TEXT, messages JSONB, response JSONB, proxy_server_request JSONB
);
CREATE TABLE "LiteLLM_VerificationToken" (
  token TEXT PRIMARY KEY, spend DOUBLE PRECISION NOT NULL DEFAULT 0, model_spend JSONB NOT NULL DEFAULT '{}',
  budget_limits JSONB, updated_at TIMESTAMP(3)`+lastActive+`
);
`+entity("LiteLLM_UserTable", "user_id")+entity("LiteLLM_TeamTable", "team_id")+
		entity("LiteLLM_OrganizationTable", "organization_id")+`
CREATE TABLE "LiteLLM_TeamMembership" (
  user_id TEXT NOT NULL, team_id TEXT NOT NULL, spend DOUBLE PRECISION NOT NULL DEFAULT 0`+totalSpend+`,
  PRIMARY KEY (user_id, team_id)
);
CREATE TABLE "LiteLLM_OrganizationMembership" (
  user_id TEXT NOT NULL, organization_id TEXT NOT NULL, spend DOUBLE PRECISION DEFAULT 0, updated_at TIMESTAMP(3),
  PRIMARY KEY (user_id, organization_id)
);
CREATE TABLE "LiteLLM_EndUserTable" (user_id TEXT PRIMARY KEY, spend DOUBLE PRECISION NOT NULL DEFAULT 0);
`+daily("LiteLLM_DailyUserSpend", "user_id")+daily("LiteLLM_DailyTeamSpend", "team_id")+
		daily("LiteLLM_DailyOrganizationSpend", "organization_id")+daily("LiteLLM_DailyEndUserSpend", "end_user_id")+`
CREATE TABLE "LiteLLM_BudgetWindowSpend" (
  entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, window_duration TEXT NOT NULL,
  window_start TIMESTAMP(3) NOT NULL, spend DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP(3) NOT NULL,
  PRIMARY KEY (entity_type, entity_id, window_duration)
);`)
	require.NoError(t, err)

	// The batch's keys, users, teams, organisations and memberships exist; end users are upserted.
	for i := range 4 {
		_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_VerificationToken" (token, budget_limits) VALUES ($1, $2)`,
			fmt.Sprintf("key-%d", i), []byte(`[{"budget_duration":"1d","max_budget":100}]`))
		require.NoError(t, err)
	}
	for u := range 3 {
		user := fmt.Sprintf("user-%d", u)
		_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_UserTable" (user_id) VALUES ($1)`, user)
		require.NoError(t, err)
		for team := range 3 {
			_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id) VALUES ($1, $2)`, user, fmt.Sprintf("team-%d", team))
			require.NoError(t, err)
		}
		for org := range 2 {
			_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id) VALUES ($1, $2)`, user, fmt.Sprintf("org-%d", org))
			require.NoError(t, err)
		}
	}
	for team := range 3 {
		_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_TeamTable" (team_id) VALUES ($1)`, fmt.Sprintf("team-%d", team))
		require.NoError(t, err)
	}
	for org := range 2 {
		_, err = pool.Exec(t.Context(), `INSERT INTO "LiteLLM_OrganizationTable" (organization_id) VALUES ($1)`, fmt.Sprintf("org-%d", org))
		require.NoError(t, err)
	}
	return pool
}

// randomSpendBatches makes batches whose keys, users, teams and models repeat within and across batches.
func randomSpendBatches(seed uint64, batches, size int) [][]*models.SpendLogEntry {
	random := rand.New(rand.NewPCG(seed, 1))
	pick := func(values ...string) string { return values[random.IntN(len(values))] }
	start := time.Date(2026, 11, 21, 23, 58, 0, 0, time.UTC)
	result := make([][]*models.SpendLogEntry, 0, batches)
	for b := range batches {
		batch := make([]*models.SpendLogEntry, 0, size)
		for i := range size {
			at := start.Add(time.Duration(random.IntN(240)) * time.Second)
			entry := &models.SpendLogEntry{
				RequestID:         fmt.Sprintf("req-%d-%d", b, i),
				AirEventID:        fmt.Sprintf("event-%d-%d", b, i),
				StartTime:         at,
				EndTime:           at.Add(time.Second),
				CallType:          pick("acompletion", "aresponses", "anthropic_messages", "aembedding"),
				APIKey:            pick("key-0", "key-0", "key-0", "key-1", "key-2", "key-3"),
				Spend:             float64(random.IntN(10000)) / 1e4,
				PromptTokens:      random.IntN(5000),
				CompletionTokens:  random.IntN(800),
				Model:             pick("gpt-5", "gpt-5-mini", "claude-sonnet"),
				ModelGroup:        pick("group-a", "group-b"),
				CustomLLMProvider: pick("openai", "anthropic"),
				UserID:            pick("user-0", "user-1", "user-2", ""),
				TeamID:            pick("team-0", "team-1", "team-2", ""),
				OrganizationID:    pick("org-0", "org-1", ""),
				EndUser:           pick("end-0", "end-1", ""),
				Status:            pick("success", "success", "failure"),
				Metadata:          `{"usage_object":{"cache_read_input_tokens":7}}`,
			}
			entry.TotalTokens = entry.PromptTokens + entry.CompletionTokens
			entry.BillingTeamID = entry.TeamID
			batch = append(batch, entry)
		}
		result = append(result, batch)
	}
	return result
}

// dumpAccountingTables returns every row of the accounting tables without the columns set by now() or gen_random_uuid().
func dumpAccountingTables(t *testing.T, pool *pgxpool.Pool) map[string][]string {
	t.Helper()
	dump := make(map[string][]string, len(accountingTables))
	for _, table := range accountingTables {
		rows, err := pool.Query(t.Context(), fmt.Sprintf(
			`SELECT (to_jsonb(t) - 'id' - 'created_at' - 'updated_at' - 'last_active')::text AS row FROM %q t ORDER BY 1`, table))
		require.NoError(t, err)
		values, err := pgx.CollectRows(rows, pgx.RowTo[string])
		require.NoError(t, err)
		dump[table] = values
	}
	return dump
}

func writeBatches(t *testing.T, pool *pgxpool.Pool, logger *Logger, batches [][]*models.SpendLogEntry) {
	t.Helper()
	for _, batch := range batches {
		tx, err := pool.BeginTx(t.Context(), pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
		require.NoError(t, err)
		inserted, err := logger.commitBatchTransaction(t.Context(), tx, batch)
		require.NoError(t, err)
		require.Len(t, inserted, len(batch))
	}
}

func accountingTestLogger(singleRoundTrip, windows bool) *Logger {
	logger := newAtomicTestLogger()
	logger.logger = testhelpers.NewTestLogger()
	logger.config.SpendUpdatesSingleRoundTrip = singleRoundTrip
	logger.config.EnforceKeyBudgetWindows = windows
	return logger
}

func TestSpendUpdatesSingleRoundTripWritesTheSameRowsInPostgres(t *testing.T) {
	for _, windows := range []bool{false, true} {
		t.Run(fmt.Sprintf("budget_windows=%v", windows), func(t *testing.T) {
			restoreSchemaFlags(t)
			seed := uint64(time.Now().UnixNano())
			t.Logf("seed %d", seed)

			oneByOne := accountingTestPool(t, false)
			writeBatches(t, oneByOne, accountingTestLogger(false, windows), randomSpendBatches(seed, 4, 60))
			batched := accountingTestPool(t, false)
			writeBatches(t, batched, accountingTestLogger(true, windows), randomSpendBatches(seed, 4, 60))

			want := dumpAccountingTables(t, oneByOne)
			assert.Len(t, want["LiteLLM_SpendLogs"], 240)
			assert.NotEmpty(t, want["LiteLLM_DailyEndUserSpend"])
			assert.Equal(t, windows, len(want["LiteLLM_BudgetWindowSpend"]) > 0)
			assert.Equal(t, want, dumpAccountingTables(t, batched))
		})
	}
}

func TestSpendUpdatesSingleRoundTripRollsBackEveryRowInPostgres(t *testing.T) {
	restoreSchemaFlags(t)
	pool := accountingTestPool(t, false)
	before := dumpAccountingTables(t, pool)
	_, err := pool.Exec(t.Context(), `ALTER TABLE "LiteLLM_DailyEndUserSpend" ADD CONSTRAINT no_spend CHECK (spend < 0)`)
	require.NoError(t, err)
	batch := randomSpendBatches(7, 1, 40)[0]
	batch[0].EndUser, batch[0].Spend = "end-0", 1

	tx, err := pool.BeginTx(t.Context(), pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	require.NoError(t, err)
	_, err = accountingTestLogger(true, true).commitBatchTransaction(t.Context(), tx, batch)

	var pgErr *pgconn.PgError
	require.ErrorAs(t, err, &pgErr)
	assert.Equal(t, "23514", pgErr.Code)
	assert.Equal(t, before, dumpAccountingTables(t, pool), "the insert, counters, windows and daily rows roll back together")
}

func TestSpendUpdatesSingleRoundTripFallsBackOnAnOlderSchemaInPostgres(t *testing.T) {
	restoreSchemaFlags(t)
	batches := randomSpendBatches(11, 1, 30)
	oneByOne := accountingTestPool(t, true)
	batched := accountingTestPool(t, true)

	for name, run := range map[string]struct {
		pool   *pgxpool.Pool
		single bool
	}{"one_by_one": {oneByOne, false}, "batched": {batched, true}} {
		restoreSchemaFlags(t)
		logger := accountingTestLogger(run.single, false)
		// last_active, then total_spend are found missing; each aborts one attempt, as in flushBatch's retries.
		for attempt := 1; ; attempt++ {
			tx, err := run.pool.BeginTx(t.Context(), pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
			require.NoError(t, err)
			_, err = logger.commitBatchTransaction(t.Context(), tx, batches[0])
			if err == nil {
				assert.Equal(t, 3, attempt, name)
				break
			}
			require.True(t, isColumnNotExist(err), "%s: %v", name, err)
			require.Less(t, attempt, 3, name)
		}
		assert.False(t, schemaTokenHasLastActive.Load(), name)
		assert.False(t, schemaTeamMemberHasTotalSpend.Load(), name)
	}
	assert.Equal(t, dumpAccountingTables(t, oneByOne), dumpAccountingTables(t, batched))
}
