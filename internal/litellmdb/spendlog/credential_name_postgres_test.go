package spendlog

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCredentialNamePostgres(t *testing.T) {
	databaseURL := os.Getenv("AIR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("AIR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	conn, err := pgx.Connect(ctx, databaseURL)
	require.NoError(t, err)
	defer func() { _ = conn.Close(ctx) }()
	_, err = conn.Exec(ctx, `CREATE TEMP TABLE "LiteLLM_SpendLogs" (
		request_id TEXT PRIMARY KEY, call_type TEXT, api_key TEXT, spend DOUBLE PRECISION CHECK (spend >= 0),
		total_tokens INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
		"startTime" TIMESTAMP, "endTime" TIMESTAMP, request_duration_ms INTEGER,
		"completionStartTime" TIMESTAMP, model TEXT, model_id TEXT, model_group TEXT,
		custom_llm_provider TEXT, api_base TEXT, "user" TEXT, metadata JSONB,
		cache_hit TEXT, cache_key TEXT, team_id TEXT, organization_id TEXT, end_user TEXT,
		requester_ip_address TEXT, session_id TEXT, status TEXT,
		messages JSONB, response JSONB, proxy_server_request JSONB
	)`)
	require.NoError(t, err)

	insert := func(enabled bool, entries ...*models.SpendLogEntry) ([]string, error) {
		t.Helper()
		tx, beginErr := conn.Begin(ctx)
		require.NoError(t, beginErr)
		defer func() { _ = tx.Rollback(ctx) }()
		ids, insertErr := insertSpendRowsReturningIDs(ctx, tx, entries, enabled)
		if insertErr != nil {
			return nil, insertErr
		}
		return ids, tx.Commit(ctx)
	}

	legacy := atomicTestEntry("legacy")
	legacy.CredentialName = "grant-legacy"
	ids, err := insert(false, legacy)
	require.NoError(t, err)
	assert.Equal(t, []string{"legacy"}, ids)

	first := atomicTestEntry("first")
	first.CredentialName = "grant'); DROP TABLE anything; --"
	_, err = insert(true, first)
	var pgErr *pgconn.PgError
	require.ErrorAs(t, err, &pgErr)
	assert.Equal(t, "42703", pgErr.Code)

	migration, err := os.ReadFile("../migrations/add_spend_log_credential_name.sql")
	require.NoError(t, err)
	for range 2 {
		_, err = conn.Exec(ctx, string(migration))
		require.NoError(t, err)
	}

	second := atomicTestEntry("second")
	second.CredentialName = "grant-2"
	ids, err = insert(true, first, second)
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{"first", "second"}, ids)

	for _, entry := range []*models.SpendLogEntry{first, second} {
		var credential, team string
		var spend float64
		var tokens int
		err = conn.QueryRow(ctx, `SELECT credential_name, team_id, spend, total_tokens
			FROM "LiteLLM_SpendLogs" WHERE request_id = $1`, entry.RequestID).Scan(&credential, &team, &spend, &tokens)
		require.NoError(t, err)
		assert.Equal(t, entry.CredentialName, credential)
		assert.Equal(t, entry.TeamID, team)
		assert.Equal(t, entry.Spend, spend)
		assert.Equal(t, entry.TotalTokens, tokens)
	}

	ids, err = insert(true, first, second)
	require.NoError(t, err)
	assert.Empty(t, ids)
	first.CredentialName = "must-not-overwrite"
	ids, err = insert(false, first)
	require.NoError(t, err)
	assert.Empty(t, ids)

	disabled := atomicTestEntry("disabled")
	disabled.CredentialName = "must-not-write"
	_, err = insert(false, disabled)
	require.NoError(t, err)
	missing := atomicTestEntry("missing")
	_, err = insert(true, missing)
	require.NoError(t, err)

	var rowCount, attributedCount int
	var totalSpend float64
	err = conn.QueryRow(ctx, `SELECT count(*), count(credential_name), sum(spend)
		FROM "LiteLLM_SpendLogs"`).Scan(&rowCount, &attributedCount, &totalSpend)
	require.NoError(t, err)
	assert.Equal(t, 5, rowCount)
	assert.Equal(t, 2, attributedCount)
	assert.Equal(t, 6.25, totalSpend)
	var credential string
	err = conn.QueryRow(ctx, `SELECT credential_name FROM "LiteLLM_SpendLogs" WHERE request_id = 'first'`).Scan(&credential)
	require.NoError(t, err)
	assert.Equal(t, "grant'); DROP TABLE anything; --", credential)

	for _, enabled := range []bool{false, true} {
		for _, count := range []int{2500, 5000} {
			t.Run(fmt.Sprintf("enabled=%t/count=%d", enabled, count), func(t *testing.T) {
				prefix := fmt.Sprintf("bulk-%t-%d-", enabled, count)
				batch := make([]*models.SpendLogEntry, count)
				wantIDs := make([]string, count)
				for i := range batch {
					wantIDs[i] = fmt.Sprintf("%s%04d", prefix, i)
					batch[i] = atomicTestEntry(wantIDs[i])
					batch[i].CredentialName = "provider-1"
				}
				ids, err := insert(enabled, batch...)
				require.NoError(t, err)
				assert.Equal(t, wantIDs, ids)
				ids, err = insert(enabled, batch...)
				require.NoError(t, err)
				assert.Empty(t, ids)

				var stored, attributed int
				var spend float64
				err = conn.QueryRow(ctx, `SELECT count(*), count(*) FILTER (WHERE credential_name = 'provider-1'), sum(spend)
					FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1`, prefix+"%").Scan(&stored, &attributed, &spend)
				require.NoError(t, err)
				assert.Equal(t, count, stored)
				assert.Equal(t, float64(count)*1.25, spend)
				if enabled {
					assert.Equal(t, count, attributed)
				} else {
					assert.Zero(t, attributed)
				}

				for _, entry := range batch {
					entry.RequestID = "rollback-" + entry.RequestID
				}
				batch[len(batch)-1].Spend = -1
				ids, err = insert(enabled, batch...)
				var constraintErr *pgconn.PgError
				require.ErrorAs(t, err, &constraintErr)
				assert.Equal(t, "23514", constraintErr.Code)
				assert.Nil(t, ids)
				err = conn.QueryRow(ctx, `SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1`, "rollback-"+prefix+"%").Scan(&stored)
				require.NoError(t, err)
				assert.Zero(t, stored)
			})
		}
	}
}
