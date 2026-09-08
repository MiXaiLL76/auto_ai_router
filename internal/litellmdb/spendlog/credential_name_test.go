package spendlog

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCredentialNameBatchParameters(t *testing.T) {
	first := atomicTestEntry("first")
	first.CredentialName = "grant'); DROP TABLE anything; --"
	first.Metadata = `{"spend_logs_metadata":{"air_event_id":"event-1"},"large_id":9007199254740993,"credential_name":"stale"}`
	second := atomicTestEntry("second")
	entries := []*models.SpendLogEntry{first, second}

	legacy, err := GetBatchParams(entries, false)
	require.NoError(t, err)
	require.Len(t, legacy, 52)
	assert.NotContains(t, legacy, first.CredentialName)

	params, err := GetBatchParams(entries, true)
	require.NoError(t, err)
	require.Len(t, params, 52)
	assert.Equal(t, legacy[:17], params[:17])
	assert.Equal(t, legacy[18:], params[18:])
	var metadata map[string]json.RawMessage
	require.NoError(t, json.Unmarshal([]byte(params[17].(string)), &metadata))
	var credential string
	require.NoError(t, json.Unmarshal(metadata["credential_name"], &credential))
	assert.Equal(t, first.CredentialName, credential)
	assert.Equal(t, "9007199254740993", string(metadata["large_id"]))
	assert.JSONEq(t, `{"air_event_id":"event-1"}`, string(metadata["spend_logs_metadata"]))
	assert.Equal(t, first.Metadata, legacy[17])
	assert.Contains(t, first.Metadata, `"credential_name":"stale"`)
	assert.NotContains(t, queries.BuildBatchInsertQuery(2), "credential_name")
}

func TestCredentialNameMetadataValidation(t *testing.T) {
	for _, metadata := range []string{"", "null", "{}"} {
		entry := atomicTestEntry("empty-metadata")
		entry.CredentialName = "provider-1"
		entry.Metadata = metadata
		params, err := GetBatchParams([]*models.SpendLogEntry{entry}, true)
		require.NoError(t, err)
		assert.JSONEq(t, `{"credential_name":"provider-1"}`, params[17].(string))
		assert.Equal(t, metadata, entry.Metadata)
	}
	for _, metadata := range []string{`{"broken":`, `[]`, `{} {}`} {
		entry := atomicTestEntry("invalid-metadata")
		entry.CredentialName = "provider-1"
		entry.Metadata = metadata
		params, err := GetBatchParams([]*models.SpendLogEntry{entry}, true)
		require.ErrorContains(t, err, "invalid-metadata")
		assert.Nil(t, params)
		assert.Equal(t, metadata, entry.Metadata)
	}
}

func TestCredentialNameSurvivesRequestIDCollision(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
			logger := newAtomicTestLogger()
			logger.config = &models.Config{LogCredentialName: enabled}
			first := atomicProviderIDEntry("chatcmpl-shared", "air-event-1")
			first.CredentialName = "grant-1"
			second := atomicProviderIDEntry("chatcmpl-shared", "air-event-2")
			second.CredentialName = "grant-2"
			tx := &atomicTestTx{
				insertResults: [][]string{{"chatcmpl-shared"}, {"air-event-2"}},
			}

			inserted, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{first, second})
			require.NoError(t, err)
			assert.Equal(t, []string{"chatcmpl-shared", "air-event-2"}, inserted)
			require.Len(t, tx.queryArgs, 2)
			for i, credential := range []string{"grant-1", "grant-2"} {
				assert.NotContains(t, tx.queries[i], "credential_name")
				require.Len(t, tx.queryArgs[i], 26)
				var metadata map[string]any
				require.NoError(t, json.Unmarshal([]byte(tx.queryArgs[i][17].(string)), &metadata))
				assert.Equal(t, fmt.Sprintf("air-event-%d", i+1), metadata["spend_logs_metadata"].(map[string]any)["air_event_id"])
				if enabled {
					assert.Equal(t, credential, metadata["credential_name"])
				} else {
					assert.NotContains(t, metadata, "credential_name")
				}
			}
			assert.Equal(t, 4, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
		})
	}
}

func TestCredentialNameLargeBatch(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, collision := range []bool{false, true} {
			for _, failLast := range []bool{false, true} {
				t.Run(fmt.Sprintf("enabled=%t/collision=%t/failLast=%t", enabled, collision, failLast), func(t *testing.T) {
					logger := newAtomicTestLogger()
					logger.config = &models.Config{LogCredentialName: enabled}
					batch := make([]*models.SpendLogEntry, 5000)
					wantIDs := make([]string, len(batch))
					for i := range batch {
						id := fmt.Sprintf("request-%04d", i)
						batch[i] = atomicTestEntry(id)
						if collision {
							batch[i] = atomicProviderIDEntry("shared-provider-id", id)
						}
						batch[i].CredentialName = "provider-1"
						wantIDs[i] = id
					}
					if collision {
						wantIDs[0] = "shared-provider-id"
					}
					tx := &parameterLimitedSpendTx{paramsPerEntry: 26}
					if failLast {
						tx.failRequestID = wantIDs[len(wantIDs)-1]
					}

					inserted, err := logger.commitBatchTransaction(context.Background(), tx, batch)
					if failLast {
						require.ErrorContains(t, err, "injected last-entry failure")
						assert.Nil(t, inserted)
						assert.True(t, tx.rolledBack)
						assert.False(t, tx.committed)
						assert.Empty(t, tx.stagedSQL)
						assert.Empty(t, tx.attemptedSQL)
						return
					}
					require.NoError(t, err)
					assert.Equal(t, wantIDs, inserted)
					assert.True(t, tx.committed)
					assert.Equal(t, 4, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
				})
			}
		}
	}
}

type parameterLimitedSpendTx struct {
	atomicTestTx
	paramsPerEntry int
	failRequestID  string
	lastRows       *atomicTestRows
}

func (tx *parameterLimitedSpendTx) Query(_ context.Context, sql string, args ...any) (pgx.Rows, error) {
	if len(args) > 65535 {
		return nil, errors.New("extended protocol limited to 65535 parameters")
	}
	if tx.lastRows != nil && !tx.lastRows.closed {
		return nil, errors.New("previous query rows are still open")
	}
	rows := make([][]any, 0, len(args)/tx.paramsPerEntry)
	for i := 0; i < len(args); i += tx.paramsPerEntry {
		if args[i] == tx.failRequestID {
			return nil, errors.New("injected last-entry failure")
		}
		rows = append(rows, []any{args[i]})
	}
	tx.stagedSQL = append(tx.stagedSQL, sql)
	tx.lastRows = &atomicTestRows{rows: rows}
	return tx.lastRows, nil
}
