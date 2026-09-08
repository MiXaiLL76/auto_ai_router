package spendlog

import (
	"context"
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCredentialNameBatchParameters(t *testing.T) {
	first := atomicTestEntry("first")
	first.CredentialName = "grant'); DROP TABLE anything; --"
	second := atomicTestEntry("second")
	entries := []*models.SpendLogEntry{first, second}

	legacy := GetBatchParams(entries, false)
	require.Len(t, legacy, 52)
	assert.NotContains(t, legacy, first.CredentialName)

	params := GetBatchParams(entries, true)
	require.Len(t, params, 54)
	assert.Equal(t, legacy[:26], params[:26])
	assert.Equal(t, first.CredentialName, params[26])
	assert.Equal(t, legacy[26:], params[27:53])
	assert.Nil(t, params[53])
	assert.NotContains(t, queries.BuildBatchInsertQuery(2, true), first.CredentialName)
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
				if enabled {
					assert.Contains(t, tx.queries[i], "credential_name")
					require.Len(t, tx.queryArgs[i], 27)
					assert.Equal(t, credential, tx.queryArgs[i][26])
				} else {
					assert.NotContains(t, tx.queries[i], "credential_name")
					assert.Len(t, tx.queryArgs[i], 26)
				}
			}
			assert.Equal(t, 4, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
		})
	}
}
