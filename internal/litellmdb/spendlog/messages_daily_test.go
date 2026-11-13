package spendlog

import (
	"bytes"
	"context"
	"log/slog"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Follow the production path from the incoming URL through the atomic writer.
// Both successful Messages requests and interrupted requests with partial cost
// used to commit their raw rows and counters without any daily projections.
func TestMessagesSpendReachesAllDailyProjections(t *testing.T) {
	for _, status := range []string{"success", "failure"} {
		t.Run(status, func(t *testing.T) {
			logger := newAtomicTestLogger()
			entry := atomicTestEntry("messages-" + status)
			entry.CallType = LiteLLMCallTypeForPath("/v1/messages")
			entry.CustomLLMProvider = "anthropic"
			entry.Status = status
			entry.Metadata = `{"usage_object":{"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}`
			assert.Equal(t, "anthropic_messages", entry.CallType)

			records, err := buildSpendLogRecords(
				[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test", nil,
			)
			require.NoError(t, err)
			require.Len(t, records, 1)
			assert.False(t, records[0].SkipDaily)
			assert.Empty(t, records[0].Endpoint, "LiteLLM has no ROUTE_ENDPOINT_MAPPING entry for anthropic_messages, so its own daily endpoint dimension is NULL")
			assert.Equal(t, entry.Spend, records[0].Spend)
			assert.Equal(t, entry.Model, records[0].Model)
			assert.Equal(t, "anthropic", records[0].CustomLLMProvider)
			assert.Equal(t, int64(7), records[0].CacheReadInputTokens)
			assert.Equal(t, int64(3), records[0].CacheCreationInputTokens)

			tx := &atomicTestTx{insertedIDs: []string{entry.RequestID}}
			_, err = logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})
			require.NoError(t, err)
			assert.True(t, tx.committed)
			for _, entity := range []string{"User", "Team", "Organization", "EndUser"} {
				assert.Equal(t, 1, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`+entity+`Spend"`))
			}

			replay := &atomicTestTx{}
			_, err = logger.commitBatchTransaction(context.Background(), replay, []*models.SpendLogEntry{entry})
			require.NoError(t, err)
			assert.True(t, replay.committed)
			assert.Empty(t, replay.attemptedSQL, "a replay must not charge counters or daily tables again")
		})
	}
}

func TestSkippedPaidDailyProjectionIsVisibleAtErrorLevel(t *testing.T) {
	for _, callType := range []string{"", "unsupported-route"} {
		t.Run("call_type="+callType, func(t *testing.T) {
			var logs bytes.Buffer
			logger := newAtomicTestLogger()
			logger.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelError}))
			entry := atomicTestEntry("missing-daily-route")
			entry.CallType = callType
			tx := &atomicTestTx{insertedIDs: []string{entry.RequestID}}

			_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})
			require.NoError(t, err, "an unknown route must not poison the raw spend batch")
			assert.True(t, tx.committed)
			assert.Zero(t, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`))
			assert.Contains(t, logs.String(), "paid spend skipped daily aggregation")
			assert.Contains(t, logs.String(), entry.RequestID)
			assert.NotContains(t, logs.String(), entry.APIKey)
		})
	}
}
