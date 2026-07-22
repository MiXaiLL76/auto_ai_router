package spendlog

import (
	"context"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildDailyProjectionDimensions(t *testing.T) {
	tests := []struct {
		name             string
		rawCallType      string
		originalCallType string
		model            string
		modelGroup       string
		provider         string
		status           string
		wantSkip         bool
		wantEndpoint     string
		wantModel        string
		wantModelGroup   string
		wantProvider     string
	}{
		{
			name: "missing effective route stays in raw only", model: "backend-model",
			modelGroup: "public-model", provider: "openai", wantSkip: true,
			wantModel: "backend-model", wantModelGroup: "public-model", wantProvider: "openai",
		},
		{
			name: "known raw route keeps success dimensions", rawCallType: "acompletion",
			model: "backend-model", modelGroup: "public-model",
			provider: "openai", wantEndpoint: "/chat/completions",
			wantModel: "backend-model", wantModelGroup: "public-model", wantProvider: "openai",
		},
		{
			name: "chat failure matches LiteLLM daily dimensions", originalCallType: "acompletion",
			model: "backend-model", modelGroup: "openai/public-model", provider: "openai",
			status: "failure", wantModel: "openai/public-model",
			wantModelGroup: "openai/public-model",
		},
		{
			name: "responses failure clears LiteLLM model group", originalCallType: "aresponses",
			model: "backend-model", modelGroup: "openai/public-model", provider: "openai",
			status: "failure", wantModel: "openai/public-model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := atomicTestEntry("req-dimensions")
			entry.CallType, entry.Model, entry.ModelGroup = tt.rawCallType, tt.model, tt.modelGroup
			entry.CustomLLMProvider = tt.provider
			if tt.originalCallType != "" {
				entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"` + tt.originalCallType + `"}}`
			}
			if tt.status != "" {
				entry.Status = tt.status
			}
			logger := newAtomicTestLogger()
			records, err := buildSpendLogRecords(
				[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
			)
			require.NoError(t, err)
			require.Len(t, records, 1)
			assert.Equal(t, tt.wantSkip, records[0].SkipDaily)
			assert.Equal(t, tt.wantEndpoint, records[0].Endpoint)
			assert.Equal(t, tt.wantModel, records[0].Model)
			assert.Equal(t, tt.wantModelGroup, records[0].ModelGroup)
			assert.Equal(t, tt.wantProvider, records[0].CustomLLMProvider)
		})
	}
}

func TestBuildSpendLogRecordsUsesUTCDateAndEntryDimensions(t *testing.T) {
	entry := atomicTestEntry("req-date")
	logger := newAtomicTestLogger()

	records, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, entry.StartTime.UTC().Format("2006-01-02"), records[0].Date)
	assert.Equal(t, entry.UserID, records[0].UserID)
	assert.Equal(t, entry.APIKey, records[0].APIKey)
	assert.Equal(t, entry.RequestID, records[0].RequestID)
}

func TestBuildSpendLogRecordsExtractsCacheTokensFromMetadata(t *testing.T) {
	entry := atomicTestEntry("req-cache")
	entry.Metadata = `{"usage_object":{
		"cache_read_input_tokens": 7,
		"cache_creation_input_tokens": 0,
		"prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 4, "cache_creation_tokens": 5}
	}}`
	logger := newAtomicTestLogger()

	records, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	// Anthropic-style top-level fields win when nonzero; zero falls through to
	// the OpenAI-compatible prompt_tokens_details fallbacks.
	assert.Equal(t, int64(7), records[0].CacheReadInputTokens)
	assert.Equal(t, int64(4), records[0].CacheCreationInputTokens)

	entry.Metadata = `{"usage_object":{"prompt_tokens_details":{"cached_tokens": 3, "cache_creation_tokens": 5}}}`
	records, err = buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)
	require.NoError(t, err)
	assert.Equal(t, int64(3), records[0].CacheReadInputTokens)
	assert.Equal(t, int64(5), records[0].CacheCreationInputTokens)
}

func TestKnownEffectiveRouteWithEmptyRawCallTypeRequiresFailureStatus(t *testing.T) {
	entry := atomicTestEntry("req-corrupt-status")
	entry.CallType, entry.Status = "", "success"
	entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"acompletion"}}`
	logger := newAtomicTestLogger()

	_, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)

	require.Error(t, err)
	assert.Contains(t, err.Error(), `empty raw LiteLLM call_type with status "success"`)
}

// LiteLLM writes failure rows with partial cost/usage (interrupted streams),
// so an empty raw call_type with nonzero spend must aggregate, not error.
func TestKnownEffectiveRouteAcceptsNonzeroFailureWithEmptyRawCallType(t *testing.T) {
	entry := atomicTestEntry("req-nonzero-empty-call-type")
	entry.CallType, entry.Status = "", "failure"
	entry.PromptTokens = 3
	entry.CompletionTokens = 2
	entry.Spend = 0.001
	entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"aresponses"}}`
	logger := newAtomicTestLogger()

	records, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}}, logger.logger, "test",
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.False(t, records[0].SkipDaily)
	assert.Equal(t, 0.001, records[0].Spend)
}

// An unknown call_type is a permanent property of the entry: it must not poison
// the batch (retry → DLQ would lose the valid rows around it). The raw row and
// entity counters commit; only the daily projection is skipped.
func TestUnknownEffectiveDailyRouteSkipsDailyButCommitsRawWrite(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-unknown-effective-route")
	entry.CallType = ""
	entry.Metadata = `{"spend_logs_metadata":{"original_call_type":"unsupported-route"}}`
	tx := &atomicTestTx{insertedIDs: []string{entry.RequestID}}
	_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})
	require.NoError(t, err)
	assert.False(t, tx.rolledBack)
	assert.True(t, tx.committed)
	assert.Equal(t, 0, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`),
		"daily projections must be skipped for an unknown call_type")
}
