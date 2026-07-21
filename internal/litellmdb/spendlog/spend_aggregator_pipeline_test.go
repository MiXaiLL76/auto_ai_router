package spendlog

import (
	"context"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadDailyProjectionDimensions(t *testing.T) {
	tests := []struct {
		name              string
		rawCallType       string
		effectiveCallType string
		model             string
		modelGroup        string
		provider          string
		mcpTool           string
		status            string
		wantSkip          bool
		wantEndpoint      string
		wantModel         string
		wantModelGroup    string
		wantProvider      string
		wantMCPTool       string
	}{
		{
			name: "missing effective route stays in raw only", model: "backend-model",
			modelGroup: "public-model", provider: "openai", wantSkip: true,
			wantModel: "backend-model", wantModelGroup: "public-model", wantProvider: "openai",
		},
		{
			name: "known raw route keeps success dimensions", rawCallType: "acompletion",
			effectiveCallType: "acompletion", model: "backend-model", modelGroup: "public-model",
			provider: "openai", mcpTool: "server/tool", wantEndpoint: "/chat/completions",
			wantModel: "backend-model", wantModelGroup: "public-model", wantProvider: "openai", wantMCPTool: "server/tool",
		},
		{
			name: "chat failure matches LiteLLM daily dimensions", effectiveCallType: "acompletion",
			model: "backend-model", modelGroup: "openai/public-model", provider: "openai",
			mcpTool: "server/tool", status: "failure", wantModel: "openai/public-model",
			wantModelGroup: "openai/public-model", wantMCPTool: "server/tool",
		},
		{
			name: "responses failure clears LiteLLM model group", effectiveCallType: "aresponses",
			model: "backend-model", modelGroup: "openai/public-model", provider: "openai",
			mcpTool: "server/tool", status: "failure", wantModel: "openai/public-model", wantMCPTool: "server/tool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := atomicTestEntry("req-dimensions")
			entry.CallType, entry.Model, entry.ModelGroup = tt.rawCallType, tt.model, tt.modelGroup
			entry.CustomLLMProvider = tt.provider
			if tt.status != "" {
				entry.Status = tt.status
			}
			row := atomicTestSpendRow(entry)
			row[6] = tt.mcpTool // mcp_namespaced_tool_name column in the selected row
			row[8] = tt.effectiveCallType
			logger := newAtomicTestLogger()
			records, err := loadUnprocessedSpendLogRecords(
				context.Background(), &atomicTestTx{spendRows: [][]any{row}}, logger.logger, "test", []string{entry.RequestID},
			)
			require.NoError(t, err)
			require.Len(t, records, 1)
			assert.Equal(t, tt.wantSkip, records[0].SkipDaily)
			assert.Equal(t, tt.wantEndpoint, records[0].Endpoint)
			assert.Equal(t, tt.wantModel, records[0].Model)
			assert.Equal(t, tt.wantModelGroup, records[0].ModelGroup)
			assert.Equal(t, tt.wantProvider, records[0].CustomLLMProvider)
			assert.Equal(t, tt.wantMCPTool, records[0].MCPNamespacedTool)
		})
	}
}

func TestKnownEffectiveRouteWithEmptyRawCallTypeRequiresFailureStatus(t *testing.T) {
	entry := atomicTestEntry("req-corrupt-status")
	entry.CallType, entry.Status = "", "success"
	row := atomicTestSpendRow(entry)
	row[8] = "acompletion"
	logger := newAtomicTestLogger()
	_, err := loadUnprocessedSpendLogRecords(
		context.Background(), &atomicTestTx{spendRows: [][]any{row}}, logger.logger, "test", []string{entry.RequestID},
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
	row := atomicTestSpendRow(entry)
	row[8] = "aresponses"
	logger := newAtomicTestLogger()

	records, err := loadUnprocessedSpendLogRecords(
		context.Background(), &atomicTestTx{spendRows: [][]any{row}}, logger.logger, "test", []string{entry.RequestID},
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.False(t, records[0].SkipDaily)
	assert.Equal(t, 0.001, records[0].Spend)
}

// An unknown call_type is a permanent property of the row: it must not poison
// the batch (retry → DLQ would lose the valid rows around it). The raw row and
// entity counters commit; only the daily projection is skipped.
func TestUnknownEffectiveDailyRouteSkipsDailyButCommitsRawWrite(t *testing.T) {
	logger := newAtomicTestLogger()
	entry := atomicTestEntry("req-unknown-effective-route")
	entry.CallType = ""
	row := atomicTestSpendRow(entry)
	row[8] = "unsupported-route"
	tx := &atomicTestTx{insertedIDs: []string{entry.RequestID}, spendRows: [][]any{row}}
	_, err := logger.commitBatchTransaction(context.Background(), tx, []*models.SpendLogEntry{entry})
	require.NoError(t, err)
	assert.False(t, tx.rolledBack)
	assert.True(t, tx.committed)
	assert.Equal(t, 0, countSQLContaining(tx.committedSQL, `INSERT INTO "LiteLLM_Daily`),
		"daily projections must be skipped for an unknown call_type")
}
