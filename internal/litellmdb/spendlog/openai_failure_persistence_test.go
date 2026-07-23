package spendlog

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIChatFailureDailyProjectionKeepsEndpointAndFailureCount(t *testing.T) {
	entry := atomicTestEntry("req-openai-chat-failure")
	entry.CallType = "acompletion"
	entry.Status = "failure"
	entry.Spend = 0
	entry.PromptTokens = 0
	entry.CompletionTokens = 0
	entry.TotalTokens = 0

	logger := newAtomicTestLogger()
	records, err := buildSpendLogRecords(
		[]insertedSpendEntry{{entry: entry, requestID: entry.RequestID}},
		logger.logger,
		"test",
	)

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.False(t, records[0].SkipDaily)
	assert.Equal(t, "/chat/completions", records[0].Endpoint)

	value := &aggregationValue{}
	value.addRecord(records[0])
	assert.Equal(t, int64(1), value.apiRequests)
	assert.Zero(t, value.successfulRequests)
	assert.Equal(t, int64(1), value.failedRequests)
}
