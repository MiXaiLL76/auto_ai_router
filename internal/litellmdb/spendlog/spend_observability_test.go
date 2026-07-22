package spendlog

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newObservabilityTestLogger(queueSize, _ int) *Logger {
	cfg := &models.Config{
		LogQueueSize:     queueSize,
		LogBatchSize:     1,
		LogFlushInterval: time.Hour,
		Logger:           testhelpers.NewTestLogger(),
	}
	cfg.ApplyDefaults()
	return &Logger{
		config:   cfg,
		logger:   cfg.Logger,
		queue:    make(chan *models.SpendLogEntry, queueSize),
		stopChan: make(chan struct{}),
	}
}

func TestFilterBatchByInsertedIDs_DeduplicatesInputBatch(t *testing.T) {
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1", APIKey: "key-1", Spend: 1},
		{RequestID: "req-1", APIKey: "key-1", Spend: 1},
		{RequestID: "req-existing", APIKey: "key-1", Spend: 10},
	}

	inserted := filterBatchByInsertedIDs(batch, []string{"req-1"})
	require.Len(t, inserted, 1)
	assert.Equal(t, "req-1", inserted[0].entry.RequestID)
	assert.Equal(t, 1.0, aggregateSpendUpdates(insertedEntries(inserted)).Tokens[entityModelKey{EntityID: "key-1"}])
}

func TestRecordCommittedBatchCountsOnlyPersistedRows(t *testing.T) {
	logger := newObservabilityTestLogger(4, 4)
	batch := []*models.SpendLogEntry{
		{RequestID: "req-1", ComparisonEligible: true},
		{RequestID: "req-1", ComparisonEligible: true},
		{RequestID: "req-existing", ComparisonEligible: false},
	}
	atomic.StoreInt64(&logger.pendingEntries, int64(len(batch)))

	logger.recordCommittedBatch(batch, []string{"req-1"})

	stats := logger.Stats()
	assert.Equal(t, uint64(1), stats.Written)
	assert.Equal(t, uint64(2), stats.Duplicates)
	assert.Equal(t, uint64(1), stats.ComparisonEligible)
	assert.Equal(t, uint64(0), stats.ComparisonIneligible)
	assert.Zero(t, stats.PendingEntries)
}

func TestComparisonWindowValidIsConservative(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Logger)
	}{
		{"dropped", func(sl *Logger) { atomic.StoreUint64(&sl.dropped, 1) }},
		{"dlq overflow", func(sl *Logger) { atomic.StoreUint64(&sl.dlqOverflow, 1) }},
		{"aggregation error", func(sl *Logger) { atomic.StoreUint64(&sl.aggregationErrors, 1) }},
		{"pending entry", func(sl *Logger) { atomic.StoreInt64(&sl.pendingEntries, 1) }},
		{"dlq not empty", func(sl *Logger) { sl.dlq = []*deadLetterBatch{{}} }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := newObservabilityTestLogger(2, 2)
			assert.True(t, logger.Stats().ComparisonWindowValid)
			tt.mutate(logger)
			assert.False(t, logger.Stats().ComparisonWindowValid)
		})
	}

	logger := newObservabilityTestLogger(2, 2)
	atomic.StoreUint64(&logger.comparisonIneligible, 1)
	assert.True(t, logger.Stats().ComparisonWindowValid,
		"ineligible rows are excluded from full financial comparison but are not transport loss")
}

func TestAtomicWriterHasNoPendingDailyPhase(t *testing.T) {
	stats := newObservabilityTestLogger(2, 1).Stats()
	assert.Zero(t, stats.PendingAggregation)
	assert.Zero(t, stats.PendingAggregationOverflow)
	assert.Zero(t, stats.AggregationLag)
}

func TestTryLogQueueFullIsImmediateAndRetainsExactEntryInDLQ(t *testing.T) {
	logger := newObservabilityTestLogger(1, 1)
	first := &models.SpendLogEntry{RequestID: "req-1"}
	second := &models.SpendLogEntry{RequestID: "req-2"}
	require.NoError(t, logger.TryLog(first))

	started := time.Now()
	require.NoError(t, logger.TryLog(second))
	assert.Less(t, time.Since(started), 100*time.Millisecond)
	stats := logger.Stats()
	assert.Zero(t, stats.Dropped, "retained queue pressure is not terminal data loss")
	assert.Equal(t, uint64(1), stats.QueueFullCount)
	assert.Equal(t, uint64(2), stats.Queued)
	assert.Equal(t, 2, stats.PendingEntries)
	assert.Equal(t, 1, stats.DLQSize)
	assert.False(t, stats.ComparisonWindowValid, "the window remains pending until DLQ recovery")
	require.Len(t, logger.dlq, 1)
	require.Len(t, logger.dlq[0].batch, 1)
	assert.Same(t, second, logger.dlq[0].batch[0])
}

func TestAggregationSuccessIsRecordedOnlyAfterAtomicCommit(t *testing.T) {
	logger := newObservabilityTestLogger(2, 2)
	before := time.Now()
	logger.recordAggregationSuccess()
	stats := logger.Stats()
	assert.Equal(t, uint64(1), stats.AggregationCount)
	assert.False(t, stats.LastAggregationTime.Before(before))
	assert.Zero(t, stats.PendingAggregation)
}

func TestRestoreFailedDLQBatchesPreservesBoundAndRecordsOverflow(t *testing.T) {
	logger := newObservabilityTestLogger(2, 2)
	for i := 0; i < 10; i++ {
		logger.dlq = append(logger.dlq, &deadLetterBatch{
			batch: []*models.SpendLogEntry{{RequestID: "new-" + string(rune('a'+i))}},
		})
	}
	failed := []*deadLetterBatch{
		{batch: []*models.SpendLogEntry{{RequestID: "old-a"}}},
		{batch: []*models.SpendLogEntry{{RequestID: "old-b"}}},
	}
	atomic.StoreInt64(&logger.pendingEntries, 12)

	logger.restoreFailedDLQBatches(failed)

	stats := logger.Stats()
	assert.Equal(t, 10, stats.DLQSize)
	assert.Equal(t, uint64(2), stats.DLQOverflow)
	assert.Equal(t, 10, stats.PendingEntries)
	assert.False(t, stats.ComparisonWindowValid)
	assert.Equal(t, "new-a", logger.dlq[0].batch[0].RequestID)
}

func TestAddToDLQOwnsBatchSlice(t *testing.T) {
	logger := newObservabilityTestLogger(2, 2)
	original := &models.SpendLogEntry{RequestID: "failed-original"}
	batch := make([]*models.SpendLogEntry, 1, 2)
	batch[0] = original

	logger.addToDLQ(batch, assert.AnError, 4)

	// The worker resets its reusable batch to [:0] after flushBatch and appends
	// new queue entries into the same backing array.
	batch = batch[:0]
	batch = append(batch, &models.SpendLogEntry{RequestID: "next-request"})
	require.Equal(t, "next-request", batch[0].RequestID)

	require.Len(t, logger.dlq, 1)
	require.Len(t, logger.dlq[0].batch, 1)
	assert.Same(t, original, logger.dlq[0].batch[0],
		"DLQ must not be mutated by reuse of the worker batch buffer")
}

func TestCommitErrorInvalidatesComparisonWindowButRemainsReplaySafe(t *testing.T) {
	logger := newObservabilityTestLogger(2, 2)
	assert.True(t, logger.Stats().ComparisonWindowValid)

	err := logger.handleCommitError(assert.AnError)

	assert.ErrorIs(t, err, assert.AnError)
	assert.Equal(t, uint64(1), logger.Stats().AggregationErrors)
	assert.False(t, logger.Stats().ComparisonWindowValid,
		"data replay is safe, but a lost acknowledgement makes process-local terminal counters ambiguous")
}
