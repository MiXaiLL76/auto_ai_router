package kafkalog

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestLogger builds a Logger with no underlying Kafka client, for testing
// queue/DLQ bookkeeping logic that doesn't touch the network (mirrors
// litellmdb/spendlog's own test idiom of a Logger{pool: nil, ...}).
func newTestLogger(queueSize int) *Logger {
	cfg := DefaultConfig()
	cfg.Brokers = []string{"kafka:9092"}
	cfg.LogQueueSize = queueSize
	cfg.ApplyDefaults()

	return &Logger{
		config:   cfg,
		logger:   cfg.Logger,
		topic:    cfg.Topic,
		queue:    make(chan *SpendEvent, queueSize),
		stopChan: make(chan struct{}),
	}
}

func TestLogger_Log_NonBlocking(t *testing.T) {
	l := newTestLogger(100)

	done := make(chan struct{})
	go func() {
		_ = l.Log(&SpendEvent{RequestID: "req-1"})
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Log() blocked for too long")
	}

	stats := l.Stats()
	assert.Equal(t, uint64(1), stats.Queued)
	assert.Equal(t, 1, stats.QueueLen)
}

func TestLogger_Log_NilEvent(t *testing.T) {
	l := newTestLogger(10)
	assert.NoError(t, l.Log(nil))
	assert.Equal(t, 0, l.Stats().QueueLen)
}

func TestLogger_Log_QueueFull(t *testing.T) {
	l := newTestLogger(1)

	assert.NoError(t, l.Log(&SpendEvent{RequestID: "req-1"}))

	start := time.Now()
	err := l.Log(&SpendEvent{RequestID: "req-2"})
	elapsed := time.Since(start)

	assert.ErrorIs(t, err, ErrQueueFull)
	assert.GreaterOrEqual(t, elapsed, 4*time.Second, "should wait close to the 5s backpressure timeout")

	stats := l.Stats()
	assert.Equal(t, uint64(1), stats.Dropped)
	assert.Equal(t, uint64(1), stats.QueueFullCount)
}

func TestLogger_AddToDLQ_Overflow(t *testing.T) {
	l := newTestLogger(10)

	for i := 0; i < dlqMaxSize+2; i++ {
		l.addToDLQ([]*SpendEvent{{RequestID: "req"}}, assert.AnError, 4)
	}

	stats := l.Stats()
	assert.Equal(t, dlqMaxSize, stats.DLQSize)
	assert.Equal(t, uint64(dlqMaxSize+2), stats.DLQCount)
	assert.Equal(t, uint64(2), stats.DLQOverflow)
}

// TestLogger_AddToDLQ_CopiesBatch guards against a data race where the DLQ
// held a reference to the caller's slice: the worker loop does
// `batch = batch[:0]` and keeps appending to the same backing array right
// after flushBatch/addToDLQ returns, which used to silently corrupt whatever
// was sitting in the DLQ (and race with the concurrent DLQ recovery worker
// reading it). addToDLQ must store an independent copy.
func TestLogger_AddToDLQ_CopiesBatch(t *testing.T) {
	l := newTestLogger(10)

	original := []*SpendEvent{{RequestID: "req-1"}, {RequestID: "req-2"}}
	l.addToDLQ(original, assert.AnError, 4)

	// Simulate the worker reusing original's backing array after addToDLQ
	// returns, the way the worker loop does with batch[:0] + append. original
	// has len=cap=2, so this append writes in place without reallocating —
	// the resulting slice header is discarded on purpose, only the shared
	// backing-array mutation matters here.
	_ = append(original[:0], &SpendEvent{RequestID: "mutated-1"}, &SpendEvent{RequestID: "mutated-2"})

	l.dlqMu.Lock()
	require.Len(t, l.dlq, 1)
	dlqBatch := l.dlq[0].batch
	l.dlqMu.Unlock()

	require.Len(t, dlqBatch, 2)
	assert.Equal(t, "req-1", dlqBatch[0].RequestID, "DLQ copy must be unaffected by the caller reusing its slice")
	assert.Equal(t, "req-2", dlqBatch[1].RequestID, "DLQ copy must be unaffected by the caller reusing its slice")
}

// TestLogger_AppendToDLQLocked_NeverExceedsCap guards the fix where flushDLQ
// used to re-add recovery-failed batches with a plain append, bypassing the
// overflow eviction that addToDLQ enforces — so the DLQ could grow past
// dlqMaxSize if new failures arrived (via addToDLQ) while flushDLQ was
// retrying. Both paths now share appendToDLQLocked, so interleaving them
// must never push DLQSize above dlqMaxSize.
func TestLogger_AppendToDLQLocked_NeverExceedsCap(t *testing.T) {
	l := newTestLogger(10)

	// Fill to capacity via the "new failure" path (addToDLQ).
	for i := 0; i < dlqMaxSize; i++ {
		l.addToDLQ([]*SpendEvent{{RequestID: "new"}}, assert.AnError, 4)
	}
	require.Equal(t, dlqMaxSize, l.Stats().DLQSize)

	// Simulate flushDLQ re-adding several batches that failed recovery while
	// the DLQ is already full — this is the exact interleaving that used to
	// overflow the cap.
	l.dlqMu.Lock()
	for i := 0; i < 5; i++ {
		l.appendToDLQLocked(&deadLetterBatch{
			batch:    []*SpendEvent{{RequestID: "retried"}},
			failedAt: time.Now(),
			attempts: 4,
		})
	}
	l.dlqMu.Unlock()

	assert.Equal(t, dlqMaxSize, l.Stats().DLQSize, "DLQ must never exceed dlqMaxSize regardless of which path appends")
}

func TestLogger_IsHealthy_DefaultsFalse(t *testing.T) {
	l := newTestLogger(10)
	assert.False(t, l.IsHealthy())

	l.healthy.Store(true)
	assert.True(t, l.IsHealthy())
}
