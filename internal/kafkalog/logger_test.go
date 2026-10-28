package kafkalog

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/twmb/franz-go/pkg/kgo"
)

// newTestLogger builds a Logger with no underlying Kafka client, for testing
// queue/DLQ bookkeeping logic that doesn't touch the network (mirrors
// litellmdb/spendlog's own test idiom of a Logger{pool: nil, ...}).
func newTestLogger(queueSize int) *Logger[*SpendEvent] {
	cfg := DefaultConfig()
	cfg.Brokers = []string{"kafka:9092"}
	cfg.LogQueueSize = queueSize
	cfg.ApplyDefaults()

	return &Logger[*SpendEvent]{
		config:   cfg,
		logger:   cfg.Logger,
		topic:    cfg.Topic,
		queue:    make(chan *SpendEvent, queueSize),
		stopChan: make(chan struct{}),
	}
}

// newTestRawBodyLogger mirrors newTestLogger for the RawBodyEvent
// instantiation of the generic Logger[T], used by manager_test.go's
// DefaultRawBodyManager tests.
func newTestRawBodyLogger(queueSize int) *Logger[*RawBodyEvent] {
	cfg := DefaultConfig()
	cfg.Brokers = []string{"kafka:9092"}
	cfg.LogQueueSize = queueSize
	cfg.ApplyDefaults()

	return &Logger[*RawBodyEvent]{
		config:   cfg,
		logger:   cfg.Logger,
		topic:    cfg.Topic,
		queue:    make(chan *RawBodyEvent, queueSize),
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
		l.appendToDLQLocked(&deadLetterBatch[*SpendEvent]{
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

// fakeProducer stands in for kgo.Client.ProduceSync: it records the model of
// every record each call carries and fails the records whose model is in
// rejected — on every call, or only on the first rejectCalls calls when that
// is set. Results come back in reverse order so callers that matched them to
// events by position would pick the wrong events.
type fakeProducer struct {
	t           *testing.T
	rejected    map[string]bool
	rejectCalls int
	calls       [][]string
}

func (f *fakeProducer) produce(_ context.Context, rs ...*kgo.Record) kgo.ProduceResults {
	rejecting := f.rejectCalls == 0 || len(f.calls) < f.rejectCalls
	call := make([]string, 0, len(rs))
	results := make(kgo.ProduceResults, 0, len(rs))
	for i := len(rs) - 1; i >= 0; i-- {
		var event SpendEvent
		require.NoError(f.t, json.Unmarshal(rs[i].Value, &event))
		call = append([]string{event.Model}, call...)
		result := kgo.ProduceResult{Record: rs[i]}
		if rejecting && f.rejected[event.Model] {
			result.Err = assert.AnError
		}
		results = append(results, result)
	}
	f.calls = append(f.calls, call)
	return results
}

// newProducingTestLogger is newTestLogger wired to a fake producer, with the
// default number of attempts but no waiting between them.
func newProducingTestLogger(t *testing.T, producer *fakeProducer) *Logger[*SpendEvent] {
	producer.t = t
	l := newTestLogger(10)
	l.produce = producer.produce
	l.backoffDurations = make([]time.Duration, len(defaultBackoffDurations))
	return l
}

// spendBatch builds events that share one request_id — as two calls can
// when the provider reuses a response id — and differ only by model.
func spendBatch(models ...string) []*SpendEvent {
	batch := make([]*SpendEvent, len(models))
	for i, model := range models {
		batch[i] = &SpendEvent{RequestID: "chatcmpl-shared", Model: model}
	}
	return batch
}

func TestFailedEvents_MatchesByRecordNotPosition(t *testing.T) {
	batch := spendBatch("a", "b")
	recordA := &kgo.Record{Key: batch[0].Key()}
	recordB := &kgo.Record{Key: batch[1].Key()}
	events := map[*kgo.Record]*SpendEvent{recordA: batch[0], recordB: batch[1]}

	failed, err := failedEvents(kgo.ProduceResults{
		{Record: recordB, Err: assert.AnError},
		{Record: recordA},
	}, events)

	require.ErrorIs(t, err, assert.AnError)
	require.Len(t, failed, 1)
	assert.Same(t, batch[1], failed[0])
}

func TestFailedEvents_AllAccepted(t *testing.T) {
	batch := spendBatch("a")
	record := &kgo.Record{Key: batch[0].Key()}

	failed, err := failedEvents(kgo.ProduceResults{{Record: record}}, map[*kgo.Record]*SpendEvent{record: batch[0]})

	require.NoError(t, err)
	assert.Empty(t, failed)
}

// TestLogger_FlushBatch_ResendsOnlyFailedRecords guards against republishing
// the records Kafka already accepted: after a partial failure the next
// attempt must carry only the rejected record.
func TestLogger_FlushBatch_ResendsOnlyFailedRecords(t *testing.T) {
	producer := &fakeProducer{rejected: map[string]bool{"b": true}, rejectCalls: 1}
	l := newProducingTestLogger(t, producer)

	l.flushBatch(spendBatch("a", "b", "c"))

	assert.Equal(t, [][]string{{"a", "b", "c"}, {"b"}}, producer.calls)
	stats := l.Stats()
	assert.Equal(t, uint64(3), stats.Produced)
	assert.Equal(t, uint64(1), stats.BatchesOK)
	assert.Zero(t, stats.Errors)
	assert.Zero(t, stats.DLQSize)
	assert.True(t, stats.Healthy)
}

func TestLogger_FlushBatch_DLQKeepsOnlyUndeliveredRecords(t *testing.T) {
	producer := &fakeProducer{rejected: map[string]bool{"b": true}}
	l := newProducingTestLogger(t, producer)
	batch := spendBatch("a", "b", "c")

	l.flushBatch(batch)

	assert.Equal(t, [][]string{{"a", "b", "c"}, {"b"}, {"b"}, {"b"}}, producer.calls)
	stats := l.Stats()
	assert.Equal(t, uint64(2), stats.Produced)
	assert.Equal(t, uint64(1), stats.Errors)
	assert.Zero(t, stats.BatchesOK)
	assert.False(t, stats.Healthy)

	l.dlqMu.Lock()
	defer l.dlqMu.Unlock()
	require.Len(t, l.dlq, 1)
	require.Len(t, l.dlq[0].batch, 1)
	assert.Same(t, batch[1], l.dlq[0].batch[0], "the DLQ must hold the original, unmodified event")
	assert.Equal(t, len(defaultBackoffDurations), l.dlq[0].attempts)
}

// TestLogger_FlushBatch_FinalAttemptOnShutdownSendsOnlyFailed covers the
// stopChan branch: shutdown skips the backoff wait and makes one last
// attempt, which must also carry only the rejected records.
func TestLogger_FlushBatch_FinalAttemptOnShutdownSendsOnlyFailed(t *testing.T) {
	producer := &fakeProducer{rejected: map[string]bool{"b": true}}
	l := newProducingTestLogger(t, producer)
	l.backoffDurations = []time.Duration{0, time.Hour, time.Hour, time.Hour}
	close(l.stopChan)

	l.flushBatch(spendBatch("a", "b", "c"))

	assert.Equal(t, [][]string{{"a", "b", "c"}, {"b"}}, producer.calls)
	stats := l.Stats()
	assert.Equal(t, uint64(2), stats.Produced)
	assert.Equal(t, uint64(1), stats.Errors)
	assert.Equal(t, 1, stats.DLQSize)
}

func TestLogger_FlushDLQ_RequeuesOnlyUndeliveredRecords(t *testing.T) {
	producer := &fakeProducer{rejected: map[string]bool{"b": true}}
	l := newProducingTestLogger(t, producer)
	batch := spendBatch("a", "b")
	failedAt := time.Now().Add(-time.Minute)
	l.dlq = []*deadLetterBatch[*SpendEvent]{{batch: batch, failedAt: failedAt, attempts: 4}}

	l.flushDLQ()

	assert.Equal(t, [][]string{{"a", "b"}}, producer.calls)
	stats := l.Stats()
	assert.Equal(t, uint64(1), stats.Produced)
	assert.Zero(t, stats.DLQRecovered)

	l.dlqMu.Lock()
	defer l.dlqMu.Unlock()
	require.Len(t, l.dlq, 1)
	require.Len(t, l.dlq[0].batch, 1)
	assert.Same(t, batch[1], l.dlq[0].batch[0])
	assert.Equal(t, failedAt, l.dlq[0].failedAt)
}

func TestLogger_FlushDLQ_RecoversWholeBatch(t *testing.T) {
	producer := &fakeProducer{}
	l := newProducingTestLogger(t, producer)
	l.dlq = []*deadLetterBatch[*SpendEvent]{{batch: spendBatch("a", "b"), failedAt: time.Now(), attempts: 4}}

	l.flushDLQ()

	stats := l.Stats()
	assert.Equal(t, uint64(2), stats.Produced)
	assert.Equal(t, uint64(1), stats.DLQRecovered)
	assert.Zero(t, stats.DLQSize)
}

// TestLogger_FlushBatch_WithoutBackoffScheduleStillProduces covers a Logger
// built without NewLogger: an unset retry schedule must fall back to the
// default one instead of sending every batch to the DLQ untried.
func TestLogger_FlushBatch_WithoutBackoffScheduleStillProduces(t *testing.T) {
	producer := &fakeProducer{t: t}
	l := newTestLogger(10)
	l.produce = producer.produce

	l.flushBatch(spendBatch("a"))

	assert.Equal(t, [][]string{{"a"}}, producer.calls)
	assert.Equal(t, uint64(1), l.Stats().Produced)
	assert.Zero(t, l.Stats().DLQSize)
}
