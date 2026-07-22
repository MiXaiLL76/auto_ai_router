package kafkalog

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sasl/plain"
	"github.com/twmb/franz-go/pkg/sasl/scram"
)

// healthCheckInterval is how often the background worker probes broker
// connectivity via Client.Ping to keep IsHealthy() current even when the
// queue is idle.
const healthCheckInterval = 15 * time.Second

// dlqRecoveryInterval mirrors spendlog's DLQ recovery cadence.
const dlqRecoveryInterval = 5 * time.Minute

// dlqMaxSize caps the in-memory dead letter queue (same bound as spendlog).
const dlqMaxSize = 10

// Stats holds kafkalog producer statistics for observability.
type Stats struct {
	QueueLen       int
	QueueCap       int
	Queued         uint64
	Produced       uint64
	Dropped        uint64
	Errors         uint64
	BatchesOK      uint64
	QueueFullCount uint64
	DLQSize        int
	DLQCount       uint64
	DLQRecovered   uint64
	DLQOverflow    uint64
	Healthy        bool
}

// deadLetterBatch represents a batch that failed to produce after all retries.
type deadLetterBatch struct {
	batch     []*SpendEvent
	failedAt  time.Time
	lastError error
	attempts  int
}

// Logger is an asynchronous Kafka producer for spend events.
//
// Mirrors internal/litellmdb/spendlog.Logger: non-blocking Log(), batching,
// retry with exponential backoff, an in-memory Dead Letter Queue, and
// graceful shutdown. Unlike spendlog, broker unavailability never blocks
// callers or drops the process — see Manager.IsHealthy.
type Logger struct {
	client *kgo.Client
	topic  string
	logger *slog.Logger
	config *Config

	queue chan *SpendEvent

	stopChan  chan struct{}
	wg        sync.WaitGroup
	shutdown  atomic.Bool
	startOnce sync.Once

	healthy atomic.Bool

	queued         uint64
	produced       uint64
	dropped        uint64
	errors         uint64
	batchesOK      uint64
	queueFullCount uint64
	dlqCount       uint64
	dlqRecovered   uint64
	dlqOverflow    uint64

	dlqMu sync.Mutex
	dlq   []*deadLetterBatch
}

// NewLogger creates a new asynchronous Kafka producer logger. The underlying
// client connects lazily — broker unavailability at construction time is not
// an error, it only keeps IsHealthy() false until a connection succeeds.
func NewLogger(cfg *Config) (*Logger, error) {
	opts := []kgo.Opt{
		kgo.SeedBrokers(cfg.Brokers...),
		kgo.ClientID(cfg.ClientID),
		kgo.DefaultProduceTopic(cfg.Topic),
	}

	if cfg.TLSEnabled {
		opts = append(opts, kgo.DialTLSConfig(&tls.Config{MinVersion: tls.VersionTLS12}))
	}

	switch cfg.SASLMechanism {
	case "PLAIN":
		auth := plain.Auth{User: cfg.SASLUsername, Pass: cfg.SASLPassword}
		opts = append(opts, kgo.SASL(auth.AsMechanism()))
	case "SCRAM-SHA-256":
		auth := scram.Auth{User: cfg.SASLUsername, Pass: cfg.SASLPassword}
		opts = append(opts, kgo.SASL(auth.AsSha256Mechanism()))
	case "SCRAM-SHA-512":
		auth := scram.Auth{User: cfg.SASLUsername, Pass: cfg.SASLPassword}
		opts = append(opts, kgo.SASL(auth.AsSha512Mechanism()))
	}

	client, err := kgo.NewClient(opts...)
	if err != nil {
		return nil, fmt.Errorf("kafkalog: create client: %w", err)
	}

	l := &Logger{
		client:   client,
		topic:    cfg.Topic,
		logger:   cfg.Logger,
		config:   cfg,
		queue:    make(chan *SpendEvent, cfg.LogQueueSize),
		stopChan: make(chan struct{}),
	}
	return l, nil
}

// Start starts the background workers, health checker and DLQ recovery loop.
// Safe to call multiple times (idempotent).
func (l *Logger) Start() {
	l.startOnce.Do(func() {
		numWorkers := l.config.LogWorkers
		if numWorkers <= 0 {
			numWorkers = 1
		}

		l.wg.Add(numWorkers + 2)
		for i := 0; i < numWorkers; i++ {
			go l.worker()
		}
		go l.healthCheckWorker()
		go l.dlqRecoveryWorker()
		l.logger.Info("[Kafka] SpendLogger started",
			"topic", l.topic,
			"queue_size", l.config.LogQueueSize,
			"batch_size", l.config.LogBatchSize,
			"flush_interval", l.config.LogFlushInterval,
			"workers", numWorkers,
		)
	})
}

// Log adds an event to the queue with backpressure handling.
// BLOCKING: waits up to 5 seconds for queue space if full.
// Returns ErrQueueFull if the timeout is reached (event not queued).
func (l *Logger) Log(event *SpendEvent) error {
	if event == nil {
		return nil
	}

	select {
	case l.queue <- event:
		atomic.AddUint64(&l.queued, 1)
		return nil
	default:
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	select {
	case l.queue <- event:
		atomic.AddUint64(&l.queued, 1)
		return nil
	case <-ctx.Done():
		atomic.AddUint64(&l.dropped, 1)
		atomic.AddUint64(&l.queueFullCount, 1)
		l.logger.Error("[Kafka] SpendLog event dropped: queue full timeout",
			"request_id", event.RequestID,
			"queue_len", len(l.queue),
			"queue_cap", cap(l.queue),
			"timeout_sec", 5,
		)
		return ErrQueueFull
	}
}

// IsHealthy reports the last known broker connectivity state.
func (l *Logger) IsHealthy() bool {
	return l.healthy.Load()
}

// Stats returns current producer statistics.
func (l *Logger) Stats() Stats {
	l.dlqMu.Lock()
	dlqSize := len(l.dlq)
	l.dlqMu.Unlock()

	return Stats{
		QueueLen:       len(l.queue),
		QueueCap:       cap(l.queue),
		Queued:         atomic.LoadUint64(&l.queued),
		Produced:       atomic.LoadUint64(&l.produced),
		Dropped:        atomic.LoadUint64(&l.dropped),
		Errors:         atomic.LoadUint64(&l.errors),
		BatchesOK:      atomic.LoadUint64(&l.batchesOK),
		QueueFullCount: atomic.LoadUint64(&l.queueFullCount),
		DLQSize:        dlqSize,
		DLQCount:       atomic.LoadUint64(&l.dlqCount),
		DLQRecovered:   atomic.LoadUint64(&l.dlqRecovered),
		DLQOverflow:    atomic.LoadUint64(&l.dlqOverflow),
		Healthy:        l.IsHealthy(),
	}
}

// Shutdown stops the logger and waits for all queued events to be flushed.
// Idempotent: safe to call multiple times.
func (l *Logger) Shutdown(ctx context.Context) error {
	if !l.shutdown.CompareAndSwap(false, true) {
		return nil
	}

	l.logger.Info("[Kafka] SpendLogger shutting down...", "pending", len(l.queue))

	close(l.stopChan)

	done := make(chan struct{})
	go func() {
		l.wg.Wait()
		close(done)
	}()

	var shutdownErr error
	select {
	case <-done:
		l.logger.Info("[Kafka] SpendLogger shutdown complete",
			"produced", atomic.LoadUint64(&l.produced),
			"dropped", atomic.LoadUint64(&l.dropped),
			"errors", atomic.LoadUint64(&l.errors),
			"dlq_size", l.dlqSize(),
		)
	case <-ctx.Done():
		l.logger.Warn("[Kafka] SpendLogger shutdown timeout", "pending", len(l.queue))
		shutdownErr = ctx.Err()
	}

	l.client.Close()
	return shutdownErr
}

func (l *Logger) dlqSize() int {
	l.dlqMu.Lock()
	defer l.dlqMu.Unlock()
	return len(l.dlq)
}

// worker is the background goroutine that batches and produces the queue.
func (l *Logger) worker() {
	defer l.wg.Done()

	batch := make([]*SpendEvent, 0, l.config.LogBatchSize)
	ticker := time.NewTicker(l.config.LogFlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-l.stopChan:
			l.drainQueue(&batch)
			if len(batch) > 0 {
				l.flushBatch(batch)
			}
			return

		case event := <-l.queue:
			batch = append(batch, event)
			if len(batch) >= l.config.LogBatchSize {
				l.flushBatch(batch)
				batch = batch[:0]
			}

		case <-ticker.C:
			if len(batch) > 0 {
				l.flushBatch(batch)
				batch = batch[:0]
			}
		}
	}
}

func (l *Logger) drainQueue(batch *[]*SpendEvent) {
	for {
		select {
		case event := <-l.queue:
			*batch = append(*batch, event)
		default:
			return
		}
	}
}

// flushBatch produces a batch to Kafka with retry and DLQ fallback.
// Retry strategy mirrors spendlog: 0s, 1s, 5s, 30s backoff, 4 attempts total.
func (l *Logger) flushBatch(batch []*SpendEvent) {
	if len(batch) == 0 {
		return
	}

	const maxAttempts = 4
	backoffDurations := []time.Duration{0, 1 * time.Second, 5 * time.Second, 30 * time.Second}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			backoff := backoffDurations[attempt]
			select {
			case <-time.After(backoff):
			case <-l.stopChan:
				if finalErr := l.produceBatch(batch); finalErr == nil {
					atomic.AddUint64(&l.produced, uint64(len(batch)))
					atomic.AddUint64(&l.batchesOK, 1)
					l.healthy.Store(true)
					return
				} else {
					lastErr = finalErr
				}
				atomic.AddUint64(&l.errors, uint64(len(batch)))
				l.healthy.Store(false)
				l.addToDLQ(batch, lastErr, attempt)
				return
			}
		}

		err := l.produceBatch(batch)
		if err == nil {
			atomic.AddUint64(&l.produced, uint64(len(batch)))
			atomic.AddUint64(&l.batchesOK, 1)
			l.healthy.Store(true)
			return
		}

		lastErr = err
		l.healthy.Store(false)
		l.logger.Warn("[Kafka] SpendLog batch produce failed",
			"attempt", attempt+1,
			"max_attempts", maxAttempts,
			"batch_size", len(batch),
			"error", err,
		)
	}

	atomic.AddUint64(&l.errors, uint64(len(batch)))
	l.addToDLQ(batch, lastErr, maxAttempts)
}

// produceBatch marshals and synchronously produces a batch of events, keyed
// by request_id so retries/reprocessing stay ordered per request.
func (l *Logger) produceBatch(batch []*SpendEvent) error {
	records := make([]*kgo.Record, 0, len(batch))
	for _, event := range batch {
		value, err := json.Marshal(event)
		if err != nil {
			l.logger.Error("[Kafka] Failed to marshal spend event, skipping",
				"request_id", event.RequestID, "error", err)
			continue
		}
		records = append(records, &kgo.Record{
			Key:   event.Key(),
			Value: value,
			Topic: l.topic,
		})
	}
	if len(records) == 0 {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results := l.client.ProduceSync(ctx, records...)
	return results.FirstErr()
}

// appendToDLQLocked appends dlb to the DLQ, evicting the oldest entry first
// if already at capacity. Must be called with dlqMu held. This is the single
// enforcement point for dlqMaxSize: addToDLQ (brand new failures) and
// flushDLQ (batches that failed recovery and go back in) both route through
// it, so the cap holds regardless of which path is writing — previously
// flushDLQ re-added failed batches with a plain append, which could push the
// DLQ past dlqMaxSize while new failures were being added concurrently by
// the worker.
func (l *Logger) appendToDLQLocked(dlb *deadLetterBatch) {
	if len(l.dlq) >= dlqMaxSize {
		dropped := l.dlq[0]
		l.dlq = l.dlq[1:]
		atomic.AddUint64(&l.dlqOverflow, 1)
		l.logger.Error("[Kafka] SpendLog DLQ overflow - batch dropped",
			"dropped_batch_size", len(dropped.batch),
			"dropped_at", dropped.failedAt,
			"dlq_size", len(l.dlq),
		)
		l.notifyFallback(dropped.batch, "dlq_overflow")
	}
	l.dlq = append(l.dlq, dlb)
}

// notifyFallback calls config.FallbackNotifier for every event in batch,
// fire-and-forget, so a batch permanently dropped from the DLQ doesn't block
// the caller (holding dlqMu) on Postgres round-trips. Request IDs are copied
// upfront since dropped.batch's backing array must not be retained/mutated
// after appendToDLQLocked returns.
func (l *Logger) notifyFallback(batch []*SpendEvent, reason string) {
	if l.config.FallbackNotifier == nil || len(batch) == 0 {
		return
	}
	requestIDs := make([]string, len(batch))
	for i, event := range batch {
		requestIDs[i] = event.RequestID
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		for _, requestID := range requestIDs {
			if err := l.config.FallbackNotifier(ctx, requestID, reason); err != nil {
				l.logger.Warn("[Kafka] Failed to flag spend log row for fallback resend",
					"request_id", requestID,
					"reason", reason,
					"error", err,
				)
			}
		}
	}()
}

func (l *Logger) addToDLQ(batch []*SpendEvent, lastErr error, attempts int) {
	l.dlqMu.Lock()
	defer l.dlqMu.Unlock()

	// Copy the batch: the caller's worker loop reuses batch's backing array via
	// batch[:0] + append immediately after this returns, and dlqRecoveryWorker
	// reads dlb.batch concurrently from a different goroutine. Without a copy,
	// both goroutines end up racing on (and corrupting) the same backing array.
	batchCopy := append([]*SpendEvent(nil), batch...)

	dlb := &deadLetterBatch{
		batch:     batchCopy,
		failedAt:  time.Now(),
		lastError: lastErr,
		attempts:  attempts,
	}

	l.appendToDLQLocked(dlb)
	atomic.AddUint64(&l.dlqCount, 1)

	l.logger.Error("[Kafka] SpendLog batch sent to Dead Letter Queue",
		"batch_size", len(batch),
		"dlq_size", len(l.dlq),
		"last_error", lastErr,
		"attempts", attempts,
	)
}

// healthCheckWorker periodically pings the brokers so IsHealthy() reflects
// connectivity even while the queue is idle (no batches being produced).
func (l *Logger) healthCheckWorker() {
	defer l.wg.Done()

	ticker := time.NewTicker(healthCheckInterval)
	defer ticker.Stop()

	l.probeHealth()

	for {
		select {
		case <-l.stopChan:
			return
		case <-ticker.C:
			l.probeHealth()
		}
	}
}

func (l *Logger) probeHealth() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := l.client.Ping(ctx)
	healthy := err == nil
	if l.healthy.Swap(healthy) != healthy {
		if healthy {
			l.logger.Info("[Kafka] Broker connectivity restored")
		} else {
			l.logger.Warn("[Kafka] Broker connectivity lost", "error", err)
		}
	}
}

// dlqRecoveryWorker periodically retries failed batches from the DLQ.
func (l *Logger) dlqRecoveryWorker() {
	defer l.wg.Done()

	ticker := time.NewTicker(dlqRecoveryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-l.stopChan:
			l.flushDLQ()
			return
		case <-ticker.C:
			l.flushDLQ()
		}
	}
}

func (l *Logger) flushDLQ() {
	l.dlqMu.Lock()
	if len(l.dlq) == 0 {
		l.dlqMu.Unlock()
		return
	}
	dlqCopy := make([]*deadLetterBatch, len(l.dlq))
	copy(dlqCopy, l.dlq)
	l.dlq = l.dlq[:0]
	l.dlqMu.Unlock()

	for _, dlb := range dlqCopy {
		err := l.produceBatch(dlb.batch)
		if err == nil {
			atomic.AddUint64(&l.produced, uint64(len(dlb.batch)))
			atomic.AddUint64(&l.batchesOK, 1)
			atomic.AddUint64(&l.dlqRecovered, 1)
			l.healthy.Store(true)
			l.logger.Warn("[Kafka] SpendLog batch recovered from DLQ",
				"batch_size", len(dlb.batch),
				"time_in_dlq", time.Since(dlb.failedAt).String(),
			)
			continue
		}

		l.dlqMu.Lock()
		l.appendToDLQLocked(dlb)
		l.dlqMu.Unlock()
	}
}
