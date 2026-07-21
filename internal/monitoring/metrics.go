package monitoring

import (
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var spendAggregationOldestUnixNano atomic.Int64

var (
	RequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_requests_total",
			Help: "Total number of requests",
		},
		[]string{"credential", "model", "endpoint", "status"},
	)

	RequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "auto_ai_router_requests_duration_seconds",
			Help:    "Request duration in seconds",
			Buckets: []float64{1, 10, 30, 60, 120, 240, 600},
		},
		[]string{"credential", "endpoint"},
	)

	AbortedRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_aborted_requests_total",
			Help: "Total number of requests aborted by the client while the response was being written",
		},
		[]string{"credential", "model", "endpoint"},
	)

	CredentialRPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_rpm_current",
			Help: "Current RPM for each credential",
		},
		[]string{"credential"},
	)

	CredentialTPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_tpm_current",
			Help: "Current TPM (tokens per minute) for each credential",
		},
		[]string{"credential"},
	)

	CredentialBanned = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_banned",
			Help: "Ban status for each credential (1 = banned, 0 = active)",
		},
		[]string{"credential"},
	)

	CredentialErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_errors_total",
			Help: "Total number of errors for each credential",
		},
		[]string{"credential"},
	)

	ModelRPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_model_rpm_current",
			Help: "Current RPM for each model within a credential",
		},
		[]string{"credential", "model"},
	)

	ModelTPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_model_tpm_current",
			Help: "Current TPM (tokens per minute) for each model within a credential",
		},
		[]string{"credential", "model"},
	)

	CredentialSelectionRejected = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_selection_rejected_total",
			Help: "Total number of times a credential was rejected during selection",
		},
		[]string{"reason"},
	)

	CredentialBanEvents = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_ban_events_total",
			Help: "Total number of ban events for credential+model pairs",
		},
		[]string{"credential", "model", "error_code"},
	)

	CredentialUnbanEvents = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_unban_events_total",
			Help: "Total number of unban events for credential+model pairs",
		},
		[]string{"credential", "model"},
	)

	InputTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_input_tokens_total",
			Help: "Total input tokens processed",
		},
		[]string{"credential", "model"},
	)

	OutputTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_output_tokens_total",
			Help: "Total output tokens generated",
		},
		[]string{"credential", "model"},
	)

	ReasoningTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_reasoning_tokens_total",
			Help: "Total reasoning tokens generated",
		},
		[]string{"credential", "model"},
	)

	CachedTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_cached_tokens_total",
			Help: "Total cached input tokens used",
		},
		[]string{"credential", "model"},
	)

	// Redis-specific metrics
	RedisConnectionErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_redis_connection_errors_total",
			Help: "Total number of Redis connection errors",
		},
		[]string{"operation"},
	)

	RedisFallbackEventsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_redis_fallback_events_total",
			Help: "Total number of times fallback to local backend occurred due to Redis errors",
		},
	)

	// Kafka publisher stats are snapshots of cumulative counters, so gauges avoid
	// double-counting when the periodic updater publishes a new snapshot.
	KafkaSpendLoggerQueuedTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_queued_total",
			Help: "Cumulative number of spend events queued for Kafka publishing",
		},
	)

	KafkaSpendLoggerProducedTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_produced_total",
			Help: "Cumulative number of spend events successfully produced to Kafka",
		},
	)

	KafkaSpendLoggerDroppedTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_dropped_total",
			Help: "Cumulative number of spend events dropped because the producer queue was full",
		},
	)

	KafkaSpendLoggerErrorsTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_errors_total",
			Help: "Cumulative number of spend events that failed to produce after all retries",
		},
	)

	KafkaSpendLoggerDLQSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_dlq_size",
			Help: "Current number of batches held in the Kafka spend logger dead letter queue",
		},
	)

	KafkaSpendLoggerHealthy = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_kafka_spend_logger_healthy",
			Help: "Kafka broker connectivity for spend-log publishing (1 = healthy, 0 = unhealthy)",
		},
	)

	SpendQueueDepth = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_queue_depth",
			Help: "Current number of spend entries waiting in the input channel",
		},
	)

	SpendPendingEntries = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_pending_entries",
			Help: "Accepted spend entries not yet resolved by the writer or DLQ",
		},
	)

	SpendPendingAggregationDepth = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_pending_aggregation_depth",
			Help: "Inserted spend batches waiting for or undergoing daily aggregation",
		},
	)

	SpendDLQSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_dlq_size",
			Help: "Current number of batches in the in-memory spend dead letter queue",
		},
	)

	SpendAggregationLagSeconds = promauto.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_aggregation_lag_seconds",
			Help: "Age in seconds of the oldest outstanding daily aggregation batch",
		},
		func() float64 {
			oldest := spendAggregationOldestUnixNano.Load()
			if oldest == 0 {
				return 0
			}
			lag := time.Since(time.Unix(0, oldest)).Seconds()
			if lag < 0 {
				return 0
			}
			return lag
		},
	)

	SpendComparisonWindowValid = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_spend_comparison_window_valid",
			Help: "Whether the current process-lifetime comparison window is transport-complete and fully aggregated",
		},
	)

	SpendDroppedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_dropped_total",
			Help: "Total spend entries dropped before persistence",
		},
	)

	SpendDLQOverflowTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_dlq_overflow_total",
			Help: "Total spend batches lost because the in-memory DLQ was full",
		},
	)

	SpendDuplicatesTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_duplicates_total",
			Help: "Total raw rows ignored by request_id ON CONFLICT",
		},
	)

	SpendAggregationErrorsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_aggregation_errors_total",
			Help: "Total terminal atomic accounting failures with an ambiguous commit outcome",
		},
	)

	SpendPendingAggregationOverflowTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_pending_aggregation_overflow_total",
			Help: "Total inserted spend batches that could not enter the daily aggregation queue",
		},
	)

	SpendComparisonRowsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_spend_comparison_rows_total",
			Help: "Newly persisted spend rows by comparison eligibility",
		},
		[]string{"eligibility"},
	)
)

// SpendSnapshot contains instantaneous spend writer state. Loss/error
// counters are recorded separately so repeated snapshots cannot double count.
type SpendSnapshot struct {
	QueueDepth            int
	PendingEntries        int
	PendingAggregation    int
	DLQSize               int
	AggregationLag        time.Duration
	ComparisonWindowValid bool
}

func ObserveSpendSnapshot(snapshot SpendSnapshot) {
	SpendQueueDepth.Set(float64(snapshot.QueueDepth))
	SpendPendingEntries.Set(float64(snapshot.PendingEntries))
	SpendPendingAggregationDepth.Set(float64(snapshot.PendingAggregation))
	SpendDLQSize.Set(float64(snapshot.DLQSize))
	if snapshot.PendingAggregation == 0 {
		spendAggregationOldestUnixNano.Store(0)
	} else {
		spendAggregationOldestUnixNano.Store(time.Now().Add(-snapshot.AggregationLag).UnixNano())
	}
	if snapshot.ComparisonWindowValid {
		SpendComparisonWindowValid.Set(1)
	} else {
		SpendComparisonWindowValid.Set(0)
	}
}

func addCounter(counter prometheus.Counter, count uint64) {
	if count > 0 {
		counter.Add(float64(count))
	}
}

func RecordSpendDropped(count uint64) {
	addCounter(SpendDroppedTotal, count)
}

func RecordSpendDLQOverflow(count uint64) {
	addCounter(SpendDLQOverflowTotal, count)
}

func RecordSpendDuplicates(count uint64) {
	addCounter(SpendDuplicatesTotal, count)
}

func RecordSpendAggregationErrors(count uint64) {
	addCounter(SpendAggregationErrorsTotal, count)
}

func RecordSpendPendingAggregationOverflow(count uint64) {
	addCounter(SpendPendingAggregationOverflowTotal, count)
}

func RecordSpendComparisonRows(eligible bool, count uint64) {
	if count == 0 {
		return
	}
	label := "ineligible"
	if eligible {
		label = "eligible"
	}
	SpendComparisonRowsTotal.WithLabelValues(label).Add(float64(count))
}

type Metrics struct {
	enabled bool
}

func New(enabled bool) *Metrics {
	return &Metrics{
		enabled: enabled,
	}
}

func (m *Metrics) isEnabled() bool {
	return m.enabled
}

// updateCredentialMetric updates a credential-level gauge metric
func (m *Metrics) updateCredentialMetric(gauge *prometheus.GaugeVec, credential string, value int) {
	if !m.isEnabled() {
		return
	}
	gauge.WithLabelValues(credential).Set(float64(value))
}

// updateModelMetric updates a model-level gauge metric
func (m *Metrics) updateModelMetric(gauge *prometheus.GaugeVec, credential, model string, value int) {
	if !m.isEnabled() {
		return
	}
	gauge.WithLabelValues(credential, model).Set(float64(value))
}

func (m *Metrics) RecordRequest(credential, endpoint, model string, statusCode int, duration time.Duration) {
	if !m.isEnabled() {
		return
	}

	status := strconv.Itoa(statusCode)
	RequestsTotal.WithLabelValues(credential, model, endpoint, status).Inc()
	RequestDuration.WithLabelValues(credential, endpoint).Observe(duration.Seconds())

	if statusCode != 200 {
		CredentialErrorsTotal.WithLabelValues(credential).Inc()
	}
}

func (m *Metrics) RecordAbortedRequest(credential, endpoint, model string) {
	if !m.isEnabled() {
		return
	}
	AbortedRequestsTotal.WithLabelValues(credential, model, endpoint).Inc()
}

func (m *Metrics) UpdateCredentialRPM(credential string, rpm int) {
	m.updateCredentialMetric(CredentialRPMCurrent, credential, rpm)
}

func (m *Metrics) UpdateCredentialTPM(credential string, tpm int) {
	m.updateCredentialMetric(CredentialTPMCurrent, credential, tpm)
}

func (m *Metrics) UpdateCredentialBanStatus(credential string, banned bool) {
	if !m.isEnabled() {
		return
	}
	value := 0.0
	if banned {
		value = 1.0
	}
	CredentialBanned.WithLabelValues(credential).Set(value)
}

func (m *Metrics) UpdateModelRPM(credential, model string, rpm int) {
	m.updateModelMetric(ModelRPMCurrent, credential, model, rpm)
}

func (m *Metrics) UpdateModelTPM(credential, model string, tpm int) {
	m.updateModelMetric(ModelTPMCurrent, credential, model, tpm)
}

func (m *Metrics) RecordTokenUsage(credential, model string, inputTokens, outputTokens, reasoningTokens, cachedTokens int) {
	if !m.isEnabled() {
		return
	}
	if inputTokens > 0 {
		InputTokensTotal.WithLabelValues(credential, model).Add(float64(inputTokens))
	}
	if outputTokens > 0 {
		OutputTokensTotal.WithLabelValues(credential, model).Add(float64(outputTokens))
	}
	if reasoningTokens > 0 {
		ReasoningTokensTotal.WithLabelValues(credential, model).Add(float64(reasoningTokens))
	}
	if cachedTokens > 0 {
		CachedTokensTotal.WithLabelValues(credential, model).Add(float64(cachedTokens))
	}
}

// RecordRedisConnectionError records a Redis connection error.
func (m *Metrics) RecordRedisConnectionError(operation string) {
	if !m.isEnabled() {
		return
	}
	RedisConnectionErrorsTotal.WithLabelValues(operation).Inc()
}

// RecordRedisFallback records a fallback event from Redis to local backend.
func (m *Metrics) RecordRedisFallback() {
	if !m.isEnabled() {
		return
	}
	RedisFallbackEventsTotal.Inc()
}

// UpdateKafkaSpendLoggerStats publishes a kafkalog producer Stats snapshot
// (queue/DLQ counters, broker health) as Prometheus metrics. Intended to be
// called periodically from a background updater, not per-request.
func (m *Metrics) UpdateKafkaSpendLoggerStats(queued, produced, dropped, errors uint64, dlqSize int, healthy bool) {
	if !m.isEnabled() {
		return
	}
	KafkaSpendLoggerQueuedTotal.Set(float64(queued))
	KafkaSpendLoggerProducedTotal.Set(float64(produced))
	KafkaSpendLoggerDroppedTotal.Set(float64(dropped))
	KafkaSpendLoggerErrorsTotal.Set(float64(errors))
	KafkaSpendLoggerDLQSize.Set(float64(dlqSize))
	h := 0.0
	if healthy {
		h = 1.0
	}
	KafkaSpendLoggerHealthy.Set(h)
}
