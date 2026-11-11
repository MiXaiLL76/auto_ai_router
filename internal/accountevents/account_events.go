// Package accountevents publishes upstream outcome events for the health
// worker. It reuses the kafkalog producer wrapper (same queueing/retry
// machinery as the spend logs) with the account-events topic; the broker,
// SASL and TLS settings come from the router's kafka section.
package accountevents

import (
	"context"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// EventsTotal counts outcome events queued for publishing.
	EventsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_account_events_total",
			Help: "Outcome events queued for the health worker",
		},
	)

	// EventsDropped counts outcome events rejected (queue full or encode error).
	EventsDropped = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_account_events_dropped_total",
			Help: "Outcome events dropped (queue full or encode error)",
		},
	)
)

// Event is one upstream outcome for the health worker. Keyed by credential
// so the worker sees all events of one account in one partition.
type Event struct {
	Credential        string `json:"credential"`
	Provider          string `json:"provider,omitempty"`
	Model             string `json:"model,omitempty"`
	StatusCode        int64  `json:"status_code"`
	RetryAfterSeconds int64  `json:"retry_after_seconds,omitempty"`
	Error             string `json:"error,omitempty"`
	EventID           string `json:"event_id,omitempty"`
	Timestamp         int64  `json:"timestamp"`
}

// Key implements kafkalog.Keyed (partitioning by credential).
func (e *Event) Key() []byte { return []byte(e.Credential) }

// Publisher wraps the kafkalog producer for account events.
type Publisher struct {
	cfg    *config.AccountEventsConfig
	logger *kafkalog.Logger[*Event]
}

// New builds the publisher over the router's kafka settings (brokers/SASL/TLS
// come from the kafka section; only the topic is account-events specific).
func New(cfg *config.AccountEventsConfig, kafkaCfg *config.KafkaConfig, log *slog.Logger) (*Publisher, error) {
	klc := &kafkalog.Config{
		Brokers:          kafkaCfg.Brokers,
		Topic:            cfg.Topic,
		ClientID:         kafkaCfg.ClientID,
		LogQueueSize:     kafkaCfg.LogQueueSize,
		LogBatchSize:     kafkaCfg.LogBatchSize,
		LogFlushInterval: kafkaCfg.LogFlushInterval,
		LogWorkers:       kafkaCfg.LogWorkers,
		TLSEnabled:       kafkaCfg.TLSEnabled,
		SASLMechanism:    kafkaCfg.SASLMechanism,
		SASLUsername:     kafkaCfg.SASLUsername,
		SASLPassword:     kafkaCfg.SASLPassword,
		TLSCACert:        kafkaCfg.TLSCACert,
		Logger:           log,
	}
	logger, err := kafkalog.NewLogger[*Event](klc)
	if err != nil {
		return nil, err
	}
	logger.Start()
	return &Publisher{cfg: cfg, logger: logger}, nil
}

// Report queues one outcome event (fires every upstream outcome, including
// successes: the worker needs them to reset attempt counters). eventID
// enables at-least-once deduplication in the worker.
func (p *Publisher) Report(credential, provider, model string, statusCode int, retryAfterSeconds int, reason string) {
	if p == nil {
		return
	}
	ev := &Event{
		Credential:        credential,
		Provider:          provider,
		Model:             model,
		StatusCode:        int64(statusCode),
		RetryAfterSeconds: int64(retryAfterSeconds),
		Error:             reason,
		EventID:           newEventID(),
		Timestamp:         time.Now().UTC().Unix(),
	}
	if err := p.logger.Log(ev); err != nil {
		EventsDropped.Inc()
		return
	}
	EventsTotal.Inc()
}

// Shutdown flushes and closes the producer.
func (p *Publisher) Shutdown(ctx context.Context) error {
	if p == nil {
		return nil
	}
	return p.logger.Shutdown(ctx)
}

func newEventID() string {
	return uuid.NewString()
}
