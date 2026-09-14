// Package kafkalog publishes an expanded copy of every SpendLogEntry to a
// Kafka topic ("air.spend_logs") for downstream ClickHouse analytics,
// alongside (not instead of) the existing LiteLLM Postgres write path. It
// also has a second, independently-toggleable write-path (ErrorBodyManager)
// that publishes raw request/response bodies for failed requests only, to
// its own topic -- kept separate from spend events because that data is
// bulky and short-retention, unlike the spend/billing rows in air.logs.
//
// Architecture mirrors internal/litellmdb/spendlog: async queue -> batch ->
// retry with backoff -> in-memory Dead Letter Queue -> graceful shutdown.
// Unlike litellmdb, Kafka availability is never required for request
// processing to proceed — see Manager.IsHealthy. Both write-paths share the
// same generic Logger[T] engine (see logger.go), just parameterized over a
// different event type and pointed at a different topic.
package kafkalog

import (
	"context"
	"log/slog"
)

// Manager is the spend-log write-path interface.
type Manager interface {
	// LogSpend queues a spend event for asynchronous publishing to Kafka.
	// Returns an error only if the event could not be queued (e.g. queue full).
	LogSpend(event *SpendEvent) error

	// IsEnabled reports whether Kafka publishing is configured on.
	IsEnabled() bool

	// IsHealthy reports current broker connectivity. Kafka being unhealthy
	// never blocks request processing — it only affects this flag and metrics.
	IsHealthy() bool

	// Stats returns producer statistics for observability.
	Stats() Stats

	// Shutdown stops the producer, flushing pending events.
	Shutdown(ctx context.Context) error
}

// ==================== NoopManager ====================

// NoopManager is a no-op implementation used when Kafka publishing is disabled.
type NoopManager struct{}

// NewNoopManager creates a new no-op manager.
func NewNoopManager() *NoopManager {
	return &NoopManager{}
}

func (n *NoopManager) LogSpend(_ *SpendEvent) error { return nil }
func (n *NoopManager) IsEnabled() bool              { return false }
func (n *NoopManager) IsHealthy() bool              { return false }
func (n *NoopManager) Stats() Stats                 { return Stats{} }
func (n *NoopManager) Shutdown(_ context.Context) error {
	return nil
}

// ==================== DefaultManager ====================

// DefaultManager is the real implementation of Manager, backed by a Kafka producer Logger.
type DefaultManager struct {
	logger *Logger[*SpendEvent]
	log    *slog.Logger
}

// New creates a new Manager instance and starts the background producer.
// Never fails hard on broker unavailability: the client connects lazily and
// IsHealthy() reflects connectivity once probed, matching the "no
// kafka.is_required" decision — Kafka issues degrade the health flag, not
// startup or request processing.
func New(cfg *Config) (Manager, error) {
	cfg.ApplyDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	logger, err := NewLogger[*SpendEvent](cfg)
	if err != nil {
		return nil, err
	}
	logger.Start()

	m := &DefaultManager{
		logger: logger,
		log:    cfg.Logger,
	}

	cfg.Logger.Info("Kafka spend logger initialized",
		"brokers", cfg.Brokers,
		"topic", cfg.Topic,
		"log_queue_size", cfg.LogQueueSize,
	)

	return m, nil
}

func (m *DefaultManager) LogSpend(event *SpendEvent) error {
	if event == nil {
		return nil
	}
	return m.logger.Log(event)
}

func (m *DefaultManager) IsEnabled() bool { return true }

func (m *DefaultManager) IsHealthy() bool { return m.logger.IsHealthy() }

func (m *DefaultManager) Stats() Stats { return m.logger.Stats() }

func (m *DefaultManager) Shutdown(ctx context.Context) error {
	m.log.Info("Shutting down Kafka spend logger...")
	err := m.logger.Shutdown(ctx)
	m.log.Info("Kafka spend logger shutdown complete")
	return err
}

// ==================== ErrorBodyManager ====================

// ErrorBodyManager is the raw-error-body write-path interface. Structurally
// identical to Manager (same lifecycle: enabled/healthy/stats/shutdown), but
// kept as its own interface -- rather than a second type parameter on
// Manager -- so a caller holding just a Manager can't accidentally call
// LogErrorBody, and vice versa; the two write-paths are independently
// enabled and genuinely optional in different ways (see
// docs/litellm-integration/kafka_spend_log.md).
type ErrorBodyManager interface {
	// LogErrorBody queues a raw request/response body event for a failed
	// request. Returns an error only if the event could not be queued.
	LogErrorBody(event *ErrorBodyEvent) error

	IsEnabled() bool
	IsHealthy() bool
	Stats() Stats
	Shutdown(ctx context.Context) error
}

// NoopErrorBodyManager is a no-op implementation used when error-body
// publishing is disabled (the default).
type NoopErrorBodyManager struct{}

// NewNoopErrorBodyManager creates a new no-op error-body manager.
func NewNoopErrorBodyManager() *NoopErrorBodyManager {
	return &NoopErrorBodyManager{}
}

func (n *NoopErrorBodyManager) LogErrorBody(_ *ErrorBodyEvent) error { return nil }
func (n *NoopErrorBodyManager) IsEnabled() bool                      { return false }
func (n *NoopErrorBodyManager) IsHealthy() bool                      { return false }
func (n *NoopErrorBodyManager) Stats() Stats                         { return Stats{} }
func (n *NoopErrorBodyManager) Shutdown(_ context.Context) error {
	return nil
}

// DefaultErrorBodyManager is the real implementation of ErrorBodyManager.
type DefaultErrorBodyManager struct {
	logger *Logger[*ErrorBodyEvent]
	log    *slog.Logger
}

// NewErrorBody creates a new ErrorBodyManager instance and starts its
// background producer. cfg is a plain kafkalog.Config pointed at the
// error-bodies topic (same brokers/TLS/SASL as the spend-log cfg is the
// common case, but that's the caller's choice, not enforced here).
func NewErrorBody(cfg *Config) (ErrorBodyManager, error) {
	cfg.ApplyDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	logger, err := NewLogger[*ErrorBodyEvent](cfg)
	if err != nil {
		return nil, err
	}
	logger.Start()

	m := &DefaultErrorBodyManager{
		logger: logger,
		log:    cfg.Logger,
	}

	cfg.Logger.Info("Kafka error-body logger initialized",
		"brokers", cfg.Brokers,
		"topic", cfg.Topic,
		"log_queue_size", cfg.LogQueueSize,
	)

	return m, nil
}

func (m *DefaultErrorBodyManager) LogErrorBody(event *ErrorBodyEvent) error {
	if event == nil {
		return nil
	}
	return m.logger.Log(event)
}

func (m *DefaultErrorBodyManager) IsEnabled() bool { return true }

func (m *DefaultErrorBodyManager) IsHealthy() bool { return m.logger.IsHealthy() }

func (m *DefaultErrorBodyManager) Stats() Stats { return m.logger.Stats() }

func (m *DefaultErrorBodyManager) Shutdown(ctx context.Context) error {
	m.log.Info("Shutting down Kafka error-body logger...")
	err := m.logger.Shutdown(ctx)
	m.log.Info("Kafka error-body logger shutdown complete")
	return err
}

// ==================== Compile-time interface checks ====================

var _ Manager = (*DefaultManager)(nil)
var _ Manager = (*NoopManager)(nil)
var _ ErrorBodyManager = (*DefaultErrorBodyManager)(nil)
var _ ErrorBodyManager = (*NoopErrorBodyManager)(nil)
