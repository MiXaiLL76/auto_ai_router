// Package kafkalog publishes an expanded copy of every SpendLogEntry to a
// Kafka topic ("air.spend_logs") for downstream ClickHouse analytics,
// alongside (not instead of) the existing LiteLLM Postgres write path.
//
// Architecture mirrors internal/litellmdb/spendlog: async queue -> batch ->
// retry with backoff -> in-memory Dead Letter Queue -> graceful shutdown.
// Unlike litellmdb, Kafka availability is never required for request
// processing to proceed — see Manager.IsHealthy.
package kafkalog

import (
	"context"
	"log/slog"
)

// Manager is the main interface for the kafkalog module.
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
	logger *Logger
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

	logger, err := NewLogger(cfg)
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

// ==================== Compile-time interface check ====================

var _ Manager = (*DefaultManager)(nil)
var _ Manager = (*NoopManager)(nil)
