package kafkalog

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"
)

// Config holds configuration for the kafkalog module.
type Config struct {
	// Brokers is the list of Kafka bootstrap addresses ("host:port").
	Brokers []string
	// Topic is the target topic for spend events (e.g. "air.spend_logs").
	Topic string
	// ClientID identifies this producer to the Kafka brokers.
	ClientID string

	// Async producer queue
	LogQueueSize     int           // Queue buffer size (default: 5000)
	LogBatchSize     int           // Batch size per ProduceSync call (default: 100)
	LogFlushInterval time.Duration // Flush interval (default: 5s)
	LogWorkers       int           // Number of parallel queue-draining workers (default: 4)

	// TLS / SASL (optional, for production Kafka clusters)
	TLSEnabled    bool
	SASLMechanism string // "" | "PLAIN" | "SCRAM-SHA-256" | "SCRAM-SHA-512"
	SASLUsername  string
	SASLPassword  string

	Logger *slog.Logger

	// FallbackNotifier, if set, is called (best-effort) for every event in a
	// batch evicted from the in-memory DLQ due to overflow (a sustained Kafka
	// outage that outlasts dlqMaxSize retries) so the caller can flag the
	// event's existing durable record (e.g. its LiteLLM_SpendLogs Postgres row)
	// for later re-send, instead of the batch being silently dropped. Never
	// blocks producing/retrying; a non-nil error is only logged.
	FallbackNotifier func(ctx context.Context, requestID, reason string) error
}

// DefaultConfig returns configuration with default values.
func DefaultConfig() *Config {
	return &Config{
		Topic:            "air.spend_logs",
		ClientID:         "auto_ai_router",
		LogQueueSize:     5000,
		LogBatchSize:     100,
		LogFlushInterval: 5 * time.Second,
		LogWorkers:       4,
	}
}

// ApplyDefaults fills zero fields with defaults.
func (c *Config) ApplyDefaults() {
	defaults := DefaultConfig()

	if c.Topic == "" {
		c.Topic = defaults.Topic
	}
	if c.ClientID == "" {
		c.ClientID = defaults.ClientID
	}
	if c.LogQueueSize == 0 {
		c.LogQueueSize = defaults.LogQueueSize
	}
	if c.LogBatchSize == 0 {
		c.LogBatchSize = defaults.LogBatchSize
	}
	if c.LogFlushInterval == 0 {
		c.LogFlushInterval = defaults.LogFlushInterval
	}
	if c.LogWorkers == 0 {
		c.LogWorkers = defaults.LogWorkers
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// Validate checks configuration validity.
func (c *Config) Validate() error {
	if len(c.Brokers) == 0 {
		return errors.New("kafkalog: at least one broker is required")
	}
	if c.Topic == "" {
		return errors.New("kafkalog: topic is required")
	}
	if c.LogQueueSize <= 0 {
		return errors.New("kafkalog: log_queue_size must be positive")
	}
	if c.LogBatchSize <= 0 {
		return errors.New("kafkalog: log_batch_size must be positive")
	}
	if c.LogFlushInterval <= 0 {
		return errors.New("kafkalog: log_flush_interval must be positive")
	}
	if c.LogWorkers <= 0 {
		return errors.New("kafkalog: log_workers must be positive")
	}
	switch c.SASLMechanism {
	case "", "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512":
	default:
		return fmt.Errorf("kafkalog: unsupported sasl_mechanism %q", c.SASLMechanism)
	}
	if c.SASLMechanism != "" && (c.SASLUsername == "" || c.SASLPassword == "") {
		return errors.New("kafkalog: sasl_username and sasl_password are required when sasl_mechanism is set")
	}
	return nil
}
