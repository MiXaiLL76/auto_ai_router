package kafkalog

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig_ApplyDefaults(t *testing.T) {
	cfg := &Config{Brokers: []string{"kafka:9092"}}
	cfg.ApplyDefaults()

	assert.Equal(t, "air.spend_logs", cfg.Topic)
	assert.Equal(t, "auto_ai_router", cfg.ClientID)
	assert.Equal(t, 5000, cfg.LogQueueSize)
	assert.Equal(t, 100, cfg.LogBatchSize)
	assert.Equal(t, 5*time.Second, cfg.LogFlushInterval)
	assert.NotNil(t, cfg.Logger)
}

func TestConfig_ApplyDefaults_PreservesSetFields(t *testing.T) {
	cfg := &Config{
		Brokers:          []string{"kafka:9092"},
		Topic:            "custom.topic",
		ClientID:         "custom-client",
		LogQueueSize:     10,
		LogBatchSize:     5,
		LogFlushInterval: time.Minute,
	}
	cfg.ApplyDefaults()

	assert.Equal(t, "custom.topic", cfg.Topic)
	assert.Equal(t, "custom-client", cfg.ClientID)
	assert.Equal(t, 10, cfg.LogQueueSize)
	assert.Equal(t, 5, cfg.LogBatchSize)
	assert.Equal(t, time.Minute, cfg.LogFlushInterval)
}

func TestConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *Config
		wantErr bool
	}{
		{
			name:    "valid minimal",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second, LogWorkers: 1},
			wantErr: false,
		},
		{
			name:    "no brokers",
			cfg:     &Config{Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second},
			wantErr: true,
		},
		{
			name:    "no topic",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second},
			wantErr: true,
		},
		{
			name:    "zero queue size",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 0, LogBatchSize: 1, LogFlushInterval: time.Second},
			wantErr: true,
		},
		{
			name:    "zero batch size",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 0, LogFlushInterval: time.Second},
			wantErr: true,
		},
		{
			name:    "zero flush interval",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: 0},
			wantErr: true,
		},
		{
			name:    "unsupported sasl mechanism",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second, SASLMechanism: "GSSAPI"},
			wantErr: true,
		},
		{
			name:    "sasl mechanism without credentials",
			cfg:     &Config{Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second, SASLMechanism: "PLAIN"},
			wantErr: true,
		},
		{
			name: "sasl mechanism with credentials",
			cfg: &Config{
				Brokers: []string{"kafka:9092"}, Topic: "t", LogQueueSize: 1, LogBatchSize: 1, LogFlushInterval: time.Second, LogWorkers: 1,
				SASLMechanism: "SCRAM-SHA-256", SASLUsername: "u", SASLPassword: "p",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
