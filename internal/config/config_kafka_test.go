package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
)

// baseValidConfigForKafkaTests returns the minimal Config needed to pass
// every non-Kafka Validate() check, so each test case only needs to vary the
// Kafka fields under test.
func baseValidConfigForKafkaTests() *Config {
	return &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{
			{Name: "test", Type: "openai", APIKey: "key", BaseURL: "http://test.com", RPM: 10},
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
	}
}

func TestConfig_Validate_Kafka(t *testing.T) {
	tests := []struct {
		name        string
		kafka       KafkaConfig
		wantErr     bool
		errContains string
	}{
		{
			name: "valid minimal config passes",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
				LogWorkers:       4,
			},
			wantErr: false,
		},
		{
			name: "missing brokers fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
			},
			wantErr:     true,
			errContains: "kafka.brokers is required",
		},
		{
			name: "missing topic fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
			},
			wantErr:     true,
			errContains: "kafka.topic is required",
		},
		{
			name: "negative log_queue_size fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     -1,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
			},
			wantErr:     true,
			errContains: "kafka.log_queue_size must be positive",
		},
		{
			name: "negative log_batch_size fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     -5,
				LogFlushInterval: 5 * time.Second,
			},
			wantErr:     true,
			errContains: "kafka.log_batch_size must be positive",
		},
		{
			name: "non-positive log_flush_interval fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 0,
			},
			wantErr:     true,
			errContains: "kafka.log_flush_interval must be positive",
		},
		{
			name: "unsupported sasl_mechanism fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
				LogWorkers:       4,
				SASLMechanism:    "GSSAPI",
			},
			wantErr:     true,
			errContains: "kafka.sasl_mechanism unsupported",
		},
		{
			name: "sasl_mechanism set without credentials fails",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
				LogWorkers:       4,
				SASLMechanism:    "PLAIN",
			},
			wantErr:     true,
			errContains: "kafka.sasl_username and kafka.sasl_password are required",
		},
		{
			name: "sasl_mechanism with credentials passes",
			kafka: KafkaConfig{
				Enabled:          true,
				Brokers:          []string{"kafka:9092"},
				Topic:            "air.spend_logs",
				LogQueueSize:     5000,
				LogBatchSize:     100,
				LogFlushInterval: 5 * time.Second,
				LogWorkers:       4,
				SASLMechanism:    "SCRAM-SHA-256",
				SASLUsername:     "user",
				SASLPassword:     "pass",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := baseValidConfigForKafkaTests()
			cfg.Kafka = tt.kafka
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfig_Validate_KafkaOnlyModeRequiresKafka(t *testing.T) {
	cfg := baseValidConfigForKafkaTests()
	cfg.LiteLLMDB.DisableSpendLogsWrite = true
	cfg.Kafka.Enabled = false

	err := cfg.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "disable_spend_logs_write=true requires kafka.enabled=true")
}

func TestKafkaConfig_UnmarshalYAML_SplitsCommaSeparatedBrokers(t *testing.T) {
	t.Setenv("KAFKA_BROKERS_TEST", "kafka1:9092,kafka2:9092, kafka3:9092 ,,")

	yamlDoc := `
enabled: true
brokers:
  - "os.environ/KAFKA_BROKERS_TEST"
topic: air.spend_logs
`
	var kafkaCfg KafkaConfig
	a := assert.New(t)
	err := yaml.Unmarshal([]byte(yamlDoc), &kafkaCfg)
	a.NoError(err)
	a.Equal([]string{"kafka1:9092", "kafka2:9092", "kafka3:9092"}, kafkaCfg.Brokers)
}
