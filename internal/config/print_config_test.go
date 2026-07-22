package config

import (
	"bytes"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestPrintConfig(t *testing.T) {
	// Create a buffer to capture log output
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:                   8080,
			MaxBodySizeMB:          10,
			ResponseBodyMultiplier: 1,
			RequestTimeout:         30 * time.Second,
			ReadTimeout:            60 * time.Second,
			WriteTimeout:           60 * time.Second,
			IdleTimeout:            120 * time.Second,
			LoggingLevel:           "info",
			MasterKey:              "secret-key",
			DefaultModelsRPM:       100,
			MaxIdleConns:           10,
			MaxIdleConnsPerHost:    5,
			IdleConnTimeout:        30 * time.Second,
			ModelPricesLink:        "https://example.com/prices.json",
			MaxProviderRetries:     3,
			SessionStickyEnabled:   true,
			SessionStickyTTL:       6,
		},
		Monitoring: MonitoringConfig{
			PrometheusEnabled: true,
			HealthCheckPath:   "/health",
			LogErrors:         true,
			ErrorsLogPath:     "/var/log/errors.log",
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts:    3,
			BanDuration:    5 * time.Minute,
			ErrorCodes:     []int{401, 403, 429},
			ErrorCodeRules: nil,
		},
		Credentials: []CredentialConfig{
			{
				Name:       "test-provider",
				Type:       ProviderTypeOpenAI,
				BaseURL:    "https://api.openai.com",
				APIKey:     "sk-test",
				RPM:        60,
				TPM:        10000,
				IsFallback: false,
			},
		},
		Models: []ModelRPMConfig{
			{
				Name:       "gpt-4",
				Credential: "test-provider",
				RPM:        50,
				TPM:        10000,
			},
		},
		ModelAlias: map[string]string{
			"gpt-4": "gpt-4o",
		},
		LiteLLMDB: LiteLLMDBConfig{
			Enabled:             true,
			DatabaseURL:         "postgres://user:pass@localhost:5432/db",
			IsRequired:          false,
			MaxConns:            20,
			MinConns:            5,
			HealthCheckInterval: 10 * time.Second,
			ConnectTimeout:      5 * time.Second,
			AuthCacheTTL:        15 * time.Minute,
			AuthCacheSize:       1000,
			LogQueueSize:        100,
			LogBatchSize:        10,
			LogFlushInterval:    1 * time.Minute,
		},
	}

	// This should not panic
	PrintConfig(logger, cfg)

	// Verify that some log output was generated
	output := buf.String()
	assert.NotEmpty(t, output)
	assert.Contains(t, output, "Configuration Loaded")
	assert.Contains(t, output, "session_sticky_enabled")
	assert.Contains(t, output, "session_sticky_ttl_minutes")
}

func TestPrintConfig_KafkaEnabled(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "secret-key",
			RequestTimeout: 30 * time.Second,
			LoggingLevel:   "info",
		},
		Monitoring: MonitoringConfig{
			PrometheusEnabled: false,
			HealthCheckPath:   "/health",
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts: 3,
			ErrorCodes:  []int{401},
		},
		Credentials: []CredentialConfig{
			{Name: "test-provider", Type: ProviderTypeOpenAI, BaseURL: "https://api.openai.com", APIKey: "sk-test", RPM: 60},
		},
		Kafka: KafkaConfig{
			Enabled:          true,
			Brokers:          []string{"kafka1:9092", "kafka2:9092"},
			Topic:            "air.spend_logs",
			ClientID:         "auto_ai_router",
			LogQueueSize:     5000,
			LogBatchSize:     100,
			LogFlushInterval: 5 * time.Second,
			SASLMechanism:    "PLAIN",
			SASLUsername:     "kafka-user",
			SASLPassword:     "super-secret-password",
		},
	}

	PrintConfig(logger, cfg)

	output := buf.String()
	assert.Contains(t, output, "kafka (ENABLED)")
	assert.Contains(t, output, "air.spend_logs")
	assert.Contains(t, output, "kafka-user")
	assert.Contains(t, output, "***REDACTED***")
	assert.NotContains(t, output, "super-secret-password", "SASL password must never be logged in the clear")
}

func TestPrintConfig_KafkaDisabled(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "secret-key",
			RequestTimeout: 30 * time.Second,
			LoggingLevel:   "info",
		},
		Monitoring: MonitoringConfig{HealthCheckPath: "/health"},
		Fail2Ban:   Fail2BanConfig{MaxAttempts: 3, ErrorCodes: []int{401}},
		Credentials: []CredentialConfig{
			{Name: "test-provider", Type: ProviderTypeOpenAI, BaseURL: "https://api.openai.com", APIKey: "sk-test", RPM: 60},
		},
	}

	PrintConfig(logger, cfg)

	output := buf.String()
	assert.Contains(t, output, "\"kafka\"")
	assert.Contains(t, output, "DISABLED")
}

func TestPrintConfig_DisabledLiteLLMDB(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "secret-key",
			RequestTimeout: 30 * time.Second,
			LoggingLevel:   "info",
		},
		Monitoring: MonitoringConfig{
			PrometheusEnabled: false,
			HealthCheckPath:   "/health",
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts: 3,
			BanDuration: 0,
			ErrorCodes:  []int{401, 403},
		},
		Credentials: []CredentialConfig{
			{
				Name:    "test-provider",
				Type:    ProviderTypeOpenAI,
				BaseURL: "https://api.openai.com",
				APIKey:  "sk-test",
				RPM:     60,
			},
		},
		LiteLLMDB: LiteLLMDBConfig{
			Enabled: false,
		},
	}

	PrintConfig(logger, cfg)

	output := buf.String()
	assert.NotEmpty(t, output)
	assert.Contains(t, output, "DISABLED")
}

func TestPrintConfig_VertexAI(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "secret-key",
			RequestTimeout: 30 * time.Second,
			LoggingLevel:   "info",
		},
		Monitoring: MonitoringConfig{
			PrometheusEnabled: false,
			HealthCheckPath:   "/health",
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts: 3,
			BanDuration: 0,
			ErrorCodes:  []int{401},
		},
		Credentials: []CredentialConfig{
			{
				Name:      "vertex-provider",
				Type:      ProviderTypeVertexAI,
				ProjectID: "my-project",
				Location:  "us-central1",
				APIKey:    "test-key",
				RPM:       60,
			},
		},
	}

	PrintConfig(logger, cfg)

	output := buf.String()
	assert.NotEmpty(t, output)
	assert.Contains(t, output, "project_id")
	assert.Contains(t, output, "location")
}

func TestPrintConfig_EmptyModels(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))

	cfg := &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "secret-key",
			RequestTimeout: 30 * time.Second,
			LoggingLevel:   "info",
		},
		Monitoring: MonitoringConfig{
			PrometheusEnabled: false,
			HealthCheckPath:   "/health",
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts: 3,
			BanDuration: 0,
			ErrorCodes:  []int{401},
		},
		Credentials: []CredentialConfig{},
		Models:      []ModelRPMConfig{},
	}

	// Should not panic with empty models
	PrintConfig(logger, cfg)

	output := buf.String()
	assert.NotEmpty(t, output)
}
