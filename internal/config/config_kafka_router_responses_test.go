package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// TestKafkaConfig_UnmarshalYAML_RouterResponses pins the parsing of the
// router-response section: the toggles, the topic default, and the
// redaction default.
func TestKafkaConfig_UnmarshalYAML_RouterResponses(t *testing.T) {
	yamlDoc := `
enabled: true
brokers:
  - "kafka:9092"
topic: air.spend_logs
router_responses:
  enabled: true
  topic: router-responses
`
	var kafkaCfg KafkaConfig
	require.NoError(t, yaml.Unmarshal([]byte(yamlDoc), &kafkaCfg))
	assert.True(t, kafkaCfg.RouterResponses.Enabled)
	assert.Equal(t, "router-responses", kafkaCfg.RouterResponses.Topic)
	assert.True(t, kafkaCfg.RouterResponses.RedactSensitiveFields,
		"redact_sensitive_fields must default to true when omitted")
}

// TestKafkaConfig_UnmarshalYAML_RouterResponsesTopicDefaultsWhenOmitted
// guards the shape a deployment would plausibly write: enabling the section
// without spelling out a topic must fall back to the documented default,
// not become empty and fail startup validation ("topic is required") --
// the same trap the raw-bodies section has its own regression test for.
func TestKafkaConfig_UnmarshalYAML_RouterResponsesTopicDefaultsWhenOmitted(t *testing.T) {
	yamlDoc := `
enabled: true
brokers:
  - "kafka:9092"
topic: air.spend_logs
router_responses:
  enabled: true
`
	var kafkaCfg KafkaConfig
	require.NoError(t, yaml.Unmarshal([]byte(yamlDoc), &kafkaCfg))
	assert.Equal(t, "router-responses", kafkaCfg.RouterResponses.Topic,
		"an omitted topic must fall back to the documented default, not become empty")

	cfg := baseValidConfigForKafkaTests()
	cfg.Kafka = kafkaCfg
	require.NoError(t, cfg.Validate(), "this exact shape must pass startup validation, not crash-loop")
}

// TestKafkaConfig_UnmarshalYAML_RouterResponsesDefaultsToDisabled pins that
// the section is off unless it is asked for: a router that says nothing
// about it must not start publishing response bodies.
func TestKafkaConfig_UnmarshalYAML_RouterResponsesDefaultsToDisabled(t *testing.T) {
	yamlDoc := `
enabled: true
brokers:
  - "kafka:9092"
topic: air.spend_logs
`
	var kafkaCfg KafkaConfig
	require.NoError(t, yaml.Unmarshal([]byte(yamlDoc), &kafkaCfg))
	assert.False(t, kafkaCfg.RouterResponses.Enabled,
		"the router-response path must be off by default")
}

// TestKafkaConfig_UnmarshalYAML_RouterResponsesRedactCanBeDisabled pins
// that the redaction toggle is readable -- the deliberate escape hatch for
// a short-lived, access-controlled debugging session.
func TestKafkaConfig_UnmarshalYAML_RouterResponsesRedactCanBeDisabled(t *testing.T) {
	yamlDoc := `
enabled: true
brokers:
  - "kafka:9092"
topic: air.spend_logs
router_responses:
  enabled: true
  redact_sensitive_fields: false
`
	var kafkaCfg KafkaConfig
	require.NoError(t, yaml.Unmarshal([]byte(yamlDoc), &kafkaCfg))
	assert.False(t, kafkaCfg.RouterResponses.RedactSensitiveFields)
}

// TestKafkaConfig_UnmarshalYAML_RouterResponsesRejectsBadValues pins that a
// mistyped toggle is a startup error, not a silently-defaulted one.
func TestKafkaConfig_UnmarshalYAML_RouterResponsesRejectsBadValues(t *testing.T) {
	for _, doc := range []string{
		"enabled: true\nrouter_responses:\n  enabled: maybe\n",
		"enabled: true\nrouter_responses:\n  enabled: true\n  redact_sensitive_fields: maybe\n",
	} {
		var k KafkaConfig
		assert.Error(t, yaml.Unmarshal([]byte(doc), &k), "doc: %s", doc)
	}
}

// kafkaWithDefaults returns a Kafka config that passes every check but the
// ones under test: the queue/batch/worker sizes are required to be
// positive, and a case that does not set them would fail on those first.
func kafkaWithDefaults() KafkaConfig {
	return KafkaConfig{
		Enabled:          true,
		Brokers:          []string{"kafka:9092"},
		Topic:            "air.spend_logs",
		LogQueueSize:     5000,
		LogBatchSize:     100,
		LogFlushInterval: 5 * time.Second,
		LogWorkers:       4,
	}
}

// TestKafkaConfig_RouterResponsesValidation pins the rules the section has
// to obey: it needs Kafka, it needs its own topic, and that topic must be
// its own -- distinct from the spend log's and from the raw-bodies'.
//
// The last one matters because the three carry different streams with
// different retentions: a response body mixed into the spend log would be
// kept for years alongside the billing rows.
func TestKafkaConfig_RouterResponsesValidation(t *testing.T) {
	tests := []struct {
		name        string
		kafka       KafkaConfig
		wantErr     bool
		errContains string
	}{
		{
			name: "disabled is always fine",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: false}
				return k
			}(),
			wantErr: false,
		},
		{
			name: "enabled with a distinct topic passes",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: true, Topic: "router-responses"}
				return k
			}(),
			wantErr: false,
		},
		{
			name: "enabled without kafka.enabled fails",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.Enabled = false
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: true, Topic: "router-responses"}
				return k
			}(),
			wantErr:     true,
			errContains: "kafka.router_responses.enabled requires kafka.enabled=true",
		},
		{
			name: "enabled without topic fails",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: true}
				return k
			}(),
			wantErr:     true,
			errContains: "kafka.router_responses.topic is required",
		},
		{
			name: "enabled with the spend-log topic fails",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: true, Topic: "air.spend_logs"}
				return k
			}(),
			wantErr:     true,
			errContains: "kafka.router_responses.topic must differ from kafka.topic",
		},
		{
			name: "enabled with the raw-bodies topic fails",
			kafka: func() KafkaConfig {
				k := kafkaWithDefaults()
				k.RawBodies = KafkaRawBodiesConfig{Enabled: true, Topic: "raw-bodies"}
				k.RouterResponses = KafkaRouterResponsesConfig{Enabled: true, Topic: "raw-bodies"}
				return k
			}(),
			wantErr:     true,
			errContains: "kafka.router_responses.topic must differ from kafka.raw_bodies.topic",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := baseValidConfigForKafkaTests()
			cfg.Kafka = tt.kafka
			err := cfg.Validate()
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}
			require.NoError(t, err)
		})
	}
}
