package accountevents

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventKeyIsCredential(t *testing.T) {
	ev := &Event{Credential: "openai-prod-1"}
	assert.Equal(t, []byte("openai-prod-1"), ev.Key())
}

// No-broker smoke test: the publisher must construct, queue reports and
// shut down without panicking (delivery fails into the retry queue, which
// is the expected degradation when Kafka is down).
func TestPublisherConstructAndReportWithoutBroker(t *testing.T) {
	cfg := &config.AccountEventsConfig{Enabled: true, Topic: "account-events"}
	kafkaCfg := &config.KafkaConfig{
		Enabled:          true,
		Brokers:          []string{"127.0.0.1:1"},
		ClientID:         "test",
		LogQueueSize:     100,
		LogBatchSize:     10,
		LogFlushInterval: 50 * time.Millisecond,
		LogWorkers:       1,
	}
	p, err := New(cfg, kafkaCfg, slog.Default())
	require.NoError(t, err, "kafka client construction must not fail without a broker")

	for i := 0; i < 3; i++ {
		p.Report("cred-1", "openai", "gpt-4o", 429, 45, "status 429")
	}
	p.Report("cred-2", "openai", "gpt-4o", 200, 0, "")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// With no broker the flush may hit the shutdown deadline — that is the
	// expected degradation, not a construction failure.
	_ = p.Shutdown(shutdownCtx)
	_ = bytes.MinRead
}

// Contract lock with the worker's consumer: these JSON keys are what the
// healthcheck-service model.Report decodes. Changing a tag here breaks the
// wire format both sides must agree on.
func TestEventJSONContract(t *testing.T) {
	ev := &Event{
		Credential:        "openai-prod-1",
		Provider:          "openai",
		Model:             "gpt-4o",
		StatusCode:        429,
		RetryAfterSeconds: 45,
		Error:             "status 429",
		EventID:           "ev-1",
		Timestamp:         1700000000,
	}
	raw, err := json.Marshal(ev)
	require.NoError(t, err)

	var decoded map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &decoded))
	for _, k := range []string{"credential", "provider", "model", "status_code",
		"retry_after_seconds", "error", "event_id", "timestamp"} {
		_, ok := decoded[k]
		assert.True(t, ok, "event field %q must be present", k)
	}
	assert.Equal(t, "429", string(decoded["status_code"]))
}
