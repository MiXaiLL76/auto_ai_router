package proxy

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/accountevents"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/require"
)

// With the kafka publisher active, reportHealth must queue events and
// recordBanSignal must skip the local fail2ban counters entirely (the
// worker owns the decisions). No broker is needed: delivery failures land
// in the producer's retry/DLQ machinery, not in the request path.
func TestKafkaPathReportAndBanSignal(t *testing.T) {
	pub, err := accountevents.New(
		&config.AccountEventsConfig{Enabled: true, Topic: "account-events"},
		&config.KafkaConfig{
			Brokers:          []string{"127.0.0.1:1"},
			ClientID:         "test",
			LogQueueSize:     32,
			LogBatchSize:     4,
			LogFlushInterval: 30 * time.Millisecond,
			LogWorkers:       1,
		},
		slog.Default(),
	)
	require.NoError(t, err)

	// balancer is intentionally nil: with the external source wired, neither
	// call should touch it.
	prx := &Proxy{events: pub, banDecisionsExternal: true}

	prx.reportHealth("openai", "cred-1", "gpt-4o", 429, 45) // queues event
	prx.recordBanSignal(&config.CredentialConfig{Name: "cred-1"}, "gpt-4o", 429)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = pub.Shutdown(ctx)
}
