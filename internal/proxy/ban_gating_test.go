package proxy

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/accountevents"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The local fail2ban must stay armed unless an external ban source is truly
// wired — a misconfigured kafka publisher alone must never disarm it.
func TestBanSignalGatingRequiresExternalSource(t *testing.T) {
	f2b := fail2ban.New(3, 0, []int{429})
	rl := ratelimit.New()
	bal := balancer.New([]config.CredentialConfig{{Name: "c", APIKey: "k", BaseURL: "http://x"}}, f2b, rl)

	pub, err := accountevents.New(
		&config.AccountEventsConfig{Enabled: true, Topic: "t"},
		&config.KafkaConfig{
			Brokers:          []string{"127.0.0.1:1"},
			ClientID:         "test",
			LogQueueSize:     8,
			LogBatchSize:     2,
			LogFlushInterval: 30 * time.Millisecond,
			LogWorkers:       1,
		},
		slog.Default(),
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = pub.Shutdown(ctx)
	})

	// Publisher present, external ban source NOT wired: local counters feed.
	prx := &Proxy{events: pub, balancer: bal, banDecisionsExternal: false}
	prx.recordBanSignal(&config.CredentialConfig{Name: "c"}, "m", 429)
	assert.Equal(t, 1, f2b.GetFailureCount("c", "m"), "local fail2ban stays armed without an external source")

	// External source wired: local counters skipped.
	prx2 := &Proxy{balancer: bal, banDecisionsExternal: true}
	prx2.recordBanSignal(&config.CredentialConfig{Name: "c2"}, "m", 429)
	assert.Equal(t, 0, f2b.GetFailureCount("c2", "m"), "external source takes over the decisions")
}
