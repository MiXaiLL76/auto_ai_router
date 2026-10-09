// Integration test against a real Redis (VALKEY_ADDR), mirroring how the
// worker writes bans. Skipped when the env var is unset.
package banreader

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valkey-io/valkey-go"
)

type valkeyClient = valkey.Client

func readerClient(t *testing.T, prefix string) (valkeyClient, config.RedisConfig, bool) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		return nil, config.RedisConfig{}, false
	}
	cfg := config.RedisConfig{
		InitAddresses:   []string{addr},
		SelectDB:        9,
		KeyPrefix:       "rl-test:", // unrelated: the reader must ignore it
		HealthKeyPrefix: prefix,
		SyncInterval:    200 * time.Millisecond,
		KeyTTL:          3600,
	}
	client, err := ratelimit.NewValkeyClient(cfg)
	require.NoError(t, err)
	return client, cfg, true
}

func seedBan(t *testing.T, client valkeyClient, cfg config.RedisConfig, member string, until int64) {
	key := cfg.HealthKeyPrefix + "openai:bans"
	err := client.Do(context.Background(),
		client.B().Zadd().Key(key).ScoreMember().ScoreMember(float64(until), member).Build()).Error()
	require.NoError(t, err)
	_ = client.Do(context.Background(), client.B().Sadd().Key(cfg.HealthKeyPrefix+"providers").Member("openai").Build()).Error()
}

func seedBanDetails(t *testing.T, client valkeyClient, cfg config.RedisConfig, member string, code int64, reason string) {
	err := client.Do(context.Background(),
		client.B().Hset().Key(cfg.HealthKeyPrefix+"openai:ban:"+member).FieldValue().
			FieldValue("until", "0").
			FieldValue("code", fmt.Sprintf("%d", code)).
			FieldValue("reason", reason).
			FieldValue("origin", "fail2ban").
			Build()).Error()
	require.NoError(t, err)
}

func TestReaderMaterializesAndLiftsBans(t *testing.T) {
	client, cfg, ok := readerClient(t, "brt:")
	if !ok {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
		return
	}
	now := time.Now().UTC().Unix()
	seedBan(t, client, cfg, "cred1|gpt-4o", now+120)
	seedBanDetails(t, client, cfg, "cred1|gpt-4o", 429, "status 429")

	f2b := fail2ban.New(3, 0, []int{})
	r := New(client, cfg, f2b, []string{"openai"}, 200*time.Millisecond, slog.Default())
	r.sync(context.Background())

	assert.True(t, f2b.IsBanned("cred1", "gpt-4o"), "model ban materialized into fail2ban")
	assert.False(t, r.IsDead("cred1"), "model ban is not an account dead")

	// Remove the ban in Redis -> next sync lifts it locally.
	err := client.Do(context.Background(),
		client.B().Zrem().Key(cfg.HealthKeyPrefix+"openai:bans").Member("cred1|gpt-4o").Build()).Error()
	require.NoError(t, err)
	r.sync(context.Background())
	assert.False(t, f2b.IsBanned("cred1", "gpt-4o"), "vanished ban is lifted")

	// Wildcard ban kills the account for the balancer's IsDead path.
	seedBan(t, client, cfg, "cred2|*", now+300)
	seedBanDetails(t, client, cfg, "cred2|*", 401, "status 401")
	r.sync(context.Background())
	assert.True(t, r.IsDead("cred2"), "wildcard ban reported as dead account")
	cred, model := splitBanKey("cred1|gpt-4o")
	assert.Equal(t, "cred1", cred)
	assert.Equal(t, "gpt-4o", model)
}

// A Redis read failure must NOT lift applied bans: a partial snapshot is
// indistinguishable from "bans disappeared", so the reader aborts the sync
// and leaves fail2ban untouched (bans live until their own expiry).
func TestReaderKeepsBansOnRedisFailure(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
		return
	}
	cfg := config.RedisConfig{
		InitAddresses:   []string{addr},
		SelectDB:        9,
		HealthKeyPrefix: "brt3:",
		KeyTTL:          3600,
	}
	seedClient, err := ratelimit.NewValkeyClient(cfg)
	require.NoError(t, err)
	t.Cleanup(func() {
		ctx := context.Background()
		// cleanup via SCAN like the other tests
		_ = seedClient.Do(ctx, seedClient.B().Scan().Cursor(0).Match("brt3:*").Count(100).Build()).Error()
	})

	now := time.Now().UTC().Unix()
	seedBan(t, seedClient, cfg, "cred1|gpt-4o", now+120)
	seedBanDetails(t, seedClient, cfg, "cred1|gpt-4o", 429, "status 429")

	readerClient, err := ratelimit.NewValkeyClient(cfg)
	require.NoError(t, err)

	f2b := fail2ban.New(3, 0, []int{})
	r := New(readerClient, cfg, f2b, []string{"openai"}, 100*time.Millisecond, slog.Default())
	r.sync(context.Background())
	assert.True(t, f2b.IsBanned("cred1", "gpt-4o"), "precondition: ban materialized")

	// Simulate Redis going dark for the reader.
	readerClient.Close()
	r.sync(context.Background())

	assert.True(t, f2b.IsBanned("cred1", "gpt-4o"),
		"a failed snapshot read must not unban live bans")
}
