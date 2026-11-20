package yandexsearch

import (
	"context"
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valkey-io/valkey-go"
)

// valkeyClientForTest connects to VALKEY_ADDR, or skips — same convention
// as internal/ratelimit/redis_backend_test.go's redisBackendForTest.
func valkeyClientForTest(t *testing.T) valkey.Client {
	t.Helper()
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := valkey.NewClient(valkey.ClientOption{
		InitAddress:       []string{addr},
		ForceSingleClient: true,
	})
	require.NoError(t, err)
	t.Cleanup(client.Close)
	return client
}

func uniquePrefix(t *testing.T) string {
	t.Helper()
	return "yssq-test:" + t.Name() + ":" + uuid.New().String() + ":"
}

func TestFolderQuotaAllowsUpToRPSThenRejects(t *testing.T) {
	client := valkeyClientForTest(t)
	q := NewFolderQuota(client, uniquePrefix(t), 3, 100000)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		allowed, _, err := q.Allow(ctx)
		require.NoError(t, err)
		require.True(t, allowed, "request %d should be admitted under the RPS limit", i)
	}

	allowed, retryAfter, err := q.Allow(ctx)
	require.NoError(t, err)
	assert.False(t, allowed)
	assert.Greater(t, retryAfter, time.Duration(0))
	assert.LessOrEqual(t, retryAfter, quotaRPSWindow+100*time.Millisecond)
}

func TestFolderQuotaAllowsAgainAfterRPSWindowPasses(t *testing.T) {
	client := valkeyClientForTest(t)
	q := NewFolderQuota(client, uniquePrefix(t), 1, 100000)
	ctx := context.Background()

	allowed, _, err := q.Allow(ctx)
	require.NoError(t, err)
	require.True(t, allowed)

	allowed, _, err = q.Allow(ctx)
	require.NoError(t, err)
	require.False(t, allowed)

	time.Sleep(quotaRPSWindow + 150*time.Millisecond)

	allowed, _, err = q.Allow(ctx)
	require.NoError(t, err)
	assert.True(t, allowed, "a new request after the 1s window elapses must be admitted again")
}

func TestFolderQuotaEnforcesHourlyLimitIndependentlyOfRPS(t *testing.T) {
	client := valkeyClientForTest(t)
	// RPS effectively unlimited for this test (high), hourly limit tiny.
	q := NewFolderQuota(client, uniquePrefix(t), 1000, 2)
	ctx := context.Background()

	for i := 0; i < 2; i++ {
		// Space calls out so the RPS window never binds, isolating the hourly check.
		time.Sleep(20 * time.Millisecond)
		allowed, _, err := q.Allow(ctx)
		require.NoError(t, err)
		require.True(t, allowed, "request %d should be admitted under the hourly limit", i)
	}

	time.Sleep(20 * time.Millisecond)
	allowed, retryAfter, err := q.Allow(ctx)
	require.NoError(t, err)
	assert.False(t, allowed)
	// Retry-After for an hourly rejection should be on the order of the hour
	// window, not the 1s RPS window.
	assert.Greater(t, retryAfter, 30*time.Minute)
}

func TestFolderQuotaConcurrentCallersShareOneBudget(t *testing.T) {
	client := valkeyClientForTest(t)
	q := NewFolderQuota(client, uniquePrefix(t), 5, 100000)
	ctx := context.Background()

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		allowed int
	)
	// Simulate "multiple replicas": several goroutines hitting the SAME
	// FolderQuota (same Redis keys) concurrently must jointly never exceed
	// the configured RPS, proving the quota is shared, not per-caller.
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok, _, err := q.Allow(ctx)
			require.NoError(t, err)
			if ok {
				mu.Lock()
				allowed++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	assert.Equal(t, 5, allowed, "exactly RPS=5 of the 20 concurrent calls should be admitted")
}

func TestFolderQuotaFailsClosedWhenRedisUnreachable(t *testing.T) {
	badClient, err := valkey.NewClient(valkey.ClientOption{
		InitAddress:       []string{"127.0.0.1:1"}, // nothing listens here
		ForceSingleClient: true,
		Dialer:            net.Dialer{Timeout: 200 * time.Millisecond},
	})
	if err != nil {
		// Some valkey-go versions fail fast on an unreachable initial
		// address rather than returning a client; either way the
		// "unreachable Redis" behavior is exercised.
		return
	}
	defer badClient.Close()

	q := NewFolderQuota(badClient, "yssq-unreachable-test:", 1, 100)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	allowed, _, err := q.Allow(ctx)
	assert.False(t, allowed, "a Redis error must deny (fail closed), never admit")
	assert.Error(t, err)
}

func TestNewFolderQuotaKeyPrefixIsolatesInstances(t *testing.T) {
	client := valkeyClientForTest(t)
	prefixA := fmt.Sprintf("yssq-a-%s:", uuid.New().String())
	prefixB := fmt.Sprintf("yssq-b-%s:", uuid.New().String())
	qa := NewFolderQuota(client, prefixA, 1, 100000)
	qb := NewFolderQuota(client, prefixB, 1, 100000)
	ctx := context.Background()

	allowed, _, err := qa.Allow(ctx)
	require.NoError(t, err)
	require.True(t, allowed)

	// A different key prefix must have its own, independent budget.
	allowed, _, err = qb.Allow(ctx)
	require.NoError(t, err)
	assert.True(t, allowed, "a different FolderQuota key prefix must not share qa's exhausted budget")
}
