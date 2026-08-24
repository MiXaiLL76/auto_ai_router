package proxy

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionStoreSetGetDelete(t *testing.T) {
	store := NewSessionStore(time.Minute)

	store.Set("session-1", "model-a", "cred-1")

	cred, ok := store.Get("session-1", "model-a")
	require.True(t, ok)
	assert.Equal(t, "cred-1", cred)

	store.Delete("session-1", "model-a")
	_, ok = store.Get("session-1", "model-a")
	assert.False(t, ok)
}

func TestSessionStoreExpiresEntries(t *testing.T) {
	store := NewSessionStore(10 * time.Millisecond)
	store.Set("session-1", "model-a", "cred-1")

	time.Sleep(20 * time.Millisecond)

	_, ok := store.Get("session-1", "model-a")
	assert.False(t, ok)
	assert.Equal(t, 0, store.Len())
}

func TestSessionStoreGetRefreshesExpiry(t *testing.T) {
	now := time.Date(2026, time.August, 24, 12, 0, 0, 0, time.UTC)
	store := NewSessionStore(10 * time.Minute)
	store.now = func() time.Time { return now }
	store.Set("session-1", "anthropic/claude-sonnet-4.6", "anthropic-cred-1")

	now = now.Add(9 * time.Minute)
	cred, ok := store.Get("session-1", "anthropic/claude-sonnet-4.6")
	require.True(t, ok)
	assert.Equal(t, "anthropic-cred-1", cred)

	now = now.Add(9 * time.Minute)
	cred, ok = store.Get("session-1", "anthropic/claude-sonnet-4.6")
	require.True(t, ok)
	assert.Equal(t, "anthropic-cred-1", cred)

	now = now.Add(11 * time.Minute)
	_, ok = store.Get("session-1", "anthropic/claude-sonnet-4.6")
	assert.False(t, ok)
}

func TestSessionStoreKeysAreIndependentPerModel(t *testing.T) {
	store := NewSessionStore(time.Minute)
	store.Set("session-1", "model-a", "cred-1")
	store.Set("session-1", "model-b", "cred-2")

	credA, okA := store.Get("session-1", "model-a")
	credB, okB := store.Get("session-1", "model-b")

	require.True(t, okA)
	require.True(t, okB)
	assert.Equal(t, "cred-1", credA)
	assert.Equal(t, "cred-2", credB)
	assert.Equal(t, 2, store.Len())
}

func TestSessionStoreStartCleanupRemovesExpiredEntries(t *testing.T) {
	store := NewSessionStore(10 * time.Millisecond)
	store.Set("session-1", "model-a", "cred-1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	go func() {
		store.StartCleanup(ctx, 5*time.Millisecond)
		close(done)
	}()

	time.Sleep(30 * time.Millisecond)
	assert.Eventually(t, func() bool {
		return store.Len() == 0
	}, 100*time.Millisecond, 5*time.Millisecond)

	cancel()
	assert.Eventually(t, func() bool {
		select {
		case <-done:
			return true
		default:
			return false
		}
	}, 100*time.Millisecond, 5*time.Millisecond)
}
