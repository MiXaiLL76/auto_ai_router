package auth

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCache_InvalidSize(t *testing.T) {
	// Test with 0 size - should default to 10000
	cache, err := NewCache(0, 60*time.Second)
	require.NoError(t, err)
	assert.NotNil(t, cache)
	assert.Equal(t, 0, cache.Len()) // Empty cache

	// Test with negative size - should default to 10000
	cache2, err := NewCache(-5, 60*time.Second)
	require.NoError(t, err)
	assert.NotNil(t, cache2)
}

func TestNewCache_InvalidTTL(t *testing.T) {
	// Test with 0 TTL - should default to 5s and not panic
	cache, err := NewCache(100, 0)
	require.NoError(t, err)
	assert.NotNil(t, cache)
	// TTL defaults to 5s, so tokens should be cached
	assert.Equal(t, 0, cache.Len())

	// Test with negative TTL - should default to 5s and not panic
	cache2, err := NewCache(100, -5*time.Second)
	require.NoError(t, err)
	assert.NotNil(t, cache2)
	// Should be able to use cache normally
	assert.Equal(t, 0, cache2.Len())
}

func TestCache_SetGet_Basic(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	token := &models.TokenInfo{
		KeyName: "test-key",
		Token:   "test-token-hash",
		UserID:  "user-123",
	}

	hashedToken := HashToken(token.Token)

	// Get before set should miss
	_, ok := cache.Get(hashedToken)
	assert.False(t, ok)

	// Set and verify hit
	cache.Set(hashedToken, token)
	retrieved, ok := cache.Get(hashedToken)
	assert.True(t, ok)
	assert.Equal(t, token.KeyName, retrieved.KeyName)
	assert.Equal(t, token.Token, retrieved.Token)
	assert.Equal(t, token.UserID, retrieved.UserID)
}

func TestCache_TTLExpiration(t *testing.T) {
	cache, err := NewCache(100, 100*time.Millisecond)
	require.NoError(t, err)

	token := &models.TokenInfo{KeyName: "test-key", Token: "test-hash"}
	hashedToken := HashToken("test-hash")

	// Set token
	cache.Set(hashedToken, token)
	retrieved, ok := cache.Get(hashedToken)
	assert.True(t, ok)
	assert.NotNil(t, retrieved)

	// Wait for TTL to expire
	time.Sleep(150 * time.Millisecond)

	// Get should return miss after TTL
	_, ok = cache.Get(hashedToken)
	assert.False(t, ok)

	// Verify token was removed from cache
	assert.Equal(t, 0, cache.Len())
}

func TestCache_Invalidate(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	token := &models.TokenInfo{KeyName: "test-key", Token: "test-hash"}
	hashedToken := HashToken("test-hash")

	cache.Set(hashedToken, token)
	assert.Equal(t, 1, cache.Len())

	// Invalidate
	cache.Invalidate(hashedToken)
	assert.Equal(t, 0, cache.Len())

	// Get should miss
	_, ok := cache.Get(hashedToken)
	assert.False(t, ok)
}

func TestCache_InvalidateAll(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	// Add multiple tokens
	for i := 0; i < 5; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		hashedToken := HashToken(token.Token)
		cache.Set(hashedToken, token)
	}

	assert.Equal(t, 5, cache.Len())

	// Invalidate all
	cache.InvalidateAll()
	assert.Equal(t, 0, cache.Len())
}

func TestCache_HitMissStats(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	token := &models.TokenInfo{KeyName: "test-key", Token: "test-hash"}
	hashedToken := HashToken("test-hash")

	// Miss
	cache.Get(hashedToken)
	stats := cache.Stats()
	assert.Equal(t, uint64(1), stats.Misses)
	assert.Equal(t, uint64(0), stats.Hits)

	// Hit
	cache.Set(hashedToken, token)
	cache.Get(hashedToken)
	stats = cache.Stats()
	assert.Equal(t, uint64(1), stats.Misses)
	assert.Equal(t, uint64(1), stats.Hits)
	assert.Equal(t, 50.0, stats.HitRate)
}

func TestCache_Stats_WithTTLMiss(t *testing.T) {
	cache, err := NewCache(100, 50*time.Millisecond)
	require.NoError(t, err)

	token := &models.TokenInfo{KeyName: "test-key", Token: "test-hash"}
	hashedToken := HashToken("test-hash")

	cache.Set(hashedToken, token)

	// Hit before expiration
	_, _ = cache.Get(hashedToken)

	// Wait for TTL
	time.Sleep(100 * time.Millisecond)

	// Miss after expiration
	_, _ = cache.Get(hashedToken)

	stats := cache.Stats()
	assert.Equal(t, uint64(1), stats.Hits)
	assert.Equal(t, uint64(1), stats.Misses)
	assert.Equal(t, 50.0, stats.HitRate)
}

func TestCache_Len(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	assert.Equal(t, 0, cache.Len())

	for i := 0; i < 5; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		hashedToken := HashToken(token.Token)
		cache.Set(hashedToken, token)
	}

	assert.Equal(t, 5, cache.Len())
}

func TestCache_ThreadSafety_ConcurrentSetGet(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	var wg sync.WaitGroup
	tokenCount := 50
	goroutineCount := 10

	// Spawn goroutines that concurrently set and get
	for g := 0; g < goroutineCount; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for i := 0; i < tokenCount; i++ {
				idx := goroutineID*100 + i
				token := &models.TokenInfo{
					KeyName: "key-" + fmt.Sprint(idx),
					Token:   "token-hash-" + fmt.Sprint(idx),
				}
				hashedToken := HashToken(token.Token)
				cache.Set(hashedToken, token)

				// Immediately get
				retrieved, ok := cache.Get(hashedToken)
				assert.True(t, ok)
				assert.Equal(t, token.KeyName, retrieved.KeyName)
			}
		}(g)
	}

	wg.Wait()

	// Verify all entries are in cache
	assert.Greater(t, cache.Len(), 0)
}

func TestCache_ThreadSafety_ConcurrentInvalidate(t *testing.T) {
	cache, err := NewCache(1000, 60*time.Second)
	require.NoError(t, err)

	// Pre-populate cache
	for i := 0; i < 100; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		hashedToken := HashToken(token.Token)
		cache.Set(hashedToken, token)
	}

	var wg sync.WaitGroup
	goroutineCount := 10

	// Spawn goroutines that concurrently invalidate
	for g := 0; g < goroutineCount; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for i := 0; i < 10; i++ {
				idx := goroutineID*10 + i
				hashedToken := HashToken("token-hash-" + fmt.Sprint(idx))
				cache.Invalidate(hashedToken)
			}
		}(g)
	}

	wg.Wait()

	// Most cache should be empty or partial
	assert.Less(t, cache.Len(), 100)
}

func TestCache_Eviction_AtMaxSize(t *testing.T) {
	cache, err := NewCache(5, 60*time.Second)
	require.NoError(t, err)

	// Add 5 tokens (at max)
	for i := 0; i < 5; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		hashedToken := HashToken(token.Token)
		cache.Set(hashedToken, token)
	}

	assert.Equal(t, 5, cache.Len())

	// Add 6th token - should trigger eviction of oldest
	token6 := &models.TokenInfo{
		KeyName: "key-6",
		Token:   "token-hash-6",
	}
	hashedToken6 := HashToken("token-hash-6")
	cache.Set(hashedToken6, token6)

	// LRU should have evicted one item
	assert.LessOrEqual(t, cache.Len(), 5)
}

func TestCache_Stats_Empty(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	stats := cache.Stats()
	assert.Equal(t, 0, stats.Size)
	assert.Equal(t, uint64(0), stats.Hits)
	assert.Equal(t, uint64(0), stats.Misses)
	// Empty cache has 0.0 hit rate (or undefined) - 0 total accesses
	assert.Equal(t, 0.0, stats.HitRate)
}

func TestCache_NilCache_SafeOperations(t *testing.T) {
	var cache *Cache

	// All operations should not panic
	token, ok := cache.Get("test")
	assert.False(t, ok)
	assert.Nil(t, token)

	cache.Set("test", &models.TokenInfo{KeyName: "test"})
	cache.Invalidate("test")
	cache.InvalidateAll()

	len := cache.Len()
	assert.Equal(t, 0, len)

	stats := cache.Stats()
	assert.Equal(t, 0, stats.Size)
}

func TestCache_AtomicMetrics(t *testing.T) {
	cache, err := NewCache(100, 60*time.Second)
	require.NoError(t, err)

	var wg sync.WaitGroup
	iterations := 100

	// Concurrent reads/writes to metrics
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				token := &models.TokenInfo{
					KeyName: "test-key",
					Token:   "test-hash",
				}
				cache.Set("test-hash", token)
				cache.Get("test-hash")
			}
		}()
	}

	wg.Wait()

	stats := cache.Stats()
	// Should have at least some hits (exact count depends on timing)
	assert.Greater(t, stats.Hits+stats.Misses, uint64(0))
}

func TestCache_HashToken(t *testing.T) {
	// Test hash consistency
	token := "sk-test-token-123"
	hash1 := HashToken(token)
	hash2 := HashToken(token)
	assert.Equal(t, hash1, hash2)

	// Different tokens should produce different hashes
	hash3 := HashToken("sk-different-token")
	assert.NotEqual(t, hash1, hash3)
}

func TestCache_PerformanceMetrics(t *testing.T) {
	cache, err := NewCache(10000, 60*time.Second)
	require.NoError(t, err)

	// Add many tokens
	tokenCount := 5000
	for i := 0; i < tokenCount; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		hashedToken := HashToken(token.Token)
		cache.Set(hashedToken, token)
	}

	assert.Equal(t, tokenCount, cache.Len())

	// Verify all can be retrieved
	for i := 0; i < tokenCount; i++ {
		hashedToken := HashToken("token-hash-" + fmt.Sprint(i))
		_, ok := cache.Get(hashedToken)
		assert.True(t, ok)
	}

	stats := cache.Stats()
	assert.Equal(t, tokenCount, stats.Size)
	assert.Equal(t, uint64(tokenCount), stats.Hits)
	assert.Equal(t, uint64(0), stats.Misses)
	assert.Equal(t, 100.0, stats.HitRate)
}

func TestCache_MixedOperations(t *testing.T) {
	cache, err := NewCache(1000, 60*time.Second)
	require.NoError(t, err)

	// Set 10 tokens
	for i := 0; i < 10; i++ {
		token := &models.TokenInfo{
			KeyName: "key-" + fmt.Sprint(i),
			Token:   "token-hash-" + fmt.Sprint(i),
		}
		cache.Set(HashToken(token.Token), token)
	}

	// Get all (hits)
	for i := 0; i < 10; i++ {
		cache.Get(HashToken("token-hash-" + fmt.Sprint(i)))
	}

	// Try to get non-existent (misses)
	cache.Get(HashToken("non-existent-1"))
	cache.Get(HashToken("non-existent-2"))

	// Invalidate some
	cache.Invalidate(HashToken("token-hash-1"))
	cache.Invalidate(HashToken("token-hash-2"))

	// Verify stats
	stats := cache.Stats()
	assert.Equal(t, 8, stats.Size) // 10 - 2 invalidated
	assert.Equal(t, uint64(10), stats.Hits)
	assert.Equal(t, uint64(2), stats.Misses)

	// 10 hits out of 12 total = 83.33%
	expectedHitRate := 10.0 / 12.0 * 100
	assert.InDelta(t, expectedHitRate, stats.HitRate, 0.1)
}

func TestCache_Get_ExpiredEntry_DoesNotEvictFreshEntry(t *testing.T) {
	cache, err := NewCache(100, 50*time.Millisecond)
	require.NoError(t, err)

	hashedToken := HashToken("test-token")

	// Set initial entry
	staleToken := &models.TokenInfo{KeyName: "stale-key", Token: "test-token"}
	cache.Set(hashedToken, staleToken)

	// Verify it's cached
	retrieved, ok := cache.Get(hashedToken)
	require.True(t, ok)
	assert.Equal(t, "stale-key", retrieved.KeyName)

	// Wait for TTL to expire
	time.Sleep(80 * time.Millisecond)

	// Now simulate the race: another goroutine refreshes the entry
	// right before/during our expired Get.
	freshToken := &models.TokenInfo{KeyName: "fresh-key", Token: "test-token"}

	var wg sync.WaitGroup

	// Goroutine 1: Set a fresh entry (simulates DB refresh)
	wg.Add(1)
	go func() {
		defer wg.Done()
		cache.Set(hashedToken, freshToken)
	}()

	// Goroutine 2: Get the expired entry (should NOT evict the fresh one)
	wg.Add(1)
	go func() {
		defer wg.Done()
		cache.Get(hashedToken)
	}()

	wg.Wait()

	// After both goroutines complete, the fresh entry must survive.
	// The Get() with expired TTL should re-check under write lock
	// and NOT remove the entry if it was refreshed by Set().
	retrieved, ok = cache.Get(hashedToken)
	assert.True(t, ok, "fresh entry should still be in cache")
	assert.Equal(t, "fresh-key", retrieved.KeyName, "fresh entry should not have been evicted by stale TTL check")
}

// Compile-time check that HashToken function exists and works
func TestHashToken_Signature(t *testing.T) {
	token := "sk-test-token"
	hash := HashToken(token)
	assert.NotEmpty(t, hash)
	assert.Greater(t, len(hash), 0)
}
