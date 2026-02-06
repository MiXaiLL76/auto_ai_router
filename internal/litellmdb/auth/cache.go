package auth

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// cachedToken holds a cached token with timestamp
type cachedToken struct {
	info     *models.TokenInfo
	cachedAt time.Time
}

// Cache is an LRU cache for token authentication
// Thread-safe, uses hashicorp/golang-lru under the hood
type Cache struct {
	cache *lru.Cache[string, *cachedToken]
	ttl   time.Duration
	mu    sync.RWMutex

	// Metrics
	hits   uint64
	misses uint64
}

// NewCache creates a new token cache
func NewCache(maxSize int, ttl time.Duration) (*Cache, error) {
	if maxSize <= 0 {
		maxSize = 10000
	}
	if ttl <= 0 {
		ttl = 60 * time.Second
	}

	cache, err := lru.New[string, *cachedToken](maxSize)
	if err != nil {
		return nil, fmt.Errorf("litellmdb: failed to create auth cache: %w", err)
	}

	return &Cache{
		cache: cache,
		ttl:   ttl,
	}, nil
}

// Get retrieves a token from cache
// Returns nil, false if token not found or TTL expired
func (c *Cache) Get(hashedToken string) (*models.TokenInfo, bool) {
	c.mu.RLock()
	cached, ok := c.cache.Get(hashedToken)
	c.mu.RUnlock()

	if !ok {
		atomic.AddUint64(&c.misses, 1)
		return nil, false
	}

	// Check TTL
	if time.Since(cached.cachedAt) > c.ttl {
		// TTL expired - remove from cache
		c.mu.Lock()
		c.cache.Remove(hashedToken)
		c.mu.Unlock()
		atomic.AddUint64(&c.misses, 1)
		return nil, false
	}

	atomic.AddUint64(&c.hits, 1)
	return cached.info, true
}

// Set adds a token to cache
func (c *Cache) Set(hashedToken string, info *models.TokenInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Add(hashedToken, &cachedToken{
		info:     info,
		cachedAt: time.Now().UTC(),
	})
}

// Invalidate removes a token from cache
func (c *Cache) Invalidate(hashedToken string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache.Remove(hashedToken)
}

// InvalidateAll clears the entire cache
func (c *Cache) InvalidateAll() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache.Purge()
}

// Stats returns cache statistics
func (c *Cache) Stats() models.AuthCacheStats {
	c.mu.RLock()
	size := c.cache.Len()
	c.mu.RUnlock()

	hits := atomic.LoadUint64(&c.hits)
	misses := atomic.LoadUint64(&c.misses)
	total := hits + misses

	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100
	}

	return models.AuthCacheStats{
		Size:    size,
		Hits:    hits,
		Misses:  misses,
		HitRate: hitRate,
	}
}

// Len returns current cache size
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cache.Len()
}
