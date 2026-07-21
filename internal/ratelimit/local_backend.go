package ratelimit

import (
	"context"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// slidingWindowBuckets is the number of one-second buckets used to approximate
// the 60-second RPM/TPM sliding window. Using fixed-size per-second counters
// instead of a per-request timestamp log turns cleanup from O(requests in the
// last minute) into O(1) amortized (bounded by this constant), independent of
// how many requests actually happened.
const slidingWindowBuckets = 60

// localCounter holds the in-process sliding-window state for one rate-limit key.
// requests/tokens are tracked as circular buffers of per-second counters rather
// than individual timestamps: bucket i holds the count for the epoch second
// stored in the matching *Seconds slot, and is lazily reset once that second
// falls out of the window. *Total always mirrors the sum of live buckets, so
// reads never need to rescan the buffers.
type localCounter struct {
	mu sync.Mutex

	reqCounts  [slidingWindowBuckets]int64
	reqSeconds [slidingWindowBuckets]int64
	reqTotal   int64
	reqLastSec int64

	tokCounts  [slidingWindowBuckets]int64
	tokSeconds [slidingWindowBuckets]int64
	tokTotal   int64
	tokLastSec int64
}

// localBackend is the in-process counterBackend implementation.
// It replicates the original RPMLimiter sliding-window logic.
type localBackend struct {
	mu       sync.RWMutex
	counters map[string]*localCounter
}

func newLocalBackend() *localBackend {
	return &localBackend{
		counters: make(map[string]*localCounter),
	}
}

// getOrCreate returns the counter for key, creating it lazily if necessary.
func (b *localBackend) getOrCreate(key string) *localCounter {
	b.mu.RLock()
	c := b.counters[key]
	b.mu.RUnlock()
	if c != nil {
		return c
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	// Double-check after acquiring write lock.
	if c = b.counters[key]; c != nil {
		return c
	}
	c = &localCounter{}
	b.counters[key] = c
	return c
}

// evictBuckets advances the window to nowSec, zeroing out any buckets whose
// second has fallen out of the trailing slidingWindowBuckets-second range and
// subtracting their contribution from total. The loop is bounded by
// slidingWindowBuckets regardless of request volume, so this is O(1) w.r.t.
// the number of requests recorded (unlike a full rescan of a timestamp log).
func evictBuckets(counts, seconds *[slidingWindowBuckets]int64, total, lastSec *int64, nowSec int64) {
	last := *lastSec
	if last == 0 {
		*lastSec = nowSec
		return
	}
	if nowSec <= last {
		return
	}
	if nowSec-last >= slidingWindowBuckets {
		*counts = [slidingWindowBuckets]int64{}
		*seconds = [slidingWindowBuckets]int64{}
		*total = 0
	} else {
		for s := last + 1; s <= nowSec; s++ {
			idx := s % slidingWindowBuckets
			*total -= counts[idx]
			counts[idx] = 0
			seconds[idx] = s
		}
	}
	*lastSec = nowSec
}

// --- RPM helpers (must be called with c.mu held) ---

func localCleanOldRequests(c *localCounter) int {
	nowSec := utils.NowUTC().Unix()
	evictBuckets(&c.reqCounts, &c.reqSeconds, &c.reqTotal, &c.reqLastSec, nowSec)
	return int(c.reqTotal)
}

func localRecordRequest(c *localCounter) {
	nowSec := utils.NowUTC().Unix()
	evictBuckets(&c.reqCounts, &c.reqSeconds, &c.reqTotal, &c.reqLastSec, nowSec)
	idx := nowSec % slidingWindowBuckets
	if c.reqSeconds[idx] != nowSec {
		c.reqCounts[idx] = 0
		c.reqSeconds[idx] = nowSec
	}
	c.reqCounts[idx]++
	c.reqTotal++
}

func localCheckRPM(c *localCounter, limit int, record bool) bool {
	current := localCleanOldRequests(c)
	if limit != -1 && current >= limit {
		return false
	}
	if record {
		localRecordRequest(c)
	}
	return true
}

// --- TPM helpers (must be called with c.mu held) ---

func localCleanOldTokens(c *localCounter) int {
	nowSec := utils.NowUTC().Unix()
	evictBuckets(&c.tokCounts, &c.tokSeconds, &c.tokTotal, &c.tokLastSec, nowSec)
	return int(c.tokTotal)
}

func localRecordTokens(c *localCounter, tokenCount int) {
	nowSec := utils.NowUTC().Unix()
	evictBuckets(&c.tokCounts, &c.tokSeconds, &c.tokTotal, &c.tokLastSec, nowSec)
	idx := nowSec % slidingWindowBuckets
	if c.tokSeconds[idx] != nowSec {
		c.tokCounts[idx] = 0
		c.tokSeconds[idx] = nowSec
	}
	c.tokCounts[idx] += int64(tokenCount)
	c.tokTotal += int64(tokenCount)
}

func localCheckTPM(c *localCounter, limit int) bool {
	if limit == -1 {
		return true
	}
	return localCleanOldTokens(c) < limit
}

// distributeAcrossBuckets spreads total evenly across all slidingWindowBuckets
// one-second buckets ending at nowSec, so a synced remote usage value decays
// gradually over the window instead of expiring all at once.
func distributeAcrossBuckets(counts, seconds *[slidingWindowBuckets]int64, nowSec, total int64) {
	base := total / slidingWindowBuckets
	remainder := total % slidingWindowBuckets
	for i := int64(0); i < slidingWindowBuckets; i++ {
		sec := nowSec - slidingWindowBuckets + 1 + i
		idx := ((sec % slidingWindowBuckets) + slidingWindowBuckets) % slidingWindowBuckets
		count := base
		if i < remainder {
			count++
		}
		counts[idx] = count
		seconds[idx] = sec
	}
}

// --- counterBackend implementation ---

// Note: localBackend methods accept context but don't use it since
// in-memory operations are fast and not cancellable.

func (b *localBackend) tryAllowRPM(_ context.Context, key string, limit int) bool {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	return localCheckRPM(c, limit, true)
}

func (b *localBackend) canAllowRPM(_ context.Context, key string, limit int) bool {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	return localCheckRPM(c, limit, false)
}

func (b *localBackend) canAllowTPM(_ context.Context, key string, limit int) bool {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	return localCheckTPM(c, limit)
}

func (b *localBackend) consumeTokens(_ context.Context, key string, tokenCount int) {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	localRecordTokens(c, tokenCount)
}

func (b *localBackend) currentRPM(_ context.Context, key string) int {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	return localCleanOldRequests(c)
}

func (b *localBackend) currentTPM(_ context.Context, key string) int {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	return localCleanOldTokens(c)
}

func (b *localBackend) tryAllowAll(_ context.Context, credKey string, credRPM, credTPM int, modelKey string, modelRPM, modelTPM int) bool {
	cred := b.getOrCreate(credKey)

	var mod *localCounter
	if modelKey != "" {
		mod = b.getOrCreate(modelKey)
	}

	// Consistent lock ordering: always lock credKey before modelKey.
	cred.mu.Lock()
	defer cred.mu.Unlock()

	if !localCheckRPM(cred, credRPM, false) {
		return false
	}
	if !localCheckTPM(cred, credTPM) {
		return false
	}

	if mod != nil {
		mod.mu.Lock()
		defer mod.mu.Unlock()

		if !localCheckRPM(mod, modelRPM, false) {
			return false
		}
		if !localCheckTPM(mod, modelTPM) {
			return false
		}
	}

	// All checks passed — record.
	localRecordRequest(cred)
	if mod != nil {
		localRecordRequest(mod)
	}
	return true
}

func (b *localBackend) setCurrentUsage(_ context.Context, key string, currentRPM, currentTPM int) {
	c := b.getOrCreate(key)
	c.mu.Lock()
	defer c.mu.Unlock()

	nowSec := utils.NowUTC().Unix()

	c.reqCounts = [slidingWindowBuckets]int64{}
	c.reqSeconds = [slidingWindowBuckets]int64{}
	c.reqTotal = 0
	c.reqLastSec = nowSec
	if currentRPM > 0 {
		distributeAcrossBuckets(&c.reqCounts, &c.reqSeconds, nowSec, int64(currentRPM))
		c.reqTotal = int64(currentRPM)
	}

	c.tokCounts = [slidingWindowBuckets]int64{}
	c.tokSeconds = [slidingWindowBuckets]int64{}
	c.tokTotal = 0
	c.tokLastSec = nowSec
	if currentTPM > 0 {
		distributeAcrossBuckets(&c.tokCounts, &c.tokSeconds, nowSec, int64(currentTPM))
		c.tokTotal = int64(currentTPM)
	}
}

func (b *localBackend) deleteKey(_ context.Context, key string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.counters, key)
}

func (b *localBackend) batchCurrentStats(ctx context.Context, keys []string) map[string][2]int {
	out := make(map[string][2]int, len(keys))
	for _, key := range keys {
		out[key] = [2]int{b.currentRPM(ctx, key), b.currentTPM(ctx, key)}
	}
	return out
}
