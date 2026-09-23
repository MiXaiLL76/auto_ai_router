package budget

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

const (
	hybridWriteBatchSize = 200
	hybridSyncBatchSize  = 1000
	hybridFlushInterval  = 100 * time.Millisecond
	// hybridMaxRetryBackoff caps the delay between flush attempts while Redis
	// keeps failing, so an outage doesn't turn into a 10/s warn-log loop.
	hybridMaxRetryBackoff = 5 * time.Second
	hybridRedisTimeout    = 10 * time.Second
)

// luaApplyDelta is the async counterpart of luaTryReserve's seeding step: it
// seeds the shared Redis counter (only when the key is absent) and applies
// delta, with no budget check — the allow/reject decision was already made
// locally, synchronously, before this ever runs.
//
// The write is idempotent per (instance, seq): KEYS[2] remembers the last seq
// this instance applied to KEYS[1], so a retry of a write whose reply was lost
// (timeout, broken connection after the server ran it) is a no-op instead of a
// double count. Returns 1 when applied, 0 when it was a duplicate.
const luaApplyDelta = `
local key = KEYS[1]
local seqkey = KEYS[2]
local ttl = tonumber(ARGV[3])
local seq = tonumber(ARGV[4])
local last = tonumber(redis.call('GET', seqkey) or '0')
if seq <= last then
  return 0
end
if redis.call('EXISTS', key) == 0 then
  redis.call('SET', key, ARGV[1])
end
redis.call('INCRBYFLOAT', key, ARGV[2])
redis.call('EXPIRE', key, ttl)
redis.call('SET', seqkey, ARGV[4], 'EX', ttl)
return 1
`

// pendingWrite is one delta sent (or about to be sent) to Redis. It is kept
// unchanged until Redis acknowledges it, so a retry re-sends the same seq and
// luaApplyDelta can drop it if the first attempt did land.
type pendingWrite struct {
	seq   uint64
	seed  float64 // value to seed the Redis key with if it's absent
	delta float64
}

// entityState is this instance's local view of one entity's budget counter.
//
// Invariant: localDelta == flushed + pending + inflight.delta.
type entityState struct {
	mu       sync.Mutex
	seeded   bool
	lastUsed time.Time
	// gen is bumped on every local reseed so in-flight Redis results and sync
	// reads taken against the previous seed are discarded.
	gen     uint64
	dbSpend float64
	// localDelta is the sum of TryReserve/Reconcile amounts this instance has
	// applied since the seed; it's what local decisions are made on.
	localDelta float64
	// flushed is the part of localDelta Redis has acknowledged.
	flushed float64
	// pending is the part of localDelta not yet handed to a write.
	pending float64
	// inflight is the write currently owed to Redis (sent or failed); it is
	// retried as-is until acknowledged.
	inflight *pendingWrite
	// repush asks the next flush to rewrite the Redis key even with nothing
	// pending — set when a sync finds the key gone while we have spend in it.
	repush bool
	// remoteOther is the estimated spend of OTHER instances:
	// redis_total - (dbSpend + flushed) as of the last sync.
	remoteOther float64
	// evicted marks a state removed from HybridBackend.states; a caller that
	// grabbed it just before eviction must look the entity up again.
	evicted bool
}

func (s *entityState) reseed(dbSpend float64) {
	s.gen++
	s.seeded = true
	s.dbSpend = dbSpend
	s.localDelta = 0
	s.flushed = 0
	s.pending = 0
	s.inflight = nil
	s.repush = false
	s.remoteOther = 0
}

// dirtyLocked reports whether the state still owes Redis a write. s.mu held.
func (s *entityState) dirtyLocked() bool {
	return s.inflight != nil || s.pending != 0 || s.repush
}

// HybridBackend makes budget-reservation decisions against local in-memory
// state (zero added latency, exact for concurrent requests on this instance —
// the per-entity mutex serializes them) and keeps a shared Redis counter
// updated asynchronously so every instance can estimate the others'
// contribution. This mirrors internal/ratelimit's HybridBackend and the same
// trade-off: overspend across the whole fleet is bounded by how much other
// instances reserve within one syncInterval, not eliminated outright — but
// that window is far narrower than the DB-snapshot check it replaces, and
// costs no synchronous Redis round trip per request.
//
// Deltas are never dropped: they accumulate per entity until Redis
// acknowledges them, and failed writes are retried (idempotently, see
// luaApplyDelta) with backoff. Entities idle for longer than ttl are evicted,
// so memory and per-sync Redis traffic track the active set, not every
// entity ever seen.
type HybridBackend struct {
	client       valkey.Client
	keyPrefix    string
	ttl          time.Duration
	ttlSeconds   string
	syncInterval time.Duration
	logger       *slog.Logger
	metrics      *monitoring.Metrics

	// instanceID namespaces this process's write sequence numbers in Redis.
	instanceID string
	seq        atomic.Uint64

	statesMu sync.Mutex
	states   map[string]*entityState
	dirty    map[string]struct{}

	// workMu serializes flush/sync/evict (normally all on the worker
	// goroutine; tests drive them directly). Fields below are guarded by it.
	workMu     sync.Mutex
	failStreak int
	retryAt    time.Time

	stopCh    chan struct{}
	closeOnce sync.Once
	wg        sync.WaitGroup
}

// NewHybridBackend wraps a Redis/Valkey client with a local in-process
// counter for budget reservations. Hot-path decisions run entirely on the
// local backend; Redis is updated asynchronously. syncInterval controls how
// often the shared counter is pulled to refresh the other-instances estimate
// (default 5s when zero).
func NewHybridBackend(client valkey.Client, keyPrefix string, ttl, syncInterval time.Duration, logger *slog.Logger, metrics *monitoring.Metrics) *HybridBackend {
	if ttl <= 0 {
		ttl = 15 * time.Minute
	}
	if syncInterval <= 0 {
		syncInterval = 5 * time.Second
	}
	if logger == nil {
		logger = slog.Default()
	}
	// EXPIRE/EX take whole seconds and 0 would delete the key outright.
	ttlSeconds := int64(ttl.Seconds())
	if ttlSeconds < 1 {
		ttlSeconds = 1
	}
	h := &HybridBackend{
		client:       client,
		keyPrefix:    keyPrefix,
		ttl:          ttl,
		ttlSeconds:   strconv.FormatInt(ttlSeconds, 10),
		syncInterval: syncInterval,
		logger:       logger,
		metrics:      metrics,
		instanceID:   newInstanceID(),
		states:       make(map[string]*entityState),
		dirty:        make(map[string]struct{}),
		stopCh:       make(chan struct{}),
	}
	h.wg.Add(1)
	go h.worker()
	return h
}

func newInstanceID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand does not fail on supported platforms; fall back to a
		// clock-derived id rather than refusing to start.
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b[:])
}

// Close stops the background worker after a final flush attempt. Call during
// server shutdown, before closing the Redis client.
func (h *HybridBackend) Close() {
	h.closeOnce.Do(func() {
		close(h.stopCh)
		h.wg.Wait()
	})
}

func (h *HybridBackend) recordRedisError(operation string, err error) {
	// Warn, not Error: a failed background write/sync is retried, so on its own
	// it's a lag in cross-instance visibility, not an operator emergency.
	h.logger.Warn("Budget hybrid backend: Redis operation failed", "operation", operation, "error", err)
	if h.metrics != nil {
		h.metrics.RecordRedisConnectionError(operation)
	}
}

// key wraps the entity in a hash tag so the counter and its per-instance seq
// keys share a cluster slot (luaApplyDelta touches both).
func (h *HybridBackend) key(entity string) string { return h.keyPrefix + "{" + entity + "}" }

func (h *HybridBackend) seqKey(entity string) string {
	return h.key(entity) + ":w:" + h.instanceID
}

// testHookLockEntity, when set by tests, runs between lockEntity's map
// lookup and its s.mu.Lock — the window an eviction can race into.
var testHookLockEntity func(entity string)

// lockEntity returns entity's live state with s.mu held, creating it if
// needed. It retries when it races with eviction so callers never write into
// a state that's no longer reachable from h.states (that write would be lost).
func (h *HybridBackend) lockEntity(entity string) *entityState {
	for {
		h.statesMu.Lock()
		s := h.states[entity]
		if s == nil {
			s = &entityState{}
			h.states[entity] = s
		}
		h.statesMu.Unlock()

		if testHookLockEntity != nil {
			testHookLockEntity(entity)
		}
		s.mu.Lock()
		if !s.evicted {
			return s
		}
		s.mu.Unlock()
	}
}

func (h *HybridBackend) lookup(entity string) *entityState {
	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	return h.states[entity]
}

func (h *HybridBackend) markDirty(entity string) {
	h.statesMu.Lock()
	h.dirty[entity] = struct{}{}
	h.statesMu.Unlock()
}

func (h *HybridBackend) takeDirty() []string {
	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	if len(h.dirty) == 0 {
		return nil
	}
	out := make([]string, 0, len(h.dirty))
	for e := range h.dirty {
		out = append(out, e)
	}
	clear(h.dirty)
	return out
}

func (h *HybridBackend) snapshotStates() map[string]*entityState {
	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	out := make(map[string]*entityState, len(h.states))
	for e, s := range h.states {
		out[e] = s
	}
	return out
}

// --- reserveBackend implementation ---

func (h *HybridBackend) tryReserve(_ context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error) {
	s := h.lockEntity(entity)
	now := time.Now()
	if !s.seeded || now.Sub(s.lastUsed) > h.ttl {
		s.reseed(dbSpend)
	}
	s.lastUsed = now
	total := s.dbSpend + s.localDelta + s.remoteOther + estimatedCost
	if maxBudget >= 0 && total > maxBudget {
		s.mu.Unlock()
		return false, nil
	}
	s.localDelta += estimatedCost
	s.pending += estimatedCost
	s.mu.Unlock()

	h.markDirty(entity)
	return true, nil
}

func (h *HybridBackend) reconcile(_ context.Context, entity string, delta float64) error {
	if delta == 0 {
		return nil
	}
	s := h.lockEntity(entity)
	if !s.seeded {
		// No prior tryReserve seeded this entity on this instance (e.g. the
		// process restarted between reserve and reconcile) — seed at 0 so the
		// delta still lands; the next TTL-driven reseed pulls the true DB spend.
		s.reseed(0)
	}
	s.lastUsed = time.Now()
	s.localDelta += delta
	s.pending += delta
	s.mu.Unlock()

	h.markDirty(entity)
	return nil
}

// --- background worker ---

func (h *HybridBackend) worker() {
	defer h.wg.Done()
	flushTicker := time.NewTicker(hybridFlushInterval)
	defer flushTicker.Stop()
	syncTicker := time.NewTicker(h.syncInterval)
	defer syncTicker.Stop()

	for {
		select {
		case <-h.stopCh:
			h.shutdownFlush()
			return
		case <-flushTicker.C:
			h.flush(false)
		case <-syncTicker.C:
			h.flush(false)
			h.doSync()
			h.evictIdle(time.Now())
		}
	}
}

func (h *HybridBackend) shutdownFlush() {
	h.flush(true)
	var entities int
	var amount float64
	for _, s := range h.snapshotStates() {
		s.mu.Lock()
		if s.dirtyLocked() {
			entities++
			amount += s.pending
			if s.inflight != nil {
				amount += s.inflight.delta
			}
		}
		s.mu.Unlock()
	}
	if entities > 0 {
		h.logger.Warn("Budget hybrid backend: unflushed budget deltas dropped at shutdown",
			"entities", entities, "amount", amount)
	}
}

type flushJob struct {
	entity string
	s      *entityState
	w      *pendingWrite
	gen    uint64
}

func (h *HybridBackend) applyDeltaCmd(entity string, w *pendingWrite) valkey.Completed {
	return h.client.B().Eval().
		Script(luaApplyDelta).Numkeys(2).
		Key(h.key(entity), h.seqKey(entity)).
		Arg(strconv.FormatFloat(w.seed, 'f', -1, 64)).
		Arg(strconv.FormatFloat(w.delta, 'f', -1, 64)).
		Arg(h.ttlSeconds).
		Arg(strconv.FormatUint(w.seq, 10)).
		Build()
}

// flush sends every dirty entity's owed write to Redis. Successful writes are
// folded into flushed; failed ones stay as inflight and the entity stays
// dirty, so the next flush retries the same (seq-deduplicated) write. force
// ignores the failure backoff (used at shutdown).
func (h *HybridBackend) flush(force bool) {
	h.workMu.Lock()
	defer h.workMu.Unlock()
	if !force && time.Now().Before(h.retryAt) {
		return
	}

	entities := h.takeDirty()
	if len(entities) == 0 {
		return
	}
	jobs := make([]flushJob, 0, len(entities))
	for _, e := range entities {
		s := h.lookup(e)
		if s == nil {
			continue
		}
		s.mu.Lock()
		if !s.evicted {
			if s.inflight == nil && (s.pending != 0 || s.repush) {
				s.inflight = &pendingWrite{
					seq:   h.seq.Add(1),
					seed:  s.dbSpend + s.flushed,
					delta: s.pending,
				}
				s.pending = 0
				s.repush = false
			}
			if s.inflight != nil {
				jobs = append(jobs, flushJob{entity: e, s: s, w: s.inflight, gen: s.gen})
			}
		}
		s.mu.Unlock()
	}

	var firstErr error
	for start := 0; start < len(jobs); start += hybridWriteBatchSize {
		chunk := jobs[start:min(start+hybridWriteBatchSize, len(jobs))]
		cmds := make([]valkey.Completed, len(chunk))
		for i, j := range chunk {
			cmds[i] = h.applyDeltaCmd(j.entity, j.w)
		}
		ctx, cancel := context.WithTimeout(context.Background(), hybridRedisTimeout)
		results := h.client.DoMulti(ctx, cmds...)
		cancel()

		for i, j := range chunk {
			err := results[i].Error()
			j.s.mu.Lock()
			// A reseed in the meantime replaced the accounting this write
			// belonged to; its result no longer means anything locally.
			if err == nil && j.s.gen == j.gen && j.s.inflight == j.w {
				j.s.flushed += j.w.delta
				j.s.inflight = nil
			}
			stillDirty := !j.s.evicted && j.s.dirtyLocked()
			j.s.mu.Unlock()
			if err != nil && firstErr == nil {
				firstErr = err
			}
			if stillDirty {
				h.markDirty(j.entity)
			}
		}
	}
	h.noteFlushResultLocked(firstErr)
}

// noteFlushResultLocked tracks consecutive flush failures and sets an
// exponential retry backoff. h.workMu held.
func (h *HybridBackend) noteFlushResultLocked(err error) {
	if err == nil {
		if h.failStreak > 0 {
			h.logger.Info("Budget hybrid backend: Redis writes recovered", "failed_attempts", h.failStreak)
		}
		h.failStreak = 0
		h.retryAt = time.Time{}
		return
	}
	h.failStreak++
	h.recordRedisError("budget_hybrid_write", err)
	backoff := hybridFlushInterval << min(h.failStreak, 6)
	if backoff > hybridMaxRetryBackoff {
		backoff = hybridMaxRetryBackoff
	}
	h.retryAt = time.Now().Add(backoff)
}

// doSync refreshes each tracked entity's estimate of OTHER instances' spend:
// redis_total - (dbSpend + flushed). Only acknowledged writes are subtracted,
// so deltas still pending locally aren't mistaken for another instance's
// absence. A key that vanished (Redis restart/eviction) while this instance
// has spend in it is scheduled for a repush so the shared counter recovers.
func (h *HybridBackend) doSync() {
	h.workMu.Lock()
	defer h.workMu.Unlock()

	states := h.snapshotStates()
	if len(states) == 0 {
		return
	}
	type syncJob struct {
		entity string
		s      *entityState
		gen    uint64
	}
	jobs := make([]syncJob, 0, len(states))
	for e, s := range states {
		s.mu.Lock()
		jobs = append(jobs, syncJob{entity: e, s: s, gen: s.gen})
		s.mu.Unlock()
	}

	var firstErr error
	for start := 0; start < len(jobs); start += hybridSyncBatchSize {
		chunk := jobs[start:min(start+hybridSyncBatchSize, len(jobs))]
		cmds := make([]valkey.Completed, len(chunk))
		for i, j := range chunk {
			cmds[i] = h.client.B().Get().Key(h.key(j.entity)).Build()
		}
		ctx, cancel := context.WithTimeout(context.Background(), hybridRedisTimeout)
		results := h.client.DoMulti(ctx, cmds...)
		cancel()

		for i, j := range chunk {
			redisTotal, err := results[i].AsFloat64()
			missing := false
			if err != nil {
				if !valkey.IsValkeyNil(err) {
					if firstErr == nil {
						firstErr = err
					}
					continue
				}
				missing = true
			}

			j.s.mu.Lock()
			if j.s.evicted || j.s.gen != j.gen {
				j.s.mu.Unlock()
				continue
			}
			own := j.s.dbSpend + j.s.flushed
			if missing {
				redisTotal = 0
				if own != 0 && j.s.inflight == nil && !j.s.repush {
					j.s.repush = true
				}
			}
			remoteOther := redisTotal - own
			if remoteOther < 0 {
				remoteOther = 0
			}
			j.s.remoteOther = remoteOther
			repush := j.s.repush
			j.s.mu.Unlock()
			if repush {
				h.markDirty(j.entity)
			}
		}
	}
	if firstErr != nil {
		h.recordRedisError("budget_hybrid_sync", firstErr)
	}
}

// evictIdle drops entities idle for longer than ttl that owe Redis nothing.
// Such an entity would be reseeded from the DB on its next use anyway, and its
// Redis key has expired (or is about to), so keeping it only costs memory and
// a GET per sync.
func (h *HybridBackend) evictIdle(now time.Time) {
	h.workMu.Lock()
	defer h.workMu.Unlock()

	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	for e, s := range h.states {
		s.mu.Lock()
		if now.Sub(s.lastUsed) > h.ttl && !s.dirtyLocked() {
			s.evicted = true
			delete(h.states, e)
			delete(h.dirty, e)
		}
		s.mu.Unlock()
	}
}
