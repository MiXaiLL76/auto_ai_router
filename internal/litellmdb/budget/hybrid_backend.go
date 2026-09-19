package budget

import (
	"context"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

const (
	hybridWriteQueueSize = 10_000
	hybridWriteBatchSize = 200
	hybridFlushInterval  = 100 * time.Millisecond
)

// luaApplyDelta is the async counterpart of luaTryReserve's seeding step: it
// seeds the shared Redis counter from db_spend (only once, when the key is
// absent) and applies delta, with no budget check — the allow/reject decision
// was already made locally, synchronously, before this ever runs.
const luaApplyDelta = `
local key = KEYS[1]
local db_spend = tonumber(ARGV[1])
local delta = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if redis.call('EXISTS', key) == 0 then
  redis.call('SET', key, db_spend)
end
redis.call('INCRBYFLOAT', key, delta)
redis.call('EXPIRE', key, ttl)
return 1
`

// deltaOp is one queued async write to the shared Redis counter.
type deltaOp struct {
	entity  string
	dbSpend float64
	delta   float64
}

// entityState is this instance's local view of one entity's budget counter:
// dbSpend is the DB snapshot it was seeded from, localDelta is the sum of
// TryReserve/Reconcile calls this instance has applied since that seed.
type entityState struct {
	mu         sync.Mutex
	seeded     bool
	seededAt   time.Time
	dbSpend    float64
	localDelta float64
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
type HybridBackend struct {
	client       valkey.Client
	keyPrefix    string
	ttl          time.Duration
	syncInterval time.Duration
	logger       *slog.Logger
	metrics      *monitoring.Metrics

	statesMu sync.Mutex
	states   map[string]*entityState

	// remoteOther holds, per entity, the estimated spend contributed by OTHER
	// instances only: redis_total - this_instance's own (dbSpend + localDelta).
	remoteMu    sync.RWMutex
	remoteOther map[string]float64

	writeQueue chan deltaOp
	stopCh     chan struct{}
	wg         sync.WaitGroup
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
	h := &HybridBackend{
		client:       client,
		keyPrefix:    keyPrefix,
		ttl:          ttl,
		syncInterval: syncInterval,
		logger:       logger,
		metrics:      metrics,
		states:       make(map[string]*entityState),
		remoteOther:  make(map[string]float64),
		writeQueue:   make(chan deltaOp, hybridWriteQueueSize),
		stopCh:       make(chan struct{}),
	}
	h.wg.Add(2)
	go h.writeWorker()
	go h.syncWorker()
	return h
}

// Close stops background goroutines. Call during server shutdown.
func (h *HybridBackend) Close() {
	close(h.stopCh)
	h.wg.Wait()
}

func (h *HybridBackend) recordRedisError(operation string, err error) {
	// Warn, not Error: a failed background write/sync degrades this entity to
	// local-only accounting until Redis recovers — by design, self-healing on
	// the next sync/flush, not an operator-actionable emergency on its own.
	h.logger.Warn("Budget hybrid backend: Redis operation failed", "operation", operation, "error", err)
	if h.metrics != nil {
		h.metrics.RecordRedisConnectionError(operation)
	}
}

func (h *HybridBackend) key(entity string) string { return h.keyPrefix + entity }

func (h *HybridBackend) getOrCreate(entity string) *entityState {
	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	s := h.states[entity]
	if s == nil {
		s = &entityState{}
		h.states[entity] = s
	}
	return s
}

func (h *HybridBackend) trackedEntities() []string {
	h.statesMu.Lock()
	defer h.statesMu.Unlock()
	keys := make([]string, 0, len(h.states))
	for k := range h.states {
		keys = append(keys, k)
	}
	return keys
}

func (h *HybridBackend) remoteFor(entity string) float64 {
	h.remoteMu.RLock()
	defer h.remoteMu.RUnlock()
	return h.remoteOther[entity]
}

func (h *HybridBackend) enqueue(op deltaOp) {
	select {
	case h.writeQueue <- op:
	default:
		// Queue full: drop. The local decision already stands; Redis (and so
		// other instances' view of this instance's spend) just lags until the
		// next successful flush — never worse than the pre-hybrid DB-snapshot
		// race this feature replaces.
	}
}

// --- reserveBackend implementation ---

func (h *HybridBackend) tryReserve(_ context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error) {
	s := h.getOrCreate(entity)
	// Read remoteOther before taking s.mu: doSync locks remoteMu then s.mu, so
	// locking s.mu first here and remoteMu second would invert that order and
	// deadlock against a concurrent sync tick.
	remoteOther := h.remoteFor(entity)
	s.mu.Lock()
	now := time.Now()
	if !s.seeded || now.Sub(s.seededAt) > h.ttl {
		s.dbSpend = dbSpend
		s.localDelta = 0
		s.seeded = true
	}
	s.seededAt = now
	total := s.dbSpend + s.localDelta + remoteOther + estimatedCost
	if maxBudget >= 0 && total > maxBudget {
		s.mu.Unlock()
		return false, nil
	}
	s.localDelta += estimatedCost
	seedSpend := s.dbSpend
	s.mu.Unlock()

	h.enqueue(deltaOp{entity: entity, dbSpend: seedSpend, delta: estimatedCost})
	return true, nil
}

func (h *HybridBackend) reconcile(_ context.Context, entity string, delta float64) error {
	if delta == 0 {
		return nil
	}
	s := h.getOrCreate(entity)
	s.mu.Lock()
	if !s.seeded {
		// No prior tryReserve seeded this entity on this instance (e.g. the
		// process restarted between reserve and reconcile) — seed at 0 so the
		// delta still lands; the next TTL-driven reseed pulls the true DB spend.
		s.seeded = true
		s.seededAt = time.Now()
	}
	s.localDelta += delta
	seedSpend := s.dbSpend
	s.mu.Unlock()

	h.enqueue(deltaOp{entity: entity, dbSpend: seedSpend, delta: delta})
	return nil
}

// writeWorker drains the async queue and applies deltas to Redis in batches.
func (h *HybridBackend) writeWorker() {
	defer h.wg.Done()
	batch := make([]deltaOp, 0, hybridWriteBatchSize)
	ttlStr := strconv.FormatInt(int64(h.ttl.Seconds()), 10)

	flush := func() {
		if len(batch) == 0 {
			return
		}
		cmds := make([]valkey.Completed, 0, len(batch))
		for _, op := range batch {
			cmds = append(cmds, h.client.B().Eval().
				Script(luaApplyDelta).Numkeys(1).
				Key(h.key(op.entity)).
				Arg(strconv.FormatFloat(op.dbSpend, 'f', -1, 64)).
				Arg(strconv.FormatFloat(op.delta, 'f', -1, 64)).
				Arg(ttlStr).
				Build())
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		results := h.client.DoMulti(ctx, cmds...)
		cancel()
		for _, res := range results {
			if err := res.Error(); err != nil {
				h.recordRedisError("budget_hybrid_write", err)
				break
			}
		}
		batch = batch[:0]
	}

	ticker := time.NewTicker(hybridFlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-h.stopCh:
			// Drain remaining ops before exit.
			for {
				select {
				case op := <-h.writeQueue:
					batch = append(batch, op)
					if len(batch) >= hybridWriteBatchSize {
						flush()
					}
				default:
					flush()
					return
				}
			}
		case op := <-h.writeQueue:
			batch = append(batch, op)
			if len(batch) >= hybridWriteBatchSize {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

// syncWorker periodically refreshes remoteOther for every tracked entity.
func (h *HybridBackend) syncWorker() {
	defer h.wg.Done()
	ticker := time.NewTicker(h.syncInterval)
	defer ticker.Stop()
	for {
		select {
		case <-h.stopCh:
			return
		case <-ticker.C:
			h.doSync()
		}
	}
}

// doSync estimates how much OTHER instances have reserved for each tracked
// entity: redis_total - this_instance's own (dbSpend + localDelta).
func (h *HybridBackend) doSync() {
	entities := h.trackedEntities()
	if len(entities) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cmds := make([]valkey.Completed, 0, len(entities))
	for _, e := range entities {
		cmds = append(cmds, h.client.B().Get().Key(h.key(e)).Build())
	}
	results := h.client.DoMulti(ctx, cmds...)

	h.remoteMu.Lock()
	defer h.remoteMu.Unlock()
	for i, e := range entities {
		redisTotal, err := results[i].AsFloat64()
		if err != nil {
			if !valkey.IsValkeyNil(err) {
				h.recordRedisError("budget_hybrid_sync", err)
				continue
			}
			redisTotal = 0
		}

		s := h.getOrCreate(e)
		s.mu.Lock()
		own := s.dbSpend + s.localDelta
		s.mu.Unlock()

		remoteOther := redisTotal - own
		if remoteOther < 0 {
			remoteOther = 0
		}
		h.remoteOther[e] = remoteOther
	}
}
