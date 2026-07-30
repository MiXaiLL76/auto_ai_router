package ratelimit

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

const (
	hybridWriteQueueSize = 10_000
	hybridWriteBatchSize = 200
	hybridFlushInterval  = 100 * time.Millisecond
)

type asyncOpKind uint8

const (
	opRPM asyncOpKind = iota
	opTPM
)

type asyncOp struct {
	kind   asyncOpKind
	key    string
	member string // uuid for RPM; "uuid:count" for TPM
	now    int64  // unix ms
}

// HybridBackend stores counters in a local in-process backend for zero-latency
// hot-path decisions and keeps Redis updated asynchronously for cross-instance
// consistency.
//
// Rate-limit decisions are made against: local_count + estimated_remote_count.
// Remote count is updated in the background every syncInterval by comparing
// the Redis total with the local total.
type HybridBackend struct {
	local        *localBackend
	remote       *RedisBackend
	syncInterval time.Duration
	log          *slog.Logger
	metrics      *monitoring.Metrics

	trackedMu sync.Mutex
	tracked   map[string]struct{}

	// remoteStats holds the estimated traffic from OTHER instances only.
	// It is periodically refreshed: remote = redis_total - local.
	remoteMu    sync.RWMutex
	remoteStats map[string][2]int // key → [rpm, tpm]

	writeQueue chan asyncOp
	stopCh     chan struct{}
	wg         sync.WaitGroup
}

// NewHybridBackend wraps remote with a local in-process counter. Hot-path
// operations run entirely on the local backend; Redis is updated asynchronously.
// syncInterval controls how often remote stats are pulled (default: 5s).
// log and metrics report failures in the background write/sync workers,
// which would otherwise fail against an unreachable Redis with no visibility
// (see redis_todo.md item 3).
func NewHybridBackend(remote *RedisBackend, syncInterval time.Duration, log *slog.Logger, metrics *monitoring.Metrics) *HybridBackend {
	if syncInterval <= 0 {
		syncInterval = 5 * time.Second
	}
	if log == nil {
		log = slog.Default()
	}
	h := &HybridBackend{
		local:        newLocalBackend(),
		remote:       remote,
		syncInterval: syncInterval,
		log:          log,
		metrics:      metrics,
		tracked:      make(map[string]struct{}),
		remoteStats:  make(map[string][2]int),
		writeQueue:   make(chan asyncOp, hybridWriteQueueSize),
		stopCh:       make(chan struct{}),
	}
	h.wg.Add(2)
	go h.writeWorker()
	go h.syncWorker()
	return h
}

// recordRedisError logs and records a background-worker failure against
// Redis. metrics may be nil in tests that construct a HybridBackend without
// a full Metrics instance.
func (h *HybridBackend) recordRedisError(operation string, err error) {
	// Warn, not Error: a failed background write/sync degrades this backend
	// to local-only counting for the affected key until Redis recovers — by
	// design, non-fatal and self-healing, not an operator-actionable
	// emergency on its own. The metric below (unaffected by log level) is
	// the real signal for alerting on a sustained outage.
	h.log.Warn("Hybrid backend: Redis operation failed", "operation", operation, "error", err)
	if h.metrics != nil {
		h.metrics.RecordRedisConnectionError(operation)
	}
}

// Close stops background goroutines. Call during server shutdown.
func (h *HybridBackend) Close() {
	close(h.stopCh)
	h.wg.Wait()
}

// Client exposes the underlying Redis client (e.g. for health checks).
func (h *HybridBackend) Client() valkey.Client { return h.remote.Client() }

func (h *HybridBackend) track(key string) {
	h.trackedMu.Lock()
	h.tracked[key] = struct{}{}
	h.trackedMu.Unlock()
}

func (h *HybridBackend) trackedKeys() []string {
	h.trackedMu.Lock()
	keys := make([]string, 0, len(h.tracked))
	for k := range h.tracked {
		keys = append(keys, k)
	}
	h.trackedMu.Unlock()
	return keys
}

func (h *HybridBackend) enqueue(op asyncOp) {
	select {
	case h.writeQueue <- op:
	default:
		// Queue full: drop. Local rate limiting remains accurate; Redis just
		// misses this event until the next sync.
	}
}

// writeWorker drains the async queue and writes to Redis in batches.
func (h *HybridBackend) writeWorker() {
	defer h.wg.Done()
	batch := make([]asyncOp, 0, hybridWriteBatchSize)

	flush := func() {
		if len(batch) == 0 {
			return
		}
		windowStr := fmt.Sprintf("%d", rpmWindow.Milliseconds())
		ttlStr := fmt.Sprintf("%d", h.remote.keyTTL)

		cmds := make([]valkey.Completed, 0, len(batch))
		for _, op := range batch {
			nowStr := fmt.Sprintf("%d", op.now)
			switch op.kind {
			case opRPM:
				cmds = append(cmds, h.remote.client.B().Eval().
					Script(luaTryAllowRPM).Numkeys(1).
					Key(h.remote.rpmKey(op.key)).
					Arg(nowStr).Arg(windowStr).
					Arg("-1"). // no limit check — just record
					Arg(op.member).
					Arg(ttlStr).
					Build())
			case opTPM:
				cmds = append(cmds, h.remote.client.B().Eval().
					Script(luaConsumeTokens).Numkeys(1).
					Key(h.remote.tpmKey(op.key)).
					Arg(nowStr).Arg(windowStr).
					Arg(op.member).
					Arg(ttlStr).
					Build())
			}
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		results := h.remote.client.DoMulti(ctx, cmds...)
		cancel()
		for _, res := range results {
			if err := res.Error(); err != nil {
				h.recordRedisError("hybrid_write", err)
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

// syncWorker pulls aggregated stats from Redis and refreshes remoteStats.
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

func (h *HybridBackend) doSync() {
	keys := h.trackedKeys()
	if len(keys) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Fetch Redis totals (all instances) and local counts in parallel. The
	// goroutine below only ever produces a result — it never calls
	// recordRedisError itself, so the outer select (the only place that
	// reports) can't double-count the same underlying failure once for the
	// goroutine's own error and again for a racing ctx.Done() timeout.
	type statsResult struct {
		total    map[string][2]int
		local    map[string][2]int
		totalErr error
	}
	ch := make(chan statsResult, 1)

	go func() {
		total, err := h.remote.batchCurrentStatsErr(ctx, keys)
		local := h.local.batchCurrentStats(ctx, keys)
		select {
		case ch <- statsResult{total, local, err}:
		case <-ctx.Done():
		}
	}()

	select {
	case <-ctx.Done():
		h.recordRedisError("hybrid_sync", ctx.Err())
		return
	case r := <-ch:
		if r.totalErr != nil {
			h.recordRedisError("hybrid_sync", r.totalErr)
		}
		h.remoteMu.Lock()
		for _, key := range keys {
			total := r.total[key]
			local := r.local[key]
			remoteRPM := total[0] - local[0]
			remoteTPM := total[1] - local[1]
			if remoteRPM < 0 {
				remoteRPM = 0
			}
			if remoteTPM < 0 {
				remoteTPM = 0
			}
			h.remoteStats[key] = [2]int{remoteRPM, remoteTPM}
		}
		h.remoteMu.Unlock()
	}
}

func (h *HybridBackend) remoteFor(key string) (rpm, tpm int) {
	h.remoteMu.RLock()
	v := h.remoteStats[key]
	h.remoteMu.RUnlock()
	return v[0], v[1]
}

// effectiveRPMLimit subtracts estimated remote traffic from the configured limit.
func (h *HybridBackend) effectiveRPMLimit(key string, limit int) int {
	if limit == -1 {
		return -1
	}
	remoteRPM, _ := h.remoteFor(key)
	eff := limit - remoteRPM
	if eff <= 0 {
		return 0
	}
	return eff
}

// effectiveTPMLimit subtracts estimated remote token usage from the configured limit.
func (h *HybridBackend) effectiveTPMLimit(key string, limit int) int {
	if limit == -1 {
		return -1
	}
	_, remoteTPM := h.remoteFor(key)
	eff := limit - remoteTPM
	if eff <= 0 {
		return 0
	}
	return eff
}

// --- counterBackend implementation ---

func (h *HybridBackend) tryAllowRPM(ctx context.Context, key string, limit int) bool {
	h.track(key)
	effLimit := h.effectiveRPMLimit(key, limit)
	if effLimit == 0 {
		return false
	}
	member := uuid.New().String()
	now := nowMS()
	if !h.local.tryAllowRPM(ctx, key, effLimit) {
		return false
	}
	h.enqueue(asyncOp{kind: opRPM, key: key, member: member, now: now})
	return true
}

func (h *HybridBackend) canAllowRPM(ctx context.Context, key string, limit int) bool {
	h.track(key)
	return h.local.canAllowRPM(ctx, key, h.effectiveRPMLimit(key, limit))
}

func (h *HybridBackend) canAllowTPM(ctx context.Context, key string, limit int) bool {
	if limit == -1 {
		return true
	}
	h.track(key)
	return h.local.canAllowTPM(ctx, key, h.effectiveTPMLimit(key, limit))
}

func (h *HybridBackend) consumeTokens(ctx context.Context, key string, tokenCount int) {
	h.track(key)
	h.local.consumeTokens(ctx, key, tokenCount)
	member := fmt.Sprintf("%s:%d", uuid.New().String(), tokenCount)
	h.enqueue(asyncOp{kind: opTPM, key: key, member: member, now: nowMS()})
}

func (h *HybridBackend) currentRPM(ctx context.Context, key string) int {
	localRPM := h.local.currentRPM(ctx, key)
	remoteRPM, _ := h.remoteFor(key)
	return localRPM + remoteRPM
}

func (h *HybridBackend) currentTPM(ctx context.Context, key string) int {
	localTPM := h.local.currentTPM(ctx, key)
	_, remoteTPM := h.remoteFor(key)
	return localTPM + remoteTPM
}

func (h *HybridBackend) tryAllowAll(
	ctx context.Context,
	credKey string, credRPM, credTPM int,
	modelKey string, modelRPM, modelTPM int,
) bool {
	h.track(credKey)
	now := nowMS()
	credMember := uuid.New().String()

	effCredRPM := h.effectiveRPMLimit(credKey, credRPM)
	effCredTPM := h.effectiveTPMLimit(credKey, credTPM)

	var effModelRPM, effModelTPM int
	var modelMember string
	if modelKey != "" {
		h.track(modelKey)
		effModelRPM = h.effectiveRPMLimit(modelKey, modelRPM)
		effModelTPM = h.effectiveTPMLimit(modelKey, modelTPM)
		modelMember = uuid.New().String()
	}

	if !h.local.tryAllowAll(ctx, credKey, effCredRPM, effCredTPM, modelKey, effModelRPM, effModelTPM) {
		return false
	}
	h.enqueue(asyncOp{kind: opRPM, key: credKey, member: credMember, now: now})
	if modelKey != "" {
		h.enqueue(asyncOp{kind: opRPM, key: modelKey, member: modelMember, now: now})
	}
	return true
}

// setCurrentUsage is forwarded to local backend only (used by proxy-credential sync).
func (h *HybridBackend) setCurrentUsage(ctx context.Context, key string, currentRPM, currentTPM int) {
	h.local.setCurrentUsage(ctx, key, currentRPM, currentTPM)
}

func (h *HybridBackend) deleteKey(ctx context.Context, key string) {
	h.trackedMu.Lock()
	delete(h.tracked, key)
	h.trackedMu.Unlock()

	h.remoteMu.Lock()
	delete(h.remoteStats, key)
	h.remoteMu.Unlock()

	h.local.deleteKey(ctx, key)
	h.remote.deleteKey(ctx, key)
}

func (h *HybridBackend) batchCurrentStats(ctx context.Context, keys []string) map[string][2]int {
	localStats := h.local.batchCurrentStats(ctx, keys)
	h.remoteMu.RLock()
	defer h.remoteMu.RUnlock()
	out := make(map[string][2]int, len(keys))
	for _, key := range keys {
		local := localStats[key]
		remote := h.remoteStats[key]
		out[key] = [2]int{local[0] + remote[0], local[1] + remote[1]}
	}
	return out
}
