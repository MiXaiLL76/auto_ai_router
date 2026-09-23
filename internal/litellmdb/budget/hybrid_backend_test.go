// Hybrid-backend tests exercise NewHybrid against a real Redis/Valkey
// (VALKEY_ADDR), mirroring reservation_test.go's integration-test convention:
// local decisions are exact and instant, Redis is only eventually consistent,
// and a second instance sharing the same Redis observes the first instance's
// spend after a sync cycle. Outage tests route the client through flakyProxy,
// a TCP proxy that can drop every connection on demand.
package budget

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/valkey-io/valkey-go"
)

func valkeyAddrForTest(t *testing.T) string {
	t.Helper()
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	return addr
}

func newTestClient(t *testing.T, addr string) valkey.Client {
	t.Helper()
	client, err := valkey.NewClient(valkey.ClientOption{
		InitAddress:       []string{addr},
		ForceSingleClient: true,
		ConnWriteTimeout:  time.Second,
		Dialer:            net.Dialer{Timeout: time.Second},
	})
	if err != nil {
		t.Fatalf("failed to create valkey client: %v", err)
	}
	t.Cleanup(client.Close)
	return client
}

func hybridClientForTest(t *testing.T) valkey.Client {
	t.Helper()
	return newTestClient(t, valkeyAddrForTest(t))
}

// uniquePrefix keeps reruns against the same Redis from seeing stale keys.
func uniquePrefix(t *testing.T) string {
	return fmt.Sprintf("test:budgethybrid:%s:%d:", t.Name(), time.Now().UnixNano())
}

// flakyProxy forwards TCP to target; setDown(true) closes every live
// connection and refuses new ones until setDown(false), simulating a Redis
// outage/network blip without touching the real server.
type flakyProxy struct {
	ln     net.Listener
	target string

	mu    sync.Mutex
	down  bool
	conns map[net.Conn]struct{}
}

func newFlakyProxy(t *testing.T, target string) *flakyProxy {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("proxy listen: %v", err)
	}
	p := &flakyProxy{ln: ln, target: target, conns: make(map[net.Conn]struct{})}
	go p.acceptLoop()
	t.Cleanup(func() {
		_ = ln.Close()
		p.setDown(true)
	})
	return p
}

func (p *flakyProxy) addr() string { return p.ln.Addr().String() }

func (p *flakyProxy) acceptLoop() {
	for {
		c, err := p.ln.Accept()
		if err != nil {
			return
		}
		p.mu.Lock()
		down := p.down
		p.mu.Unlock()
		if down {
			_ = c.Close()
			continue
		}
		upstream, err := net.Dial("tcp", p.target)
		if err != nil {
			_ = c.Close()
			continue
		}
		p.mu.Lock()
		p.conns[c] = struct{}{}
		p.conns[upstream] = struct{}{}
		p.mu.Unlock()
		go func() { _, _ = io.Copy(upstream, c); _ = upstream.Close() }()
		go func() { _, _ = io.Copy(c, upstream); _ = c.Close() }()
	}
}

func (p *flakyProxy) setDown(down bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.down = down
	if down {
		for c := range p.conns {
			_ = c.Close()
		}
		clear(p.conns)
	}
}

type stateView struct {
	localDelta, flushed, pending, remoteOther float64
	inflight, repush                          bool
}

func viewEntity(h *HybridBackend, entity string) (stateView, bool) {
	s := h.lookup(entity)
	if s == nil {
		return stateView{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return stateView{
		localDelta:  s.localDelta,
		flushed:     s.flushed,
		pending:     s.pending,
		remoteOther: s.remoteOther,
		inflight:    s.inflight != nil,
		repush:      s.repush,
	}, true
}

// track makes h follow entity (so syncs pull it) without reserving anything.
func track(h *HybridBackend, entity string) {
	s := h.lockEntity(entity)
	if !s.seeded {
		s.reseed(0)
	}
	s.lastUsed = time.Now()
	s.mu.Unlock()
}

func redisValue(t *testing.T, client valkey.Client, key string) (float64, bool) {
	t.Helper()
	v, err := client.Do(context.Background(), client.B().Get().Key(key).Build()).AsFloat64()
	if valkey.IsValkeyNil(err) {
		return 0, false
	}
	if err != nil {
		t.Fatalf("GET %s: %v", key, err)
	}
	return v, true
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool, format string, args ...any) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !cond() {
		t.Fatalf(format, args...)
	}
}

func TestHybrid_TryReserve_LocalDecisionIsInstantAndExact(t *testing.T) {
	client := hybridClientForTest(t)
	r := NewHybrid(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer r.Close()
	ctx := context.Background()
	entity := "token:local"

	allowed, err := r.TryReserve(ctx, entity, 90, 5, 100)
	if err != nil || !allowed {
		t.Fatalf("expected first reservation under budget to be allowed: allowed=%v err=%v", allowed, err)
	}
	// 90 (seed) + 5 (first) + 5 (second) = 100, still within budget.
	allowed, err = r.TryReserve(ctx, entity, 90, 5, 100)
	if err != nil || !allowed {
		t.Fatalf("expected second reservation to fit exactly at budget: allowed=%v err=%v", allowed, err)
	}
	// One more unit pushes it over — must reject without a Redis round trip.
	allowed, err = r.TryReserve(ctx, entity, 90, 1, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected reservation over budget to be rejected by local state")
	}
}

func TestHybrid_Reconcile_AdjustsLocalState(t *testing.T) {
	client := hybridClientForTest(t)
	r := NewHybrid(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer r.Close()
	ctx := context.Background()
	entity := "token:reconcile"

	allowed, err := r.TryReserve(ctx, entity, 50, 40, 100)
	if err != nil || !allowed {
		t.Fatalf("setup reservation failed: allowed=%v err=%v", allowed, err)
	}
	if err := r.Reconcile(ctx, entity, -30); err != nil {
		t.Fatalf("Reconcile error: %v", err)
	}
	// 50 + (40-30) + 39 = 99, still fits.
	allowed, err = r.TryReserve(ctx, entity, 50, 39, 100)
	if err != nil || !allowed {
		t.Fatalf("expected room after reconcile reduced the counter: allowed=%v err=%v", allowed, err)
	}
	if err := r.Reconcile(ctx, entity, 50); err != nil {
		t.Fatalf("Reconcile error: %v", err)
	}
	// Now well over budget.
	allowed, err = r.TryReserve(ctx, entity, 50, 1, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected rejection after reconcile pushed counter over budget")
	}
}

func TestHybrid_CrossInstance_SyncPropagatesOtherInstanceSpend(t *testing.T) {
	client := hybridClientForTest(t)
	entity := "token:crossinstance"
	prefix := uniquePrefix(t)

	// Two independent "instances" (routers) sharing the same Redis, with a
	// fast sync interval so the test doesn't need to wait long.
	instanceA := NewHybrid(client, prefix, time.Minute, 50*time.Millisecond, nil, nil)
	defer instanceA.Close()
	instanceB := NewHybrid(client, prefix, time.Minute, 50*time.Millisecond, nil, nil)
	defer instanceB.Close()
	ctx := context.Background()

	// Instance A reserves 60 against a starting DB spend of 0 (max 100).
	allowed, err := instanceA.TryReserve(ctx, entity, 0, 60, 100)
	if err != nil || !allowed {
		t.Fatalf("instance A reservation failed: allowed=%v err=%v", allowed, err)
	}

	// B must track the entity before its sync worker will pull it, and this
	// must not itself count as a reservation, so the assertion below is purely
	// about A's synced contribution.
	backendB := instanceB.backend.(*HybridBackend)
	track(backendB, entity)

	waitFor(t, 3*time.Second, func() bool {
		v, _ := viewEntity(backendB, entity)
		return v.remoteOther >= 60
	}, "instance B never observed instance A's reservation via Redis sync")

	// 0 (B's dbSpend) + 0 (B's own localDelta) + 60 (A, synced) + 45 > 100.
	allowed, err = instanceB.TryReserve(ctx, entity, 0, 45, 100)
	if err != nil {
		t.Fatalf("instance B TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected instance B to see instance A's synced spend and reject")
	}
}

// Before the fix, doSync subtracted this instance's whole localDelta —
// including deltas not yet written to Redis — from the Redis total, so a
// sync racing a flush under-estimated the other instances by the unflushed
// amount. Only acknowledged (flushed) spend may be subtracted; whichever way
// the worker's flush interleaves, B's 60 must come out intact.
func TestHybrid_Sync_PendingLocalDeltasDoNotMaskOtherInstances(t *testing.T) {
	client := hybridClientForTest(t)
	prefix := uniquePrefix(t)
	entity := "token:mask"
	ctx := context.Background()

	a := NewHybridBackend(client, prefix, time.Minute, time.Hour, nil, nil)
	defer a.Close()
	b := NewHybridBackend(client, prefix, time.Minute, time.Hour, nil, nil)
	defer b.Close()

	if ok, _ := b.tryReserve(ctx, entity, 0, 60, -1); !ok {
		t.Fatal("B reservation rejected")
	}
	b.flush(true)

	if ok, _ := a.tryReserve(ctx, entity, 0, 30, -1); !ok {
		t.Fatal("A reservation rejected")
	}
	a.doSync()
	v, _ := viewEntity(a, entity)
	if v.remoteOther != 60 {
		t.Fatalf("A's estimate of other instances = %v, want 60 (pending own deltas leaked into it)", v.remoteOther)
	}
}

// A Redis outage must not lose deltas: they stay owed locally, are retried
// once Redis is back, and land exactly once.
func TestHybrid_RedisOutage_DeltasRetriedNotLost(t *testing.T) {
	addr := valkeyAddrForTest(t)
	direct := newTestClient(t, addr)
	proxy := newFlakyProxy(t, addr)
	viaProxy := newTestClient(t, proxy.addr())
	ctx := context.Background()
	entity := "token:outage"

	h := NewHybridBackend(viaProxy, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	key := h.key(entity)

	if ok, _ := h.tryReserve(ctx, entity, 100, 10, -1); !ok {
		t.Fatal("reservation rejected")
	}
	waitFor(t, 3*time.Second, func() bool { v, _ := redisValue(t, direct, key); return v == 110 },
		"initial delta never reached Redis")

	proxy.setDown(true)
	for i := 0; i < 4; i++ {
		if ok, _ := h.tryReserve(ctx, entity, 100, 5, -1); !ok {
			t.Fatal("reservation rejected during outage")
		}
	}
	if err := h.reconcile(ctx, entity, -3); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	waitFor(t, 5*time.Second, func() bool {
		h.workMu.Lock()
		defer h.workMu.Unlock()
		return h.failStreak >= 2
	}, "expected flushes to fail while the proxy is down")

	if v, _ := redisValue(t, direct, key); v != 110 {
		t.Fatalf("Redis changed during outage: %v", v)
	}
	v, _ := viewEntity(h, entity)
	if owed := v.localDelta - v.flushed; owed != 17 || v.flushed != 10 {
		t.Fatalf("local state lost the outage deltas: %+v", v)
	}

	proxy.setDown(false)
	waitFor(t, 10*time.Second, func() bool { v, _ := redisValue(t, direct, key); return v == 127 },
		"outage deltas were not retried into Redis after recovery")

	// Let a few more flush ticks run: the retried write must not be applied twice.
	time.Sleep(3 * hybridFlushInterval)
	if got, _ := redisValue(t, direct, key); got != 127 {
		t.Fatalf("Redis total = %v after recovery, want 127", got)
	}
	v, _ = viewEntity(h, entity)
	if v.flushed != 27 || v.pending != 0 || v.inflight {
		t.Fatalf("expected everything acknowledged after recovery: %+v", v)
	}
}

// The ambiguous failure: Redis ran the write but the reply was lost. The
// retry carries the same seq, so luaApplyDelta must drop it.
func TestHybrid_RetryOfAlreadyAppliedWriteIsNotDoubleCounted(t *testing.T) {
	addr := valkeyAddrForTest(t)
	direct := newTestClient(t, addr)
	proxy := newFlakyProxy(t, addr)
	viaProxy := newTestClient(t, proxy.addr())
	ctx := context.Background()
	entity := "token:dup"

	h := NewHybridBackend(viaProxy, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	key := h.key(entity)

	proxy.setDown(true)
	if ok, _ := h.tryReserve(ctx, entity, 50, 7, -1); !ok {
		t.Fatal("reservation rejected")
	}
	h.flush(true) // fails, leaving the write in flight with a fixed seq

	s := h.lookup(entity)
	s.mu.Lock()
	w := s.inflight
	s.mu.Unlock()
	if w == nil {
		t.Fatal("expected a failed write to stay in flight")
	}
	// Pretend the server did run it before the connection dropped.
	if n, err := direct.Do(ctx, h.applyDeltaCmd(entity, w)).AsInt64(); err != nil || n != 1 {
		t.Fatalf("direct apply: n=%v err=%v", n, err)
	}

	proxy.setDown(false)
	waitFor(t, 10*time.Second, func() bool {
		v, _ := viewEntity(h, entity)
		return !v.inflight && v.flushed == 7
	}, "retry was never acknowledged")
	if got, _ := redisValue(t, direct, key); got != 57 {
		t.Fatalf("Redis total = %v, want 57 (retry double-counted)", got)
	}
}

func TestHybrid_ApplyDelta_IdempotentPerSeq(t *testing.T) {
	client := hybridClientForTest(t)
	ctx := context.Background()
	h := NewHybridBackend(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	entity := "token:seq"

	w1 := &pendingWrite{seq: h.seq.Add(1), seed: 50, delta: 7}
	for i, want := range []int64{1, 0} {
		n, err := client.Do(ctx, h.applyDeltaCmd(entity, w1)).AsInt64()
		if err != nil || n != want {
			t.Fatalf("attempt %d: n=%v err=%v, want %v", i, n, err, want)
		}
	}
	w2 := &pendingWrite{seq: h.seq.Add(1), seed: 999, delta: 3}
	if n, err := client.Do(ctx, h.applyDeltaCmd(entity, w2)).AsInt64(); err != nil || n != 1 {
		t.Fatalf("next seq: n=%v err=%v", n, err)
	}
	// Seed only applies to an absent key.
	if got, _ := redisValue(t, client, h.key(entity)); got != 60 {
		t.Fatalf("Redis total = %v, want 60", got)
	}
	ttl, err := client.Do(ctx, client.B().Ttl().Key(h.seqKey(entity)).Build()).AsInt64()
	if err != nil || ttl <= 0 {
		t.Fatalf("seq key must carry a TTL: ttl=%v err=%v", ttl, err)
	}
}

// A counter key that vanished from Redis (restart without persistence,
// eviction, manual flush) is restored from this instance's acknowledged spend
// on the next sync instead of silently reading as 0 forever.
func TestHybrid_RedisKeyLoss_Repushed(t *testing.T) {
	client := hybridClientForTest(t)
	ctx := context.Background()
	h := NewHybridBackend(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	entity := "token:lost"
	key := h.key(entity)

	if ok, _ := h.tryReserve(ctx, entity, 100, 20, -1); !ok {
		t.Fatal("reservation rejected")
	}
	h.flush(true)
	if got, _ := redisValue(t, client, key); got != 120 {
		t.Fatalf("Redis total = %v, want 120", got)
	}

	if err := client.Do(ctx, client.B().Del().Key(key).Build()).Error(); err != nil {
		t.Fatalf("DEL: %v", err)
	}
	h.doSync()
	if v, _ := viewEntity(h, entity); !v.repush {
		t.Fatalf("expected sync to schedule a repush for the vanished key: %+v", v)
	}
	h.flush(true)
	if got, ok := redisValue(t, client, key); !ok || got != 120 {
		t.Fatalf("Redis total = %v (present=%v) after repush, want 120", got, ok)
	}
	if v, _ := viewEntity(h, entity); v.repush || v.inflight || v.flushed != 20 {
		t.Fatalf("unexpected state after repush: %+v", v)
	}
}

func TestHybrid_EvictsIdleEntities(t *testing.T) {
	client := hybridClientForTest(t)
	ctx := context.Background()
	ttl := time.Minute
	h := NewHybridBackend(client, uniquePrefix(t), ttl, time.Hour, nil, nil)
	defer h.Close()

	for _, e := range []string{"token:idle1", "token:idle2"} {
		if ok, _ := h.tryReserve(ctx, e, 0, 1, -1); !ok {
			t.Fatal("reservation rejected")
		}
	}
	h.flush(true)

	// Recently used entities survive.
	h.evictIdle(time.Now())
	if _, ok := viewEntity(h, "token:idle1"); !ok {
		t.Fatal("active entity was evicted")
	}

	// An idle entity that still owes Redis a write must survive too.
	if ok, _ := h.tryReserve(ctx, "token:owing", 0, 1, -1); !ok {
		t.Fatal("reservation rejected")
	}
	h.evictIdle(time.Now().Add(2 * ttl))

	for _, e := range []string{"token:idle1", "token:idle2"} {
		if _, ok := viewEntity(h, e); ok {
			t.Fatalf("%s should have been evicted after idling past ttl", e)
		}
	}
	if _, ok := viewEntity(h, "token:owing"); !ok {
		t.Fatal("entity with unflushed deltas was evicted")
	}

	// An evicted entity comes back reseeded from the caller's DB value.
	if ok, _ := h.tryReserve(ctx, "token:idle1", 500, 2, -1); !ok {
		t.Fatal("reservation rejected")
	}
	s := h.lookup("token:idle1")
	s.mu.Lock()
	dbSpend, localDelta := s.dbSpend, s.localDelta
	s.mu.Unlock()
	if dbSpend != 500 || localDelta != 2 {
		t.Fatalf("expected a fresh seed after eviction: dbSpend=%v localDelta=%v", dbSpend, localDelta)
	}
}

// Eviction racing reservations must never strand a delta in an orphaned
// state: every reserved unit reaches Redis exactly once. Run with -race.
func TestHybrid_ConcurrentReserveAndEvict_NoLostDeltas(t *testing.T) {
	client := hybridClientForTest(t)
	ctx := context.Background()
	h := NewHybridBackend(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	entity := "token:race"

	const workers, perWorker = 8, 1000
	stop := make(chan struct{})
	evictorDone := make(chan struct{})
	go func() {
		defer close(evictorDone)
		for {
			select {
			case <-stop:
				return
			default:
				h.flush(true)
				// Far-future "now" makes every clean entity evictable.
				h.evictIdle(time.Now().Add(time.Hour))
			}
		}
	}()

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				if ok, err := h.tryReserve(ctx, entity, 0, 1, -1); !ok || err != nil {
					t.Errorf("reservation rejected: ok=%v err=%v", ok, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	<-evictorDone
	h.flush(true)

	if got, _ := redisValue(t, client, h.key(entity)); got != workers*perWorker {
		t.Fatalf("Redis total = %v, want %v", got, workers*perWorker)
	}
}

// Deterministic version of the race above: the entity is evicted exactly
// between lockEntity's map lookup and its s.mu.Lock. The reservation must
// retry onto a fresh state rather than write into the orphaned one.
func TestHybrid_ReserveRacingEviction_RetriesOntoLiveState(t *testing.T) {
	client := hybridClientForTest(t)
	ctx := context.Background()
	h := NewHybridBackend(client, uniquePrefix(t), time.Minute, time.Hour, nil, nil)
	defer h.Close()
	entity := "token:evictwindow"

	if ok, _ := h.tryReserve(ctx, entity, 0, 1, -1); !ok {
		t.Fatal("reservation rejected")
	}
	h.flush(true)

	fired := false
	testHookLockEntity = func(string) {
		if !fired {
			fired = true
			h.evictIdle(time.Now().Add(time.Hour))
		}
	}
	defer func() { testHookLockEntity = nil }()

	if ok, _ := h.tryReserve(ctx, entity, 0, 2, -1); !ok {
		t.Fatal("reservation rejected")
	}
	if !fired {
		t.Fatal("hook never ran")
	}
	v, ok := viewEntity(h, entity)
	if !ok || v.localDelta != 2 {
		t.Fatalf("reservation did not land on the live state: present=%v %+v", ok, v)
	}
	h.flush(true)
	if got, _ := redisValue(t, client, h.key(entity)); got != 3 {
		t.Fatalf("Redis total = %v, want 3 (delta stranded in evicted state)", got)
	}
}

func TestHybrid_NilClient_NoOp(t *testing.T) {
	ctx := context.Background()
	r := NewHybrid(nil, "test:budgethybrid:nilclient:", time.Minute, time.Second, nil, nil)
	defer r.Close()
	allowed, err := r.TryReserve(ctx, "e", 100, 100, 1)
	if err != nil || !allowed {
		t.Fatalf("nil-client hybrid Reserver TryReserve should be no-op allow: allowed=%v err=%v", allowed, err)
	}
	if err := r.Reconcile(ctx, "e", 5); err != nil {
		t.Fatalf("nil-client hybrid Reserver Reconcile should be no-op: %v", err)
	}
}
