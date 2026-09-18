// Hybrid-backend tests exercise NewHybrid against a real Redis/Valkey
// (VALKEY_ADDR), mirroring reservation_test.go's integration-test convention:
// local decisions are exact and instant, Redis is only eventually consistent,
// and a second instance sharing the same Redis observes the first instance's
// spend after a sync cycle.
package budget

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/valkey-io/valkey-go"
)

func hybridClientForTest(t *testing.T) valkey.Client {
	t.Helper()
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := valkey.NewClient(valkey.ClientOption{
		InitAddress:       []string{addr},
		ForceSingleClient: true,
	})
	if err != nil {
		t.Fatalf("failed to create valkey client: %v", err)
	}
	t.Cleanup(client.Close)
	return client
}

func TestHybrid_TryReserve_LocalDecisionIsInstantAndExact(t *testing.T) {
	client := hybridClientForTest(t)
	r := NewHybrid(client, "test:budgethybrid:local:", time.Minute, time.Hour, nil, nil)
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
	r := NewHybrid(client, "test:budgethybrid:reconcile:", time.Minute, time.Hour, nil, nil)
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
	prefix := "test:budgethybrid:cross:"

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
	// must not itself count as a reservation (checking remoteFor directly,
	// rather than looping TryReserve, keeps B's own localDelta at 0 so the
	// assertion below is purely about A's synced contribution).
	backendB := instanceB.backend.(*HybridBackend)
	backendB.getOrCreate(entity)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && backendB.remoteFor(entity) < 60 {
		time.Sleep(20 * time.Millisecond)
	}
	if got := backendB.remoteFor(entity); got < 60 {
		t.Fatalf("instance B never observed instance A's reservation via Redis sync; remoteOther=%v", got)
	}

	// 0 (B's dbSpend) + 0 (B's own localDelta) + 60 (A, synced) + 45 > 100.
	allowed, err = instanceB.TryReserve(ctx, entity, 0, 45, 100)
	if err != nil {
		t.Fatalf("instance B TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected instance B to see instance A's synced spend and reject")
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
