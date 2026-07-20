package budget

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/valkey-io/valkey-go"
)

// reserverForTest builds a Reserver backed by a real Redis/Valkey from
// VALKEY_ADDR, mirroring internal/ratelimit's integration-test convention.
// Skips the test when the variable is unset.
func reserverForTest(t *testing.T, prefix string) *Reserver {
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
	return New(client, prefix, time.Minute, nil)
}

func TestTryReserve_SeedsFromDBSpendAndAllows(t *testing.T) {
	r := reserverForTest(t, "test:budget:seed:")
	ctx := context.Background()
	entity := "token:seed"

	// db_spend=90, est=5, max=100 → 95 <= 100 → allowed, seeded from DB.
	allowed, err := r.TryReserve(ctx, entity, 90, 5, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if !allowed {
		t.Fatal("expected first reservation under budget to be allowed")
	}
}

func TestTryReserve_RejectsAndRollsBack(t *testing.T) {
	r := reserverForTest(t, "test:budget:reject:")
	ctx := context.Background()
	entity := "token:reject"

	// db_spend=99, est=5, max=100 → 104 > 100 → rejected, rolled back to 99.
	allowed, err := r.TryReserve(ctx, entity, 99, 5, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected reservation over budget to be rejected")
	}

	// After rollback the counter should be back at 99, so a cheap request fits.
	allowed, err = r.TryReserve(ctx, entity, 99, 0.5, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if !allowed {
		t.Fatal("expected cheap reservation after rollback to be allowed (rollback failed)")
	}
}

func TestTryReserve_UnlimitedAlwaysAllows(t *testing.T) {
	r := reserverForTest(t, "test:budget:unlimited:")
	ctx := context.Background()

	allowed, err := r.TryReserve(ctx, "token:unlimited", 1e9, 1e9, -1)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if !allowed {
		t.Fatal("expected unlimited budget (max < 0) to always allow")
	}
}

func TestReconcile_AdjustsCounter(t *testing.T) {
	r := reserverForTest(t, "test:budget:reconcile:")
	ctx := context.Background()
	entity := "token:reconcile"

	// Seed at 50 with a 40 reservation → 90.
	allowed, err := r.TryReserve(ctx, entity, 50, 40, 100)
	if err != nil || !allowed {
		t.Fatalf("setup reservation failed: allowed=%v err=%v", allowed, err)
	}

	// Actual cost was 10, reserved 40 → delta -30 → counter 60. Room for 39 more.
	if err := r.Reconcile(ctx, entity, -30); err != nil {
		t.Fatalf("Reconcile error: %v", err)
	}
	allowed, err = r.TryReserve(ctx, entity, 50, 39, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if !allowed {
		t.Fatal("expected room after reconcile reduced the counter")
	}

	// Increase the counter past budget and confirm a subsequent reserve rejects.
	if err := r.Reconcile(ctx, entity, 50); err != nil {
		t.Fatalf("Reconcile error: %v", err)
	}
	allowed, err = r.TryReserve(ctx, entity, 50, 1, 100)
	if err != nil {
		t.Fatalf("TryReserve error: %v", err)
	}
	if allowed {
		t.Fatal("expected rejection after reconcile pushed counter over budget")
	}
}

func TestNilReserver_NoOp(t *testing.T) {
	ctx := context.Background()

	// Nil *Reserver.
	var nilReserver *Reserver
	allowed, err := nilReserver.TryReserve(ctx, "e", 0, 1, 1)
	if err != nil || !allowed {
		t.Fatalf("nil Reserver TryReserve should be no-op allow: allowed=%v err=%v", allowed, err)
	}
	if err := nilReserver.Reconcile(ctx, "e", 1); err != nil {
		t.Fatalf("nil Reserver Reconcile should be no-op: %v", err)
	}

	// Reserver with nil client.
	rc := New(nil, "test:budget:nilclient:", time.Minute, nil)
	allowed, err = rc.TryReserve(ctx, "e", 100, 100, 1)
	if err != nil || !allowed {
		t.Fatalf("nil-client Reserver TryReserve should be no-op allow: allowed=%v err=%v", allowed, err)
	}
	if err := rc.Reconcile(ctx, "e", 5); err != nil {
		t.Fatalf("nil-client Reserver Reconcile should be no-op: %v", err)
	}
}
