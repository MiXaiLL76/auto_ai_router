// Package budget provides atomic, Redis-backed budget pre-reservation to close
// the pre-check-vs-actual-spend TOCTOU race described in todo_auth_billing.md P1.4.
//
// A reservation seeds a per-entity counter from the authoritative DB spend value
// (only when it hasn't been seeded yet), atomically adds the request's
// estimated max cost, and rejects the request if the new total would exceed the
// budget. After the real cost is known the caller reconciles the reservation to
// the true cost. When no backend is configured every method is a safe no-op so
// the feature degrades to the legacy DB-snapshot budget check.
//
// Two backends are available: New() talks to Redis synchronously (exact,
// +1 RTT before and after every request); NewHybrid() decides locally and
// syncs to Redis asynchronously (near-zero added latency, ~syncInterval
// cross-instance drift) — see HybridBackend's doc comment for the trade-off.
package budget

import (
	"context"
	"log/slog"
	"time"

	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

// Reserver performs atomic budget reservations, backed by either direct Redis
// or a HybridBackend.
type Reserver struct {
	backend reserveBackend
}

// New creates a Reserver that talks to Redis/Valkey synchronously. A nil
// client yields a no-op Reserver (all methods succeed without touching
// Redis), so callers can wire it unconditionally.
func New(client valkey.Client, keyPrefix string, ttl time.Duration) *Reserver {
	if client == nil {
		return &Reserver{}
	}
	if ttl <= 0 {
		ttl = 15 * time.Minute
	}
	return &Reserver{backend: newRedisReserveBackend(client, keyPrefix, ttl)}
}

// NewHybrid creates a Reserver backed by a HybridBackend: reservation
// decisions are made locally (no added request latency) and Redis is updated
// asynchronously in batches, so it's cheap enough to enable by default. A nil
// client yields a no-op Reserver.
func NewHybrid(client valkey.Client, keyPrefix string, ttl, syncInterval time.Duration, logger *slog.Logger, metrics *monitoring.Metrics) *Reserver {
	if client == nil {
		return &Reserver{}
	}
	return &Reserver{backend: NewHybridBackend(client, keyPrefix, ttl, syncInterval, logger, metrics)}
}

// TryReserve atomically seeds the counter from dbSpend (only if it hasn't been
// seeded yet), adds estimatedCost, and checks against maxBudget. Returns
// allowed=false and leaves estimatedCost unapplied if the new total would
// exceed maxBudget. maxBudget < 0 means unlimited (always allowed, still
// tracks spend). A nil client/Reserver is a no-op that allows the request.
func (r *Reserver) TryReserve(ctx context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error) {
	if r == nil || r.backend == nil {
		return true, nil
	}
	return r.backend.tryReserve(ctx, entity, dbSpend, estimatedCost, maxBudget)
}

// Reconcile adjusts a reserved amount to the true cost: delta = actualCost -
// reservedEstimate. Call exactly once per successful TryReserve (on success,
// failure, or provider error) or the reservation leaks and permanently inflates
// the counter. A nil client/Reserver, or delta == 0, is a no-op.
func (r *Reserver) Reconcile(ctx context.Context, entity string, delta float64) error {
	if r == nil || r.backend == nil || delta == 0 {
		return nil
	}
	return r.backend.reconcile(ctx, entity, delta)
}

// closer is implemented by backends that own background goroutines (currently
// only HybridBackend) and need to stop them during server shutdown.
type closer interface{ Close() }

// Close stops the backend's background goroutines, if it has any. Safe to
// call on a nil Reserver or one backed by direct Redis (both are no-ops).
func (r *Reserver) Close() {
	if r == nil {
		return
	}
	if c, ok := r.backend.(closer); ok {
		c.Close()
	}
}
