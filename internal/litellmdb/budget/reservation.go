// Package budget provides atomic, Redis-backed budget pre-reservation to close
// the pre-check-vs-actual-spend TOCTOU race described in todo_auth_billing.md P1.4.
//
// A reservation seeds a per-entity counter from the authoritative DB spend value
// (only when the Redis key does not yet exist), atomically adds the request's
// estimated max cost, and rejects the request if the new total would exceed the
// budget. After the real cost is known the caller reconciles the reservation to
// the true cost. When no Redis client is configured every method is a safe no-op
// so the feature degrades to the legacy DB-snapshot budget check.
package budget

import (
	"context"
	"log/slog"
	"strconv"
	"time"

	"github.com/valkey-io/valkey-go"
)

// commandTimeout caps a single Redis command when the parent context has no
// tighter deadline.
const commandTimeout = 3 * time.Second

// luaTryReserve atomically seeds the counter from db_spend (only when the key is
// absent), adds est_cost, refreshes the TTL, and checks against max_budget.
// Rolls back the increment and returns 0 when the new total exceeds max_budget.
// max_budget < 0 means unlimited (always allowed, spend still tracked).
// Returns 1 if allowed, 0 if rejected.
const luaTryReserve = `
local key = KEYS[1]
local db_spend = tonumber(ARGV[1])
local est_cost = tonumber(ARGV[2])
local max_budget = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
if redis.call('EXISTS', key) == 0 then
  redis.call('SET', key, db_spend)
end
local new_val = redis.call('INCRBYFLOAT', key, est_cost)
redis.call('EXPIRE', key, ttl)
if max_budget >= 0 and tonumber(new_val) > max_budget then
  redis.call('INCRBYFLOAT', key, -est_cost)
  return 0
end
return 1
`

// Reserver performs atomic budget reservations against Redis/Valkey.
type Reserver struct {
	client    valkey.Client
	keyPrefix string
	ttl       time.Duration
	logger    *slog.Logger
}

// New creates a Reserver. A nil client yields a no-op Reserver (all methods
// succeed without touching Redis), so callers can wire it unconditionally.
func New(client valkey.Client, keyPrefix string, ttl time.Duration, logger *slog.Logger) *Reserver {
	if ttl <= 0 {
		ttl = 15 * time.Minute
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Reserver{
		client:    client,
		keyPrefix: keyPrefix,
		ttl:       ttl,
		logger:    logger,
	}
}

// enabled reports whether the Reserver is backed by a live Redis client.
func (r *Reserver) enabled() bool {
	return r != nil && r.client != nil
}

func (r *Reserver) cmdCtx(parent context.Context) (context.Context, context.CancelFunc) {
	if d, ok := parent.Deadline(); ok && time.Until(d) <= commandTimeout {
		return parent, func() {}
	}
	return context.WithTimeout(parent, commandTimeout)
}

// TryReserve atomically seeds the counter from dbSpend (only if the key doesn't
// exist yet), adds estimatedCost, and checks against maxBudget. Returns
// allowed=false and rolls back the increment if the new total exceeds maxBudget.
// maxBudget < 0 means unlimited (always allowed, still tracks spend).
// A nil client/Reserver is a no-op that allows the request.
func (r *Reserver) TryReserve(ctx context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error) {
	if !r.enabled() {
		return true, nil
	}
	cmdCtx, cancel := r.cmdCtx(ctx)
	defer cancel()

	res, err := r.client.Do(cmdCtx, r.client.B().Eval().
		Script(luaTryReserve).
		Numkeys(1).
		Key(r.keyPrefix+entity).
		Arg(strconv.FormatFloat(dbSpend, 'f', -1, 64)).
		Arg(strconv.FormatFloat(estimatedCost, 'f', -1, 64)).
		Arg(strconv.FormatFloat(maxBudget, 'f', -1, 64)).
		Arg(strconv.FormatInt(int64(r.ttl.Seconds()), 10)).
		Build()).AsInt64()
	if err != nil {
		return false, err
	}
	return res == 1, nil
}

// Reconcile adjusts a reserved amount to the true cost: delta = actualCost -
// reservedEstimate. Call exactly once per successful TryReserve (on success,
// failure, or provider error) or the reservation leaks and permanently inflates
// the counter. A nil client/Reserver, or delta == 0, is a no-op.
func (r *Reserver) Reconcile(ctx context.Context, entity string, delta float64) error {
	if !r.enabled() || delta == 0 {
		return nil
	}
	cmdCtx, cancel := r.cmdCtx(ctx)
	defer cancel()

	key := r.keyPrefix + entity
	if err := r.client.Do(cmdCtx, r.client.B().Incrbyfloat().
		Key(key).
		Increment(delta).
		Build()).Error(); err != nil {
		return err
	}
	// Best-effort TTL refresh; a stale TTL is harmless (the key just expires and
	// reseeds from DB on next use), so an error here is not propagated.
	_ = r.client.Do(cmdCtx, r.client.B().Expire().
		Key(key).
		Seconds(int64(r.ttl.Seconds())).
		Build()).Error()
	return nil
}
