package budget

import (
	"context"
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

// redisReserveBackend implements reserveBackend directly against Redis/Valkey:
// every TryReserve/Reconcile is one synchronous round trip (Lua-atomic).
type redisReserveBackend struct {
	client    valkey.Client
	keyPrefix string
	ttl       time.Duration
}

func newRedisReserveBackend(client valkey.Client, keyPrefix string, ttl time.Duration) *redisReserveBackend {
	return &redisReserveBackend{client: client, keyPrefix: keyPrefix, ttl: ttl}
}

func (b *redisReserveBackend) key(entity string) string { return b.keyPrefix + entity }

func (b *redisReserveBackend) cmdCtx(parent context.Context) (context.Context, context.CancelFunc) {
	if d, ok := parent.Deadline(); ok && time.Until(d) <= commandTimeout {
		return parent, func() {}
	}
	return context.WithTimeout(parent, commandTimeout)
}

func (b *redisReserveBackend) tryReserve(ctx context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error) {
	cmdCtx, cancel := b.cmdCtx(ctx)
	defer cancel()

	res, err := b.client.Do(cmdCtx, b.client.B().Eval().
		Script(luaTryReserve).
		Numkeys(1).
		Key(b.key(entity)).
		Arg(strconv.FormatFloat(dbSpend, 'f', -1, 64)).
		Arg(strconv.FormatFloat(estimatedCost, 'f', -1, 64)).
		Arg(strconv.FormatFloat(maxBudget, 'f', -1, 64)).
		Arg(strconv.FormatInt(int64(b.ttl.Seconds()), 10)).
		Build()).AsInt64()
	if err != nil {
		return false, err
	}
	return res == 1, nil
}

func (b *redisReserveBackend) reconcile(ctx context.Context, entity string, delta float64) error {
	cmdCtx, cancel := b.cmdCtx(ctx)
	defer cancel()

	key := b.key(entity)
	if err := b.client.Do(cmdCtx, b.client.B().Incrbyfloat().
		Key(key).
		Increment(delta).
		Build()).Error(); err != nil {
		return err
	}
	// Best-effort TTL refresh; a stale TTL is harmless (the key just expires and
	// reseeds from DB on next use), so an error here is not propagated.
	_ = b.client.Do(cmdCtx, b.client.B().Expire().
		Key(key).
		Seconds(int64(b.ttl.Seconds())).
		Build()).Error()
	return nil
}
