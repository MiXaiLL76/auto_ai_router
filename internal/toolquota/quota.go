// Package toolquota admits the router's own tool calls against the
// providers' per-account QPS limits.
//
// The providers count their limits per ACCOUNT, not per key, and each tool
// has its own pool: traffic to one tool does not consume another's quota.
// The pools are therefore named, and each pool's key is the account's.
//
// The limiter keeps the pools in Redis, so the limit is shared by the
// router's replicas and by the platforms, and falls back to in-process
// counters when Redis is unavailable. The fallback is weaker (each replica
// has its own budget) but never blocks the feature outright; a Redis error
// degrades the limit the same way, so a Redis outage does not fail every
// search.
package toolquota

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/valkey-io/valkey-go"
)

// quotaWindow is the sliding window the limit is counted over: one second.
const quotaWindow = time.Second

// quotaKeyTTLSeconds is just an idle-cleanup safety net -- the sliding
// window's ZREMRANGEBYSCORE is what actually enforces the limit.
const quotaKeyTTLSeconds = 5

// luaQuota atomically admits one request against a sliding-window ZSET: a
// request is admitted only if fewer than limit requests fall inside the
// window.
//
// KEYS[1] = the zset key
// ARGV[1] = now (unix ms), ARGV[2] = window ms, ARGV[3] = limit,
// ARGV[4] = member id, ARGV[5] = key TTL seconds.
// Returns {admitted(0/1), oldest_member_score_ms}.
const luaQuota = `
local key     = KEYS[1]
local now     = tonumber(ARGV[1])
local window  = tonumber(ARGV[2])
local limit   = tonumber(ARGV[3])
local member  = ARGV[4]
local ttl     = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local score = 0
  if oldest[2] then score = tonumber(oldest[2]) end
  return {0, score}
end

redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
return {1, 0}
`

// scriptQuota runs luaQuota via EVALSHA, falling back to EVAL on NOSCRIPT.
var scriptQuota = valkey.NewLuaScript(luaQuota)

// Quota admits tool calls against per-account QPS limits.
//
// The pools are named, and each pool is keyed by the ACCOUNT (not the key):
// the providers count the limit per account regardless of which key made the
// call, and the router must not exceed it by spreading traffic over several
// keys of the same account -- hence one shared Redis key per account and
// pool.
type Quota struct {
	client valkey.Client
	// keyPrefix namespaces the Redis keys.
	keyPrefix string
	// limits are the per-pool limits, keyed by the pool's name. A limit of
	// zero or less means unlimited.
	limits map[string]int

	// localMu guards the fallback counters used when Redis is unavailable.
	localMu sync.Mutex
	local   map[string]*localWindow
}

// localWindow is the fallback sliding window: the timestamps of the
// admitted requests inside the current second.
type localWindow struct {
	stamps []int64
}

// NewQuota builds a Quota over the named pools. A nil client selects the
// local fallback.
//
// A limit of zero or less is unlimited: the pool admits every call.
func NewQuota(client valkey.Client, keyPrefix string, limits map[string]int) *Quota {
	copied := make(map[string]int, len(limits))
	for pool, limit := range limits {
		copied[pool] = limit
	}
	return &Quota{
		client:    client,
		keyPrefix: keyPrefix,
		limits:    copied,
		local:     make(map[string]*localWindow),
	}
}

// Allow admits one call against the named pool, reporting whether it was
// admitted and, when it was not, how long the caller should wait.
func (q *Quota) Allow(ctx context.Context, pool, account string) (bool, time.Duration, error) {
	limit := q.limits[pool]
	if limit <= 0 {
		return true, 0, nil
	}
	key := q.keyPrefix + pool + ":" + account
	if q.client == nil {
		return q.allowLocal(key, limit), 0, nil
	}
	now := time.Now()
	reply := scriptQuota.Exec(ctx, q.client,
		[]string{key},
		[]string{
			strconv.FormatInt(now.UnixMilli(), 10),
			strconv.FormatInt(quotaWindow.Milliseconds(), 10),
			strconv.Itoa(limit),
			uuid.NewString(),
			strconv.Itoa(quotaKeyTTLSeconds),
		})
	parts, err := reply.ToArray()
	if err != nil || len(parts) < 2 {
		// Redis is unavailable: degrade to the local counter rather than
		// failing every search. The limit is then per replica instead of
		// per account, which is weaker but keeps the feature working.
		return q.allowLocal(key, limit), 0, nil
	}
	admitted, err := parts[0].ToInt64()
	if err != nil {
		return q.allowLocal(key, limit), 0, nil
	}
	if admitted == 1 {
		return true, 0, nil
	}
	oldest, err := parts[1].ToInt64()
	if err != nil {
		return false, quotaWindow, nil
	}
	wait := quotaWindow - time.Duration(now.UnixMilli()-oldest)*time.Millisecond
	if wait < 0 {
		wait = 0
	}
	return false, wait, nil
}

// allowLocal admits one call against the in-process fallback window.
func (q *Quota) allowLocal(key string, limit int) bool {
	q.localMu.Lock()
	defer q.localMu.Unlock()
	now := time.Now().UnixMilli()
	cutoff := now - quotaWindow.Milliseconds()
	window := q.local[key]
	if window == nil {
		window = &localWindow{}
		q.local[key] = window
	}
	// The stamps inside the window are kept in their own slice, so the
	// filter does not write into the slice it reads.
	kept := make([]int64, 0, len(window.stamps)+1)
	for _, stamp := range window.stamps {
		if stamp > cutoff {
			kept = append(kept, stamp)
		}
	}
	if len(kept) >= limit {
		window.stamps = kept
		return false
	}
	kept = append(kept, now)
	window.stamps = kept
	return true
}
