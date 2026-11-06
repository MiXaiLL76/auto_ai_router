package deepseektools

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/valkey-io/valkey-go"
)

// DeepSeek's account-level QPS limits for the two tools, per the vendor's
// guide: the limits are per account (not per key), the two are independent
// (traffic to one does not consume the other's quota), and Fetch is
// additionally capped per domain.
const (
	// DefaultSearchQPS is the account-level Search limit.
	DefaultSearchQPS = 100
	// DefaultFetchQPS is the account-level Fetch limit.
	DefaultFetchQPS = 25
)

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

// Quota admits Search and Fetch calls against per-account QPS limits.
//
// The two tools have separate pools, and each pool is keyed by the DeepSeek
// ACCOUNT (not the key): DeepSeek counts the limit per account regardless of
// which key made the call, and the router must not exceed it by spreading
// traffic over several keys of the same account -- hence one shared Redis key
// per account and tool.
//
// A nil client means no Redis: the limiter then keeps a local in-process
// count. That is weaker (each replica has its own budget) but never blocks
// the feature outright; a Redis error is treated the same way, so a Redis
// outage degrades the limit rather than failing every search.
type Quota struct {
	client valkey.Client
	// keyPrefix namespaces the Redis keys.
	keyPrefix string
	// searchQPS and fetchQPS are the per-account limits, -1 for unlimited.
	searchQPS int
	fetchQPS  int

	// localMu guards the fallback counters used when Redis is unavailable.
	localMu sync.Mutex
	local   map[string]*localWindow
}

// localWindow is the fallback sliding window: the timestamps of the
// admitted requests inside the current second.
type localWindow struct {
	stamps []int64
}

// NewQuota builds a Quota. A nil client selects the local fallback.
func NewQuota(client valkey.Client, keyPrefix string, searchQPS, fetchQPS int) *Quota {
	if searchQPS <= 0 {
		searchQPS = DefaultSearchQPS
	}
	if fetchQPS <= 0 {
		fetchQPS = DefaultFetchQPS
	}
	return &Quota{
		client:    client,
		keyPrefix: keyPrefix,
		searchQPS: searchQPS,
		fetchQPS:  fetchQPS,
		local:     make(map[string]*localWindow),
	}
}

// AllowSearch admits one Search call for the given account.
func (q *Quota) AllowSearch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.allow(ctx, "search", account, q.searchQPS)
}

// AllowFetch admits one Fetch call for the given account.
func (q *Quota) AllowFetch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.allow(ctx, "fetch", account, q.fetchQPS)
}

// allow admits one call against the named pool, reporting whether it was
// admitted and, when it was not, how long the caller should wait.
func (q *Quota) allow(ctx context.Context, pool, account string, limit int) (bool, time.Duration, error) {
	if limit <= 0 {
		return true, 0, nil
	}
	key := q.keyPrefix + pool + ":" + account
	if q.client == nil {
		return q.allowLocal(key, limit), 0, nil
	}
	now := time.Now()
	reply := q.client.Do(ctx, q.client.B().Eval().
		Script(luaQuota).
		Numkeys(1).
		Key(key).
		Arg(strconv.FormatInt(now.UnixMilli(), 10)).
		Arg(strconv.FormatInt(quotaWindow.Milliseconds(), 10)).
		Arg(strconv.Itoa(limit)).
		Arg(uuid.NewString()).
		Arg(strconv.Itoa(quotaKeyTTLSeconds)).
		Build())
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
