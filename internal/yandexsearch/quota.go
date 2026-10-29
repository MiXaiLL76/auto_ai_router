package yandexsearch

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/valkey-io/valkey-go"
)

const (
	quotaRPSWindow  = time.Second
	quotaHourWindow = time.Hour

	// Key TTLs are just an idle-cleanup safety net — ZREMRANGEBYSCORE in the
	// script is what actually enforces the sliding window.
	quotaRPSKeyTTLSeconds  = 5
	quotaHourKeyTTLSeconds = int64(quotaHourWindow/time.Second) + 300
)

// luaFolderQuota atomically admits one request against two sliding-window
// ZSETs sharing the quota's key prefix — the same proven sliding-window
// primitive internal/ratelimit/redis_backend.go uses for RPM/TPM (ZSET +
// ZREMRANGEBYSCORE, no fixed-window boundary-burst problem), sized here for
// a per-second AND per-hour cap checked together, atomically, in one round
// trip: a request is admitted only if BOTH windows have room.
//
// KEYS[1] = rps zset key, KEYS[2] = hour zset key
// ARGV[1] = now (unix ms), ARGV[2] = rps window ms, ARGV[3] = rps limit,
// ARGV[4] = hour window ms, ARGV[5] = hour limit, ARGV[6] = member id,
// ARGV[7] = rps key TTL seconds, ARGV[8] = hour key TTL seconds.
// Returns {admitted(0/1), blocked_by("rps"|"hour"|""), oldest_member_score_ms}.
const luaFolderQuota = `
local rpsKey    = KEYS[1]
local hourKey   = KEYS[2]
local now       = tonumber(ARGV[1])
local rpsWindow = tonumber(ARGV[2])
local rpsLimit  = tonumber(ARGV[3])
local hourWindow = tonumber(ARGV[4])
local hourLimit  = tonumber(ARGV[5])
local member    = ARGV[6]
local rpsTTL    = tonumber(ARGV[7])
local hourTTL   = tonumber(ARGV[8])

redis.call('ZREMRANGEBYSCORE', rpsKey, 0, now - rpsWindow)
redis.call('ZREMRANGEBYSCORE', hourKey, 0, now - hourWindow)

local rpsCount = redis.call('ZCARD', rpsKey)
if rpsCount >= rpsLimit then
  local oldest = redis.call('ZRANGE', rpsKey, 0, 0, 'WITHSCORES')
  local score = 0
  if oldest[2] then score = tonumber(oldest[2]) end
  return {0, 'rps', score}
end

local hourCount = redis.call('ZCARD', hourKey)
if hourCount >= hourLimit then
  local oldest = redis.call('ZRANGE', hourKey, 0, 0, 'WITHSCORES')
  local score = 0
  if oldest[2] then score = tonumber(oldest[2]) end
  return {0, 'hour', score}
end

redis.call('ZADD', rpsKey, now, member)
redis.call('EXPIRE', rpsKey, rpsTTL)
redis.call('ZADD', hourKey, now, member)
redis.call('EXPIRE', hourKey, hourTTL)
return {1, '', 0}
`

// ErrFolderQuotaUnavailable wraps a Redis error from a FolderQuota check.
var ErrFolderQuotaUnavailable = errors.New("yandexsearch: shared folder quota unavailable")

// FolderQuota enforces Yandex's own quota for one folder_id — 1 RPS plus an
// hourly cap, confirmed to be scoped to the folder, not per caller — shared
// across every router replica via Redis. Unlike the per-key RPM limiter
// (internal/ratelimit.RPMLimiter, in-process, fine for per-client fairness)
// this guards a scarce external resource the whole fleet shares, so it
// fails CLOSED on a Redis error (see Allow) rather than open.
type FolderQuota struct {
	client      valkey.Client
	rpsKey      string
	hourKey     string
	rps         int
	hourlyLimit int
}

// NewFolderQuota builds a FolderQuota. keyPrefix should be dedicated to
// this feature (not shared with other Redis consumers) since it owns two
// raw ZSET keys under it.
func NewFolderQuota(client valkey.Client, keyPrefix string, rps, hourlyLimit int) *FolderQuota {
	return &FolderQuota{
		client:      client,
		rpsKey:      keyPrefix + "rps",
		hourKey:     keyPrefix + "hour",
		rps:         rps,
		hourlyLimit: hourlyLimit,
	}
}

// Allow atomically admits one request against the shared RPS+hourly quota.
//
// On a Redis error, Allow denies (allowed=false) rather than failing open:
// over-admitting here risks Yandex throttling or banning the whole
// account's folder_id for every client, not just the current request — a
// worse outcome than this one endpoint returning 503 while Redis recovers.
// This deliberately diverges from internal/litellmdb/budget.Reserver's
// fail-open convention (which protects a per-caller budget, where
// availability reasonably wins).
func (q *FolderQuota) Allow(ctx context.Context) (allowed bool, retryAfter time.Duration, err error) {
	if q == nil || q.client == nil {
		return false, 0, fmt.Errorf("%w: not configured", ErrFolderQuotaUnavailable)
	}
	now := time.Now().UnixMilli()
	member := uuid.New().String()

	reply := q.client.Do(ctx, q.client.B().Eval().
		Script(luaFolderQuota).
		Numkeys(2).
		Key(q.rpsKey).
		Key(q.hourKey).
		Arg(fmt.Sprintf("%d", now)).
		Arg(fmt.Sprintf("%d", quotaRPSWindow.Milliseconds())).
		Arg(fmt.Sprintf("%d", q.rps)).
		Arg(fmt.Sprintf("%d", quotaHourWindow.Milliseconds())).
		Arg(fmt.Sprintf("%d", q.hourlyLimit)).
		Arg(member).
		Arg(fmt.Sprintf("%d", quotaRPSKeyTTLSeconds)).
		Arg(fmt.Sprintf("%d", quotaHourKeyTTLSeconds)).
		Build())

	parts, err := reply.ToArray()
	if err != nil {
		return false, 0, fmt.Errorf("%w: %w", ErrFolderQuotaUnavailable, err)
	}
	if len(parts) != 3 {
		return false, 0, fmt.Errorf("%w: unexpected response shape", ErrFolderQuotaUnavailable)
	}

	admitted, err := parts[0].ToInt64()
	if err != nil {
		return false, 0, fmt.Errorf("%w: %w", ErrFolderQuotaUnavailable, err)
	}
	if admitted == 1 {
		return true, 0, nil
	}

	blockedBy, _ := parts[1].ToString()
	oldestMS, _ := parts[2].ToInt64()
	window := quotaHourWindow
	if blockedBy == "rps" {
		window = quotaRPSWindow
	}
	retryMS := oldestMS + window.Milliseconds() - now
	if retryMS < 0 {
		retryMS = 0
	}
	return false, time.Duration(retryMS) * time.Millisecond, nil
}
