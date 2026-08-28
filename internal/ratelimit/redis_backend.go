package ratelimit

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/valkey-io/valkey-go"
)

const (
	rpmWindow = 60 * time.Second
	tpmWindow = 60 * time.Second

	// maxRetries is the maximum number of extra attempts on transient network errors.
	maxRetries = 2

	// defaultCommandTimeout caps the duration of a single Redis command when no
	// tighter deadline already exists on the context.
	defaultCommandTimeout = 3 * time.Second
)

// Lua script: atomically check+record RPM in a ZSET sliding window.
// KEYS[1] = rpm key
// ARGV[1] = now (unix ms as string)
// ARGV[2] = window size in ms
// ARGV[3] = limit (-1 = unlimited)
// ARGV[4] = unique member id
// Returns 1 if allowed, 0 if rejected.
const luaTryAllowRPM = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if limit ~= -1 and count >= limit then
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, tonumber(ARGV[5]))
return 1
`

// Lua script: check RPM without recording.
const luaCanAllowRPM = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if limit ~= -1 and count >= limit then
  return 0
end
return 1
`

// Lua script: check TPM (sum of token counts in window) without recording.
// Token entries are stored as "uuid:count" members with score = timestamp_ms.
const luaCanAllowTPM = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
if limit == -1 then return 1 end
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local members = redis.call('ZRANGE', key, 0, -1)
local total = 0
for _, m in ipairs(members) do
  local sep = string.find(m, ':', 1, true)
  if sep then
    total = total + tonumber(string.sub(m, sep + 1)) or 0
  end
end
if total >= limit then return 0 end
return 1
`

// Lua script: record token consumption.
// ARGV[4] = "uuid:count" member
//
//nolint:gosec // G101: false positive — Lua script text, not a credential
const luaConsumeTokens = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local member = ARGV[3]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, tonumber(ARGV[4]))
return 1
`

// Lua script: get current RPM count.
const luaCurrentRPM = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
return redis.call('ZCARD', key)
`

// Lua script: get current TPM sum.
const luaCurrentTPM = `
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local members = redis.call('ZRANGE', key, 0, -1)
local total = 0
for _, m in ipairs(members) do
  local sep = string.find(m, ':', 1, true)
  if sep then
    total = total + tonumber(string.sub(m, sep + 1)) or 0
  end
end
return total
`

// Lua script: atomic TryAllowAll — check cred RPM+TPM + optional model RPM+TPM,
// record cred+model RPM only if all checks pass.
//
// KEYS[1] = cred rpm key
// KEYS[2] = cred tpm key
// KEYS[3] = model rpm key (present only when model limits are configured)
// KEYS[4] = model tpm key (present only when model limits are configured)
// ARGV[1] = now ms
// ARGV[2] = window ms
// ARGV[3] = cred rpm limit
// ARGV[4] = cred tpm limit
// ARGV[5] = model rpm limit
// ARGV[6] = model tpm limit
// ARGV[7] = cred rpm member (uuid)
// ARGV[8] = model rpm member (uuid)
// ARGV[9] = ttl (seconds)
// Returns 1 allowed, 0 rejected.
const luaTryAllowAll = `
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local ttl    = tonumber(ARGV[9])

local function check_rpm(key, limit)
  redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
  local count = redis.call('ZCARD', key)
  if limit ~= -1 and count >= limit then return false end
  return true
end

local function check_tpm(key, limit)
  if limit == -1 then return true end
  redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
  local members = redis.call('ZRANGE', key, 0, -1)
  local total = 0
  for _, m in ipairs(members) do
    local sep = string.find(m, ':', 1, true)
    if sep then
      total = total + (tonumber(string.sub(m, sep + 1)) or 0)
    end
  end
  if total >= limit then return false end
  return true
end

local function record_rpm(key, member)
  redis.call('ZADD', key, now, member)
  redis.call('EXPIRE', key, ttl)
end

-- Check all limits first.
if not check_rpm(KEYS[1], tonumber(ARGV[3])) then return 0 end
if not check_tpm(KEYS[2], tonumber(ARGV[4])) then return 0 end
if #KEYS >= 3 then
  if not check_rpm(KEYS[3], tonumber(ARGV[5])) then return 0 end
  if not check_tpm(KEYS[4], tonumber(ARGV[6])) then return 0 end
end

-- All passed — record RPM.
record_rpm(KEYS[1], ARGV[7])
if #KEYS >= 3 then
  record_rpm(KEYS[3], ARGV[8])
end
return 1
`

// RedisBackend implements counterBackend using Valkey/Redis.
// It is exported so that the underlying valkey.Client can be reused by other
// packages (e.g. responsestore) via Client().
type RedisBackend struct {
	client         valkey.Client
	keyPrefix      string
	keyTTL         int           // seconds
	commandTimeout time.Duration // per-command deadline cap
}

// NewValkeyClient creates a valkey.Client from RedisConfig.
// Callers that need to share one connection across multiple subsystems should
// call this once and pass the resulting client to each subsystem's constructor.
func NewValkeyClient(cfg config.RedisConfig) (valkey.Client, error) {
	opt := valkey.ClientOption{
		InitAddress:       cfg.InitAddresses,
		Username:          cfg.Username,
		Password:          cfg.Password,
		SelectDB:          cfg.SelectDB,
		ForceSingleClient: cfg.ForceSingleClient,
		ConnWriteTimeout:  cfg.ConnWriteTimeout,
		Dialer:            net.Dialer{Timeout: cfg.ConnectTimeout},
	}
	if cfg.TLSEnabled {
		opt.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}
	client, err := valkey.NewClient(opt)
	if err != nil {
		return nil, fmt.Errorf("redis: failed to create client: %w", err)
	}
	return client, nil
}

// NewRedisBackend creates a Redis/Valkey-backed counterBackend from config.
func NewRedisBackend(cfg config.RedisConfig) (*RedisBackend, error) {
	client, err := NewValkeyClient(cfg)
	if err != nil {
		return nil, err
	}
	keyTTL := cfg.KeyTTL
	if keyTTL == 0 {
		keyTTL = 120
	}
	cmdTimeout := cfg.CommandTimeout
	if cmdTimeout == 0 {
		cmdTimeout = defaultCommandTimeout
	}
	b := NewRedisBackendFromClientWithTTL(client, cfg.KeyPrefix, keyTTL)
	b.commandTimeout = cmdTimeout
	return b, nil
}

// NewRedisBackendFromClient wraps an existing valkey.Client with default TTL and command timeout.
// Useful when a single client is shared between rate limiting and other stores.
func NewRedisBackendFromClient(client valkey.Client, keyPrefix string) *RedisBackend {
	return NewRedisBackendFromClientWithTTL(client, keyPrefix, 120)
}

// NewRedisBackendFromClientWithTTL wraps an existing valkey.Client with custom TTL.
func NewRedisBackendFromClientWithTTL(client valkey.Client, keyPrefix string, keyTTL int) *RedisBackend {
	return &RedisBackend{
		client:         client,
		keyPrefix:      keyPrefix,
		keyTTL:         keyTTL,
		commandTimeout: defaultCommandTimeout,
	}
}

// Client returns the underlying valkey.Client so it can be shared with other components.
func (b *RedisBackend) Client() valkey.Client { return b.client }

// Close shuts down the underlying Valkey client.
func (b *RedisBackend) Close() { b.client.Close() }

// Ping performs a health check on the Redis connection.
func (b *RedisBackend) Ping(ctx context.Context) error {
	return b.client.Do(ctx, b.client.B().Ping().Build()).Error()
}

func (b *RedisBackend) rpmKey(key string) string { return b.keyPrefix + "rpm:" + b.hashTag(key) }
func (b *RedisBackend) tpmKey(key string) string { return b.keyPrefix + "tpm:" + b.hashTag(key) }

// hashTag wraps key in a Redis hash tag so that all keys for the same credential
// land in the same hash slot. This is required by valkey-go's multi-key EVAL slot
// validation (enforced even on single-node deployments).
//
// Credential keys ("c:foo")     → "{c:foo}"
// Model keys     ("m:foo:bar")  → "{c:foo}:m:foo:bar"  (inherits cred's slot)
// Other keys                    → "{key}"
func (b *RedisBackend) hashTag(key string) string {
	if strings.HasPrefix(key, "m:") {
		rest := key[2:] // "credname:modelname"
		if i := strings.IndexByte(rest, ':'); i != -1 {
			return "{c:" + rest[:i] + "}:" + key
		}
	}
	return "{" + key + "}"
}

func nowMS() int64 { return time.Now().UTC().UnixMilli() }

// cmdCtx returns a context bounded by b.commandTimeout when the parent has no
// tighter deadline. Always call the returned cancel function.
func (b *RedisBackend) cmdCtx(parent context.Context) (context.Context, context.CancelFunc) {
	if b.commandTimeout <= 0 {
		return parent, func() {}
	}
	if d, ok := parent.Deadline(); ok && time.Until(d) <= b.commandTimeout {
		return parent, func() {}
	}
	return context.WithTimeout(parent, b.commandTimeout)
}

// isTransientError returns true when the error indicates a recoverable network
// condition and retrying the operation is safe.
//
// Timeouts are NOT retried: a timed-out write may have already been committed
// on the server, so retrying could cause double-counting. Context cancellation
// means the caller has given up and should not be retried either.
func isTransientError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	// io.EOF: server closed the connection; safe to reconnect and retry.
	if errors.Is(err, io.EOF) {
		return true
	}
	// net.Error with Timeout() == true: read timed out, command may have executed.
	// net.Error with Timeout() == false: connection setup failed; safe to retry.
	var netErr net.Error
	if errors.As(err, &netErr) {
		return !netErr.Timeout()
	}
	return false
}

// doWithRetry executes fn, retrying up to maxRetries times on transient errors.
// Each attempt uses a per-command timeout via cmdCtx.
//
// The uuid member used by write operations (tryAllowRPM, tryAllowAll) is
// captured by the caller before invoking doWithRetry. Because Redis ZADD
// with the same member only updates its score, a retry after a silent success
// does not double-count the entry — the net effect is a single recorded event
// with the latest timestamp.
func (b *RedisBackend) doWithRetry(ctx context.Context, fn func(context.Context) (int64, error)) (int64, error) {
	var (
		res int64
		err error
	)
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt) * 20 * time.Millisecond
			select {
			case <-ctx.Done():
				return 0, ctx.Err()
			case <-time.After(backoff):
			}
		}
		cmdCtx, cancel := b.cmdCtx(ctx)
		res, err = fn(cmdCtx)
		cancel()
		if err == nil || !isTransientError(err) {
			return res, err
		}
	}
	return res, err
}

func (b *RedisBackend) tryAllowRPM(ctx context.Context, key string, limit int) bool {
	now := nowMS()
	member := uuid.New().String()
	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return b.client.Do(ctx, b.client.B().Eval().
			Script(luaTryAllowRPM).
			Numkeys(1).
			Key(b.rpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", rpmWindow.Milliseconds())).
			Arg(fmt.Sprintf("%d", limit)).
			Arg(member).
			Arg(fmt.Sprintf("%d", b.keyTTL)).
			Build()).AsInt64()
	})
	if err != nil {
		return false
	}
	return res == 1
}

func (b *RedisBackend) canAllowRPM(ctx context.Context, key string, limit int) bool {
	now := nowMS()
	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return b.client.Do(ctx, b.client.B().Eval().
			Script(luaCanAllowRPM).
			Numkeys(1).
			Key(b.rpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", rpmWindow.Milliseconds())).
			Arg(fmt.Sprintf("%d", limit)).
			Build()).AsInt64()
	})
	if err != nil {
		return false
	}
	return res == 1
}

func (b *RedisBackend) canAllowTPM(ctx context.Context, key string, limit int) bool {
	if limit == -1 {
		return true
	}
	now := nowMS()
	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return b.client.Do(ctx, b.client.B().Eval().
			Script(luaCanAllowTPM).
			Numkeys(1).
			Key(b.tpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", tpmWindow.Milliseconds())).
			Arg(fmt.Sprintf("%d", limit)).
			Build()).AsInt64()
	})
	if err != nil {
		return false
	}
	return res == 1
}

func (b *RedisBackend) consumeTokens(ctx context.Context, key string, tokenCount int) {
	now := nowMS()
	member := fmt.Sprintf("%s:%d", uuid.New().String(), tokenCount)
	_, _ = b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return 0, b.client.Do(ctx, b.client.B().Eval().
			Script(luaConsumeTokens).
			Numkeys(1).
			Key(b.tpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", tpmWindow.Milliseconds())).
			Arg(member).
			Arg(fmt.Sprintf("%d", b.keyTTL)).
			Build()).Error()
	})
}

func (b *RedisBackend) currentRPM(ctx context.Context, key string) int {
	now := nowMS()
	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return b.client.Do(ctx, b.client.B().Eval().
			Script(luaCurrentRPM).
			Numkeys(1).
			Key(b.rpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", rpmWindow.Milliseconds())).
			Build()).AsInt64()
	})
	if err != nil {
		return 0
	}
	return int(res)
}

func (b *RedisBackend) currentTPM(ctx context.Context, key string) int {
	now := nowMS()
	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		return b.client.Do(ctx, b.client.B().Eval().
			Script(luaCurrentTPM).
			Numkeys(1).
			Key(b.tpmKey(key)).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", tpmWindow.Milliseconds())).
			Build()).AsInt64()
	})
	if err != nil {
		return 0
	}
	return int(res)
}

func (b *RedisBackend) tryAllowAll(
	ctx context.Context,
	credKey string, credRPM, credTPM int,
	modelKey string, modelRPM, modelTPM int,
) bool {
	now := nowMS()
	credMember := uuid.New().String()
	modelMember := uuid.New().String()

	credRPMKey := b.rpmKey(credKey)
	credTPMKey := b.tpmKey(credKey)

	res, err := b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
		evalCmd := b.client.B().Eval().
			Script(luaTryAllowAll)

		if modelKey != "" {
			modRPMKey := b.rpmKey(modelKey)
			modTPMKey := b.tpmKey(modelKey)
			return b.client.Do(ctx, evalCmd.
				Numkeys(4).
				Key(credRPMKey).
				Key(credTPMKey).
				Key(modRPMKey).
				Key(modTPMKey).
				Arg(fmt.Sprintf("%d", now)).
				Arg(fmt.Sprintf("%d", rpmWindow.Milliseconds())).
				Arg(fmt.Sprintf("%d", credRPM)).
				Arg(fmt.Sprintf("%d", credTPM)).
				Arg(fmt.Sprintf("%d", modelRPM)).
				Arg(fmt.Sprintf("%d", modelTPM)).
				Arg(credMember).
				Arg(modelMember).
				Arg(fmt.Sprintf("%d", b.keyTTL)).
				Build()).AsInt64()
		}
		return b.client.Do(ctx, evalCmd.
			Numkeys(2).
			Key(credRPMKey).
			Key(credTPMKey).
			Arg(fmt.Sprintf("%d", now)).
			Arg(fmt.Sprintf("%d", rpmWindow.Milliseconds())).
			Arg(fmt.Sprintf("%d", credRPM)).
			Arg(fmt.Sprintf("%d", credTPM)).
			Arg(fmt.Sprintf("%d", modelRPM)).
			Arg(fmt.Sprintf("%d", modelTPM)).
			Arg(credMember).
			Arg(modelMember).
			Arg(fmt.Sprintf("%d", b.keyTTL)).
			Build()).AsInt64()
	})
	if err != nil {
		return false
	}
	return res == 1
}

// batchCurrentStats returns [rpm, tpm] for each key in a single pipeline round-trip.
func (b *RedisBackend) batchCurrentStats(ctx context.Context, keys []string) map[string][2]int {
	out, _ := b.batchCurrentStatsErr(ctx, keys)
	return out
}

// batchCurrentStatsErr is like batchCurrentStats but also surfaces the first
// Redis command error encountered, for callers that need connectivity
// visibility (e.g. HybridBackend.doSync — see redis_todo.md item 3).
func (b *RedisBackend) batchCurrentStatsErr(ctx context.Context, keys []string) (map[string][2]int, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	now := nowMS()
	nowStr := fmt.Sprintf("%d", now)
	windowStr := fmt.Sprintf("%d", rpmWindow.Milliseconds())

	cmds := make([]valkey.Completed, 0, len(keys)*2)
	for _, key := range keys {
		cmds = append(cmds,
			b.client.B().Eval().
				Script(luaCurrentRPM).Numkeys(1).
				Key(b.rpmKey(key)).Arg(nowStr).Arg(windowStr).
				Build(),
			b.client.B().Eval().
				Script(luaCurrentTPM).Numkeys(1).
				Key(b.tpmKey(key)).Arg(nowStr).Arg(windowStr).
				Build(),
		)
	}

	cmdCtx, cancel := b.cmdCtx(ctx)
	defer cancel()

	results := b.client.DoMulti(cmdCtx, cmds...)

	out := make(map[string][2]int, len(keys))
	var firstErr error
	for i, key := range keys {
		var rpm, tpm int64
		if v, err := results[i*2].AsInt64(); err == nil {
			rpm = v
		} else if firstErr == nil {
			firstErr = err
		}
		if v, err := results[i*2+1].AsInt64(); err == nil {
			tpm = v
		} else if firstErr == nil {
			firstErr = err
		}
		out[key] = [2]int{int(rpm), int(tpm)}
	}
	return out, firstErr
}

// setCurrentUsage is a no-op for the Redis backend: all replicas write to the
// shared Redis instance directly, so remote-sync is unnecessary.
func (b *RedisBackend) setCurrentUsage(_ context.Context, _ string, _, _ int) {}

func (b *RedisBackend) deleteKey(ctx context.Context, key string) {
	for _, redisKey := range []string{b.rpmKey(key), b.tpmKey(key)} {
		_, _ = b.doWithRetry(ctx, func(ctx context.Context) (int64, error) {
			return 0, b.client.Do(ctx, b.client.B().Del().Key(redisKey).Build()).Error()
		})
	}
}
