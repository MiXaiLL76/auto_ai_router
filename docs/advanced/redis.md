# Redis / Valkey Integration

Auto AI Router supports an optional Redis (or [Valkey](https://valkey.io/)) backend that enables three features when running multiple replicas:

| Feature                                                          | Without Redis                                                                       | With Redis                                                         |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Rate limiting** (RPM/TPM)                                      | Per-pod counters — each replica enforces limits independently                       | Global counters — limits enforced across the whole cluster         |
| **Budget reservation** (`litellm_db.enforce_budget_reservation`) | DB-snapshot check only (see [LiteLLM auth](../litellm-integration/litellm_auth.md)) | Atomic pre-reservation, closing the pre-check-vs-actual-spend race |
| **Response storage** (`store: true`)                             | Local bbolt file — not accessible from other pods                                   | Shared Redis — any replica can retrieve stored responses           |

If Redis is not configured, all three fall back to their original in-process (or DB-snapshot-only) implementations automatically.

## When to use Redis

| Scenario                              | Recommended mode                                  |
| ------------------------------------- | ------------------------------------------------- |
| Single replica, Redis low-latency     | `hybrid: false` (direct Redis, exact counts)      |
| Single replica, Redis high-latency    | `hybrid: true` (local decisions, async sync)      |
| Multiple replicas, need exact limits  | `hybrid: false`                                   |
| Multiple replicas, slight drift is OK | `hybrid: true` (better throughput, ~5 s accuracy) |
| No Redis at all                       | Omit `redis` section — pure in-process counters   |

Single-replica deployments that don't need shared response storage can omit Redis entirely.

## Configuration

Add a `redis` section to `config.yaml`:

```yaml
redis:
  enabled: true
  addresses:
    - "redis:6379"            # host:port of your Redis/Valkey instance
  password: "os.environ/REDIS_PASSWORD"   # optional; supports env variable syntax
  key_prefix: "rl:"           # namespace prefix for all keys (default: "rl:")
  force_single_client: true   # set false only for Redis Cluster
  connect_timeout: 5s
  conn_write_timeout: 10s
  command_timeout: 3s         # per-command deadline cap (default: 3s)
  key_ttl: 120                # rate-limit key TTL in seconds (default: 120)
  hybrid: false               # see "Hybrid mode" below
  sync_interval: 5s           # hybrid only: how often to pull remote counts
```

### All Parameters

| Parameter             | Type     | Default | Description                                           |
| --------------------- | -------- | ------- | ----------------------------------------------------- |
| `enabled`             | bool     | `false` | Enable Redis backend                                  |
| `addresses`           | []string | —       | One or more `host:port` addresses                     |
| `username`            | string   | —       | Redis ACL username (optional)                         |
| `password`            | string   | —       | Redis AUTH password (optional)                        |
| `select_db`           | int      | `0`     | Redis database index                                  |
| `key_prefix`          | string   | `"rl:"` | Prefix prepended to every key                         |
| `tls_enabled`         | bool     | `false` | Enable TLS                                            |
| `connect_timeout`     | duration | `5s`    | TCP dial timeout                                      |
| `conn_write_timeout`  | duration | `10s`   | Per-connection write/pipeline timeout                 |
| `force_single_client` | bool     | `false` | Skip cluster detection (use for single-node)          |
| `command_timeout`     | duration | `3s`    | Maximum duration for a single Redis command           |
| `key_ttl`             | int      | `120`   | Rate-limit key TTL in seconds                         |
| `hybrid`              | bool     | `false` | Enable hybrid mode (see below)                        |
| `sync_interval`       | duration | `5s`    | Hybrid only: interval between Redis sync pulls        |
| `min_idle_conns`      | int      | `10`    | Minimum idle connections (reserved for future use)    |
| `max_idle_conns`      | int      | `100`   | Maximum idle connections (reserved for future use)    |
| `max_conn_lifetime`   | duration | `30m`   | Maximum connection lifetime (reserved for future use) |

All string values support the `os.environ/VAR_NAME` syntax for environment variable substitution.

### Minimal config for single-node Valkey

```yaml
redis:
  enabled: true
  addresses:
    - "valkey:6379"
  force_single_client: true
```

### Config with password via environment variable

```yaml
redis:
  enabled: true
  addresses:
    - "os.environ/REDIS_ADDRESS"
  password: "os.environ/REDIS_PASSWORD"
  force_single_client: true
```

## Startup Health Check

On startup, the router connects to Redis and immediately performs a `PING` health check:

- If Redis is **reachable** → rate limiter and response store use Redis.
- If Redis is **unreachable** (connection error or ping timeout) → both features silently fall back to their in-process implementations. The server starts normally.

## Hybrid Mode

`hybrid` applies to both the RPM/TPM rate limiter and budget reservation (`litellm_db.enforce_budget_reservation`) — one flag, one `sync_interval`, backed by independent `HybridBackend` implementations (`internal/ratelimit` and `internal/litellmdb/budget` respectively) that share the same design.

When `hybrid: true`, the router uses a **HybridBackend** that combines an in-process local counter with an asynchronous Redis sync:

```
Request arrives
      │
      ▼
Local counter (in-memory, <1 µs)
  • TryAllowAll / tryAllowRPM / canAllowTPM
  • Decision is instant — no network call
      │
      ├── allowed ──► enqueue write op
      │                    │
      │               Background writeWorker
      │               (batches 200 ops, flushes every 100 ms via DoMulti pipeline)
      │                    │
      │                    ▼
      │               Redis (async, non-blocking)
      │
      └── denied ──► return 429 immediately
```

**Background sync** (every `sync_interval`, default 5 s):

1. Fetch total RPM/TPM for all tracked keys from Redis (one pipeline round-trip).
2. Subtract local counts → `remote_count = redis_total − local_total`.
3. Store `remote_count` in memory.
4. Next rate-limit check uses `effective_limit = limit − remote_count`, so traffic from other replicas is accounted for.

### Trade-offs

| Property                    | Direct Redis (`hybrid: false`) | Hybrid (`hybrid: true`)                        |
| --------------------------- | ------------------------------ | ---------------------------------------------- |
| Latency per request         | +1 RTT to Redis                | ~0 (in-memory)                                 |
| `/health` endpoint latency  | 1 pipeline RTT (batched)       | ~0 (in-memory)                                 |
| Cross-instance accuracy     | Exact (atomic Lua scripts)     | ±`sync_interval` drift (default ±5 s)          |
| Redis unavailability impact | Requests blocked until timeout | Continues with local counters                  |
| Write load on Redis         | 1–2 commands per request       | Batched async; typically 1 pipeline per 100 ms |

Use `hybrid: false` when you need hard rate-limit enforcement across replicas with zero tolerance for drift. Use `hybrid: true` when latency matters more than exact cross-replica synchronisation.

### Budget reservation specifics

The budget `HybridBackend` differs from the rate limiter's in how it writes to Redis, because a lost budget delta never ages out of a sliding window — it would permanently skew every replica's view of the entity's spend:

- **No dropped writes.** Deltas accumulate per entity in memory until Redis acknowledges them; there is no bounded queue to overflow. A failed flush keeps the delta owed and retries with exponential backoff (100 ms → 5 s cap). Only deltas still unacknowledged at shutdown are lost (logged as a warning).
- **Idempotent retries.** Each write carries a per-replica sequence number, remembered in `<key_prefix>litellmbudget:{<entity>}:w:<replica-id>`. A retry of a write whose reply was lost (timeout, dropped connection) is ignored by Redis instead of being counted twice.
- **Own spend vs. other replicas.** Sync subtracts only this replica's *acknowledged* spend from the Redis total, so deltas still in flight never mask other replicas' spend.
- **Key loss recovery.** If sync finds the counter `<key_prefix>litellmbudget:{<entity>}` gone (Redis restart without persistence, eviction) while this replica has spend in it, the next flush re-seeds it with this replica's DB snapshot plus acknowledged spend.
- **Idle eviction.** Entities unused for longer than `litellm_db.budget_reservation_ttl` that owe Redis nothing are dropped from memory and from the sync pipeline; the next request reseeds them from the DB.

## Key Layout

All keys are namespaced under `key_prefix` (default `rl:`):

| Key pattern                            | Used for                                    |
| -------------------------------------- | ------------------------------------------- |
| `rl:rpm:{c:credname}`                  | Credential request count ZSET (sliding 60s) |
| `rl:tpm:{c:credname}`                  | Credential token count ZSET (sliding 60s)   |
| `rl:rpm:{c:credname}:m:credname:model` | Model request count ZSET (sliding 60s)      |
| `rl:tpm:{c:credname}:m:credname:model` | Model token count ZSET (sliding 60s)        |
| `rl:response:{id}`                     | Stored Responses API entry                  |

The `{c:credname}` portion is a Redis **hash tag** — it ensures all four keys for a given credential (`cred rpm`, `cred tpm`, `model rpm`, `model tpm`) land in the same hash slot. This is required by valkey-go's multi-key `EVAL` slot validation, which is enforced even on single-node deployments.

Rate-limit keys expire after `key_ttl` seconds of inactivity (default **120 seconds**) via Redis `EXPIRE`. Response keys use the TTL from the `ttl` field of the request, or persist indefinitely when `ttl: 0`.

## How Rate Limiting Works in Redis

Each request is recorded atomically using a **Lua script** that runs entirely on the Redis server:

1. Remove entries older than 60 seconds from the sorted set (`ZREMRANGEBYSCORE`)
2. Count remaining entries (`ZCARD`)
3. If count ≥ limit → reject (return 0)
4. Add new entry with current timestamp as score and a UUID as member (`ZADD`)
5. Reset key TTL (`EXPIRE`)

The `TryAllowAll` check (credential RPM + credential TPM + model RPM + model TPM) is a single Lua script that validates all four counters atomically before recording anything — no TOCTOU race conditions across replicas.

Token consumption (`ConsumeTokens`) stores entries as `uuid:count` members so the TPM check can sum token counts with `ZRANGE` inside a Lua script.

### Batched Reads (Health & Metrics)

The `/health` endpoint and the metrics updater need current RPM/TPM for every credential and model. Instead of issuing one Redis command per key, the router sends all read commands in a single **pipeline** (`DoMulti`) — one network round-trip regardless of the number of credentials or models configured. This keeps `/health` response time proportional to Redis RTT, not to the size of the configuration.

## Timeouts

Two independent timeout layers protect against slow Redis:

| Layer             | Config field      | Default | Scope                                                                |
| ----------------- | ----------------- | ------- | -------------------------------------------------------------------- |
| Operation timeout | —                 | `30s`   | Applied by the rate limiter when the request context has no deadline |
| Command timeout   | `command_timeout` | `3s`    | Applied per Redis command inside the backend                         |

The command timeout is applied only when the parent context deadline is farther away than `command_timeout`. This ensures a single slow Redis call does not block a request for the full operation timeout.

## Retry Behavior

Redis operations are automatically retried on **transient network errors** (connection reset, broken pipe, `io.EOF`). Up to **2 retries** are attempted with a short exponential backoff (20 ms, 40 ms).

Retries are **not** performed on:

- Context cancellation or deadline exceeded (the caller already gave up)
- Network timeouts during command execution (the command may have already been committed on the server)
- Redis protocol errors (e.g., `WRONGTYPE`, script errors)

**Idempotency of write operations:** Each rate-limit entry uses a UUID as the ZSET member. If a retry sends the same command after a silent success, Redis `ZADD` updates the score (timestamp) of the existing member rather than inserting a duplicate — so requests are never double-counted.

> In hybrid mode, writes go through the async queue and are not retried individually. If a batch fails the affected entries are simply lost. The next sync cycle will re-read the true Redis total and correct the remote-count estimate.

## Memory Sizing

A rough guide for the rate-limit keyspace:

- Each request adds one entry to the RPM sorted set (UUID string ≈ 50 bytes + overhead ≈ ~100 bytes per entry).
- At 1000 RPM sustained across 10 credentials × 20 models = 200 keys, each holding up to 60 000 entries ≈ **~1.2 GB** in the worst case. In practice, entries expire within 60 seconds so live memory is much lower.

For the response store, size depends on average response payload. A typical 2 KB response at 10 000 stored responses ≈ **~20 MB**.

Start with `--maxmemory 256mb` and adjust based on observed usage.

## Limitations

- **Redis Cluster**: only standalone and basic single-node deployments are supported. Cluster mode is not supported (keys in multi-key Lua scripts must share a hash slot).
- **Sentinel**: not supported. Use a load-balancer in front of Redis for HA.
- **Pool settings** (`min_idle_conns`, `max_idle_conns`, `max_conn_lifetime`): parsed and reserved for future use; the valkey-go client manages its own connection pool internally.
- **Hybrid mode and response store**: `hybrid` only affects rate limiting and budget reservation. The response store always talks to Redis directly (reads must be synchronous).
