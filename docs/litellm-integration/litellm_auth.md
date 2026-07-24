# LiteLLM Auth & Billing Flow

How Auto AI Router authenticates a request against a LiteLLM Postgres database, enforces budgets/rate limits/model access, and logs spend afterwards. See [litellm_db.md](litellm_db.md) for connection/config basics and [kafka_spend_log.md](kafka_spend_log.md) for the analytics write-path.

## Overview

```mermaid
flowchart LR
    Client([Client]) -->|Bearer sk-...| Proxy[Auto AI Router]
    Proxy -->|1: authenticateRequest| Auth[Token validation<br/>+ budget snapshot]
    Proxy -->|2: parse body| Model[Model resolution<br/>alias -> real model]
    Proxy -->|3: orchestrateRequest| Enforce[Model allow-list<br/>+ budget reservation<br/>+ RPM/TPM]
    Proxy -->|4| Upstream[(Provider)]
    Proxy -.->|5: async, after response| SpendLog[(LiteLLM_SpendLogs<br/>+ aggregated spend)]

    Auth -.-> DB[(LiteLLM Postgres)]
    Enforce -.-> Redis[(Redis / Valkey<br/>optional)]
```

Everything below runs only when `litellm_db.enabled: true`. With it disabled, Auto AI Router falls back to YAML-only credentials with no per-key auth or billing.

## 1. Token validation (`authenticateRequest`)

`internal/proxy/orchestrator.go` → `authenticateRequest()` runs first, before the request body is even read.

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Auto AI Router
    participant Cache as Auth cache<br/>(in-memory LRU)
    participant DB as LiteLLM Postgres

    Client->>Proxy: Authorization: Bearer sk-...
    alt token == master_key
        Proxy->>Proxy: isMasterKey() (constant-time compare)
        Proxy-->>Proxy: AdminContext, bypass everything below
    else regular key
        Proxy->>Proxy: SHA-256(token)
        Proxy->>Cache: Get(hash)
        alt cache hit (within auth_cache_ttl)
            Cache-->>Proxy: TokenInfo
        else cache miss / expired
            Proxy->>DB: QueryValidateTokenWithHierarchy(hash)<br/>single JOIN: Token+User+Team+Org+Memberships
            DB-->>Proxy: TokenInfo
            Proxy->>Cache: Set(hash, TokenInfo)
        end
        Proxy->>Proxy: TokenInfo.Validate("")<br/>blocked -> expired -> budget hierarchy
        alt invalid
            Proxy-->>Client: 401 / 402 (blocked, expired, or budget exceeded)
        end
    end
```

- The hash (never the raw key) is what's stored in Postgres and used as the cache key.
- `Validate()` checks, in order: `Blocked` → `Expires` → token budget → team → team-member → org → user → org-member budget (embedded budgets compare `Spend > MaxBudget`; org uses `LiteLLM_BudgetTable`).
- The model isn't known yet at this point (`Validate("")` — model check is skipped), so the pre-check here is deliberately budget/expiry-only.
- Cache invalidation is TTL-only (`auth_cache_ttl`, default 5s). There is no push invalidation from LiteLLM admin actions (`InvalidateToken`/`InvalidateAll` exist but nothing calls them yet) — a blocked/rebudgeted key can stay valid in a hot cache for up to `auth_cache_ttl`.
- The cache is per-instance, not shared across replicas.

## 2. Model resolution + allow-list + budget/rate-limit enforcement (`orchestrateRequest`)

Once auth succeeds, the body is parsed and the model alias resolved to its real provider-facing name, then three checks run in order.

```mermaid
flowchart TD
    A[authenticateRequest OK] --> B[readRequestBodyAndSelectModel<br/>modelID = alias, realModelID = provider name]
    B --> C{Admin scope<br/>master key?}
    C -->|yes| G[Proceed to credential selection]
    C -->|no| D[Model allow-list check]
    D -->|not allowed| D1[403 Forbidden]
    D -->|allowed| E[enforceBudgetAndRateLimits]
    E -->|budget/RPM/TPM exceeded| E1[402 / 429]
    E -->|ok| G
```

### 2a. Model allow-list

`TokenInfo.IsAnyModelAllowed()` (`internal/litellmdb/models/models.go`) resolves LiteLLM's sentinel values before checking membership:

- `all-proxy-models` in `VerificationToken.models` → any model allowed.
- `all-team-models` → inherits the parent team's `LiteLLM_TeamTable.models` list (no team → unrestricted).
- Empty list → unrestricted (LiteLLM's own default).

**Alias equivalence.** The same provider model is often exposed under several route aliases in `config.yaml` (e.g. `claude-haiku-4.5` and `anthropic/claude-haiku-4.5` both resolving to `claude-haiku-4-5-20251001` on the same credential — see `config.yaml.example`). A key restricted to one such alias is meant to allow the underlying model, not that one spelling. `orchestrateRequest` therefore calls `modelManager.GetAliasesForModel(modelID, realModelID)` to build the full alias group for the requested model (across every credential that serves it) and checks the key's allow-list against *any* of them, not just the literal string the client sent.

```mermaid
flowchart LR
    Req["Client requests\nanthropic/claude-haiku-4.5"] --> Resolve[GetAliasesForModel]
    Resolve --> Group["{anthropic/claude-haiku-4.5,\nclaude-haiku-4.5,\nclaude-haiku-4-5-20251001}"]
    Group --> Check{"Key.models contains\nany of these?"}
    Check -->|yes| Allow[Allowed]
    Check -->|no| Deny[403]
```

### 2b. Budget reservation + RPM/TPM (`enforceBudgetAndRateLimits`)

Opt-in, Redis-backed, closes the pre-check-vs-actual-spend race that the DB-snapshot check in §1 leaves open (a burst of concurrent requests can all pass the snapshot check before any of their spend is written back). No-op when either flag is off or Redis is disabled — the DB-snapshot check from `Validate()` remains the only protection in that case.

```mermaid
sequenceDiagram
    participant Proxy as Auto AI Router
    participant Redis
    participant Provider

    Note over Proxy: For every hierarchy level<br/>(token, user, team, org, team-member, org-member)

    Proxy->>Proxy: estimateRequestCost(modelID, realModelID, body)<br/>price registry x estimated max tokens
    Proxy->>Redis: TryReserve(entity, dbSpend, estCost, maxBudget)<br/>Lua: seed from dbSpend if key absent -> INCRBYFLOAT -> compare
    alt over budget
        Redis-->>Proxy: rejected
        Proxy->>Redis: Reconcile(-reservedAmount) for already-reserved levels
        Proxy-->>Proxy: 402 Payment Required
    else within budget
        Redis-->>Proxy: allowed
        Proxy->>Redis: AddCredentialWithTPM + TryAllowAllCtx (RPM/TPM)
        alt rate limited
            Proxy-->>Proxy: 429 Too Many Requests
        else ok
            Proxy->>Provider: forward request
            Provider-->>Proxy: response + real usage
            Proxy->>Redis: Reconcile(actualCost - reservedAmount)<br/>+ ConsumeTokensCtx(realTokens) for TPM
        end
    end
```

Key properties:

- **Seed-once-per-TTL**: the Redis counter is only seeded from the authoritative DB `spend` value when the key doesn't already exist; after that it's purely Redis-side until the TTL (`budget_reservation_ttl`, default 15m) expires and it reseeds. This bounds staleness after an out-of-band DB change (e.g. an admin resetting budget).
- **Reconcile exactly once**: guarded by `RequestLogContext.budgetReconciled`. The real call site is `logSpendToLiteLLMDB` (after the true cost is computed); a `defer` in `ProxyRequest` calls it a second time with cost `0` as a safety net for paths that never reach a credential (early failures) — the guard makes that a no-op once real reconciliation already happened.
- **Fail open on Redis errors**: a `TryReserve` error logs a warning and allows the request — the DB-snapshot check remains the backstop.
- **RPM/TPM** uses a second `ratelimit.RPMLimiter` instance (namespace `litellmauth:`, separate from the per-credential/provider limiter) keyed by `token:<hash>`, `user:<id>`, `team:<id>`, `org:<id>`, `teammember:<team>:<user>`, `orgmember:<org>:<user>`.

## 3. Spend logging (post-call, async)

Doesn't block the client response.

```mermaid
sequenceDiagram
    participant Proxy as Auto AI Router
    participant Queue as In-memory queue
    participant Worker as Background worker
    participant DB as LiteLLM Postgres

    Proxy->>Proxy: compute cost (PriceRegistry x real usage)
    Proxy->>Proxy: reconcileBudgetAndRateLimits(realCost)
    Proxy->>Queue: enqueue SpendLogEntry
    Note over Worker: every log_flush_interval (default 5s)<br/>or when log_batch_size reached
    Worker->>DB: INSERT SpendLogs ON CONFLICT DO NOTHING<br/>+ UPDATE spend = spend + $1 (per hierarchy level)<br/>single transaction
    alt DB error
        Worker->>Worker: retry with exponential backoff + jitter
        Worker->>Worker: after max attempts -> Dead Letter Queue (cap 10 batches)
        Note over Worker: oldest batch dropped on DLQ overflow<br/>(logged + counted in DLQDropped stat)
    end
```

Row-lock ordering: `updateTokens`/`updateUsers`/`updateTeams`/`updateOrgs`/`updateTeamMembers`/`updateOrgMembers` iterate map keys in sorted order (`sortedKeys()`) so that two concurrent batches touching overlapping rows always take their `UPDATE` locks in the same order — avoids Postgres deadlocks (`40P01`) that unordered map iteration would otherwise risk.

## 4. WebSocket (`/v1/responses`, upgrade)

Authentication happens **before** the WebSocket upgrade, not after:

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Auto AI Router

    Client->>Proxy: GET /v1/responses (Upgrade: websocket)
    Proxy->>Proxy: authenticateRequest() — same as §1<br/>(blocked/expired/budget snapshot)
    alt invalid
        Proxy-->>Client: 401 / 402 (plain HTTP, no upgrade)
    else valid
        Proxy->>Client: 101 Switching Protocols
        loop per response.create message
            Client->>Proxy: {"type": "response.create", ...}
            Proxy->>Proxy: build internal HTTP request,<br/>clone Authorization header,<br/>ProxyRequest() — full §1+§2+§3 path
            Proxy-->>Client: response events
        end
    end
```

The per-message path re-runs the entire auth+billing flow (model allow-list, budget reservation, spend logging) exactly as the plain HTTP endpoints do — the pre-upgrade check only gates connection establishment so an unauthenticated client can't hold an open socket for free.

## 5. `/v1/responses/{id}` (GET, stored response retrieval)

Lighter-weight than the LLM-call endpoints, but still validates the token when LiteLLM DB is enabled: non-master-key requests call `ValidateToken()` (blocked/expired/budget — same as §1) before the ownership check (`apiKeyHash` must match the response's owner). If the DB is unavailable, this degrades to ownership-only so reads stay available during an outage.

## Configuration

```yaml
litellm_db:
  enabled: true
  auth_cache_ttl: 20s                        # staleness window for blocked/budget/model changes
  auth_cache_size: 10000
  enforce_budget_reservation: false          # opt-in: Redis atomic budget reservation (§2b)
  enforce_key_rate_limits: false             # opt-in: Redis per-key/user/team/org RPM/TPM (§2b)
  budget_reservation_ttl: 15m                # Redis counter TTL / reseed-from-DB interval
  default_estimated_completion_tokens: 1000  # used when a request has no max_tokens

redis:
  enabled: true   # required for enforce_budget_reservation / enforce_key_rate_limits to take effect
  addresses:
    - "os.environ/REDIS_ADDRESS"
```

| Parameter                             | Type     | Default | Description                                                                                |
| ------------------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------ |
| `auth_cache_ttl`                      | duration | 5s      | How long a validated token is trusted from the in-memory cache before re-querying Postgres |
| `auth_cache_size`                     | int      | 10000   | Max entries in the auth LRU cache                                                          |
| `enforce_budget_reservation`          | bool     | false   | Atomic Redis budget pre-reservation (§2b). No-op if `redis.enabled: false`                 |
| `enforce_key_rate_limits`             | bool     | false   | Atomic Redis RPM/TPM per key/user/team/org (§2b). No-op if `redis.enabled: false`          |
| `budget_reservation_ttl`              | duration | 15m     | TTL of a Redis budget counter before it reseeds from DB `spend`                            |
| `default_estimated_completion_tokens` | int      | 1000    | Completion-token estimate for budget reservation when `max_tokens` is absent               |

Both `enforce_*` flags default to `false`: enabling them changes production enforcement behavior (requests can start failing with 402/429 that previously passed), so it's an explicit opt-in. The model allow-list check (§2a) has no such flag — it is always enforced for non-admin keys once `litellm_db.enabled: true`.

## Known limitations

- **Cache invalidation is TTL-only.** Blocking a key or lowering its budget in the LiteLLM admin panel takes up to `auth_cache_ttl` to take effect on a given replica (up to 15m for the *spend counter itself* if it's already tracked in Redis — see `budget_reservation_ttl`).
- **Streaming reconciliation on error paths.** If a streaming request fails after the provider already produced billable tokens but before the response reaches `logSpendToLiteLLMDB`, the `defer` safety net reconciles the Redis reservation at cost `0` (releases the full reservation). This can't cause overspend (it only makes the Redis counter *more* permissive than reality), but it can let the Redis counter drift below the true DB spend until the next TTL-driven reseed.
- **No compensating logic for aborted streams** when `drain_upstream_on_abort: false` — cost is billed on estimated/partial usage, with no later reconciliation against the provider's own usage records.
