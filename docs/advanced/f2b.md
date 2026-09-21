# Fail2Ban

Fail2Ban protects the rest of a credential pool from one upstream that has gone bad. When a specific `credential + model` pair fails repeatedly with a configured HTTP status, the router stops sending that pair traffic for a configurable duration — round-robin and fallback selection simply skip it, the same way they skip a rate-limited or manually disabled credential.

```yaml
fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]
  # error_code_rules:
  #   - code: 429
  #     max_attempts: 5
  #     ban_duration: 5m
```

## Parameters

| Parameter              | Type   | Description                                                                        |
| ---------------------- | ------ | ---------------------------------------------------------------------------------- |
| `max_attempts`         | int    | Failed attempts (at the configured codes) before a credential+model pair is banned |
| `ban_duration`         | string | Ban duration (`permanent`, or a duration like `5m`, `1h`)                          |
| `error_codes`          | []int  | HTTP status codes that count toward the ban threshold                              |
| `error_code_rules`     | []rule | Per-error-code override of `max_attempts`/`ban_duration` (see below)               |
| `credential_overrides` | map    | Per-credential override of `error_codes`/`error_code_rules` (see below)            |

A response outside `error_codes` is never counted, no matter how often it repeats — Fail2Ban only reacts to codes it's told to watch. A `2xx` response always resets the failure counter for that pair, regardless of `error_codes`.

## Per-Error-Code Rules

Override `max_attempts` and `ban_duration` for specific error codes, globally:

```yaml
fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]
  error_code_rules:
    - code: 429      # Rate limit errors
      max_attempts: 5
      ban_duration: 5m
```

A code without its own rule falls back to the top-level `max_attempts`/`ban_duration`.

## Per-Credential Overrides

"What counts as a failure" is not universal across upstreams. A credential that goes straight to the real provider signals a rate limit or outage with `429` or `5xx`, and the defaults above catch that. A credential that sits behind a reseller or aggregator can signal an entirely different condition — e.g. a blocked account — with a plain `400` and a generic error body. `400` is not in the default `error_codes` on purpose: it's also the code an ordinary malformed client request gets, and banning on it globally would ban unrelated credential+model pairs on the first bad request from any client.

`credential_overrides` lets you widen (or otherwise change) `error_codes`/`error_code_rules` for one specific credential without touching the global settings that apply to everyone else:

```yaml
fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]
  credential_overrides:
    cometapi-pool-1:
      error_codes: [400, 429, 500, 502, 503, 504]
      error_code_rules:
        - code: 400
          max_attempts: 3
          ban_duration: 15m
```

The key is the credential's `name` under `credentials:`.

### Precedence

- **`error_codes`**: if a credential has a `credential_overrides` entry with a non-empty `error_codes` list, that list **fully replaces** the global `error_codes` for that credential — it is not merged. A code you want tracked for this credential must be listed here explicitly, even if it's already in the global list.
- **`error_code_rules`**: resolved per status code, most specific first — a rule in the credential's own `error_code_rules` for that code wins; otherwise the global `error_code_rules` entry for that code applies; otherwise the top-level `max_attempts`/`ban_duration` apply. Rules for codes the credential didn't override keep using the global rule, so you only need to list the codes that actually need different thresholds.
- A credential with **no** `credential_overrides` entry behaves exactly as before this feature existed — global settings apply unchanged.

Only credentials with an actual quirk need an entry here; leave the rest out.

## Manual Bans (Admin API)

Fail2Ban reacts to errors after the fact. Sometimes an operator needs to take a credential out of rotation *before* anything fails — draining a replica ahead of a scale-in, pulling a key that is being rotated, pausing one model on one backend. Three endpoints do that. They are always available and accept **only the master key**: LiteLLM DB keys get `403`, and a request without credentials gets `401`.

| Endpoint          | Purpose                                     |
| ----------------- | ------------------------------------------- |
| `POST /api/ban`   | Ban a credential (all models, or one model) |
| `POST /api/unban` | Lift bans                                   |
| `GET /api/bans`   | List active bans, automatic and manual      |

Requests and responses are JSON. Unknown request fields are rejected with `400` — a typo such as `ttl_seconds` must not silently turn a temporary ban into a permanent one.

### Ban

```bash
# All models of a credential, for 30 minutes
curl -X POST http://router:8080/api/ban \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"credential": "k8s_n9_1xH200", "ttl": "30m", "reason": "air-scaler drain"}'

# One model only, until it is explicitly unbanned
curl -X POST http://router:8080/api/ban \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"credential": "k8s_n9_1xH200", "model": "llama-3-70b"}'
```

| Field        | Required | Description                                                                                    |
| ------------ | -------- | ---------------------------------------------------------------------------------------------- |
| `credential` | yes      | Credential `name`. Unknown names return `404`.                                                 |
| `model`      | no       | A model ID bans that `credential + model` pair. Omitted, empty or `"*"` bans **every** model.  |
| `ttl`        | no       | Duration such as `30m` or `2h`, must be positive. **Omitted means the ban lasts until unban.** |
| `reason`     | no       | Free text, stored as `admin: <reason>` and shown in `/health`, `/api/bans` and logs.           |

The response describes the ban that was created:

```json
{
  "credential": "k8s_n9_1xH200",
  "model": "*",
  "origin": "admin",
  "reason": "admin: air-scaler drain",
  "error_code": 0,
  "since": "2026-09-21T10:00:00Z",
  "permanent": false,
  "until": "2026-09-21T10:30:00Z"
}
```

`model: "*"` in a response means the ban covers every model of the credential. For a permanent ban `permanent` is `true` and `until` is absent.

**Whole-credential bans cover models the router has not seen yet.** A ban on `credential|*` is checked on every routing decision, so it also applies to models that a proxy/AIR credential learns from its upstream later. Nothing is enumerated when the ban is created.

**A manual ban always wins.** It replaces any existing ban on the same key — shorter or longer, automatic or manual — so an operator can override a long automatic ban without unbanning first. Failure counters are not touched.

**There is no maximum `ttl`.** Without a `ttl` the ban stays until you lift it with `/api/unban`.

!!! warning "Bans live in memory"
A router restart drops every ban, permanent ones included. A controller that drains a replica should re-issue the ban on each cycle rather than assume it survived.

### Unban

```bash
# Lift everything on the credential (the wildcard ban and per-model bans)
curl -X POST http://router:8080/api/unban \
  -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"credential": "k8s_n9_1xH200"}'
```

The response is `{"credential": "...", "model": "...", "removed": N}`, where `removed` counts the active bans that were lifted.

- Without `model`, **all** bans on the credential are lifted — manual and automatic, wildcard and per-model.
- With `model`, only the ban on that exact key is lifted. Unbanning `llama-3-70b` does **not** lift a credential-wide ban; unban with `"model": "*"` (or without `model`) for that.
- Unban is idempotent: nothing to lift still returns `200` with `removed: 0`. An unknown credential returns `404`.

### List

`GET /api/bans` returns every active (non-expired) ban, ordered by credential and model:

```json
{
  "bans": [
    {
      "credential": "k8s_n9_1xH200",
      "model": "*",
      "origin": "admin",
      "reason": "admin: air-scaler drain",
      "error_code": 0,
      "since": "2026-09-21T10:00:00Z",
      "permanent": false,
      "until": "2026-09-21T10:30:00Z"
    },
    {
      "credential": "openai-2",
      "model": "gpt-4o",
      "origin": "fail2ban",
      "error_code": 429,
      "since": "2026-09-21T09:58:00Z",
      "permanent": false,
      "until": "2026-09-21T10:03:00Z"
    }
  ]
}
```

`origin` is `admin` for bans created through the API and `fail2ban` for automatic ones (including provider-quota bans). `error_code` is the HTTP status that triggered an automatic ban, and `0` for a manual one.

### Example: graceful scale-in

1. `POST /api/ban` for the replica being removed — new requests stop going to it.
2. Wait until its in-flight requests finish (or a timeout passes). Requests already running are not interrupted.
3. Scale the replica down, then `POST /api/unban` when it comes back.

A controller talking to a router that predates these endpoints gets `404` and can fall back to hard scale-in.

## Monitoring Bans

Ban and unban events are exported as Prometheus counters (`auto_ai_router_credential_ban_events_total`, `auto_ai_router_credential_unban_events_total`, labelled by credential and model — ban events also carry the error code). Automatic bans are logged at `ERROR` level — losing a credential shrinks routing capacity for its models, so it's worth alerting on. Manual bans are deliberate, so they are logged at `WARN` ("Credential banned by admin") with `error_code` `0`; a wildcard ban shows up with model `*`.

`/health` explains why a credential is banned:

| Field        | Where                | Description                                                                                                 |
| ------------ | -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ban_origin` | credential and model | `admin` or `fail2ban`                                                                                       |
| `ban_reason` | credential           | Reason of the ban that best explains the state: a manual ban if there is one, otherwise the longest-lasting |
| `ban_until`  | credential           | When that ban expires; absent for a permanent ban                                                           |

A credential-wide ban marks every model of the credential as banned in `/health`. If a model also has its own, more specific ban, that one is reported for the model.

The `/vhealth` dashboard reads these fields, so no master key or `/api/bans` call is needed to see manual bans there:

- a banned credential card shows the reason and when the ban ends (`until 17:29 · 30m left`, or `until unban` for a permanent ban);
- manual bans are drawn in the warning colour with a `banned · admin` badge, automatic ones in the error colour with the error counts;
- in the models table the badge stays short (`banned · admin`) and the reason and expiry are in its tooltip;
- the `banned` counter in the overview shows how many credentials are banned by an admin.
