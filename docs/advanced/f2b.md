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

## Monitoring Bans

Ban and unban events are exported as Prometheus counters (`auto_ai_router_credential_ban_events_total`, `auto_ai_router_credential_unban_events_total`, labelled by credential and model — ban events also carry the error code) and logged at `ERROR` level — losing a credential shrinks routing capacity for its models, so it's worth alerting on.
