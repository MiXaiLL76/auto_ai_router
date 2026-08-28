# Configuration

Auto AI Router is configured via a YAML file passed with the `-config` flag.

See the full example in [`config.yaml.example`](https://github.com/MiXaiLL76/auto_ai_router/blob/main/config.yaml.example).

## Full Example

```yaml
server:
  port: 8080
  max_body_size_mb: 100
  response_body_multiplier: 10
  response_compatibility: native
  request_timeout: 60s
  write_timeout: 60s
  idle_timeout: 2m
  idle_conn_timeout: 120s
  max_idle_conns: 200
  max_idle_conns_per_host: 20
  logging_level: info
  master_key: "sk-your-master-key-here"
  default_models_rpm: -1
  credential_name_as_team_id: false
  model_prices_link: ""
  model_prices_sync_interval: 5m
  response_headers:
    mode: passthrough

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]
  # error_code_rules:
  #   - code: 429
  #     max_attempts: 5
  #     ban_duration: 5m

monitoring:
  prometheus_enabled: true
  log_errors: false
  errors_log_path: "logs/logs.jsonl"

credentials:
  - name: "openai_main"
    type: "openai"
    api_key: "sk-proj-xxxxx"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-project-id"
    location: "global"
    credentials_file: "path/to/service-account.json"
    rpm: 100
    tpm: 50000

  - name: "gemini_studio"
    type: "gemini"
    api_key: "os.environ/GEMINI_API_KEY"
    base_url: "https://generativelanguage.googleapis.com"
    rpm: 60
    tpm: -1

  - name: "air_fallback"
    type: "air"
    base_url: "http://backup-router.local:8080"
    api_key: "sk-remote-master-key"
    rpm: 200
    tpm: 100000
    is_fallback: true

models:
  - name: "gpt-4o"
    credential: openai_main
    rpm: 100
    tpm: 50000
  - name: "gemini-2.5-pro"
    credential: vertex_ai
    rpm: 100
    tpm: 50000

litellm_db:
  enabled: false
  is_required: false
  database_url: "os.environ/LITELLM_DATABASE_URL"
  max_conns: 25
  min_conns: 5
  health_check_interval: 10s
  connect_timeout: 5s
  auth_cache_ttl: 20s
  auth_cache_size: 10000
  log_queue_size: 5000
  log_batch_size: 100
  log_flush_interval: 5s
  include_team_spend_in_user_spend: true
  log_retry_attempts: 3
  log_retry_delay: 1s
```

## Server Parameters

| Parameter                    | Type     | Default     | Description                                                                                           |
| ---------------------------- | -------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| `port`                       | int      | 8080        | Listen port                                                                                           |
| `max_body_size_mb`           | int      | 100         | Maximum request body size (MB)                                                                        |
| `response_body_multiplier`   | int      | 10          | Response body limit = max_body_size_mb * this value                                                   |
| `response_compatibility`     | string   | native      | Response contract: `native` or LiteLLM-compatible `litellm`                                           |
| `request_timeout`            | duration | 60s         | Request timeout                                                                                       |
| `write_timeout`              | duration | 60s         | HTTP server write timeout                                                                             |
| `idle_timeout`               | duration | 2m          | HTTP server idle timeout (default: 2 * write_timeout)                                                 |
| `idle_conn_timeout`          | duration | 120s        | Idle connection timeout for keep-alive connections                                                    |
| `max_idle_conns`             | int      | 200         | Maximum idle connections                                                                              |
| `max_idle_conns_per_host`    | int      | 20          | Maximum idle connections per host                                                                     |
| `logging_level`              | string   | info        | Logging level: `info`, `debug`, `error`                                                               |
| `master_key`                 | string   | —           | **Required.** Master key for client authentication                                                    |
| `default_models_rpm`         | int      | -1          | Default RPM limit for models (-1 = unlimited)                                                         |
| `credential_name_as_team_id` | bool     | false       | Use the selected provider credential name as `team_id` in spend logs when auth has no team assignment |
| `model_prices_link`          | string   | —           | URL or file path to model prices JSON                                                                 |
| `model_prices_sync_interval` | duration | 5m          | How often model prices are re-fetched from `model_prices_link`                                        |
| `proxy_health_timeout`       | duration | 15s         | Timeout for fetching `/health` from remote proxy credentials                                          |
| `response_headers.mode`      | string   | passthrough | Response header policy. Supported values are `passthrough` and `allowlist`                            |
| `tiktoken_enabled`           | bool     | true        | Local tiktoken-based fallback token estimation for streaming responses (see below)                    |

The `allowlist` mode forwards `Content-Type`, `Cache-Control`, `Retry-After`, `Content-Disposition`, `Content-Range`, `Last-Modified`, and `Location`. Transport headers are generated by the router. Other upstream response headers are removed.

When `credential_name_as_team_id` is enabled, spend logs use the selected provider credential name as `team_id` only when the authenticated key does not provide a `team_id`. An explicit team assignment always takes precedence. This is intended for billing-aggregator AIR instances that group spend by provider credential; leave it disabled for embedded routers and migration helpers.

`model_prices_sync_interval` controls how often the price source is re-read after the initial startup load. It accepts any Go duration string (`30s`, `15m`, `1h`) and is ignored when `model_prices_link` is empty. A failed refresh keeps the previously loaded prices and is retried on the next tick; a missing or non-positive value falls back to `5m`. See [Model Pricing](../litellm-integration/pricing.md#refresh-interval) for guidance on picking a value.

When PostgreSQL or Kafka spend logging is enabled, every routed model must have a price entry. If neither its real provider name nor its alias can be priced, the router returns `503 Model pricing unavailable` before contacting the provider. Use an explicit zero-price entry for intentionally free models.

`tiktoken_enabled` controls the local prompt/completion token estimator used as a fallback for streaming responses when a provider doesn't report usage (e.g. the stream is cut before the final usage chunk, or the provider omits token counts entirely), and for budget-reservation cost estimates ahead of the request. Two different costs apply while it's on: the **prompt-token estimate and the final completion-token BPE count are lazy** — computed only if a stream actually finishes without provider-reported usage — so they're free for providers that do report usage. However, **per-chunk delta-text accumulation runs on every streaming chunk of every stream** while this flag is on, regardless of whether the provider ultimately reports usage, since the accumulator has to keep pace with the stream in case it's needed at the end.

Set to `false` to skip local estimation entirely if all your configured providers always report usage and you'd rather not carry any tiktoken-related per-chunk cost. This has three effects, not just spend logging: spend logs show `0` prompt/completion tokens (instead of an estimate) for streaming requests where a provider omits usage; the streaming-fallback total that feeds `rateLimiter.ConsumeTokens`/`ConsumeModelTokens` is zeroed the same way; and budget-reservation cost estimates (`server.litellm_db.enforce_budget_reservation`) undercount, reflecting only the completion-token portion — reservation itself still runs, it just can't account for prompt-token cost without local tokenization.

## Fail2Ban Parameters

The `fail2ban` block bans a `credential + model` pair after repeated failures at configured HTTP status codes, so a broken or rate-limited upstream stops eating a share of round-robin traffic. See [Fail2Ban](../advanced/f2b.md) for the full parameter reference, per-error-code rules, and per-credential overrides (for upstreams — e.g. resellers/aggregators — that signal failure with a different status code than the rest of the pool).

## Monitoring Parameters

| Parameter            | Type   | Description                             |
| -------------------- | ------ | --------------------------------------- |
| `prometheus_enabled` | bool   | Enable Prometheus metrics on `/metrics` |
| `log_errors`         | bool   | Enable error logging to file            |
| `errors_log_path`    | string | Path to error log file                  |

!!! note
The `/health` endpoint is always available and cannot be disabled or reconfigured.

## Credentials

Each credential defines a connection to an LLM provider. See [Providers](../providers/index.md) for details on each type.

Common fields for all credentials:

| Field              | Type   | Description                                                                                                                      |
| ------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `name`             | string | Unique credential identifier                                                                                                     |
| `type`             | string | Provider type: `openai`, `anthropic`, `cometapi`, `vertex-ai`, `gemini`, `bedrock`, `proxy`                                      |
| `rpm`              | int    | Requests per minute limit (-1 = unlimited)                                                                                       |
| `tpm`              | int    | Tokens per minute limit (-1 = unlimited)                                                                                         |
| `is_fallback`      | bool   | Use as fallback when primary credentials are exhausted                                                                           |
| `reasoning_only`   | bool   | Route only requests that explicitly enable reasoning/thinking                                                                    |
| `scopes`           | list   | Optional client scopes allowed to use and see this credential                                                                    |
| `denied_scopes`    | list   | Optional client scopes that must not use or see this credential                                                                  |
| `forbidden_scopes` | list   | Alias for `denied_scopes`                                                                                                        |
| `openai_proto`     | bool   | `cometapi` only: use CometAPI's OpenAI-compatible wire protocol ([details](../providers/cometapi.md#openai-protocol-mode))       |
| `google_proto`     | bool   | `cometapi` only: use CometAPI's Google GenAI-compatible wire protocol ([details](../providers/cometapi.md#google-protocol-mode)) |

### Scoped credential visibility

Credential scopes let one shared AIR instance expose different providers to different API keys.
If `scopes` is omitted or empty, the credential remains available to everyone. If `scopes`
is set, a request can use the credential only when the request API key has at least one
matching scope. `denied_scopes` always wins over `scopes`.

```yaml
credentials:
  - name: shared-openai
    type: openai
    api_key: "os.environ/OPENAI_KEY"
    base_url: "https://api.openai.com/v1"
    rpm: -1

  - name: team-a-openai
    type: openai
    api_key: "os.environ/TEAM_A_OPENAI_KEY"
    base_url: "https://api.openai.com/v1"
    rpm: -1
    scopes: [team-a]

  - name: all-except-team-a
    type: openai
    api_key: "os.environ/SHARED_KEY"
    base_url: "https://api.openai.com/v1"
    rpm: -1
    denied_scopes: [team-a]
```

For LiteLLM DB auth, request scopes are read from API key metadata:

```json
{
  "air_scopes": ["team-a"],
  "air_denied_scopes": ["premium"]
}
```

If `air_scopes` is not set, AIR falls back to the LiteLLM API key name
(`key_name`, then `key_alias`). For example, a key named `team-a` gets the `team-a`
scope automatically. The master key bypasses scope filters.

For DB-loaded credentials, add the same fields to `LiteLLM_CredentialsTable.credential_info`:

```json
{
  "custom_llm_provider": "openai",
  "air_scopes": ["team-a"],
  "air_denied_scopes": ["premium"]
}
```

Scope filtering applies to routing, retry/fallback selection, `/health`, `/v1/models`,
`/trace`, `/vhealth`, and `/vtrace`.

In both LiteLLM API-key metadata and `LiteLLM_CredentialsTable.credential_info`,
`air_forbidden_scopes` is accepted as an alias for `air_denied_scopes`.

Set `reasoning_only: true` on a credential when its provider accepts only reasoning traffic:

```yaml
credentials:
  - name: anthropic-reasoning
    type: anthropic
    api_key: "os.environ/ANTHROPIC_API_KEY"
    base_url: "https://api.anthropic.com"
    reasoning_only: true
```

The filter recognizes enabled `reasoning_effort`, `reasoning`, `thinking`,
`thinking_budget`, and `thinking_level` parameters. Requests that omit reasoning or
explicitly disable it skip the credential during initial selection, sticky routing,
retries, and fallback selection. For DB-loaded credentials, use
`"air_reasoning_only": true` in `credential_info`.

## Models

The `models` section binds specific models to credentials and optionally sets per-model rate limits.

```yaml
models:
  - name: "gpt-4o"
    credential: openai_main
    rpm: 100
    tpm: 50000
```

By default, all models are available through all credentials. Use the `models` section to restrict which credentials serve which models.

By default, models can also be declared directly inside a credential via the `models:` field — they are automatically extracted and added to the global models list with the credential name pre-filled.

See [Load Balancing](../advanced/balancing.md) for details on multi-credential routing.

## YAML Anchors for Models

When many credentials share the same set of models, YAML anchors eliminate repetition.
Define a template once with `&anchor-name` and reference it with `*anchor-name`.

### List anchor in `x-model-templates`

The `x-model-templates` top-level key is a dedicated namespace for anchor definitions. It is not processed by the router — its sole purpose is to hold anchors so they can be referenced elsewhere.

```yaml
x-model-templates:
  vertex-base-models: &vertex-base-models
    - name: gemini-2.5-flash
      rpm: 100
      tpm: 50000
    - name: gemini-2.5-pro
      rpm: 50
      tpm: 100000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-1"
    location: "global"
    credentials_file: "keys/proj-1.json"
    rpm: 100
    models: *vertex-base-models   # expands to the full list

  - name: "vertex_v2"
    type: "vertex-ai"
    project_id: "proj-2"
    location: "global"
    credentials_file: "keys/proj-2.json"
    rpm: 100
    models: *vertex-base-models   # same list, credential set to "vertex_v2"
```

Each model copy automatically gets the parent credential name injected, so no manual `credential:` field is needed inside the template.

### Single-model anchor

An anchor can also target a single model mapping and be used as an item in a `models:` list:

```yaml
x-model-templates:
  flash: &flash
    name: gemini-2.5-flash
    rpm: 100
    tpm: 50000

credentials:
  - name: "vertex_v1"
    type: "vertex-ai"
    project_id: "proj-1"
    location: "global"
    credentials_file: "keys/proj-1.json"
    rpm: 100
    models:
      - *flash               # single model from anchor
      - name: gemini-2.5-pro # inline model
        rpm: 50
        tpm: 100000
```

### Expanding a list anchor inside the top-level `models:` section

A list anchor can be expanded inline within the top-level `models:` sequence. The router flattens the result so all items end up as a flat list:

```yaml
x-model-templates:
  shared-models: &shared-models
    - name: gemini-2.5-flash
      credential: vertex_v1
      rpm: 100
      tpm: 50000
    - name: gemini-2.5-pro
      credential: vertex_v1
      rpm: 50
      tpm: 100000

models:
  - *shared-models        # expands and flattens both items into the list
  - name: gpt-4o
    credential: openai_main
    rpm: 60
    tpm: 80000
```

### Supported combinations

| Syntax                   | Location                        | Result                                        |
| ------------------------ | ------------------------------- | --------------------------------------------- |
| `models: *list-anchor`   | inside a credential             | list items added with that credential name    |
| `- *list-anchor`         | inside a credential's `models:` | list items added with that credential name    |
| `- *single-model-anchor` | inside a credential's `models:` | single model added with that credential name  |
| `- *list-anchor`         | top-level `models:`             | list expanded and flattened into the sequence |
