# Configuration

Auto AI Router is configured via a YAML file passed with the `-config` flag.

See the full example in [`config.yaml.example`](https://github.com/MiXaiLL76/auto_ai_router/blob/main/config.yaml.example).

## Full Example

```yaml
server:
  port: 8080
  max_body_size_mb: 100
  response_body_multiplier: 10
  request_timeout: 60s
  write_timeout: 60s
  idle_timeout: 2m
  idle_conn_timeout: 120s
  max_idle_conns: 200
  max_idle_conns_per_host: 20
  logging_level: info
  master_key: "sk-your-master-key-here"
  default_models_rpm: -1
  model_prices_link: ""

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

  - name: "proxy_fallback"
    type: "proxy"
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
  log_retry_attempts: 3
  log_retry_delay: 1s
```

## Server Parameters

| Parameter                  | Type     | Default | Description                                           |
| -------------------------- | -------- | ------- | ----------------------------------------------------- |
| `port`                     | int      | 8080    | Listen port                                           |
| `max_body_size_mb`         | int      | 100     | Maximum request body size (MB)                        |
| `response_body_multiplier` | int      | 10      | Response body limit = max_body_size_mb * this value   |
| `request_timeout`          | duration | 60s     | Request timeout                                       |
| `write_timeout`            | duration | 60s     | HTTP server write timeout                             |
| `idle_timeout`             | duration | 2m      | HTTP server idle timeout (default: 2 * write_timeout) |
| `idle_conn_timeout`        | duration | 120s    | Idle connection timeout for keep-alive connections    |
| `max_idle_conns`           | int      | 200     | Maximum idle connections                              |
| `max_idle_conns_per_host`  | int      | 20      | Maximum idle connections per host                     |
| `logging_level`            | string   | info    | Logging level: `info`, `debug`, `error`               |
| `master_key`               | string   | —       | **Required.** Master key for client authentication    |
| `default_models_rpm`       | int      | -1      | Default RPM limit for models (-1 = unlimited)         |
| `model_prices_link`        | string   | —       | URL or file path to model prices JSON                 |

## Fail2Ban Parameters

| Parameter          | Type   | Description                                                           |
| ------------------ | ------ | --------------------------------------------------------------------- |
| `max_attempts`     | int    | Maximum failed attempts before banning a credential                   |
| `ban_duration`     | string | Ban duration (`permanent` for permanent, or duration like `5m`, `1h`) |
| `error_codes`      | []int  | HTTP status codes that trigger ban counting                           |
| `error_code_rules` | []rule | Per-error-code override rules (see example below)                     |

### Per-Error-Code Rules

Override `max_attempts` and `ban_duration` for specific error codes:

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

| Field         | Type   | Description                                                          |
| ------------- | ------ | -------------------------------------------------------------------- |
| `name`        | string | Unique credential identifier                                         |
| `type`        | string | Provider type: `openai`, `anthropic`, `vertex-ai`, `gemini`, `proxy` |
| `rpm`         | int    | Requests per minute limit (-1 = unlimited)                           |
| `tpm`         | int    | Tokens per minute limit (-1 = unlimited)                             |
| `is_fallback` | bool   | Use as fallback when primary credentials are exhausted               |

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
