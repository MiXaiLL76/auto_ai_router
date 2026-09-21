# LiteLLM Database Integration

Auto AI Router can integrate with a LiteLLM PostgreSQL database for spend logging and API key authentication.

Provider-based team attribution is controlled by the server-level `credential_name_as_team_id` setting, independently of the database connection settings below.

## Configuration

```yaml
litellm_db:
  enabled: true
  is_required: false
  database_url: "os.environ/LITELLM_DATABASE_URL"
  max_conns: 25
  min_conns: 5
  log_queue_size: 5000
  include_team_spend_in_user_spend: true
```

## Parameters

| Parameter                          | Type     | Default | Description                                                       |
| ---------------------------------- | -------- | ------- | ----------------------------------------------------------------- |
| `enabled`                          | bool     | false   | Enable LiteLLM DB integration                                     |
| `is_required`                      | bool     | false   | Fail startup if DB connection fails                               |
| `database_url`                     | string   | —       | PostgreSQL connection string (supports env variables)             |
| `max_conns`                        | int      | 25      | Maximum database connections                                      |
| `min_conns`                        | int      | 5       | Minimum database connections                                      |
| `health_check_interval`            | duration | 10s     | DB health check interval                                          |
| `connect_timeout`                  | duration | 5s      | Connection timeout                                                |
| `auth_cache_ttl`                   | duration | 20s     | Auth cache TTL                                                    |
| `auth_cache_size`                  | int      | 10000   | Auth cache size                                                   |
| `log_queue_size`                   | int      | 5000    | Spend log queue size                                              |
| `log_batch_size`                   | int      | 100     | Spend log batch size                                              |
| `log_flush_interval`               | duration | 5s      | Spend log flush interval                                          |
| `include_team_spend_in_user_spend` | bool     | true    | Include team-bound events in the cumulative user spend projection |
| `log_retry_attempts`               | int      | 3       | Retry attempts on log insert failure                              |
| `log_retry_delay`                  | duration | 1s      | Delay between retry attempts                                      |

## Features

- **Spend logging** — records token usage, costs, and request metadata
- **Daily aggregation** — aggregates spend by user, team, organization, end user, agent, and tags
- **API key auth** — validates API keys against LiteLLM verification tokens
- **Batch processing** — logs are batched and flushed periodically for performance
- **Dead Letter Queue** — failed log inserts are captured for later retry

## Models from the LiteLLM database

AIR loads deployments from `LiteLLM_ProxyModelTable` and the credentials they are bound
to, so a fleet managed through the LiteLLM admin UI can be served without repeating it
in `config.yaml`. The mapping follows LiteLLM's own semantics:

- **Credential binding** — a deployment is bound through `litellm_params.litellm_credential_name`
  (stored encrypted with the master key). A credential without a provider of its own takes it
  from the deployments that use it.
- **Provider** — `hosted_vllm` and `vllm` become the [`vllm`](../providers/vllm.md) type.
  The `hosted_vllm/` prefix in `litellm_params.model` is stripped before the request reaches
  vLLM; other slashes (`Qwen/Qwen3-8B`) are part of the model id and are kept. A vLLM
  deployment with only an inline `api_base` (no API key) is accepted too.
- **Several deployments, one name** — deployments sharing a `model_name` form one pool.
- **Aliases** — `router_settings.model_group_alias` from `LiteLLM_Config` is imported as public
  model aliases. As in LiteLLM the alias is resolved to its target group before a deployment is
  picked, so both names share one set of deployments and the target's rate limits, balancer
  state and price; the alias is still listed in `/v1/models`. A key whose model list contains
  either name may use the alias. A real model group with the alias's name wins over the alias,
  and a `public_model_alias` from the config wins over the imported one.
- **Skipped** — deployments with `blocked = true`, and `mode: rerank` deployments (`/rerank`
  is not supported).
- **Not imported** — `router_settings.fallbacks`. AIR fails over between credentials of one
  model, not between models, so `coder-ultra -> qwen-ultra` style rules are ignored.
- **Default parameters** — see [vLLM](../providers/vllm.md#default-parameters).

## Identifying the user

LiteLLM's `user_header_mappings` is replaced by fixed headers. The `-Email` headers set the
**end user** (`LiteLLM_EndUserTable`, `LiteLLM_DailyEndUserSpend`); the `-Id` headers set the
**user** recorded in the spend log and `LiteLLM_DailyUserSpend`. The `-Id` headers apply only to
service keys that have no owner (`user_id` is empty); on a key that has an owner they are
ignored and the owner is recorded. The first header present (in this order) wins; empty, longer than 256 bytes, or non-printable
values are ignored.

| Purpose  | Headers, highest priority first                                                    |
| -------- | ---------------------------------------------------------------------------------- |
| End user | `X-AIR-User-Email`, `X-End-User`, `X-OpenWebUI-User-Email`, `X-AirClaw-User-Email` |
| User id  | `X-AIR-User-Id`, `X-OpenWebUI-User-Id`, `X-AirClaw-User-Id`                        |

There is no fallback to the request body (`user`, `metadata.user_id`) and the key owner's
email is never used as an end user. The headers affect statistics only: authentication and
budget checks keep using the key's own user. They are taken on trust, so expose service keys
only to clients that set them honestly (typically behind OpenWebUI or another gateway).

## Model names in spend records

As in LiteLLM, `model` holds the deployment's provider-facing name and `model_group` the name
the client asked for. A request for the alias `qwen-ultra` served by `qwen397b-int4` is
recorded as `model = qwen397b-int4`, `model_group = qwen-ultra`; both keys of the daily tables
are filled accordingly. For a failed request that never reached a deployment, `model` falls
back to the requested name.

## Database URL

The connection string follows the standard PostgreSQL format:

```
postgresql://user:password@host:5432/litellm
```

Use environment variables for security:

```yaml
litellm_db:
  database_url: "os.environ/LITELLM_DATABASE_URL"
```

```bash
export LITELLM_DATABASE_URL="postgresql://user:password@localhost:5432/litellm"
```
