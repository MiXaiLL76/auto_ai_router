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
| `log_credential_name`             | bool     | false   | Write the resolved provider name to `metadata.credential_name` |
| `include_team_spend_in_user_spend` | bool     | true    | Include team-bound events in the cumulative user spend projection |
| `log_retry_attempts`               | int      | 3       | Retry attempts on log insert failure                              |
| `log_retry_delay`                  | duration | 1s      | Delay between retry attempts                                      |

## Features

- **Spend logging** — records token usage, costs, and request metadata
- **Daily aggregation** — aggregates spend by user, team, organization, end user, agent, and tags
- **API key auth** — validates API keys against LiteLLM verification tokens
- **Batch processing** — logs are batched and flushed periodically for performance
- **Dead Letter Queue** — failed log inserts are captured for later retry

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

## Provider credential metadata

Enable `litellm_db.log_credential_name` to record the selected provider credential
as `credential_name` inside `LiteLLM_SpendLogs.metadata`. This uses the existing
LiteLLM schema and requires no migration.

```yaml
litellm_db:
  enabled: true
  database_url: os.environ/LITELLM_DATABASE_URL
  log_credential_name: true
```

AIR uses the same resolved credential name as the prefix of `model_id`: the actual
upstream credential when available, otherwise the selected local credential.
Existing metadata fields are preserved; the resolved name replaces any existing
`metadata.credential_name` value for the new event. When disabled (the default),
metadata is passed through unchanged. An unknown credential adds no field.
Previously stored rows are not rewritten when the flag changes or an event is replayed.

Client teams, model IDs, usage, spend, and daily aggregation remain unchanged.
For reports by provider, group the raw spend logs by `metadata->>'credential_name'`
and model. For example, with a bounded reporting window:

```sql
SELECT metadata->>'credential_name' AS credential_name, model, SUM(spend) AS spend
FROM "LiteLLM_SpendLogs"
WHERE "startTime" >= $1 AND "startTime" < $2
GROUP BY metadata->>'credential_name', model;
```

The existing daily tables do not gain a provider-credential dimension. Kafka
already records the local credential in `credential_name` and the upstream name
in `credential_actual_credential_name`, independently of this PostgreSQL option.
