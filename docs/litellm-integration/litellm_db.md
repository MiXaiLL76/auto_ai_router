# LiteLLM Database Integration

Auto AI Router can integrate with a LiteLLM PostgreSQL database for spend logging and API key authentication.

## Configuration

```yaml
litellm_db:
  enabled: true
  is_required: false
  database_url: "os.environ/LITELLM_DATABASE_URL"
  max_conns: 25
  min_conns: 5
  log_queue_size: 5000
```

## Parameters

| Parameter               | Type     | Default | Description                                           |
| ----------------------- | -------- | ------- | ----------------------------------------------------- |
| `enabled`               | bool     | false   | Enable LiteLLM DB integration                         |
| `is_required`           | bool     | false   | Fail startup if DB connection fails                   |
| `database_url`          | string   | —       | PostgreSQL connection string (supports env variables) |
| `max_conns`             | int      | 25      | Maximum database connections                          |
| `min_conns`             | int      | 5       | Minimum database connections                          |
| `health_check_interval` | duration | 10s     | DB health check interval                              |
| `connect_timeout`       | duration | 5s      | Connection timeout                                    |
| `auth_cache_ttl`        | duration | 20s     | Auth cache TTL                                        |
| `auth_cache_size`       | int      | 10000   | Auth cache size                                       |
| `log_queue_size`        | int      | 5000    | Spend log queue size                                  |
| `log_batch_size`        | int      | 100     | Spend log batch size                                  |
| `log_flush_interval`    | duration | 5s      | Spend log flush interval                              |
| `log_retry_attempts`    | int      | 3       | Retry attempts on log insert failure                  |
| `log_retry_delay`       | duration | 1s      | Delay between retry attempts                          |

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
