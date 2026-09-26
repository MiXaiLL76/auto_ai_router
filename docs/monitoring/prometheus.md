# Prometheus Metrics

Enable Prometheus metrics in the config:

```yaml
monitoring:
  prometheus_enabled: true
```

Metrics are available at `/metrics`.

The same metrics can also be **pushed** to an OTLP collector instead of (or in addition to) being scraped — see [OpenTelemetry](otel.md).

## Available Metrics

| Metric                                     | Type      | Description                                                |
| ------------------------------------------ | --------- | ---------------------------------------------------------- |
| `auto_ai_router_credential_rpm_current`    | Gauge     | Current RPM usage per credential                           |
| `auto_ai_router_credential_tpm_current`    | Gauge     | Current TPM usage per credential                           |
| `auto_ai_router_credential_banned`         | Gauge     | Ban status per credential (1 = banned)                     |
| `auto_ai_router_requests_total`            | Counter   | Total requests processed                                   |
| `auto_ai_router_requests_duration_seconds` | Histogram | Request latency distribution                               |
| `auto_ai_router_aborted_requests_total`    | Counter   | Client-aborted requests by credential, model, and endpoint |

## Per-Key Metrics

Opt-in request counters per LiteLLM API key. Requires `litellm_db.enabled: true` (keys are identified through LiteLLM token validation) and `prometheus_enabled` or `otel.enabled`; otherwise the flag is ignored with a startup warning.

```yaml
monitoring:
  prometheus_enabled: true
  key_metrics:
    enabled: true
    # Owner labels on auto_ai_router_key_info. Default: all but user_email.
    info_labels: [key_alias, user_id, team_id, team_alias, organization_id]
    max_keys: 5000 # distinct keys; later keys are counted as key="__other__" (0 = unlimited)
    idle_ttl: 24h # drop series of keys without traffic this long (0 = never)
```

| Metric                              | Type    | Labels                    | Description                                 |
| ----------------------------------- | ------- | ------------------------- | ------------------------------------------- |
| `auto_ai_router_key_requests_total` | Counter | `key`, `status`           | Requests per key by the final HTTP status   |
| `auto_ai_router_key_info`           | Gauge   | `key`, `<info_labels...>` | Owner metadata per key, value is always `1` |

- `key` is the first 12 hex characters of the key's sha256 hash (find the key with `WHERE token LIKE '<key>%'`), or `master` for the router master key. The full hash is never exposed: LiteLLM auth accepts a hashed token as a bearer credential.
- `status` is the HTTP status the client received, so budget (402) and key RPM/TPM (429) rejections are counted. Requests rejected during authentication (unknown, expired or blocked key) have no key and are not counted. A streaming response that failed mid-stream after a 200 counts as 200.
- Every authenticated API call counts, including `GET /v1/models`. Each WebSocket Responses turn counts as one request; the `101` upgrade itself is not counted.
- Owner labels are only on `auto_ai_router_key_info`; join them in PromQL on `(key, instance)`. Every replica exports its own info series for the keys it served, so joining on `key` alone fails with "many-to-many matching" as soon as there is more than one replica. Changing a key's alias replaces its info series, so the join stays one-to-one.
- The join drops `key="__other__"` (no info series); query `auto_ai_router_key_requests_total{key="__other__"}` directly to see overflow traffic.

`info_labels` values: `key_alias`, `user_id`, `user_email`, `team_id`, `team_alias`, `organization_id`. `/metrics` is usually unauthenticated inside the cluster, so `user_email` is opt-in.

Example queries:

```promql
# Requests per minute per key (top 10), with alias
topk(10,
  sum by (key) (rate(auto_ai_router_key_requests_total[5m])) * 60
  * on (key, instance) group_left (key_alias) auto_ai_router_key_info
)

# Requests per minute by team
sum by (team_alias) (
  rate(auto_ai_router_key_requests_total[5m]) * 60
  * on (key, instance) group_left (team_alias) auto_ai_router_key_info
)

# Error share per user (4xx + 5xx)
sum by (user_id) (
  rate(auto_ai_router_key_requests_total{status=~"[45].."}[5m])
  * on (key, instance) group_left (user_id) auto_ai_router_key_info
)
/
sum by (user_id) (
  rate(auto_ai_router_key_requests_total[5m])
  * on (key, instance) group_left (user_id) auto_ai_router_key_info
)
```

## Proxy Credential Exclusion

Proxy credentials are **not** included in Prometheus metrics. Their statistics are available through the `/health` endpoint and are synchronized from the remote `/health` endpoint every 30 seconds.

## Scrape Configuration

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'auto-ai-router'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
```
