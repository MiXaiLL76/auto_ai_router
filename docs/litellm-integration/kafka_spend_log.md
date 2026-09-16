# Kafka Spend Log (SpendLog → Kafka → ClickHouse)

Auto AI Router can publish an extended copy of every spend event to Kafka, in addition to (or instead of) writing it to the LiteLLM PostgreSQL database. From Kafka the event flows into ClickHouse for analytics via a standard ClickHouse `Kafka` table engine and a materialized view — no custom consumer service is required. PostgreSQL remains the source of truth for auth (`ValidateToken`), budgets, and the LiteLLM UI; the Kafka/ClickHouse path is a separate, independently-enabled write path for analytics.

For the full design rationale (gap analysis against real LiteLLM data, config decisions, open-question resolutions), see the design document: `auto_ai_router_kafka_spend_log_tz.md` in the repository root.

## Configuration

```yaml
kafka:
  enabled: os.environ/KAFKA_ENABLED
  brokers:
    - "os.environ/KAFKA_BROKERS"      # "kafka1:9092,kafka2:9092"
  topic: "air.spend_logs"
  client_id: "auto_ai_router"

  log_queue_size: 5000
  log_batch_size: 100
  log_flush_interval: 5s

  tls_enabled: false
  sasl_mechanism: ""                   # "" | "PLAIN" | "SCRAM-SHA-256" | "SCRAM-SHA-512"
  sasl_username: "os.environ/KAFKA_SASL_USERNAME"
  sasl_password: "os.environ/KAFKA_SASL_PASSWORD"

  # Separate, independently-toggleable write-path: raw request/response
  # bodies, published to their own topic. See "Raw bodies" below for why
  # this isn't just a field on the spend event.
  raw_bodies:
    enabled: os.environ/KAFKA_RAW_BODIES_ENABLED   # default: false
    topic: "raw-bodies"
    store_raw_body: os.environ/KAFKA_RAW_BODIES_STORE_RAW_BODY       # default: false
    store_only_errors: os.environ/KAFKA_RAW_BODIES_STORE_ONLY_ERRORS # default: true
    redact_sensitive_fields: os.environ/KAFKA_RAW_BODIES_REDACT_SENSITIVE_FIELDS # default: true

litellm_db:
  enabled: true
  database_url: "os.environ/LITELLM_DATABASE_URL"
  disable_spend_logs_write: os.environ/DISABLE_PG_SPEND_LOGS   # default: false
```

| Environment variable    | Purpose                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `KAFKA_ENABLED`         | Enables the `kafka:` config section / event publishing                                                                  |
| `KAFKA_BROKERS`         | Comma-separated list of Kafka bootstrap brokers                                                                         |
| `DISABLE_PG_SPEND_LOGS` | If `true`, stops spend log writes to Postgres (`litellm_db.disable_spend_logs_write`); auth/keys/budgets are unaffected |

`kafka.enabled` and `litellm_db.disable_spend_logs_write` are independent flags:

| `kafka.enabled` | `disable_spend_logs_write` | Behavior                                                               |
| --------------- | -------------------------- | ---------------------------------------------------------------------- |
| false           | false                      | Postgres only (current default behavior)                               |
| true            | false                      | Dual-write: Postgres + Kafka                                           |
| true            | true                       | Kafka only; Postgres receives auth traffic only                        |
| false           | true                       | Invalid — rejected at config validation (spend would be lost entirely) |

Kafka availability is not treated as critical for production traffic: there is no `is_required`-style flag, and an unreachable Kafka cluster does not block startup or request handling. Producer health is reflected via the manager's `IsHealthy()` state (surfaced in health/metrics), while the manager retries delivery in the background.

## Event schema

The event is a flat JSON document (no nested objects), published to the `air.spend_logs` topic keyed by `request_id`. It builds on the existing `SpendLogEntry` used for Postgres, but expands the single `Metadata` JSON blob into typed, flat fields and adds credential/server/timing information that Postgres currently discards.

| Field                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Type                | Description                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | --------------------------------------------------------------------- |
| `request_id`                                                                                                                                                                                                                                                                                                                                                                                                                                           | string              | Request UUID; also the Kafka message key                              |
| `start_time` / `end_time`                                                                                                                                                                                                                                                                                                                                                                                                                              | timestamp           | Request start/end                                                     |
| `completion_start_time`                                                                                                                                                                                                                                                                                                                                                                                                                                | timestamp, nullable | TTFT — time of first streamed token, null if not streaming            |
| `duration_ms`                                                                                                                                                                                                                                                                                                                                                                                                                                          | uint                | `end_time - start_time` in milliseconds                               |
| `ttft_ms`                                                                                                                                                                                                                                                                                                                                                                                                                                              | uint, nullable      | `completion_start_time - start_time` in milliseconds                  |
| `call_type`                                                                                                                                                                                                                                                                                                                                                                                                                                            | string              | API endpoint, e.g. `/v1/chat/completions`                             |
| `api_base`                                                                                                                                                                                                                                                                                                                                                                                                                                             | string              | Upstream base URL the request was actually sent to                    |
| `status` / `http_status` / `error_message` / `error_class`                                                                                                                                                                                                                                                                                                                                                                                             | string/int          | Outcome of the request                                                |
| `model` / `real_model` / `model_id` / `model_group`                                                                                                                                                                                                                                                                                                                                                                                                    | string              | Requested alias, resolved model, credential-qualified id, model group |
| `credential_name` / `credential_type` / `credential_base_url` / `credential_is_proxy_request` / `credential_actual_credential_name`                                                                                                                                                                                                                                                                                                                    | string/bool         | Which credential served the request                                   |
| `server_router_id` / `server_version` / `server_commit`                                                                                                                                                                                                                                                                                                                                                                                                | string              | Which AIR instance/build handled the request                          |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `audio_input_tokens`, `audio_output_tokens`, `cached_input_tokens`, `cached_audio_input_tokens`, `cache_creation_tokens`, `cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `cached_output_tokens`, `reasoning_tokens`, `accepted_prediction_tokens`, `rejected_prediction_tokens`, `image_count`, `image_tokens`, `output_image_tokens`, `web_search_requests`, `web_search_context_size` | uint/string         | Token and tool usage breakdown                                        |
| `input_cost`, `output_cost`, `audio_input_cost`, `audio_output_cost`, `reasoning_cost`, `cached_input_cost`, `cache_creation_cost`, `cached_output_cost`, `prediction_cost`, `image_cost`, `web_search_cost`, `total_cost`                                                                                                                                                                                                                             | float               | Cost breakdown matching the token/tool fields above                   |
| `api_key_hash`                                                                                                                                                                                                                                                                                                                                                                                                                                         | string              | SHA-256 of the API key, same hashing as Postgres                      |
| `user_id` / `team_id` / `organization_id` / `end_user` / `key_alias` / `user_alias` / `team_alias`                                                                                                                                                                                                                                                                                                                                                     | string              | Identity/attribution fields                                           |
| `requester_ip` / `session_id` / `overhead_ms`                                                                                                                                                                                                                                                                                                                                                                                                          | string/float        | Request origin and router overhead                                    |
| `body_captured` / `body_request_bytes` / `body_response_bytes`                                                                                                                                                                                                                                                                                                                                                                                         | bool/uint           | **Placeholder fields — see "Out of scope" below**                     |

See section 4 of the design document for the complete field-by-field JSON example.

## ClickHouse schema

A reference DDL — a `Kafka`-engine table reading `air.spend_logs`, a `MergeTree` table, and a `MATERIALIZED VIEW` connecting the two — is provided at `clickhouse/init/01_spend_logs.sql`. It matches the design document's section 8 schema (columns line up 1:1 with the flat JSON event above).

This DDL is a reference/example for local development and for DBAs setting up a production ClickHouse cluster — Auto AI Router itself does not create, own, or administer this schema. Retention (`TTL`) and Kafka topic partitioning are likewise left to whoever operates the target cluster; the router's only responsibility is producing well-formed events to the topic.

### Upgrading an existing ClickHouse pipeline

The init DDL runs only for an empty ClickHouse data directory. Before deploying a router that publishes the cache and Web Search fields, pause AIR Kafka publishing and apply [`clickhouse/migrations/002_cache_web_search_columns.sql`](../../clickhouse/migrations/002_cache_web_search_columns.sql).

The migration detaches the materialized view, adds the fields to both the MergeTree and Kafka tables, then reattaches the view. For replicated production tables, add the cluster-specific `ON CLUSTER` clause required by your deployment.

After re-enabling publishing, send one synthetic event with non-zero `cached_audio_input_tokens`, `cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `web_search_requests`, and `web_search_cost`. Verify that the row appears in `air.spend_logs` and that `system.kafka_consumers` reports no parse exceptions before completing the rollout.

PostgreSQL intentionally retains the upstream LiteLLM schema. Detailed cache and Web Search values live in `LiteLLM_SpendLogs.metadata`; the typed fields are available in Kafka and ClickHouse.

**Replication:** the `air.spend_logs` table uses plain `MergeTree` because `docker-compose.kafka.yml` runs a single, unreplicated ClickHouse node. For a production cluster with more than one replica, swap it for `ReplicatedMergeTree` (or a `Replicated` database engine) once Keeper/ZooKeeper and `{shard}`/`{replica}` macros are set up — otherwise a node failure loses spend/billing data. See the comment above the `CREATE TABLE air.spend_logs` statement in `clickhouse/init/01_spend_logs.sql` for the exact engine syntax.

## Running locally

A self-contained Kafka (KRaft, single node) + ClickHouse stack for local development is provided in `docker-compose.kafka.yml` at the repository root. It auto-applies the ClickHouse schema above on first start via the standard `docker-entrypoint-initdb.d` mechanism.

```bash
docker compose -f docker-compose.kafka.yml up -d
```

This exposes Kafka on `localhost:9092` and ClickHouse on `localhost:8123` (HTTP) / `localhost:9000` (native). Point `KAFKA_BROKERS=localhost:9092` (from the host) or `kafka:9092`/`kafka:29092` (from another container on the same compose network) at it to test the pipeline end to end.

## Out of scope: request/response bodies for successful requests

Capturing and storing request/response bodies for *every* request (success included) is still out of scope — it would make `air.spend_logs` both heavier and much longer-retained than it needs to be, and is a distinct problem from "help me debug this one failure" (see "Raw bodies" below, which covers that narrower case). The event fields `body_captured`, `body_request_bytes`, and `body_response_bytes` remain reserved placeholders: `body_captured: false`, byte counts `0`, no delivery/storage/PII-handling logic behind them.

## Raw bodies (separate write-path)

The raw **provider response** body for a failed request is published separately from the spend event, to its own Kafka topic (`kafka.raw_bodies`, default topic `raw-bodies`, staged in ClickHouse as `air.raw_bodies_kafka`/`air.raw_bodies_read`) as a flat `kafkalog.RawBodyEvent`:

| Field              | Type              | Description                                                                                                         |
| ------------------ | ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `request_id`       | string            | Same value as the matching `SpendEvent.request_id` / `air.errors.request_id`; also the Kafka message key             |
| `server_router_id` | string            | Same value as the matching spend event. Join on `(request_id, server_router_id)`, not `request_id` alone — see below |
| `start_time`       | timestamp         | Same value as the matching `SpendEvent.start_time` — when this router started processing the request                 |
| `end_time`         | timestamp         | Same value as the matching `SpendEvent.end_time` — when this router finished processing it; for a failure row, effectively when the error was finalized/detected |
| `http_status`      | int               | HTTP status of the request. Usually ≥400 on a failure row, but **not always** — a mid-stream SSE error (provider returns HTTP 200, then sends an error event inside the stream) is a genuine failure with a 2xx `http_status`         |
| `error_class`      | string, omitempty | Same classification as `SpendEvent.error_class`, gated on the same canonical success/failure outcome (not on `http_status` directly, for the 2xx-mid-stream-failure reason above). Empty on success rows (see `store_only_errors` below) |
| `response_body`    | string, omitempty | Raw upstream provider error body, capped at 16 KiB (uncapped relative to `error_message`'s 512 bytes)                 |
| `client_response_body` | string, omitempty | What the router actually sent back to the client for this failure, capped the same way as `response_body`         |
| `request_body`     | string, omitempty | The client's own request body, **with prompt/message content redacted** (see below). Only populated when `kafka.raw_bodies.store_raw_body: true`        |

**`response_body` and `client_response_body` are usually different values, on purpose.** `maskedUpstreamErrorBody` (`internal/proxy/errors.go`) replaces the provider's own error text with a short, pre-vetted message for essentially every 4xx/5xx response — unconditionally, not gated by credential type — specifically so provider internals are never echoed back to the client. `response_body` is what the provider actually said; `client_response_body` is what the client was told instead. They're identical only when a mid-stream error is detected *after* the response has already committed and streamed those exact bytes to the client live — at that point there's nothing left to mask in hindsight.

**`request_body` is a separate, explicit opt-in — off by default — and even then never carries prompt content by default.** `kafka.raw_bodies.store_raw_body` (default `false`) must be turned on deliberately for `request_body` to ever be non-empty; leaving it off (the default) preserves the original failure-response-only design exactly. When it is on, `redactRequestBodyForLogging` (`internal/proxy/proxy_helpers.go`) still strips the actual conversation content before it ever reaches `logCtx`: `messages`, `system`, `prompt`, `input`, `contents`, and `instructions` are replaced with a shape-preserving placeholder (turn count and `role` kept, `content` replaced with `"[REDACTED]"`) at any depth in the JSON. Everything else — `model`, `tools`, `tool_choice`, `temperature`, `max_tokens`, `stream`, `response_format`, and so on — is left untouched, since those describe the request's shape, not its content. If the body isn't valid JSON (multipart, binary, malformed), nothing is captured at all rather than risk shipping something unredacted.

`kafka.raw_bodies.redact_sensitive_fields` (default `true`) is the toggle that controls this: set it to `false` and `request_body` captures the request verbatim, prompt included. This is a deliberate escape hatch, not a recommended default — it exists for a short-lived, access-controlled debugging session where the actual prompt is genuinely needed, at the cost of reintroducing exactly the privacy exposure the redaction exists to avoid. If you need to reproduce a specific failure's actual prompt without turning this off, correlate `request_id` with your own request logging outside AIR under whatever consent/retention rules already govern that data.

Three independent toggles control scope, all under `kafka.raw_bodies`:

| Toggle | Default | Effect when changed |
| --- | --- | --- |
| `store_raw_body` | `false` | `true` additionally captures the client's request body into `request_body` |
| `store_only_errors` | `true` | `false` publishes an event for *every* request, not just failures — `error_class`/`response_body`/`client_response_body` stay empty on success rows; mainly useful once `store_raw_body` is also on and the goal is capturing requests generally, not just failures |
| `redact_sensitive_fields` | `true` | `false` disables the prompt/message redaction above — `request_body` then carries the request verbatim. Only meaningful when `store_raw_body` is also `true` |

With all three left at their defaults, behavior is unchanged from the original design: failure-only, provider/client response bodies only, no request content.

This is deliberately **not** a field on `SpendEvent`/`air.spend_logs`:

- Spend/billing rows are kept for a long time (retention measured in months/years) and are meant to stay light; raw bodies are bulky and only useful for a short debugging window, so they need their own, independently configurable ClickHouse retention (`TTL`) — a separate table gives you that for free, a shared one doesn't.
- It's independently toggleable (`kafka.raw_bodies.enabled`) precisely so it can be turned on temporarily while debugging without touching the always-on spend-log path, and turned back off (or left permanently off, the default) without affecting spend/billing at all.
- By default it only fires for failures (`status == "failure"`) — no additional exposure of completion content for successful traffic beyond what already exists today, unless `store_only_errors` is explicitly disabled.

**Join back to the spend event / `air.errors` on `(request_id, server_router_id)`, not `request_id` alone.** In a chained deployment (one AIR instance proxying to another as an upstream credential), `request_id` is derived from the upstream provider's own response id and is echoed back through every hop unchanged — two different hops logging the same logical request in the same millisecond can share a `request_id`. `server_router_id` (the hostname/pod of the specific instance that logged the row) disambiguates them; see the `ORDER BY` on `air.logs` in your ClickHouse schema for the same reasoning applied to the spend table.

A reference join view for a ClickHouse deployment with the matching `air.errors`/`air.raw_bodies` tables:

```sql
CREATE VIEW air.errors_with_raw AS
SELECT e.*, b.response_body, b.client_response_body, b.request_body
FROM air.errors AS e
LEFT JOIN air.raw_bodies AS b
    ON e.request_id = b.request_id AND e.server_router_id = b.server_router_id;
```

## Implementation notes

The Kafka producer/event-builder lives in `internal/kafkalog` (manager, event, config, async logger). Both write-paths share one generic `Logger[T]` engine (`T` is a pointer type implementing `Keyed`, i.e. `*SpendEvent` or `*RawBodyEvent`) — same queue/batch/retry/DLQ/health-check machinery, just parameterized over the event type and pointed at a different topic (`Manager`/`DefaultManager` for spend events, `RawBodyManager`/`DefaultRawBodyManager` for error bodies).

Both are wired into the existing spend-logging call site in `internal/proxy/proxy_log.go` (`logSpendToLiteLLMDB`), which already runs from every place a request's log is finalized — no changes to individual call sites are needed to add either Kafka publish. The raw-body publish additionally requires `kafka.raw_bodies.enabled`, plus either `status == "failure"` or `store_only_errors: false`.
