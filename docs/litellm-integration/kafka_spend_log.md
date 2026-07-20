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

| Field                                                                                                                                                                                                                                                                                                           | Type                | Description                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------- |
| `request_id`                                                                                                                                                                                                                                                                                                    | string              | Request UUID; also the Kafka message key                              |
| `start_time` / `end_time`                                                                                                                                                                                                                                                                                       | timestamp           | Request start/end                                                     |
| `completion_start_time`                                                                                                                                                                                                                                                                                         | timestamp, nullable | TTFT — time of first streamed token, null if not streaming            |
| `duration_ms`                                                                                                                                                                                                                                                                                                   | uint                | `end_time - start_time` in milliseconds                               |
| `ttft_ms`                                                                                                                                                                                                                                                                                                       | uint, nullable      | `completion_start_time - start_time` in milliseconds                  |
| `call_type`                                                                                                                                                                                                                                                                                                     | string              | API endpoint, e.g. `/v1/chat/completions`                             |
| `api_base`                                                                                                                                                                                                                                                                                                      | string              | Upstream base URL the request was actually sent to                    |
| `status` / `http_status` / `error_message` / `error_class`                                                                                                                                                                                                                                                      | string/int          | Outcome of the request                                                |
| `model` / `real_model` / `model_id` / `model_group`                                                                                                                                                                                                                                                             | string              | Requested alias, resolved model, credential-qualified id, model group |
| `credential_name` / `credential_type` / `credential_base_url` / `credential_is_proxy_request` / `credential_actual_credential_name`                                                                                                                                                                             | string/bool         | Which credential served the request                                   |
| `server_router_id` / `server_version` / `server_commit`                                                                                                                                                                                                                                                         | string              | Which AIR instance/build handled the request                          |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `audio_input_tokens`, `audio_output_tokens`, `cached_input_tokens`, `cache_creation_tokens`, `cached_output_tokens`, `reasoning_tokens`, `accepted_prediction_tokens`, `rejected_prediction_tokens`, `image_count`, `image_tokens`, `output_image_tokens` | uint                | Token usage breakdown                                                 |
| `input_cost`, `output_cost`, `audio_input_cost`, `audio_output_cost`, `reasoning_cost`, `cached_input_cost`, `cache_creation_cost`, `cached_output_cost`, `prediction_cost`, `image_cost`, `total_cost`                                                                                                         | float               | Cost breakdown matching the token fields above                        |
| `api_key_hash`                                                                                                                                                                                                                                                                                                  | string              | SHA-256 of the API key, same hashing as Postgres                      |
| `user_id` / `team_id` / `organization_id` / `end_user` / `key_alias` / `user_alias` / `team_alias`                                                                                                                                                                                                              | string              | Identity/attribution fields                                           |
| `requester_ip` / `session_id` / `overhead_ms`                                                                                                                                                                                                                                                                   | string/float        | Request origin and router overhead                                    |
| `body_captured` / `body_request_bytes` / `body_response_bytes`                                                                                                                                                                                                                                                  | bool/uint           | **Placeholder fields — see "Out of scope" below**                     |

See section 4 of the design document for the complete field-by-field JSON example.

## ClickHouse schema

A reference DDL — a `Kafka`-engine table reading `air.spend_logs`, a `MergeTree` table, and a `MATERIALIZED VIEW` connecting the two — is provided at `clickhouse/init/01_spend_logs.sql`. It matches the design document's section 8 schema (columns line up 1:1 with the flat JSON event above).

This DDL is a reference/example for local development and for DBAs setting up a production ClickHouse cluster — Auto AI Router itself does not create, own, or administer this schema. Retention (`TTL`) and Kafka topic partitioning are likewise left to whoever operates the target cluster; the router's only responsibility is producing well-formed events to the topic.

**Replication:** the `air.spend_logs` table uses plain `MergeTree` because `docker-compose.kafka.yml` runs a single, unreplicated ClickHouse node. For a production cluster with more than one replica, swap it for `ReplicatedMergeTree` (or a `Replicated` database engine) once Keeper/ZooKeeper and `{shard}`/`{replica}` macros are set up — otherwise a node failure loses spend/billing data. See the comment above the `CREATE TABLE air.spend_logs` statement in `clickhouse/init/01_spend_logs.sql` for the exact engine syntax.

## Running locally

A self-contained Kafka (KRaft, single node) + ClickHouse stack for local development is provided in `docker-compose.kafka.yml` at the repository root. It auto-applies the ClickHouse schema above on first start via the standard `docker-entrypoint-initdb.d` mechanism.

```bash
docker compose -f docker-compose.kafka.yml up -d
```

This exposes Kafka on `localhost:9092` and ClickHouse on `localhost:8123` (HTTP) / `localhost:9000` (native). Point `KAFKA_BROKERS=localhost:9092` (from the host) or `kafka:9092`/`kafka:29092` (from another container on the same compose network) at it to test the pipeline end to end.

## Out of scope: request/response bodies

Capturing and storing request/response bodies is explicitly out of scope for this feature — it is planned as a separate design/PR. The event fields `body_captured`, `body_request_bytes`, and `body_response_bytes` exist in the schema as reserved placeholders only: they are populated with mock/zero values (`body_captured: false`, byte counts `0`), with no delivery, storage, or PII-handling logic behind them.

## Implementation notes

The Kafka producer/event-builder lives in `internal/kafkalog` (manager, event, config, async logger). It is wired into the existing spend-logging call site in `internal/proxy/proxy_log.go` (`logSpendToLiteLLMDB`), which already runs from every place a request's log is finalized — no changes to individual call sites are needed to add the Kafka publish.
