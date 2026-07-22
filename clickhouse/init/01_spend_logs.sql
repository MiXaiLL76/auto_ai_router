-- ClickHouse schema for the Kafka -> ClickHouse spend-log analytics pipeline
-- (air.spend_logs). This is the reference DDL from
-- auto_ai_router_kafka_spend_log_tz.md, section 8, verbatim except for
-- `kafka_broker_list`, which is adapted to the `kafka` service name/port used
-- by docker-compose.kafka.yml (the internal PLAINTEXT listener, kafka:29092).
--
-- Mounted into clickhouse-server via docker-entrypoint-initdb.d, so it only
-- runs automatically on the container's first start (empty data directory).
--
-- As noted in the TZ (section 8): AIR itself does not create or administer
-- this schema in production -- this file is a reference example for local
-- dev/testing and for DBAs, not an automated migration run by the router.
-- Retention (TTL) and Kafka topic partitioning are likewise out of scope for
-- AIR and are left to whoever operates the target ClickHouse/Kafka cluster.
--
-- Column list below matches internal/kafkalog.SpendEvent (internal/kafkalog/event.go)
-- field-for-field. Columns backed by a Go field with `,omitempty` are Nullable,
-- since JSONEachRow will see the key entirely absent (not `null`) for a zero value.
-- ClickHouse's `CREATE TABLE ... (LIKE other_table)` used as a parenthesized column
-- list is not valid DDL (that's MySQL/Postgres syntax) and fails with
-- SYNTAX_ERROR (code 62) -- both tables below spell out the same column list
-- explicitly instead.

CREATE DATABASE IF NOT EXISTS air;

CREATE TABLE air.spend_logs_kafka
(
    request_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    duration_ms UInt32,
    ttft_ms Nullable(UInt32),

    call_type String,
    api_base String,
    status LowCardinality(String),
    http_status UInt16,
    error_message Nullable(String),
    error_class LowCardinality(Nullable(String)),

    model String,
    real_model String,
    model_id String,
    model_group String,

    credential_name LowCardinality(String),
    credential_type LowCardinality(String),
    credential_base_url String,
    credential_is_proxy_request UInt8,
    credential_actual_credential_name Nullable(String),

    server_router_id LowCardinality(String),
    server_version String,
    server_commit String,

    prompt_tokens UInt32,
    completion_tokens UInt32,
    total_tokens UInt32,
    audio_input_tokens UInt32,
    audio_output_tokens UInt32,
    cached_input_tokens UInt32,
    cache_creation_tokens UInt32,
    cached_output_tokens UInt32,
    reasoning_tokens UInt32,
    accepted_prediction_tokens UInt32,
    rejected_prediction_tokens UInt32,
    image_count UInt32,
    image_tokens UInt32,
    output_image_tokens UInt32,

    input_cost Float64,
    output_cost Float64,
    audio_input_cost Float64,
    audio_output_cost Float64,
    reasoning_cost Float64,
    cached_input_cost Float64,
    cache_creation_cost Float64,
    cached_output_cost Float64,
    prediction_cost Float64,
    image_cost Float64,
    total_cost Float64,

    api_key_hash String,
    user_id String,
    team_id String,
    organization_id String,
    end_user String,
    key_alias Nullable(String),
    user_alias Nullable(String),
    team_alias Nullable(String),

    requester_ip String,
    session_id String,
    overhead_ms Float64,
    body_captured UInt8,              -- всегда 0 пока; поле-заглушка под будущий PR
    body_request_bytes UInt32,        -- всегда 0 пока; поле-заглушка под будущий PR
    body_response_bytes UInt32        -- всегда 0 пока; поле-заглушка под будущий PR
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:29092',
    kafka_topic_list = 'air.spend_logs',
    kafka_group_name = 'clickhouse_air_spend_logs',
    kafka_format = 'JSONEachRow',
    -- Go's json.Marshal writes RFC3339 timestamps (e.g. "2026-07-15T10:00:00.000Z"),
    -- which DateTime64 does not parse by default -- best_effort is required, or
    -- every message ends up in the `_error` stream despite being well-formed.
    date_time_input_format = 'best_effort',
    -- Matches the "air.spend_logs" topic's partition count (2, see
    -- docker-compose.kafka.yml's KAFKA_NUM_PARTITIONS for local dev). In
    -- production this must track whatever the topic is actually provisioned
    -- with -- more consumers than partitions just sit idle.
    kafka_num_consumers = 2,
    kafka_handle_error_mode = 'stream';   -- невалидные сообщения не теряются молча

-- Plain MergeTree is correct here because docker-compose.kafka.yml stands up
-- a single, unreplicated ClickHouse node (no Keeper/ZooKeeper, no {shard}/
-- {replica} macros configured) -- ReplicatedMergeTree would either fail to
-- create or provide zero actual redundancy with one replica.
--
-- For a production cluster with more than one ClickHouse replica, swap the
-- engine for ReplicatedMergeTree (or use a `Replicated` database engine so
-- every table under it is replicated automatically) so a node failure
-- doesn't lose spend/billing data. Typical form, once Keeper and macros are
-- configured on the cluster:
--
--   ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/air/spend_logs', '{replica}')
--
-- This is exactly the kind of cluster-topology decision the TZ (section 8,
-- decision 11.3) leaves to whoever operates the target ClickHouse cluster --
-- AIR's reference DDL intentionally stays engine-agnostic about it beyond
-- this comment.
CREATE TABLE air.spend_logs
(
    request_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    duration_ms UInt32,
    ttft_ms Nullable(UInt32),

    call_type String,
    api_base String,
    status LowCardinality(String),
    http_status UInt16,
    error_message Nullable(String),
    error_class LowCardinality(Nullable(String)),

    model String,
    real_model String,
    model_id String,
    model_group String,

    credential_name LowCardinality(String),
    credential_type LowCardinality(String),
    credential_base_url String,
    credential_is_proxy_request UInt8,
    credential_actual_credential_name Nullable(String),

    server_router_id LowCardinality(String),
    server_version String,
    server_commit String,

    prompt_tokens UInt32,
    completion_tokens UInt32,
    total_tokens UInt32,
    audio_input_tokens UInt32,
    audio_output_tokens UInt32,
    cached_input_tokens UInt32,
    cache_creation_tokens UInt32,
    cached_output_tokens UInt32,
    reasoning_tokens UInt32,
    accepted_prediction_tokens UInt32,
    rejected_prediction_tokens UInt32,
    image_count UInt32,
    image_tokens UInt32,
    output_image_tokens UInt32,

    input_cost Float64,
    output_cost Float64,
    audio_input_cost Float64,
    audio_output_cost Float64,
    reasoning_cost Float64,
    cached_input_cost Float64,
    cache_creation_cost Float64,
    cached_output_cost Float64,
    prediction_cost Float64,
    image_cost Float64,
    total_cost Float64,

    api_key_hash String,
    user_id String,
    team_id String,
    organization_id String,
    end_user String,
    key_alias Nullable(String),
    user_alias Nullable(String),
    team_alias Nullable(String),

    requester_ip String,
    session_id String,
    overhead_ms Float64,
    body_captured UInt8,
    body_request_bytes UInt32,
    body_response_bytes UInt32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(start_time)
ORDER BY (start_time, team_id, model)
-- TTL expressions must resolve to DateTime/Date, not DateTime64 -- toDateTime()
-- truncates to seconds, which is fine at a 90-day retention granularity.
TTL toDateTime(start_time) + INTERVAL 90 DAY;  -- пример; конкретное значение и владение таблицей — на стороне DBA/CH-кластера, не AIR

CREATE MATERIALIZED VIEW air.spend_logs_mv TO air.spend_logs AS
SELECT * FROM air.spend_logs_kafka;
