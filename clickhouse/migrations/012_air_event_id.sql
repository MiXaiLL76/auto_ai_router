-- Apply after 011_attempts.sql. Preserve stored rows and Kafka consumer groups.
-- Match the Kafka settings below to the deployment before applying.

ALTER TABLE air.spend_logs ADD COLUMN IF NOT EXISTS air_event_id String AFTER request_id;
ALTER TABLE air.attempts ADD COLUMN IF NOT EXISTS air_event_id String AFTER request_id;

DROP TABLE IF EXISTS air.spend_logs_mv;
DROP TABLE IF EXISTS air.spend_logs_kafka;

CREATE TABLE air.spend_logs_kafka
(
    request_id String,
    air_event_id String,
    trace_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    duration_ms UInt32,
    ttft_ms Nullable(UInt32),
    upstream_send_ms Nullable(UInt32),
    retry_ms Nullable(UInt32),
    conn_wait_ms Nullable(UInt32),
    conn_reused Nullable(UInt8),
    ttfb_ms Nullable(UInt32),
    upstream_attempts UInt8,
    exit LowCardinality(String),
    exit_rtt_ms Nullable(UInt32),
    exit_connect_ms Nullable(UInt32),

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
    service_name LowCardinality(String),

    prompt_tokens UInt32,
    completion_tokens UInt32,
    total_tokens UInt32,
    audio_input_tokens UInt32,
    audio_output_tokens UInt32,
    cached_input_tokens UInt32,
    cached_audio_input_tokens UInt32,
    cache_creation_tokens UInt32,
    cache_creation_5m_tokens UInt32,
    cache_creation_1h_tokens UInt32,
    cache_creation_12h_tokens UInt32,
    cache_type LowCardinality(Nullable(String)),
    cached_output_tokens UInt32,
    reasoning_tokens UInt32,
    accepted_prediction_tokens UInt32,
    rejected_prediction_tokens UInt32,
    image_count UInt32,
    image_tokens UInt32,
    video_input_tokens UInt32,
    output_image_tokens UInt32,
    web_search_requests UInt32,
    web_search_context_size Nullable(String),
    x_search_calls UInt32,
    x_search_posts UInt32,
    x_search_profiles UInt32,
    code_execution_calls UInt32,
    attachment_search_calls UInt32,
    collections_search_calls UInt32,
    mcp_calls UInt32,
    image_tool_generations UInt32,
    image_tool_edits UInt32,

    input_cost Float64,
    output_cost Float64,
    audio_input_cost Float64,
    audio_output_cost Float64,
    reasoning_cost Float64,
    cached_input_cost Float64,
    explicit_cache_read_cost Float64,
    cache_creation_cost Float64,
    cached_output_cost Float64,
    prediction_cost Float64,
    image_cost Float64,
    video_input_cost Float64,
    web_search_cost Float64,
    x_search_cost Float64,
    code_execution_cost Float64,
    attachment_search_cost Float64,
    collections_search_cost Float64,
    image_tool_cost Float64,
    tool_usage_cost Float64,
    total_cost Float64,
    provider_reported_cost Nullable(Float64),

    api_key_hash String,
    user_id String,
    team_id String,
    organization_id String,
    end_user String,
    key_alias Nullable(String),
    user_alias Nullable(String),
    team_alias Nullable(String),
    organization_alias Nullable(String),

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
CREATE MATERIALIZED VIEW air.spend_logs_mv TO air.spend_logs AS
SELECT * REPLACE (if(tool_usage_cost = 0, web_search_cost, tool_usage_cost) AS tool_usage_cost)
FROM air.spend_logs_kafka;

DROP TABLE IF EXISTS air.attempts_mv;
DROP TABLE IF EXISTS air.attempts_kafka;

CREATE TABLE air.attempts_kafka
(
    request_id String,
    air_event_id String,
    trace_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    service_name LowCardinality(String),
    server_router_id LowCardinality(String),
    call_type String,
    organization_id String,
    team_id String,
    api_key_hash String,
    model_group String,
    status LowCardinality(String),
    http_status UInt16,
    attempts String
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:29092',
    kafka_topic_list = 'air.spend_logs',
    kafka_group_name = 'clickhouse_air_attempts',
    kafka_format = 'JSONEachRow',
    date_time_input_format = 'best_effort',
    input_format_json_read_arrays_as_strings = 1,
    kafka_num_consumers = 2,
    kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW air.attempts_mv TO air.attempts AS
SELECT
    start_time,
    end_time,
    request_id,
    air_event_id,
    trace_id,
    service_name,
    server_router_id,
    call_type,
    organization_id,
    team_id,
    api_key_hash,
    model_group,
    request_status,
    request_http_status,
    toUInt8(number) AS attempt,
    a.credential AS credential_name,
    a.exit AS exit,
    a.kind AS kind,
    a.retry_reason AS retry_reason,
    a.http_status AS http_status,
    a.upstream_request_id AS upstream_request_id,
    a.error_type AS error_type,
    a.error_message AS error_message,
    ifNull(parseDateTime64BestEffortOrNull(a.start_time, 3), start_time) AS attempt_time,
    a.duration_ms AS duration_ms,
    a.ttfb_ms AS ttfb_ms,
    a.conn_reused AS conn_reused,
    a.is_final AS is_final
FROM
(
    SELECT
        start_time,
        end_time,
        request_id,
        air_event_id,
        trace_id,
        service_name,
        server_router_id,
        call_type,
        organization_id,
        team_id,
        api_key_hash,
        model_group,
        status AS request_status,
        http_status AS request_http_status,
        JSONExtract(attempts, 'Array(Tuple(credential String, exit String, kind String, retry_reason String, http_status UInt16, upstream_request_id String, error_type String, error_message String, start_time String, duration_ms UInt32, ttfb_ms Nullable(UInt32), conn_reused Nullable(UInt8), is_final UInt8))') AS list
    FROM air.attempts_kafka
    WHERE length(_error) = 0
)
ARRAY JOIN list AS a, arrayEnumerate(list) AS number;

-- Rows from older producers have an empty ID. Keep every such row.
-- Retries of the same Kafka event carry identical rows, so any copy suffices.
CREATE VIEW IF NOT EXISTS air.spend_logs_deduplicated AS
SELECT * FROM air.spend_logs WHERE air_event_id = ''
UNION ALL
SELECT * FROM
(
    SELECT * FROM air.spend_logs WHERE air_event_id != ''
    LIMIT 1 BY air_event_id
);

CREATE VIEW IF NOT EXISTS air.attempts_deduplicated AS
SELECT * FROM air.attempts WHERE air_event_id = ''
UNION ALL
SELECT * FROM
(
    SELECT * FROM air.attempts WHERE air_event_id != ''
    LIMIT 1 BY air_event_id, attempt
);
