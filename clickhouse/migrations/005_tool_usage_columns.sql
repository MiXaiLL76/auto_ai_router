-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to the
-- built-in server-side tool columns (xAI server_side_tool_usage_details):
--   x_search_calls, x_search_posts, x_search_profiles,
--   code_execution_calls, attachment_search_calls, collections_search_calls,
--   mcp_calls, image_tool_generations, image_tool_edits -- per-tool usage;
--   x_search_cost, code_execution_cost, attachment_search_cost,
--   collections_search_cost, image_tool_cost        -- per-tool charges;
--   tool_usage_cost        -- sum of web_search_cost and the charges above,
--                             already contained in total_cost;
--   provider_reported_cost -- the provider's own cost of the request (xAI
--                             cost_in_usd_ticks, aggregators' usage.cost),
--                             for reconciliation only, NULL when not reported.
--
-- On the MergeTree table tool_usage_cost defaults to web_search_cost, so rows
-- written before this migration (when web search was the only priced tool)
-- read back the right total. That DEFAULT only covers existing parts: an
-- event from an AIR pod that predates these fields still reaches the
-- MergeTree table through the Kafka table, which fills the missing field with
-- an explicit 0. The materialized view therefore falls back to web_search_cost
-- for such events, so tool_usage_cost stays right through the rollout window
-- (an event from a current pod never has tool_usage_cost below
-- web_search_cost, which it includes).
--
-- The Kafka table engine does not support ALTER ... ADD COLUMN (ClickHouse
-- fails with NOT_IMPLEMENTED), so air.spend_logs_kafka and the materialized
-- view are dropped and recreated; only the MergeTree table is altered in
-- place. No events are lost: consumer offsets are committed in Kafka under
-- kafka_group_name, so the recreated table resumes where the old one stopped
-- as long as kafka_group_name is unchanged.
--
-- The CREATE below uses the reference SETTINGS from
-- clickhouse/init/01_spend_logs.sql. Before running this on a real cluster,
-- copy your current SETTINGS from `SHOW CREATE TABLE air.spend_logs_kafka`
-- (broker list, topic, group name, consumer count) into it, and add the
-- ON CLUSTER clause your deployment needs.
--
-- Run it before (or together with) rolling out the AIR version that emits
-- these fields, with AIR Kafka publishing paused. Safe to re-run on its own
-- only while it is the last applied migration: never run 005 again once
-- 006_video_input_columns.sql has been applied, for the same reason
-- 002_cache_web_search_columns.sql's doc comment spells out -- it would
-- rebuild air.spend_logs_kafka from only 005's column set, narrowing it
-- back below whatever the later migration added, and break ingestion with
-- NUMBER_OF_COLUMNS_DOESNT_MATCH until that later migration is re-applied.

DROP TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS x_search_calls UInt32 DEFAULT 0 AFTER web_search_context_size,
    ADD COLUMN IF NOT EXISTS x_search_posts UInt32 DEFAULT 0 AFTER x_search_calls,
    ADD COLUMN IF NOT EXISTS x_search_profiles UInt32 DEFAULT 0 AFTER x_search_posts,
    ADD COLUMN IF NOT EXISTS code_execution_calls UInt32 DEFAULT 0 AFTER x_search_profiles,
    ADD COLUMN IF NOT EXISTS attachment_search_calls UInt32 DEFAULT 0 AFTER code_execution_calls,
    ADD COLUMN IF NOT EXISTS collections_search_calls UInt32 DEFAULT 0 AFTER attachment_search_calls,
    ADD COLUMN IF NOT EXISTS mcp_calls UInt32 DEFAULT 0 AFTER collections_search_calls,
    ADD COLUMN IF NOT EXISTS image_tool_generations UInt32 DEFAULT 0 AFTER mcp_calls,
    ADD COLUMN IF NOT EXISTS image_tool_edits UInt32 DEFAULT 0 AFTER image_tool_generations,
    ADD COLUMN IF NOT EXISTS x_search_cost Float64 DEFAULT 0 AFTER web_search_cost,
    ADD COLUMN IF NOT EXISTS code_execution_cost Float64 DEFAULT 0 AFTER x_search_cost,
    ADD COLUMN IF NOT EXISTS attachment_search_cost Float64 DEFAULT 0 AFTER code_execution_cost,
    ADD COLUMN IF NOT EXISTS collections_search_cost Float64 DEFAULT 0 AFTER attachment_search_cost,
    ADD COLUMN IF NOT EXISTS image_tool_cost Float64 DEFAULT 0 AFTER collections_search_cost,
    ADD COLUMN IF NOT EXISTS tool_usage_cost Float64 DEFAULT web_search_cost AFTER image_tool_cost,
    ADD COLUMN IF NOT EXISTS provider_reported_cost Nullable(Float64) AFTER total_cost;

DROP TABLE IF EXISTS air.spend_logs_kafka;

CREATE TABLE air.spend_logs_kafka
(
    request_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    duration_ms UInt32,
    ttft_ms Nullable(UInt32),
    upstream_send_ms Nullable(UInt32),

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
    cached_audio_input_tokens UInt32,
    cache_creation_tokens UInt32,
    cache_creation_5m_tokens UInt32,
    cache_creation_1h_tokens UInt32,
    cache_type LowCardinality(Nullable(String)),
    cached_output_tokens UInt32,
    reasoning_tokens UInt32,
    accepted_prediction_tokens UInt32,
    rejected_prediction_tokens UInt32,
    image_count UInt32,
    image_tokens UInt32,
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

    requester_ip String,
    session_id String,
    overhead_ms Float64,
    body_captured UInt8,
    body_request_bytes UInt32,
    body_response_bytes UInt32
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
    kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW air.spend_logs_mv TO air.spend_logs AS
SELECT * REPLACE (if(tool_usage_cost = 0, web_search_cost, tool_usage_cost) AS tool_usage_cost)
FROM air.spend_logs_kafka;
