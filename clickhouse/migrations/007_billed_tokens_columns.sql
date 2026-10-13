-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to the
-- event schema that reports the token counts the cost calculator derives
-- itself instead of taking them from the usage columns (exact rules in
-- docs/litellm-integration/pricing.md, "Spend storage contract"):
--   billed_input_tokens       -- regular input billed (input_cost)
--   billed_output_tokens      -- regular output billed (output_cost)
--   billed_image_tokens       -- input image tokens billed (part of image_cost)
--   billed_video_input_tokens -- input video tokens billed (video_input_cost)
-- Rows written before this migration, and events from AIR pods that predate
-- these fields, read back 0.
--
-- The Kafka table engine does not support ALTER ... ADD COLUMN (ClickHouse
-- fails with NOT_IMPLEMENTED) -- same issue as 002_cache_web_search_columns.sql,
-- same fix: air.spend_logs_kafka and the materialized view are dropped and
-- recreated; only the MergeTree table is altered in place. No events are
-- lost: consumer offsets live in Kafka under kafka_group_name, so the
-- recreated table resumes where the old one stopped as long as
-- kafka_group_name is unchanged. The recreated view keeps
-- 005_tool_usage_columns.sql's tool_usage_cost fallback for events from AIR
-- pods that predate that field.
--
-- The CREATE below uses the reference SETTINGS from
-- clickhouse/init/01_spend_logs.sql. Before running this on a real cluster,
-- copy your current SETTINGS from `SHOW CREATE TABLE air.spend_logs_kafka`
-- (broker list, topic, group name, consumer count) into it, and add the
-- ON CLUSTER clause your deployment needs.
--
-- Run it after 006_video_input_columns.sql: the Kafka table below carries
-- 006's columns too. Pause AIR Kafka publishing before running this
-- migration. Safe to re-run on its own only while it is the last applied
-- migration: never run 007 again once a later migration has been applied,
-- for the same reason 002_cache_web_search_columns.sql's doc comment spells
-- out -- it would rebuild air.spend_logs_kafka from only 007's column set,
-- narrowing it back below whatever the later migration added, and break
-- ingestion with NUMBER_OF_COLUMNS_DOESNT_MATCH until that later migration
-- is re-applied.

DROP TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS billed_input_tokens UInt32 DEFAULT 0 AFTER provider_reported_cost,
    ADD COLUMN IF NOT EXISTS billed_output_tokens UInt32 DEFAULT 0 AFTER billed_input_tokens,
    ADD COLUMN IF NOT EXISTS billed_image_tokens UInt32 DEFAULT 0 AFTER billed_output_tokens,
    ADD COLUMN IF NOT EXISTS billed_video_input_tokens UInt32 DEFAULT 0 AFTER billed_image_tokens;

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
    billed_input_tokens UInt32,
    billed_output_tokens UInt32,
    billed_image_tokens UInt32,
    billed_video_input_tokens UInt32,

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
