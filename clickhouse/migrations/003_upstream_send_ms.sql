-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to include
-- the upstream_send_ms column introduced with the time-to-upstream-send
-- metric (router processing time before the first provider attempt).
--
-- The Kafka table engine does not support ALTER ... ADD COLUMN (ClickHouse
-- fails with NOT_IMPLEMENTED) -- same issue as 002_cache_web_search_columns.sql,
-- same fix: air.spend_logs_kafka is dropped and recreated instead of altered.
-- No events are lost: consumer offsets live in Kafka under kafka_group_name,
-- so the recreated table resumes where the old one stopped as long as
-- kafka_group_name is unchanged. air.spend_logs (the MergeTree table) is
-- only ALTERed, never dropped.
--
-- Pause AIR Kafka publishing before running this migration. Safe to re-run
-- on its own -- but never run it again after
-- 004_explicit_cache_columns.sql has already been applied: it rebuilds
-- air.spend_logs_kafka from only this migration's column set, narrowing it
-- back below the columns 004 already added (see
-- 002_cache_web_search_columns.sql's doc comment for the confirmed failure
-- mode this causes). Apply 002/003/004 forward, in order, never backward.

DROP TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS upstream_send_ms Nullable(UInt32) AFTER ttft_ms;

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
    cached_output_tokens UInt32,
    reasoning_tokens UInt32,
    accepted_prediction_tokens UInt32,
    rejected_prediction_tokens UInt32,
    image_count UInt32,
    image_tokens UInt32,
    output_image_tokens UInt32,
    web_search_requests UInt32,
    web_search_context_size Nullable(String),

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
    web_search_cost Float64,
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
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:29092',
    kafka_topic_list = 'air.spend_logs',
    kafka_group_name = 'clickhouse_air_spend_logs',
    kafka_format = 'JSONEachRow',
    date_time_input_format = 'best_effort',
    kafka_num_consumers = 2,
    kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW air.spend_logs_mv TO air.spend_logs AS
SELECT * FROM air.spend_logs_kafka;
