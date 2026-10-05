-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to the
-- cache/web-search event schema introduced by PR #86.
--
-- The Kafka table engine does not support ALTER ... ADD COLUMN (ClickHouse
-- fails with NOT_IMPLEMENTED) -- confirmed against a real 24.8 instance when
-- this migration, as originally written with a plain ALTER TABLE
-- air.spend_logs_kafka, was found to error out and leave the pipeline
-- half-migrated (air.spend_logs altered, air.spend_logs_kafka not, the
-- materialized view detached and never reattached). air.spend_logs_kafka is
-- dropped and recreated instead, matching 004_explicit_cache_columns.sql's
-- approach (and llmarena/terraform's tf/clickhouse/migrations, which hit
-- and solved the exact same problem for its own equivalent table). No
-- events are lost: consumer offsets live in Kafka under kafka_group_name,
-- so the recreated table resumes where the old one stopped as long as
-- kafka_group_name is unchanged. air.spend_logs (the MergeTree table) is
-- only ALTERed, never dropped.
--
-- Pause AIR Kafka publishing before running this migration. Safe to re-run
-- on its own (e.g. if it errored out partway) -- but never run it again
-- after 003_upstream_send_ms.sql/004_explicit_cache_columns.sql have
-- already been applied: it rebuilds air.spend_logs_kafka from only this
-- migration's column set, narrowing it back below the columns those later
-- migrations already added. air.spend_logs (which only ever gets ADD
-- COLUMN, never a rebuild) keeps every column, so the two tables end up
-- with a different column count -- every message the Kafka table then
-- tries to pass to the materialized view fails with
-- NUMBER_OF_COLUMNS_DOESNT_MATCH (confirmed against a real instance) until
-- the later migration(s) are re-applied to rebuild air.spend_logs_kafka
-- with the full column set again. Apply 002/003/004 forward, in order,
-- never backward.

DROP TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS cached_audio_input_tokens UInt32 DEFAULT 0 AFTER cached_input_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_5m_tokens UInt32 DEFAULT 0 AFTER cache_creation_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_1h_tokens UInt32 DEFAULT 0 AFTER cache_creation_5m_tokens,
    ADD COLUMN IF NOT EXISTS web_search_requests UInt32 DEFAULT 0 AFTER output_image_tokens,
    ADD COLUMN IF NOT EXISTS web_search_context_size Nullable(String) AFTER web_search_requests,
    ADD COLUMN IF NOT EXISTS web_search_cost Float64 DEFAULT 0 AFTER image_cost;

DROP TABLE IF EXISTS air.spend_logs_kafka;

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
