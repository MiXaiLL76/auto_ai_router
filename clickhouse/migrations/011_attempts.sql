-- Add air.attempts to an existing air.spend_logs pipeline: a row for every
-- upstream attempt of a request, from the spend event's `attempts`, beside
-- the request's own columns:
--   attempt             -- its number in the request, from 1
--   credential_name     -- the credential it was sent to
--   exit                -- the exit it left through; '' without one
--   kind                -- first, retry or fallback
--   retry_reason        -- why the router went on to another attempt; '' when
--                          it did not
--   http_status         -- the provider's status; 0 without a response
--   upstream_request_id -- the provider's id of the request
--   error_type          -- the type of the error that left it without a
--                          response
--   error_message       -- that error's text
--   attempt_time        -- when it was sent
--   duration_ms         -- its send to the next attempt's, or to the request's
--                          end
--   ttfb_ms             -- its request written to the first byte of the
--                          response
--   conn_reused         -- 1 when its connection was a pooled one
--   is_final            -- 1 on the attempt the request's air.spend_logs row
--                          describes

DROP TABLE IF EXISTS air.attempts_mv;

DROP TABLE IF EXISTS air.attempts_kafka;

CREATE TABLE IF NOT EXISTS air.attempts
(
    start_time DateTime64(3),
    end_time DateTime64(3),
    request_id String,
    trace_id String,
    service_name LowCardinality(String),
    server_router_id LowCardinality(String),
    call_type String,
    organization_id String,
    team_id String,
    api_key_hash String,
    model_group String,
    request_status LowCardinality(String),
    request_http_status UInt16,

    attempt UInt8,
    credential_name String,
    exit LowCardinality(String),
    kind LowCardinality(String),
    retry_reason LowCardinality(String),
    http_status UInt16,
    upstream_request_id String,
    error_type LowCardinality(String),
    error_message String,
    attempt_time DateTime64(3),
    duration_ms UInt32,
    ttfb_ms Nullable(UInt32),
    conn_reused Nullable(UInt8),
    is_final UInt8
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(start_time)
ORDER BY (start_time, request_id, attempt)
TTL toDateTime(start_time) + INTERVAL 90 DAY;

-- A second reader of the topic, in a consumer group of its own: the list is
-- read as text and parsed in the view, so nothing in it can fail a message.
CREATE TABLE air.attempts_kafka
(
    request_id String,
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
