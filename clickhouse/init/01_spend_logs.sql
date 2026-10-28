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
-- field-for-field, but for `attempts`, which air.attempts_kafka reads (at the
-- end of this file), and `air_event_id`, which this schema does not store (see
-- "Deduplication" in docs/litellm-integration/kafka_spend_log.md). Columns
-- backed by a Go field with `,omitempty` are Nullable, since JSONEachRow will
-- see the key entirely absent (not `null`) for a zero value.
-- ClickHouse's `CREATE TABLE ... (LIKE other_table)` used as a parenthesized column
-- list is not valid DDL (that's MySQL/Postgres syntax) and fails with
-- SYNTAX_ERROR (code 62) -- both tables below spell out the same column list
-- explicitly instead.

CREATE DATABASE IF NOT EXISTS air;

CREATE TABLE air.spend_logs_kafka
(
    request_id String,
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
CREATE TABLE air.spend_logs
(
    request_id String,
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

-- An event from an AIR version that predates tool_usage_cost arrives with it
-- 0; web search was the only priced tool then (see
-- clickhouse/migrations/005_tool_usage_columns.sql).
CREATE MATERIALIZED VIEW air.spend_logs_mv TO air.spend_logs AS
SELECT * REPLACE (if(tool_usage_cost = 0, web_search_cost, tool_usage_cost) AS tool_usage_cost)
FROM air.spend_logs_kafka;

-- Separate, independently-toggleable write-path (kafka.raw_bodies): raw
-- provider response for a *failed* request only, on its own topic so
-- retention/enablement can be managed independently of air.spend_logs (see
-- docs/litellm-integration/kafka_spend_log.md, "Raw bodies").
--
-- response_body / client_response_body are usually NOT the same value:
-- maskedUpstreamErrorBody replaces the provider's own error text with a
-- short, pre-vetted message for essentially every 4xx/5xx (unconditionally,
-- not credential-specific), so provider internals are never echoed back to
-- the client. They're equal only when a mid-stream error was detected after
-- the response had already committed -- nothing left to mask, the client
-- already got those exact bytes live.
--
-- response_body itself holds the provider's OWN error text verbatim -- it is
-- the only field that says WHY a request failed ("usage limit reached",
-- "content policy violation", ...), so it is published as the provider wrote
-- it. The one exception: CometAPI/ProMan credentials have been seen to echo
-- the caller's own credential back inside their error body, so for those two
-- it is reduced to "[REDACTED body bytes=N]".
-- client_response_body -- what the client actually received -- is published
-- verbatim too: the router already masked that body before writing it to the
-- client (maskedUpstreamErrorBody -> classifiedBadRequestError ->
-- upstreamerror.ClassifyBadRequest), so it holds a classified message from a
-- fixed list, a code and a param -- never the provider's free text. The
-- CometAPI/ProMan exception applies here as well: on a mid-stream failure
-- the client-facing body IS the provider's raw bytes (they were relayed
-- before the router could mask them), so publishing it verbatim would
-- publish the credential with them.
-- Named air.raw_bodies_kafka/air.raw_bodies, not air.error_bodies_kafka/
-- air.error_bodies: neither table is error-only once
-- kafka.raw_bodies.store_only_errors is set to false (they can then carry a
-- row for every request, error or not) -- StoreOnlyErrors still defaults to
-- true, so out of the box both only ever hold failures.
CREATE TABLE air.raw_bodies_kafka
(
    request_id String,
    server_router_id String,
    start_time DateTime64(3),
    -- When this router finished processing the request -- for a failure
    -- row, effectively when the error was finalized/detected. Same value
    -- as the matching air.spend_logs.end_time for this request.
    end_time DateTime64(3),
    http_status UInt16,
    error_class Nullable(String),
    response_body Nullable(String),
    client_response_body Nullable(String),
    -- Only populated when kafka.raw_bodies.store_raw_body is enabled
    -- (default false), and even then the router redacts prompt/message
    -- content before publishing by default (messages/system/prompt/input/
    -- contents/instructions replaced with a role/count-preserving
    -- placeholder, every other client-written string -- tool descriptions,
    -- user, metadata, ... -- masked, model/tool names/parameter shape
    -- kept) -- see redactRequestBodyForLogging in the router's
    -- proxy_helpers.go. kafka.raw_bodies.redact_sensitive_fields=false is
    -- an explicit escape hatch that captures this verbatim instead.
    request_body Nullable(String)
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:29092',
    kafka_topic_list = 'raw-bodies',
    kafka_group_name = 'clickhouse_raw_bodies',
    kafka_format = 'JSONEachRow',
    date_time_input_format = 'best_effort',
    kafka_num_consumers = 2,
    kafka_handle_error_mode = 'stream';

-- Plain MergeTree for the same single-node-docker-compose reason as
-- air.spend_logs above; swap for ReplicatedMergeTree in a real cluster.
CREATE TABLE air.raw_bodies
(
    request_id String,
    server_router_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    http_status UInt16,
    error_class Nullable(String),
    response_body Nullable(String),
    client_response_body Nullable(String),
    request_body Nullable(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(start_time)
-- server_router_id is part of the key, not just request_id: request_id is
-- derived from the upstream provider's own response id and is echoed back
-- unchanged through every hop of a chained deployment, so two different
-- hops logging the same logical request in the same millisecond can share a
-- request_id -- see air.logs' ORDER BY for the identical reasoning.
ORDER BY (start_time, request_id, server_router_id)
-- Short, independent retention: this is bulky debugging data, not
-- billing/analytics -- pick whatever window your incident/debugging process
-- actually needs, it has no bearing on air.spend_logs' own TTL.
TTL toDateTime(start_time) + INTERVAL 14 DAY;  -- пример; на усмотрение DBA/CH-кластера

CREATE MATERIALIZED VIEW air.raw_bodies_mv TO air.raw_bodies AS
SELECT * FROM air.raw_bodies_kafka;

-- Join back to the matching air.spend_logs / air.errors row on
-- (request_id, server_router_id), not request_id alone, for the same
-- collision reason as the ORDER BY above.
CREATE VIEW air.spend_logs_with_raw_errors AS
SELECT s.*, b.response_body, b.client_response_body, b.request_body
FROM air.spend_logs AS s
LEFT JOIN air.raw_bodies AS b
    ON s.request_id = b.request_id AND s.server_router_id = b.server_router_id
WHERE s.status = 'failure';

-- A row for every upstream attempt of a request, from the event's `attempts`,
-- beside the request's own columns (see
-- clickhouse/migrations/011_attempts.sql for each column).
CREATE TABLE air.attempts
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
