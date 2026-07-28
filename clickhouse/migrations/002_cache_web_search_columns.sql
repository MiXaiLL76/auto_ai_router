-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to the
-- cache/web-search event schema introduced by PR #86.
--
-- Pause AIR Kafka publishing before running this migration. The materialized
-- view is detached first so the Kafka engine cannot consume a new-schema event
-- while the two table definitions differ.

DETACH TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS cached_audio_input_tokens UInt32 DEFAULT 0 AFTER cached_input_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_5m_tokens UInt32 DEFAULT 0 AFTER cache_creation_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_1h_tokens UInt32 DEFAULT 0 AFTER cache_creation_5m_tokens,
    ADD COLUMN IF NOT EXISTS web_search_requests UInt32 DEFAULT 0 AFTER output_image_tokens,
    ADD COLUMN IF NOT EXISTS web_search_context_size Nullable(String) AFTER web_search_requests,
    ADD COLUMN IF NOT EXISTS web_search_cost Float64 DEFAULT 0 AFTER image_cost;

ALTER TABLE air.spend_logs_kafka
    ADD COLUMN IF NOT EXISTS cached_audio_input_tokens UInt32 DEFAULT 0 AFTER cached_input_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_5m_tokens UInt32 DEFAULT 0 AFTER cache_creation_tokens,
    ADD COLUMN IF NOT EXISTS cache_creation_1h_tokens UInt32 DEFAULT 0 AFTER cache_creation_5m_tokens,
    ADD COLUMN IF NOT EXISTS web_search_requests UInt32 DEFAULT 0 AFTER output_image_tokens,
    ADD COLUMN IF NOT EXISTS web_search_context_size Nullable(String) AFTER web_search_requests,
    ADD COLUMN IF NOT EXISTS web_search_cost Float64 DEFAULT 0 AFTER image_cost;

ATTACH TABLE air.spend_logs_mv;
