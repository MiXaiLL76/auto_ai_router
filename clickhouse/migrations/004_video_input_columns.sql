-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to the
-- event schema that reports input video apart from input images
-- (video_input_tokens / video_input_cost, Gemini Embedding 2 billing).
--
-- Pause AIR Kafka publishing before running this migration. The materialized
-- view is detached first so the Kafka engine cannot consume a new-schema event
-- while the two table definitions differ.

DETACH TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS video_input_tokens UInt32 DEFAULT 0 AFTER image_tokens,
    ADD COLUMN IF NOT EXISTS video_input_cost Float64 DEFAULT 0 AFTER image_cost;

ALTER TABLE air.spend_logs_kafka
    ADD COLUMN IF NOT EXISTS video_input_tokens UInt32 DEFAULT 0 AFTER image_tokens,
    ADD COLUMN IF NOT EXISTS video_input_cost Float64 DEFAULT 0 AFTER image_cost;

ATTACH TABLE air.spend_logs_mv;
