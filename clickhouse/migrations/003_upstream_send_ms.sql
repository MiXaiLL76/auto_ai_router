-- Upgrade an existing air.spend_logs Kafka -> MergeTree pipeline to include
-- the upstream_send_ms column introduced with the time-to-upstream-send
-- metric (router processing time before the first provider attempt).
--
-- Pause AIR Kafka publishing before running this migration. The materialized
-- view is detached first so the Kafka engine cannot consume a new-schema event
-- while the two table definitions differ.

DETACH TABLE IF EXISTS air.spend_logs_mv;

ALTER TABLE air.spend_logs
    ADD COLUMN IF NOT EXISTS upstream_send_ms Nullable(UInt32) AFTER ttft_ms;

ALTER TABLE air.spend_logs_kafka
    ADD COLUMN IF NOT EXISTS upstream_send_ms Nullable(UInt32) AFTER ttft_ms;

ATTACH TABLE air.spend_logs_mv;
