CREATE TABLE IF NOT EXISTS air_video_jobs (
  id text PRIMARY KEY,
  organization_id text NOT NULL,
  idempotency_key text NOT NULL,
  request_hash text NOT NULL,
  request jsonb NOT NULL,
  principal jsonb NOT NULL,
  public_status text NOT NULL,
  state text NOT NULL,
  version bigint NOT NULL DEFAULT 1,
  provider_job_id text NOT NULL DEFAULT '',
  cancel_requested boolean NOT NULL DEFAULT false,
  result_url text NOT NULL DEFAULT '',
  result_content_type text NOT NULL DEFAULT '',
  result_size bigint NOT NULL DEFAULT 0,
  result_etag text NOT NULL DEFAULT '',
  reservation_id text NOT NULL DEFAULT '',
  reservation_handle text NOT NULL DEFAULT '',
  quoted_amount text NOT NULL DEFAULT '',
  currency text NOT NULL DEFAULT '',
  error_code text NOT NULL DEFAULT '',
  error_message text NOT NULL DEFAULT '',
  lease_owner text NOT NULL DEFAULT '',
  lease_expires_at timestamptz,
  next_run_at timestamptz NOT NULL DEFAULT now(),
  attempts integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS air_video_jobs_work ON air_video_jobs (next_run_at, created_at)
  WHERE state NOT IN ('completed','failed','cancelled','submission_unknown');
CREATE UNIQUE INDEX IF NOT EXISTS air_video_provider_job ON air_video_jobs (provider_job_id)
  WHERE provider_job_id <> '';
CREATE TABLE IF NOT EXISTS air_video_uploads (
  id text PRIMARY KEY,
  organization_id text NOT NULL,
  purpose text NOT NULL,
  mime text NOT NULL,
  size_bytes bigint NOT NULL,
  sha256 text NOT NULL,
  object_key text NOT NULL UNIQUE,
  state text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS air_video_uploads_expiry ON air_video_uploads (expires_at)
  WHERE state = 'created';
