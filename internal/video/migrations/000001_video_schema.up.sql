BEGIN;

CREATE TEMP TABLE _air_video_jobs (
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
) ON COMMIT DROP;
CREATE INDEX IF NOT EXISTS _air_video_jobs_work ON _air_video_jobs (next_run_at, created_at)
  WHERE state NOT IN ('completed','failed','cancelled','submission_unknown');
CREATE UNIQUE INDEX IF NOT EXISTS _air_video_provider_job ON _air_video_jobs (provider_job_id)
  WHERE provider_job_id <> '';
CREATE TEMP TABLE _air_video_uploads (
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
) ON COMMIT DROP;
CREATE INDEX IF NOT EXISTS _air_video_uploads_expiry ON _air_video_uploads (expires_at)
  WHERE state = 'created';

DO $migration$
DECLARE
  table_name text;
  expected regclass;
  actual regclass;
  target_schema text := current_schema();
  expected_definition jsonb;
  actual_definition jsonb;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['air_video_jobs', 'air_video_uploads']
  LOOP
    expected := format('pg_temp.%I', '_' || table_name)::regclass;
    actual := to_regclass(format('%I.%I', target_schema, table_name));
    IF actual IS NULL THEN
      EXECUTE format('CREATE TABLE %I.%I (LIKE %s INCLUDING DEFAULTS INCLUDING CONSTRAINTS)', target_schema, table_name, expected);
      EXECUTE format('ALTER TABLE %I.%I ADD PRIMARY KEY (id)', target_schema, table_name);
      IF table_name = 'air_video_jobs' THEN
        EXECUTE format('ALTER TABLE %I.%I ADD UNIQUE (organization_id, idempotency_key)', target_schema, table_name);
        EXECUTE format('CREATE INDEX air_video_jobs_work ON %I.%I (next_run_at, created_at)
          WHERE state NOT IN (''completed'',''failed'',''cancelled'',''submission_unknown'')', target_schema, table_name);
        EXECUTE format('CREATE UNIQUE INDEX air_video_provider_job ON %I.%I (provider_job_id)
          WHERE provider_job_id <> ''''', target_schema, table_name);
      ELSE
        EXECUTE format('ALTER TABLE %I.%I ADD UNIQUE (object_key)', target_schema, table_name);
        EXECUTE format('CREATE INDEX air_video_uploads_expiry ON %I.%I (expires_at)
          WHERE state = ''created''', target_schema, table_name);
      END IF;
      actual := to_regclass(format('%I.%I', target_schema, table_name));
    END IF;

    IF (SELECT relkind <> 'r' OR relpersistence <> 'p' OR relrowsecurity OR relforcerowsecurity
          FROM pg_class WHERE oid = actual) THEN
      RAISE EXCEPTION 'Incompatible video table %', table_name;
    END IF;

    SELECT jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = expected),
           jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = actual)
      INTO expected_definition, actual_definition
      FROM (
        SELECT a.attrelid AS relation,
               jsonb_build_array(a.attname, a.atttypid, a.atttypmod, a.attnotnull,
                 a.attcollation, a.attidentity, a.attgenerated,
                 pg_get_expr(d.adbin, d.adrelid)) AS definition
          FROM pg_attribute a
          LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
         WHERE a.attrelid IN (expected, actual) AND a.attnum > 0 AND NOT a.attisdropped
      ) columns;
    IF expected_definition IS DISTINCT FROM actual_definition THEN
      RAISE EXCEPTION 'Incompatible video columns in %', table_name;
    END IF;

    SELECT jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = expected),
           jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = actual)
      INTO expected_definition, actual_definition
      FROM (
        SELECT conrelid AS relation, pg_get_constraintdef(oid) AS definition
          FROM pg_constraint WHERE conrelid IN (expected, actual)
      ) constraints;
    IF expected_definition IS DISTINCT FROM actual_definition THEN
      RAISE EXCEPTION 'Incompatible video constraints in %', table_name;
    END IF;

    SELECT jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = expected),
           jsonb_agg(definition ORDER BY definition) FILTER (WHERE relation = actual)
      INTO expected_definition, actual_definition
      FROM (
        SELECT i.indrelid AS relation,
               jsonb_build_array(c.relam, i.indnkeyatts, i.indisunique, i.indisprimary, i.indisvalid,
                 i.indisready, i.indnullsnotdistinct, i.indclass::text,
                 i.indcollation::text, i.indoption::text,
                 (SELECT jsonb_agg(pg_get_indexdef(i.indexrelid, n, true) ORDER BY n)
                    FROM generate_series(1, i.indnatts) n),
                 pg_get_expr(i.indpred, i.indrelid)) AS definition
          FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
         WHERE i.indrelid IN (expected, actual)
      ) indexes;
    IF expected_definition IS DISTINCT FROM actual_definition THEN
      RAISE EXCEPTION 'Incompatible video indexes in %', table_name;
    END IF;
  END LOOP;
END
$migration$;

COMMIT;
