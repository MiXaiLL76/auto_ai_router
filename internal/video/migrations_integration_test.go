package video

import (
	"context"
	_ "embed"
	"net/url"
	"os"
	"sync"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/legacy_video_schema.sql
var legacyVideoSchema string

func migrationTestDatabase(t *testing.T) (string, *pgxpool.Pool) {
	t.Helper()
	databaseURL := os.Getenv("AIR_VIDEO_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("AIR_VIDEO_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	schema := "migration_test_" + uuid.NewString()
	_, err = admin.Exec(t.Context(), "CREATE SCHEMA "+pgx.Identifier{schema}.Sanitize())
	require.NoError(t, err)
	t.Cleanup(func() {
		_, cleanupErr := admin.Exec(context.Background(), "DROP SCHEMA "+pgx.Identifier{schema}.Sanitize()+" CASCADE")
		require.NoError(t, cleanupErr)
		admin.Close()
	})
	parsed, err := url.Parse(databaseURL)
	require.NoError(t, err)
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	databaseURL = parsed.String()
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	t.Cleanup(pool.Close)
	return databaseURL, pool
}

func TestVideoMigrationsFreshAndConcurrent(t *testing.T) {
	databaseURL, pool := migrationTestDatabase(t)
	require.Error(t, NewPostgresStore(pool).CheckSchema(t.Context()))
	var history *string
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT to_regclass('air_schema_migrations')::text`).Scan(&history))
	require.Nil(t, history)
	const callers = 6
	errors := make(chan error, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Go(func() { errors <- MigrateDatabase(t.Context(), databaseURL) })
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		require.NoError(t, err)
	}
	require.NoError(t, MigrateDatabase(t.Context(), databaseURL))
	require.NoError(t, NewPostgresStore(pool).CheckSchema(t.Context()))
	var count int
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM air_schema_migrations WHERE version=1 AND NOT dirty`).Scan(&count))
	require.Equal(t, 1, count)
	for _, name := range []string{"air_video_jobs_work", "air_video_provider_job", "air_video_uploads_expiry"} {
		var index *string
		require.NoError(t, pool.QueryRow(t.Context(), `SELECT to_regclass($1)::text`, name).Scan(&index))
		require.NotNil(t, index, name)
	}
	_, err := pool.Exec(t.Context(), `UPDATE air_schema_migrations SET dirty=true`)
	require.NoError(t, err)
	require.Error(t, NewPostgresStore(pool).CheckSchema(t.Context()))
	_, err = pool.Exec(t.Context(), `UPDATE air_schema_migrations SET version=2,dirty=false`)
	require.NoError(t, err)
	require.Error(t, NewPostgresStore(pool).CheckSchema(t.Context()))
}

func TestVideoMigrationsAdoptExistingTables(t *testing.T) {
	databaseURL, pool := migrationTestDatabase(t)
	_, err := pool.Exec(t.Context(), legacyVideoSchema)
	require.NoError(t, err)
	_, err = pool.Exec(t.Context(), `
	  INSERT INTO air_video_uploads VALUES ('upload','org','video_input_image','image/png',1,'hash','key','created',now(),now());
	  INSERT INTO air_video_jobs (id,organization_id,idempotency_key,request_hash,request,principal,public_status,state)
	    VALUES ('job','org','idempotency','hash','{}','{}','completed','completed');
	  CREATE TABLE "_prisma_migrations" (id text PRIMARY KEY);
	  INSERT INTO "_prisma_migrations" VALUES ('unrelated');`)
	require.NoError(t, err)
	var before, after string
	const snapshot = `SELECT jsonb_build_array((SELECT row_to_json(u) FROM air_video_uploads u),
	  (SELECT row_to_json(j) FROM air_video_jobs j))::text`
	require.NoError(t, pool.QueryRow(t.Context(), snapshot).Scan(&before))
	var originalOID, migratedOID uint32
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT 'air_video_uploads'::regclass::oid`).Scan(&originalOID))
	require.NoError(t, MigrateDatabase(t.Context(), databaseURL))
	require.NoError(t, pool.QueryRow(t.Context(), snapshot).Scan(&after))
	require.Equal(t, before, after)
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT 'air_video_uploads'::regclass::oid`).Scan(&migratedOID))
	require.Equal(t, originalOID, migratedOID)
	var prismaID string
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT id FROM "_prisma_migrations"`).Scan(&prismaID))
	require.Equal(t, "unrelated", prismaID)
	require.NoError(t, MigrateDatabase(t.Context(), databaseURL))
	config, err := pgxpool.ParseConfig(databaseURL)
	require.NoError(t, err)
	config.ConnConfig.RuntimeParams["default_transaction_read_only"] = "on"
	readOnly, err := pgxpool.NewWithConfig(t.Context(), config)
	require.NoError(t, err)
	t.Cleanup(readOnly.Close)
	require.NoError(t, NewPostgresStore(readOnly).CheckSchema(t.Context()))
}

func TestVideoMigrationsRejectIncompatibleSchema(t *testing.T) {
	for name, change := range map[string]string{
		"type":       `ALTER TABLE air_video_jobs ALTER COLUMN attempts TYPE bigint`,
		"null":       `ALTER TABLE air_video_jobs ALTER COLUMN attempts DROP NOT NULL`,
		"default":    `ALTER TABLE air_video_jobs ALTER COLUMN attempts SET DEFAULT 2`,
		"constraint": `ALTER TABLE air_video_jobs DROP CONSTRAINT air_video_jobs_organization_id_idempotency_key_key`,
		"index":      `DROP INDEX air_video_provider_job; CREATE INDEX air_video_provider_job ON air_video_jobs(provider_job_id)`,
		"predicate":  `DROP INDEX air_video_jobs_work; CREATE INDEX air_video_jobs_work ON air_video_jobs(next_run_at,created_at) WHERE state <> 'completed'`,
	} {
		t.Run(name, func(t *testing.T) {
			databaseURL, pool := migrationTestDatabase(t)
			_, err := pool.Exec(t.Context(), legacyVideoSchema)
			require.NoError(t, err)
			_, err = pool.Exec(t.Context(), change)
			require.NoError(t, err)
			_, err = pool.Exec(t.Context(), `INSERT INTO air_video_uploads VALUES ('upload','org','video_input_image','image/png',1,'hash','key','created',now(),now())`)
			require.NoError(t, err)
			require.ErrorContains(t, MigrateDatabase(t.Context(), databaseURL), "Incompatible video")
			var uploadID string
			require.NoError(t, pool.QueryRow(t.Context(), `SELECT id FROM air_video_uploads`).Scan(&uploadID))
			require.Equal(t, "upload", uploadID)
			var dirty bool
			require.NoError(t, pool.QueryRow(t.Context(), `SELECT dirty FROM air_schema_migrations`).Scan(&dirty))
			require.True(t, dirty)
			require.Error(t, NewPostgresStore(pool).CheckSchema(t.Context()))
			require.ErrorContains(t, MigrateDatabase(t.Context(), databaseURL), "Dirty database")
		})
	}
}
