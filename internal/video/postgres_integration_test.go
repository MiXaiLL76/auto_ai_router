package video

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
)

func TestPostgresStoreIdempotencyIsolationAndLeaseFencing(t *testing.T) {
	databaseURL := os.Getenv("AIR_VIDEO_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("AIR_VIDEO_TEST_DATABASE_URL is not set")
	}
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	t.Cleanup(pool.Close)

	store := NewPostgresStore(pool)
	require.NoError(t, store.Migrate(t.Context()))
	org := "video-test-" + uuid.NewString()
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), `DELETE FROM air_video_uploads WHERE organization_id=$1`, org)
		_, _ = pool.Exec(context.Background(), `DELETE FROM air_video_jobs WHERE organization_id=$1`, org)
	})

	principal := Principal{
		OrganizationID:     org,
		PriceProfileID:     "r8",
		PriceProfileSHA256: "sha",
		RatePerSecond:      "0.07",
		Currency:           "USD",
	}
	request := CreateRequest{Model: "runway/gen4.5", Prompt: "test", DurationSeconds: 5, AspectRatio: "16:9"}
	const callers = 12
	jobs := make(chan *Job, callers)
	errors := make(chan error, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			job, _, createErr := store.CreateJob(t.Context(), principal, "same-key", request, "same-hash")
			if createErr != nil {
				errors <- createErr
				return
			}
			jobs <- job
		}()
	}
	wg.Wait()
	close(jobs)
	close(errors)
	for createErr := range errors {
		require.NoError(t, createErr)
	}
	var id string
	for job := range jobs {
		if id == "" {
			id = job.ID
		}
		require.Equal(t, id, job.ID)
	}

	job, err := store.GetJob(t.Context(), org, id)
	require.NoError(t, err)
	require.Equal(t, StateReserving, job.State)
	job, err = store.ActivateJob(t.Context(), job, Reservation{ID: id, Handle: id, Amount: "0.35", Currency: "USD"})
	require.NoError(t, err)

	claimed, err := store.ClaimJob(t.Context(), "worker-a", time.Now().UTC(), 5*time.Minute)
	require.NoError(t, err)
	require.Equal(t, id, claimed.ID)
	require.Equal(t, "worker-a", claimed.LeaseOwner)
	require.Greater(t, claimed.Version, job.Version)

	_, err = store.GetJob(t.Context(), "foreign-org", id)
	require.ErrorIs(t, err, ErrNotFound)
	_, err = store.Transition(t.Context(), job, StateSubmitting, JobUpdate{})
	require.ErrorIs(t, err, ErrStateChanged)
}
