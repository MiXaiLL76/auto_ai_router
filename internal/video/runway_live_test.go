package video

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRunwayLiveParity(t *testing.T) {
	if os.Getenv("AIR_VIDEO_LIVE_RUNWAY") != "1" {
		t.Skip("AIR_VIDEO_LIVE_RUNWAY is not enabled")
	}
	runwayKey := os.Getenv("AIR_VIDEO_TEST_RUNWAY_API_KEY")
	accessKey := os.Getenv("AIR_VIDEO_TEST_S3_ACCESS_KEY")
	secretKey := os.Getenv("AIR_VIDEO_TEST_S3_SECRET_KEY")
	if runwayKey == "" || accessKey == "" || secretKey == "" {
		t.Fatal("live Runway and S3 credentials are required")
	}
	objects, err := NewS3Store(S3Config{
		Endpoint: "https://s3.twcstorage.ru", Region: "ru-1",
		Bucket: "a73def9f-143e-4c0e-a7c8-eb36cae1a4be", Prefix: "air-video-live-test",
		AccessKey: accessKey, SecretKey: secretKey, ArtifactProxyURL: os.Getenv("HTTPS_PROXY"),
	})
	require.NoError(t, err)
	provider, err := NewRunwayClient(RunwayConfig{
		BaseURL: "https://api.dev.runwayml.com", APIVersion: "2024-11-06", APIKey: runwayKey,
		Models: map[string]string{"runway/gen4_turbo": "gen4_turbo", "runway/gen4.5": "gen4.5"},
	})
	require.NoError(t, err)
	if taskIDs := strings.Fields(os.Getenv("AIR_VIDEO_LIVE_TASK_IDS")); len(taskIDs) > 0 {
		for index, taskID := range taskIDs {
			result, pollErr := provider.Poll(t.Context(), taskID)
			require.NoError(t, pollErr)
			require.Equal(t, StatusCompleted, result.Status)
			job := &Job{ID: "existing-" + taskID, OrganizationID: "live-test"}
			object, fetchErr := objects.FetchResult(t.Context(), job, result.ResultURL)
			require.NoError(t, fetchErr, "task index %d", index)
			require.Greater(t, object.Size, int64(12))
			require.NoError(t, objects.delete(t.Context(), resultKey(job.OrganizationID, job.ID)))
		}
		return
	}
	liveProvider := &loggingProvider{RunwayClient: provider, t: t}
	liveObjects := &loggingObjectStore{S3Store: objects, t: t}
	repository := NewMemoryStore()
	service, err := NewService(ServiceConfig{Store: repository, Objects: liveObjects, Billing: NoopBilling{}})
	require.NoError(t, err)
	worker, err := NewWorker(WorkerConfig{
		Store: repository, Provider: liveProvider, Objects: liveObjects, Billing: NoopBilling{},
		ID: "live-worker", PollInterval: 5 * time.Second, LeaseTTL: 5 * time.Minute,
	})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "live-test", PriceProfileID: "live", PriceProfileSHA256: "live",
		RatePerSecond: "0.07", Currency: "USD",
	}

	upload := createLiveImageUpload(t, service, objects)
	imageJob, _, err := service.Create(t.Context(), principal, "live-gen4-turbo-image", CreateRequest{
		Model: "runway/gen4_turbo", Prompt: "Slow camera movement over a calm blue abstract field",
		InputImageID: upload.ID, DurationSeconds: 5, AspectRatio: "1:1",
	})
	require.NoError(t, err)
	textJob, _, err := service.Create(t.Context(), principal, "live-gen45-text", CreateRequest{
		Model: "runway/gen4.5", Prompt: "A calm ocean horizon with slow natural cloud movement",
		DurationSeconds: 5, AspectRatio: "16:9",
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 12*time.Minute)
	defer cancel()
	driveLiveJobs(t, ctx, worker, service, principal.OrganizationID, imageJob.ID, textJob.ID)
	for _, job := range []*Job{imageJob, textJob} {
		current, getErr := service.Get(t.Context(), principal.OrganizationID, job.ID)
		require.NoError(t, getErr)
		require.Equalf(t, StateCompleted, current.State, "model %s code %s message %s", current.Request.Model, current.ErrorCode, current.ErrorMessage)
		object, openErr := service.Content(t.Context(), principal.OrganizationID, job.ID)
		require.NoError(t, openErr)
		require.Greater(t, object.Size, int64(12))
		require.NoError(t, object.Body.Close())
		require.NoError(t, objects.delete(t.Context(), resultKey(principal.OrganizationID, job.ID)))
	}
	require.NoError(t, objects.DeleteUpload(t.Context(), upload))
}

type loggingProvider struct {
	*RunwayClient
	t *testing.T
}

func (p *loggingProvider) Submit(ctx context.Context, job *Job, image string) (string, error) {
	id, err := p.RunwayClient.Submit(ctx, job, image)
	p.t.Logf("submit model %s provider job %s error %v", job.Request.Model, id, err)
	return id, err
}

func (p *loggingProvider) Poll(ctx context.Context, id string) (ProviderResult, error) {
	result, err := p.RunwayClient.Poll(ctx, id)
	p.t.Logf("poll provider job %s status %s error %v", id, result.Status, err)
	return result, err
}

type loggingObjectStore struct {
	*S3Store
	t *testing.T
}

func (s *loggingObjectStore) FetchResult(ctx context.Context, job *Job, rawURL string) (Object, error) {
	object, err := s.S3Store.FetchResult(ctx, job, rawURL)
	s.t.Logf("store model %s bytes %d type %s error %v", job.Request.Model, object.Size, object.ContentType, err)
	return object, err
}

func createLiveImageUpload(t *testing.T, service *Service, objects *S3Store) *Upload {
	t.Helper()
	frame := image.NewRGBA(image.Rect(0, 0, 640, 640))
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			frame.SetRGBA(x, y, color.RGBA{R: 20, G: 90, B: 180, A: 255})
		}
	}
	var encoded bytes.Buffer
	require.NoError(t, png.Encode(&encoded, frame))
	sum := sha256.Sum256(encoded.Bytes())
	upload, err := service.CreateUpload(t.Context(), "live-test", UploadRequest{
		Purpose: "video_input_image", MIME: "image/png", SizeBytes: int64(encoded.Len()),
		SHA256: hex.EncodeToString(sum[:]),
	})
	require.NoError(t, err)
	require.NoError(t, service.PutUpload(t.Context(), "live-test", upload.ID, "image/png", bytes.NewReader(encoded.Bytes())))
	upload, err = service.CompleteUpload(t.Context(), "live-test", upload.ID)
	require.NoError(t, err)
	t.Cleanup(func() { _ = objects.DeleteUpload(context.Background(), upload) })
	return upload
}

func driveLiveJobs(t *testing.T, ctx context.Context, worker *Worker, service *Service, organizationID string, jobIDs ...string) {
	t.Helper()
	for {
		allTerminal := true
		for _, id := range jobIDs {
			job, err := service.Get(ctx, organizationID, id)
			require.NoError(t, err)
			if !IsTerminal(job.State) {
				allTerminal = false
			}
		}
		if allTerminal {
			return
		}
		err := worker.RunOnce(ctx)
		if err != nil && err != ErrNotFound {
			t.Fatalf("video worker failed %v", err)
		}
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-time.After(time.Second):
		}
	}
}
