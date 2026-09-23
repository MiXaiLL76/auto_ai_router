package video

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type testResolver map[string]Principal

func (r testResolver) ResolvePrincipal(w http.ResponseWriter, req *http.Request, model string) (Principal, error) {
	p, ok := r[strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer ")]
	if !ok {
		http.Error(w, "authentication required", http.StatusUnauthorized)
		return Principal{}, ErrAuthentication
	}
	if model != "" && model != "runway/gen4.5" && model != "runway/gen4_turbo" {
		return Principal{}, ErrAuthentication
	}
	return p, nil
}

type testBilling struct {
	mu                          sync.Mutex
	reserved, settled, released int
}

func (b *testBilling) Reserve(_ context.Context, j *Job, _ string) (Reservation, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.reserved++
	return Reservation{ID: "res_" + j.ID, Handle: "opaque", Amount: j.QuotedAmount, Currency: "USD"}, nil
}
func (b *testBilling) Settle(_ context.Context, _ *Job, _ string) error {
	b.mu.Lock()
	b.settled++
	b.mu.Unlock()
	return nil
}
func (b *testBilling) Release(_ context.Context, _ *Job, _ string) error {
	b.mu.Lock()
	b.released++
	b.mu.Unlock()
	return nil
}

func TestVideoEndToEnd(t *testing.T) {
	video := []byte("deterministic-video")
	polls := 0
	providerHTTP := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer runway-secret" || r.Header.Get("X-Runway-Version") != "2024-11-06" {
			t.Errorf("missing provider headers")
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/text_to_video":
			if r.Header.Get("Idempotency-Key") == "" {
				t.Errorf("missing provider idempotency key")
			}
			_, _ = io.WriteString(w, `{"id":"task-1"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/tasks/task-1":
			polls++
			if polls == 1 {
				_, _ = io.WriteString(w, `{"id":"task-1","status":"RUNNING"}`)
			} else {
				_, _ = io.WriteString(w, `{"id":"task-1","status":"SUCCEEDED","output":["data:video/mp4;base64,`+base64.StdEncoding.EncodeToString(video)+`"]}`)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer providerHTTP.Close()
	provider, err := NewRunwayClient(RunwayConfig{BaseURL: providerHTTP.URL, APIKey: "runway-secret"})
	if err != nil {
		t.Fatal(err)
	}
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	billing := new(testBilling)
	svc, err := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: billing})
	if err != nil {
		t.Fatal(err)
	}
	p := Principal{OrganizationID: "org-a", APIKeyHash: "hash-a", UserID: "user-a", TeamID: "team-a", BillingTeamID: "billing-a", PriceProfileID: "r8", PriceProfileSHA256: "sha256:price", RatePerSecond: "0.07", Currency: "USD"}
	h, err := NewHandler(HandlerConfig{Service: svc, Resolver: testResolver{"key-a": p, "key-b": {OrganizationID: "org-b"}}, UploadSigningKey: bytes.Repeat([]byte("k"), 32)})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"model":"runway/gen4.5","prompt":"a calm sea","duration_seconds":5}`
	request := func(token, idem string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/videos", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Idempotency-Key", idem)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	first := request("key-a", "stable")
	if first.Code != http.StatusOK {
		t.Fatalf("create status %d body %s", first.Code, first.Body.String())
	}
	var created map[string]any
	if err = json.Unmarshal(first.Body.Bytes(), &created); err != nil {
		t.Fatal(err)
	}
	id := created["id"].(string)
	second := request("key-a", "stable")
	var replay map[string]any
	_ = json.Unmarshal(second.Body.Bytes(), &replay)
	if replay["id"] != id {
		t.Fatalf("idempotent replay created %v", replay["id"])
	}
	if billing.reserved != 1 {
		t.Fatalf("reservations %d", billing.reserved)
	}
	foreign := httptest.NewRequest(http.MethodGet, "/v1/videos/"+id, nil)
	foreign.Header.Set("Authorization", "Bearer key-b")
	foreignRec := httptest.NewRecorder()
	h.ServeHTTP(foreignRec, foreign)
	if foreignRec.Code != http.StatusNotFound {
		t.Fatalf("cross-org status %d", foreignRec.Code)
	}
	base := time.Now().UTC().Add(time.Hour)
	tick := 0
	now := func() time.Time { tick++; return base.Add(time.Duration(tick) * time.Second) }
	worker, err := NewWorker(WorkerConfig{Store: repo, Provider: provider, Objects: objects, Billing: billing, ID: "worker", PollInterval: time.Nanosecond, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if err = worker.RunOnce(t.Context()); err != nil {
			t.Fatalf("worker step %d: %v", i, err)
		}
	}
	get := httptest.NewRequest(http.MethodGet, "/v1/videos/"+id, nil)
	get.Header.Set("Authorization", "Bearer key-a")
	getRec := httptest.NewRecorder()
	h.ServeHTTP(getRec, get)
	if getRec.Code != http.StatusOK || !strings.Contains(getRec.Body.String(), `"status":"completed"`) {
		t.Fatalf("get status %d body %s", getRec.Code, getRec.Body.String())
	}
	content := httptest.NewRequest(http.MethodGet, "/v1/videos/"+id+"/content", nil)
	content.Header.Set("Authorization", "Bearer key-a")
	contentRec := httptest.NewRecorder()
	h.ServeHTTP(contentRec, content)
	if contentRec.Code != http.StatusOK || !bytes.Equal(contentRec.Body.Bytes(), video) {
		t.Fatalf("content status %d body %q", contentRec.Code, contentRec.Body.Bytes())
	}
	require.Equal(t, `"`+digest(video)+`"`, contentRec.Header().Get("ETag"))
	rangeRequest := httptest.NewRequest(http.MethodGet, "/v1/videos/"+id+"/content", nil)
	rangeRequest.Header.Set("Authorization", "Bearer key-a")
	rangeRequest.Header.Set("Range", "bytes=0-4")
	rangeResponse := httptest.NewRecorder()
	h.ServeHTTP(rangeResponse, rangeRequest)
	require.Equal(t, http.StatusPartialContent, rangeResponse.Code)
	require.Equal(t, video[:5], rangeResponse.Body.Bytes())
	headRequest := httptest.NewRequest(http.MethodHead, "/v1/videos/"+id+"/content", nil)
	headRequest.Header.Set("Authorization", "Bearer key-a")
	headResponse := httptest.NewRecorder()
	h.ServeHTTP(headResponse, headRequest)
	require.Equal(t, http.StatusOK, headResponse.Code)
	require.Empty(t, headResponse.Body.Bytes())
	if billing.settled != 1 || billing.released != 0 {
		t.Fatalf("billing settled=%d released=%d", billing.settled, billing.released)
	}
}

func TestUploadLifecycleAndLimits(t *testing.T) {
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	svc, _ := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: NoopBilling{}})
	p := Principal{OrganizationID: "org-a"}
	h, _ := NewHandler(HandlerConfig{Service: svc, Resolver: testResolver{"key-a": p}, UploadSigningKey: bytes.Repeat([]byte("s"), 32)})
	png := append([]byte{137, 80, 78, 71, 13, 10, 26, 10}, []byte("payload")...)
	sum := sha256.Sum256(png)
	createBody := `{"purpose":"video_input_image","mime":"image/png","size_bytes":` + strconv.Itoa(len(png)) + `,"sha256":"` + hex.EncodeToString(sum[:]) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/media/uploads", strings.NewReader(createBody))
	req.Header.Set("Authorization", "Bearer key-a")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create upload %d %s", rec.Code, rec.Body.String())
	}
	var created map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &created)
	uploadURL := created["upload_url"].(string)
	id := created["id"].(string)
	put := httptest.NewRequest(http.MethodPut, uploadURL, bytes.NewReader(png))
	put.Header.Set("Authorization", "Bearer key-a")
	put.Header.Set("Content-Type", "image/png")
	putRec := httptest.NewRecorder()
	h.ServeHTTP(putRec, put)
	if putRec.Code != http.StatusOK {
		t.Fatalf("put %d %s", putRec.Code, putRec.Body.String())
	}
	complete := httptest.NewRequest(http.MethodPost, "/v1/media/uploads/"+id+"/complete", nil)
	complete.Header.Set("Authorization", "Bearer key-a")
	completeRec := httptest.NewRecorder()
	h.ServeHTTP(completeRec, complete)
	if completeRec.Code != http.StatusOK {
		t.Fatalf("complete %d %s", completeRec.Code, completeRec.Body.String())
	}
	bad := UploadRequest{Purpose: "video_input_image", MIME: "image/png", SizeBytes: MaxImageBytes + 1, SHA256: strings.Repeat("a", 64)}
	if _, err := svc.CreateUpload(t.Context(), "org-a", bad); err != ErrInvalid {
		t.Fatalf("oversize upload error %v", err)
	}
}

func TestLegacyInputReferenceRemainsInert(t *testing.T) {
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	billing := new(testBilling)
	service, err := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: billing})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "org-a", PriceProfileID: "r8", PriceProfileSHA256: "sha",
		RatePerSecond: "0.07", Currency: "USD",
	}
	job, duplicate, err := service.Create(t.Context(), principal, "legacy-input-reference", CreateRequest{
		Model: "runway/gen4.5", Prompt: "test", DurationSeconds: 5,
		InputReference: map[string]any{"file_id": "file_123"},
	})
	require.NoError(t, err)
	require.False(t, duplicate)
	require.Empty(t, job.Request.InputImageID)
	require.Empty(t, job.Request.InputImageURL)
	require.Equal(t, map[string]any{"file_id": "file_123"}, job.Request.InputReference)
	require.Equal(t, 1, billing.reserved)
}

func TestRunwayRatioCompatibility(t *testing.T) {
	require.Equal(t, "720:1280", runwayRatioFromSize("720x1280"))
	require.Equal(t, "1280:720", runwayRatioFromSize("1920x1080"))
	require.Equal(t, "832:1104", runwayRatio("3:4"))
}

func TestRunwayRequestValidationPrecedesBilling(t *testing.T) {
	repo := NewMemoryStore()
	billing := new(testBilling)
	service, err := NewService(ServiceConfig{
		Store: repo, Objects: NewMemoryObjectStore(), Billing: billing,
	})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "org-a", PriceProfileID: "r8", PriceProfileSHA256: "sha",
		RatePerSecond: "0.07", Currency: "USD",
	}

	invalid := []CreateRequest{
		{Model: "runway/gen4_turbo", Prompt: "image required", DurationSeconds: 5, AspectRatio: "16:9"},
		{Model: "runway/gen4.5", Prompt: "too short", DurationSeconds: 1, AspectRatio: "16:9"},
		{Model: "runway/gen4.5", Prompt: "too long", DurationSeconds: 11, AspectRatio: "16:9"},
		{Model: "runway/gen4.5", Prompt: "text mode ratio", DurationSeconds: 5, AspectRatio: "1:1"},
		{Model: "runway/gen4_turbo", Prompt: "unknown file", DurationSeconds: 5, AspectRatio: "1:1", InputImageID: "file_123"},
		{Model: "runway/gen4_turbo", Prompt: "legacy reference is not image input", DurationSeconds: 5, AspectRatio: "1:1", InputReference: map[string]any{"file_id": "file_123"}},
		{Model: "runway/unsupported", Prompt: "unsupported", DurationSeconds: 5, AspectRatio: "16:9"},
	}
	for index, request := range invalid {
		_, _, createErr := service.Create(t.Context(), principal, "invalid-"+strconv.Itoa(index), request)
		require.Error(t, createErr, "request %d", index)
	}
	require.Zero(t, billing.reserved)

	valid := []CreateRequest{
		{Model: "runway/gen4.5", Prompt: "text landscape", DurationSeconds: 2, AspectRatio: "16:9"},
		{Model: "runway/gen4.5", Prompt: "text portrait", DurationSeconds: 10, AspectRatio: "9:16"},
		{Model: "runway/gen4.5", Prompt: "image square", DurationSeconds: 5, AspectRatio: "1:1", InputImageURL: "https://example.com/frame.png"},
		{Model: "runway/gen4_turbo", Prompt: "image wide", DurationSeconds: 5, AspectRatio: "1584:672", InputImageURL: "https://example.com/frame.png"},
	}
	for index, request := range valid {
		_, _, createErr := service.Create(t.Context(), principal, "valid-"+strconv.Itoa(index), request)
		require.NoError(t, createErr, "request %d", index)
	}
	require.Equal(t, len(valid), billing.reserved)
}

func TestUploadedImageDataURILimit(t *testing.T) {
	for _, mime := range []string{"image/png", "image/jpeg", "image/webp"} {
		prefixLength := len("data:" + mime + ";base64,")
		require.LessOrEqual(t, prefixLength+base64.StdEncoding.EncodedLen(MaxImageBytes), MaxImageDataURIBytes)
		require.Greater(t, prefixLength+base64.StdEncoding.EncodedLen(MaxImageBytes+1), MaxImageDataURIBytes)
	}
}

func TestRunwaySubmitModesAndDeleteCancellation(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		switch requests {
		case 1:
			require.Equal(t, http.MethodPost, r.Method)
			require.Equal(t, "/v1/image_to_video", r.URL.Path)
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			require.Equal(t, "gen4_turbo", payload["model"])
			require.Equal(t, "960:960", payload["ratio"])
			require.Equal(t, float64(2), payload["duration"])
			require.Equal(t, "https://example.com/frame.png", payload["promptImage"])
			_, _ = io.WriteString(w, `{"id":"turbo-task"}`)
		case 2:
			require.Equal(t, http.MethodPost, r.Method)
			require.Equal(t, "/v1/text_to_video", r.URL.Path)
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			require.Equal(t, "gen4.5", payload["model"])
			require.Equal(t, "720:1280", payload["ratio"])
			require.Equal(t, float64(10), payload["duration"])
			require.NotContains(t, payload, "promptImage")
			_, _ = io.WriteString(w, `{"id":"gen45-task"}`)
		case 3:
			require.Equal(t, http.MethodDelete, r.Method)
			require.Equal(t, "/v1/tasks/turbo-task", r.URL.Path)
			w.WriteHeader(http.StatusNoContent)
		case 4:
			require.Equal(t, http.MethodDelete, r.Method)
			require.Equal(t, "/v1/tasks/already-gone", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		default:
			t.Fatalf("unexpected provider request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	client, err := NewRunwayClient(RunwayConfig{BaseURL: server.URL, APIKey: "secret"})
	require.NoError(t, err)
	turboID, err := client.Submit(t.Context(), &Job{ID: "job-turbo", Request: CreateRequest{
		Model: "runway/gen4_turbo", Prompt: "square", DurationSeconds: 2, AspectRatio: "1:1",
	}}, "https://example.com/frame.png")
	require.NoError(t, err)
	require.Equal(t, "turbo-task", turboID)
	gen45ID, err := client.Submit(t.Context(), &Job{ID: "job-gen45", Request: CreateRequest{
		Model: "runway/gen4.5", Prompt: "portrait", DurationSeconds: 10, AspectRatio: "9:16",
	}}, "")
	require.NoError(t, err)
	require.Equal(t, "gen45-task", gen45ID)
	require.NoError(t, client.Cancel(t.Context(), turboID))
	require.NoError(t, client.Cancel(t.Context(), "already-gone"))
	require.Equal(t, 4, requests)
}

func TestCancelBeforeSubmitReleasesWithoutProviderCall(t *testing.T) {
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	billing := new(testBilling)
	service, err := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: billing})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "org-a", PriceProfileID: "r8", PriceProfileSHA256: "sha",
		RatePerSecond: "0.07", Currency: "USD",
	}
	job, _, err := service.Create(t.Context(), principal, "cancel", CreateRequest{
		Model: "runway/gen4.5", Prompt: "test", DurationSeconds: 5,
	})
	require.NoError(t, err)
	job, err = service.Cancel(t.Context(), principal.OrganizationID, job.ID)
	require.NoError(t, err)
	require.Equal(t, StateCancelRequested, job.State)
	provider := &neverProvider{}
	worker, err := NewWorker(WorkerConfig{
		Store: repo, Provider: provider, Objects: objects, Billing: billing,
		ID: "cancel-worker", Now: func() time.Time { return time.Now().UTC().Add(time.Hour) },
	})
	require.NoError(t, err)
	require.NoError(t, worker.RunOnce(t.Context()))
	require.NoError(t, worker.RunOnce(t.Context()))
	job, err = service.Get(t.Context(), principal.OrganizationID, job.ID)
	require.NoError(t, err)
	require.Equal(t, StatusCancelled, job.Status)
	require.Equal(t, 1, billing.released)
	require.Equal(t, 0, provider.calls)
}

type neverProvider struct{ calls int }

func (p *neverProvider) Submit(context.Context, *Job, string) (string, error) {
	p.calls++
	return "", errors.New("unexpected submit")
}
func (p *neverProvider) Poll(context.Context, string) (ProviderResult, error) {
	p.calls++
	return ProviderResult{}, errors.New("unexpected poll")
}
func (p *neverProvider) Cancel(context.Context, string) error {
	p.calls++
	return errors.New("unexpected cancel")
}

func TestAmbiguousSubmitIsNotRetried(t *testing.T) {
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	billing := new(testBilling)
	service, err := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: billing})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "org-a", PriceProfileID: "r8", PriceProfileSHA256: "sha",
		RatePerSecond: "0.07", Currency: "USD",
	}
	job, _, err := service.Create(t.Context(), principal, "ambiguous", CreateRequest{
		Model: "runway/gen4.5", Prompt: "test", DurationSeconds: 5,
	})
	require.NoError(t, err)
	provider := &neverProvider{}
	worker, err := NewWorker(WorkerConfig{
		Store: repo, Provider: provider, Objects: objects, Billing: billing,
		ID: "ambiguous-worker", Now: func() time.Time { return time.Now().UTC().Add(time.Hour) },
	})
	require.NoError(t, err)
	require.NoError(t, worker.RunOnce(t.Context()))
	require.NoError(t, worker.RunOnce(t.Context()))
	job, err = service.Get(t.Context(), principal.OrganizationID, job.ID)
	require.NoError(t, err)
	require.Equal(t, StateSubmissionUnknown, job.State)
	require.Equal(t, StatusFailed, job.Status)
	require.Equal(t, 1, provider.calls)
	require.ErrorIs(t, worker.RunOnce(t.Context()), ErrNotFound)
	require.Equal(t, 1, provider.calls)
}

func TestRecoveredSubmittingJobIsNotSubmittedAgain(t *testing.T) {
	repo := NewMemoryStore()
	objects := NewMemoryObjectStore()
	billing := new(testBilling)
	service, err := NewService(ServiceConfig{Store: repo, Objects: objects, Billing: billing})
	require.NoError(t, err)
	principal := Principal{
		OrganizationID: "org-a", PriceProfileID: "r8", PriceProfileSHA256: "sha",
		RatePerSecond: "0.07", Currency: "USD",
	}
	job, _, err := service.Create(t.Context(), principal, "crash-window", CreateRequest{
		Model: "runway/gen4.5", Prompt: "test", DurationSeconds: 5,
	})
	require.NoError(t, err)
	now := time.Now().UTC().Add(time.Hour)
	claimed, err := repo.ClaimJob(t.Context(), "worker-before-crash", now, 5*time.Minute)
	require.NoError(t, err)
	ready, err := repo.Transition(t.Context(), claimed, StateReadyToSubmit, JobUpdate{})
	require.NoError(t, err)
	claimed, err = repo.ClaimJob(t.Context(), "worker-before-crash", now.Add(time.Second), 5*time.Minute)
	require.NoError(t, err)
	require.Equal(t, ready.ID, claimed.ID)
	submitting, err := repo.Transition(t.Context(), claimed, StateSubmitting, JobUpdate{RetainLease: true})
	require.NoError(t, err)
	require.Equal(t, StateSubmitting, submitting.State)

	provider := &neverProvider{}
	worker, err := NewWorker(WorkerConfig{
		Store: repo, Provider: provider, Objects: objects, Billing: billing,
		ID: "worker-after-crash", Now: func() time.Time { return now.Add(10 * time.Minute) },
	})
	require.NoError(t, err)
	require.NoError(t, worker.RunOnce(t.Context()))
	job, err = service.Get(t.Context(), principal.OrganizationID, job.ID)
	require.NoError(t, err)
	require.Equal(t, StateSubmissionUnknown, job.State)
	require.Zero(t, provider.calls)
}
