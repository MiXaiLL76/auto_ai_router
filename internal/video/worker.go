package video

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/google/uuid"
)

type WorkerConfig struct {
	Store                                Store
	Provider                             Provider
	Objects                              ObjectStore
	Billing                              Billing
	ID                                   string
	LeaseTTL, PollInterval, IdleInterval time.Duration
	MaxAttempts                          int
	Now                                  func() time.Time
}
type Worker struct{ cfg WorkerConfig }

func NewWorker(cfg WorkerConfig) (*Worker, error) {
	if cfg.Store == nil || cfg.Provider == nil || cfg.Objects == nil || cfg.Billing == nil {
		return nil, ErrInvalid
	}
	if cfg.ID == "" {
		cfg.ID = "video-worker-" + uuid.NewString()
	}
	if cfg.LeaseTTL < 5*time.Minute {
		cfg.LeaseTTL = 5 * time.Minute
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 5 * time.Second
	}
	if cfg.IdleInterval <= 0 {
		cfg.IdleInterval = time.Second
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 20
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Worker{cfg: cfg}, nil
}
func (w *Worker) Run(ctx context.Context) error {
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			err := w.RunOnce(ctx)
			delay := time.Duration(0)
			if err == ErrNotFound {
				delay = w.cfg.IdleInterval
			} else if err != nil {
				delay = time.Second
			}
			timer.Reset(delay)
		}
	}
}
func (w *Worker) RunOnce(ctx context.Context) error {
	j, err := w.cfg.Store.ClaimJob(ctx, w.cfg.ID, w.cfg.Now().UTC(), w.cfg.LeaseTTL)
	if err != nil {
		return err
	}
	return w.process(ctx, j)
}
func (w *Worker) process(ctx context.Context, j *Job) error {
	if j.CancelRequested && j.State == StateReserving {
		_, err := w.cfg.Store.Transition(ctx, j, StateCancelRequested, JobUpdate{})
		return err
	}
	if j.CancelRequested && (j.State == StateReadyToSubmit || j.State == StateSubmitting) {
		code := "cancelled"
		_, err := w.cfg.Store.Transition(ctx, j, StateReleasing, JobUpdate{ErrorCode: &code})
		return err
	}
	if j.CancelRequested && (j.State == StateSubmitted || j.State == StateProcessing) {
		_, err := w.cfg.Store.Transition(ctx, j, StateCancelRequested, JobUpdate{})
		return err
	}
	switch j.State {
	case StateReserving:
		reservation, err := w.cfg.Billing.Reserve(ctx, j, j.ID+":reserve")
		if err != nil {
			return w.retryOrFail(ctx, j, err)
		}
		if reservation.Amount == "" {
			reservation.Amount = j.QuotedAmount
		}
		if reservation.Currency == "" {
			reservation.Currency = j.Currency
		}
		_, err = w.cfg.Store.ActivateJob(ctx, j, reservation)
		return err
	case StateQueued:
		_, err := w.cfg.Store.Transition(ctx, j, StateReadyToSubmit, JobUpdate{})
		return err
	case StateReadyToSubmit:
		started, err := w.cfg.Store.Transition(ctx, j, StateSubmitting, JobUpdate{RetainLease: true})
		if err != nil {
			return err
		}
		return w.submit(ctx, started)
	case StateSubmitting:
		return w.markSubmissionUnknown(ctx, j, errors.New("submission lease expired before provider task was persisted"))
	case StateSubmitted, StateProcessing:
		return w.poll(ctx, j)
	case StateStoring:
		return w.store(ctx, j)
	case StateSettling:
		return w.settle(ctx, j)
	case StateCancelRequested:
		to := StateCancelling
		if j.ProviderJobID == "" {
			to = StateReleasing
		}
		code := "cancelled"
		_, err := w.cfg.Store.Transition(ctx, j, to, JobUpdate{ErrorCode: &code})
		return err
	case StateCancelling:
		return w.cancel(ctx, j)
	case StateReleasing:
		return w.release(ctx, j)
	default:
		return ErrStateChanged
	}
}
func (w *Worker) submit(ctx context.Context, j *Job) error {
	image := j.Request.InputImageURL
	if strings.HasPrefix(j.Request.InputImageID, "upl_") {
		u, err := w.cfg.Store.GetUpload(ctx, j.OrganizationID, j.Request.InputImageID)
		if err != nil {
			return w.retryOrFail(ctx, j, err)
		}
		o, err := w.cfg.Objects.OpenUpload(ctx, u)
		if err != nil {
			return w.retryOrFail(ctx, j, err)
		}
		defer func() { _ = o.Body.Close() }()
		data, err := io.ReadAll(io.LimitReader(o.Body, MaxImageBytes+1))
		if err != nil || int64(len(data)) > MaxImageBytes {
			return w.retryOrFail(ctx, j, ErrInvalid)
		}
		image = "data:" + o.ContentType + ";base64," + base64.StdEncoding.EncodeToString(data)
		if len(image) > MaxImageDataURIBytes {
			return w.retryOrFail(ctx, j, ErrInvalid)
		}
	} else if j.Request.InputImageID != "" {
		image = j.Request.InputImageID
	}
	id, err := w.cfg.Provider.Submit(ctx, j, image)
	if err != nil {
		return w.markSubmissionUnknown(ctx, j, err)
	}
	next := w.cfg.Now().Add(w.cfg.PollInterval)
	_, err = w.cfg.Store.Transition(ctx, j, StateSubmitted, JobUpdate{ProviderJobID: &id, NextRunAt: &next})
	return err
}

func (w *Worker) markSubmissionUnknown(ctx context.Context, job *Job, cause error) error {
	code := "submission_unknown"
	message := "provider submission outcome is unknown"
	_, err := w.cfg.Store.Transition(ctx, job, StateSubmissionUnknown, JobUpdate{
		ErrorCode: &code, ErrorMessage: &message,
	})
	if err != nil {
		return fmt.Errorf("provider submission outcome unknown: %v; persist ambiguity: %w", cause, err)
	}
	return nil
}
func (w *Worker) poll(ctx context.Context, j *Job) error {
	r, err := w.cfg.Provider.Poll(ctx, j.ProviderJobID)
	if err != nil {
		return w.retryOrFail(ctx, j, err)
	}
	next := w.cfg.Now().Add(w.cfg.PollInterval)
	switch r.Status {
	case StatusQueued, StatusInProgress:
		_, err = w.cfg.Store.Transition(ctx, j, StateProcessing, JobUpdate{NextRunAt: &next})
	case StatusCompleted:
		_, err = w.cfg.Store.Transition(ctx, j, StateStoring, JobUpdate{ResultURL: &r.ResultURL})
	case StatusFailed:
		code := first(r.ErrorCode, "provider_failed")
		message := first(r.ErrorMessage, "video generation failed")
		_, err = w.cfg.Store.Transition(ctx, j, StateReleasing, JobUpdate{ErrorCode: &code, ErrorMessage: &message})
	case StatusCancelled:
		code := "cancelled"
		_, err = w.cfg.Store.Transition(ctx, j, StateReleasing, JobUpdate{ErrorCode: &code})
	default:
		err = ErrStateChanged
	}
	return err
}
func (w *Worker) store(ctx context.Context, j *Job) error {
	o, err := w.cfg.Objects.FetchResult(ctx, j, j.ResultURL)
	if err != nil {
		return w.retryOrFail(ctx, j, err)
	}
	_, err = w.cfg.Store.Transition(ctx, j, StateSettling, JobUpdate{ResultContentType: &o.ContentType, ResultSize: &o.Size, ResultETag: &o.ETag})
	return err
}
func (w *Worker) settle(ctx context.Context, j *Job) error {
	if err := w.cfg.Billing.Settle(ctx, j, j.ID+":settle"); err != nil {
		return w.retryOrFail(ctx, j, err)
	}
	_, err := w.cfg.Store.Transition(ctx, j, StateCompleted, JobUpdate{})
	return err
}
func (w *Worker) cancel(ctx context.Context, j *Job) error {
	if err := w.cfg.Provider.Cancel(ctx, j.ProviderJobID); err != nil {
		return w.retryOrFail(ctx, j, err)
	}
	code := "cancelled"
	_, err := w.cfg.Store.Transition(ctx, j, StateReleasing, JobUpdate{ErrorCode: &code})
	return err
}
func (w *Worker) release(ctx context.Context, j *Job) error {
	if err := w.cfg.Billing.Release(ctx, j, j.ID+":release"); err != nil {
		return w.retryOrFail(ctx, j, err)
	}
	to := StateFailed
	if j.ErrorCode == "cancelled" {
		to = StateCancelled
	}
	_, err := w.cfg.Store.Transition(ctx, j, to, JobUpdate{})
	return err
}
func (w *Worker) retryOrFail(ctx context.Context, j *Job, cause error) error {
	attempt := j.Attempts + 1
	if attempt < w.cfg.MaxAttempts || j.State == StateSettling || j.State == StateReleasing {
		next := w.cfg.Now().Add(retryDelay(attempt))
		_, err := w.cfg.Store.Transition(ctx, j, j.State, JobUpdate{NextRunAt: &next, IncrementAttempts: true})
		return err
	}
	code := "internal_error"
	message := "video processing failed"
	_, err := w.cfg.Store.Transition(ctx, j, StateReleasing, JobUpdate{ErrorCode: &code, ErrorMessage: &message})
	if err != nil {
		return fmt.Errorf("%v: %w", cause, err)
	}
	return nil
}
func retryDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	if attempt > 6 {
		attempt = 6
	}
	return time.Duration(1<<uint(attempt-1)) * time.Second
}
func first(v, fallback string) string {
	if v != "" {
		return v
	}
	return fallback
}

type Cleaner struct {
	store    Store
	objects  ObjectStore
	interval time.Duration
	now      func() time.Time
	onError  func(error)
}

func NewCleaner(store Store, objects ObjectStore, interval time.Duration, errorHandlers ...func(error)) *Cleaner {
	if interval <= 0 {
		interval = time.Minute
	}
	var onError func(error)
	if len(errorHandlers) > 0 {
		onError = errorHandlers[0]
	}
	return &Cleaner{store: store, objects: objects, interval: interval, now: time.Now, onError: onError}
}
func (c *Cleaner) RunOnce(ctx context.Context) error {
	uploads, err := c.store.ClaimExpiredUploads(ctx, c.now().UTC(), 100)
	if err != nil {
		return err
	}
	for i := range uploads {
		if err = c.objects.DeleteUpload(ctx, &uploads[i]); err != nil {
			return err
		}
		if err = c.store.ExpireUpload(ctx, uploads[i].OrganizationID, uploads[i].ID); err != nil {
			return err
		}
	}
	return nil
}
func (c *Cleaner) Run(ctx context.Context) error {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()
	for {
		if err := c.RunOnce(ctx); err != nil {
			if c.onError != nil {
				c.onError(err)
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
