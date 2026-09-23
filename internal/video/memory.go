package video

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

type MemoryStore struct {
	mu          sync.Mutex
	jobs        map[string]*Job
	idempotency map[string]string
	uploads     map[string]*Upload
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{jobs: map[string]*Job{}, idempotency: map[string]string{}, uploads: map[string]*Upload{}}
}

func (s *MemoryStore) CreateJob(_ context.Context, p Principal, idem string, req CreateRequest, hash string) (*Job, bool, error) {
	if p.OrganizationID == "" || idem == "" {
		return nil, false, ErrInvalid
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key := p.OrganizationID + "\x00" + idem
	if id := s.idempotency[key]; id != "" {
		j := s.jobs[id]
		if j.RequestHash != hash {
			return nil, false, ErrConflict
		}
		return cloneJob(j), true, nil
	}
	now := time.Now().UTC()
	amount, err := quote(p.RatePerSecond, req.DurationSeconds)
	if err != nil {
		return nil, false, err
	}
	j := &Job{ID: "vid_" + uuid.NewString(), OrganizationID: p.OrganizationID, IdempotencyKey: idem, RequestHash: hash, Request: req,
		Principal: p, Status: StatusQueued, State: StateReserving, Version: 1, NextRunAt: now.Add(30 * time.Second), QuotedAmount: amount, Currency: p.Currency, CreatedAt: now, UpdatedAt: now}
	s.jobs[j.ID] = j
	s.idempotency[key] = j.ID
	return cloneJob(j), false, nil
}
func (s *MemoryStore) ActivateJob(_ context.Context, claimed *Job, r Reservation) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := s.jobs[claimed.ID]
	if j == nil || j.Version != claimed.Version || j.State != StateReserving {
		return nil, ErrStateChanged
	}
	j.ReservationID = r.ID
	j.ReservationHandle = r.Handle
	j.QuotedAmount = r.Amount
	j.Currency = r.Currency
	j.State = StateQueued
	j.Status = StatusQueued
	j.Version++
	j.NextRunAt = time.Now().UTC()
	j.LeaseOwner = ""
	j.LeaseExpiresAt = nil
	j.UpdatedAt = j.NextRunAt
	return cloneJob(j), nil
}
func (s *MemoryStore) RejectJob(_ context.Context, claimed *Job, code, message string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := s.jobs[claimed.ID]
	if j == nil || j.Version != claimed.Version || j.State != StateReserving {
		return nil, ErrStateChanged
	}
	j.State = StateFailed
	j.Status = StatusFailed
	j.ErrorCode = code
	j.ErrorMessage = message
	j.Version++
	j.UpdatedAt = time.Now().UTC()
	return cloneJob(j), nil
}

func (s *MemoryStore) GetJob(_ context.Context, org, id string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := s.jobs[id]
	if j == nil || j.OrganizationID != org {
		return nil, ErrNotFound
	}
	return cloneJob(j), nil
}

func (s *MemoryStore) RequestCancel(_ context.Context, org, id string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := s.jobs[id]
	if j == nil || j.OrganizationID != org {
		return nil, ErrNotFound
	}
	if IsTerminal(j.State) || j.State == StateCancelRequested || j.State == StateCancelling {
		return cloneJob(j), nil
	}
	if j.LeaseOwner != "" && j.LeaseExpiresAt != nil && j.LeaseExpiresAt.After(time.Now().UTC()) {
		j.CancelRequested = true
		return cloneJob(j), nil
	}
	if j.State == StateSubmitting {
		j.CancelRequested = true
		return cloneJob(j), nil
	}
	if j.State == StateStoring || j.State == StateSettling || j.State == StateReleasing {
		return cloneJob(j), nil
	}
	j.State = StateCancelRequested
	j.Status = StatusInProgress
	j.Version++
	j.LeaseOwner = ""
	j.LeaseExpiresAt = nil
	j.NextRunAt = time.Now().UTC()
	j.UpdatedAt = j.NextRunAt
	return cloneJob(j), nil
}

func (s *MemoryStore) ClaimJob(_ context.Context, owner string, now time.Time, lease time.Duration) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var candidates []*Job
	for _, j := range s.jobs {
		if !IsTerminal(j.State) && !j.NextRunAt.After(now) && (j.LeaseExpiresAt == nil || j.LeaseExpiresAt.Before(now)) {
			candidates = append(candidates, j)
		}
	}
	if len(candidates) == 0 {
		return nil, ErrNotFound
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].NextRunAt.Before(candidates[j].NextRunAt) })
	j := candidates[0]
	expires := now.Add(lease)
	j.LeaseOwner = owner
	j.LeaseExpiresAt = &expires
	j.Version++
	j.UpdatedAt = now
	return cloneJob(j), nil
}

func (s *MemoryStore) Transition(_ context.Context, claimed *Job, to State, u JobUpdate) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := s.jobs[claimed.ID]
	if j == nil || j.OrganizationID != claimed.OrganizationID {
		return nil, ErrNotFound
	}
	if j.Version != claimed.Version || j.State != claimed.State || j.LeaseOwner != claimed.LeaseOwner || !CanTransition(j.State, to) {
		return nil, ErrStateChanged
	}
	set := func(p *string, dst *string) {
		if p != nil {
			*dst = *p
		}
	}
	set(u.ProviderJobID, &j.ProviderJobID)
	set(u.ResultURL, &j.ResultURL)
	set(u.ResultContentType, &j.ResultContentType)
	set(u.ResultETag, &j.ResultETag)
	set(u.ReservationID, &j.ReservationID)
	set(u.ReservationHandle, &j.ReservationHandle)
	set(u.QuotedAmount, &j.QuotedAmount)
	set(u.Currency, &j.Currency)
	set(u.ErrorCode, &j.ErrorCode)
	set(u.ErrorMessage, &j.ErrorMessage)
	if u.ResultSize != nil {
		j.ResultSize = *u.ResultSize
	}
	if u.NextRunAt != nil {
		j.NextRunAt = *u.NextRunAt
	} else {
		j.NextRunAt = time.Now().UTC()
	}
	actualTo := to
	if to == StateSubmitted && j.CancelRequested {
		actualTo = StateCancelRequested
	}
	if u.IncrementAttempts {
		j.Attempts++
	} else if j.State != actualTo {
		j.Attempts = 0
	}
	j.State = actualTo
	j.Status = publicStatus(actualTo)
	j.Version++
	if !u.RetainLease {
		j.LeaseOwner = ""
		j.LeaseExpiresAt = nil
	}
	j.UpdatedAt = time.Now().UTC()
	return cloneJob(j), nil
}

func (s *MemoryStore) CreateUpload(_ context.Context, org string, r UploadRequest, now time.Time, ttl time.Duration) (*Upload, error) {
	if org == "" || r.validate() != nil {
		return nil, ErrInvalid
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := "upl_" + uuid.NewString()
	u := &Upload{ID: id, OrganizationID: org, Purpose: r.Purpose, MIME: r.MIME, SizeBytes: r.SizeBytes, SHA256: r.SHA256, ObjectKey: safePart(org) + "/uploads/" + id + "/object", State: "created", CreatedAt: now, ExpiresAt: now.Add(ttl)}
	s.uploads[id] = u
	return cloneUpload(u), nil
}
func (s *MemoryStore) GetUpload(_ context.Context, org, id string) (*Upload, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.uploads[id]
	if u == nil || u.OrganizationID != org {
		return nil, ErrNotFound
	}
	return cloneUpload(u), nil
}
func (s *MemoryStore) CompleteUpload(_ context.Context, org, id string) (*Upload, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.uploads[id]
	if u == nil || u.OrganizationID != org {
		return nil, ErrNotFound
	}
	if u.State == "completed" {
		return cloneUpload(u), nil
	}
	if u.State != "created" || time.Now().After(u.ExpiresAt) {
		return nil, ErrNotReady
	}
	u.State = "completed"
	return cloneUpload(u), nil
}
func (s *MemoryStore) ClaimExpiredUploads(_ context.Context, now time.Time, limit int) ([]Upload, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Upload
	for _, u := range s.uploads {
		if (u.State == "created" || u.State == "expiring") && u.ExpiresAt.Before(now) {
			u.State = "expiring"
			out = append(out, *cloneUpload(u))
			if len(out) >= limit {
				return out, nil
			}
		}
	}
	return out, nil
}
func (s *MemoryStore) ExpireUpload(_ context.Context, org, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.uploads[id]
	if u == nil || u.OrganizationID != org {
		return ErrNotFound
	}
	u.State = "expired"
	return nil
}

func cloneJob(in *Job) *Job {
	if in == nil {
		return nil
	}
	out := *in
	if in.Request.Seed != nil {
		seed := *in.Request.Seed
		out.Request.Seed = &seed
	}
	if in.LeaseExpiresAt != nil {
		expiry := *in.LeaseExpiresAt
		out.LeaseExpiresAt = &expiry
	}
	return &out
}
func cloneUpload(in *Upload) *Upload {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}
