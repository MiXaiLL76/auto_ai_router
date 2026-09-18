package video

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"strings"
	"time"

	"github.com/google/uuid"
)

type ServiceConfig struct {
	Store     Store
	Objects   ObjectStore
	Billing   Billing
	UploadTTL time.Duration
	Now       func() time.Time
}

type Service struct {
	store     Store
	objects   ObjectStore
	billing   Billing
	uploadTTL time.Duration
	now       func() time.Time
}

func NewService(cfg ServiceConfig) (*Service, error) {
	if cfg.Store == nil || cfg.Objects == nil || cfg.Billing == nil {
		return nil, ErrInvalid
	}
	if cfg.UploadTTL <= 0 {
		cfg.UploadTTL = 15 * time.Minute
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Service{store: cfg.Store, objects: cfg.Objects, billing: cfg.Billing, uploadTTL: cfg.UploadTTL, now: cfg.Now}, nil
}

func (s *Service) Create(ctx context.Context, p Principal, idem string, request CreateRequest) (*Job, bool, error) {
	req, err := request.normalized()
	if err != nil {
		return nil, false, err
	}
	if err = validCreatePrincipal(p); err != nil {
		return nil, false, err
	}
	if idem == "" {
		idem = "auto-" + uuid.NewString()
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, false, err
	}
	sum := sha256.Sum256(raw)
	hash := hex.EncodeToString(sum[:])
	if strings.HasPrefix(req.InputImageID, "upl_") {
		upload, getErr := s.store.GetUpload(ctx, p.OrganizationID, req.InputImageID)
		if getErr != nil || upload.State != "completed" || upload.Purpose != "video_input_image" {
			return nil, false, ErrInvalid
		}
	}
	job, duplicate, err := s.store.CreateJob(ctx, p, idem, req, hash)
	if err != nil {
		return nil, false, err
	}
	amount := job.QuotedAmount
	reservation := Reservation{Amount: amount, Currency: p.Currency}
	if duplicate {
		return job, true, nil
	}
	reservation, err = s.billing.Reserve(ctx, job, job.ID+":reserve")
	if err != nil {
		_, _ = s.store.RejectJob(ctx, job, "insufficient_quota", "usage reservation failed")
		return nil, false, ErrQuota
	}
	if reservation.Amount == "" {
		reservation.Amount = amount
	}
	if reservation.Currency == "" {
		reservation.Currency = p.Currency
	}
	id := job.ID
	job, err = s.store.ActivateJob(ctx, job, reservation)
	if errors.Is(err, ErrStateChanged) {
		current, getErr := s.store.GetJob(ctx, p.OrganizationID, id)
		if getErr == nil && current.RequestHash == hash {
			return current, true, nil
		}
	}
	return job, duplicate, err
}

func (s *Service) Get(ctx context.Context, org, id string) (*Job, error) {
	return s.store.GetJob(ctx, org, id)
}
func (s *Service) Cancel(ctx context.Context, org, id string) (*Job, error) {
	return s.store.RequestCancel(ctx, org, id)
}
func (s *Service) Content(ctx context.Context, org, id string) (*Object, error) {
	j, err := s.store.GetJob(ctx, org, id)
	if err != nil {
		return nil, err
	}
	if j.State != StateCompleted {
		return nil, ErrNotReady
	}
	return s.objects.OpenResult(ctx, j)
}
func (s *Service) CreateUpload(ctx context.Context, org string, r UploadRequest) (*Upload, error) {
	if err := r.validate(); err != nil {
		return nil, err
	}
	return s.store.CreateUpload(ctx, org, r, s.now().UTC(), s.uploadTTL)
}
func (s *Service) PutUpload(ctx context.Context, org, id, contentType string, body io.Reader) error {
	u, err := s.store.GetUpload(ctx, org, id)
	if err != nil {
		return err
	}
	if u.State == "completed" {
		return nil
	}
	if u.State != "created" || !u.ExpiresAt.After(s.now()) {
		return ErrNotFound
	}
	return s.objects.PutUpload(ctx, u, contentType, body)
}
func (s *Service) CompleteUpload(ctx context.Context, org, id string) (*Upload, error) {
	u, err := s.store.GetUpload(ctx, org, id)
	if err != nil {
		return nil, err
	}
	if u.State == "completed" {
		return u, nil
	}
	if !u.ExpiresAt.After(s.now()) {
		return nil, ErrNotFound
	}
	if err = s.objects.VerifyUpload(ctx, u); err != nil {
		return nil, err
	}
	return s.store.CompleteUpload(ctx, org, id)
}

func validCreatePrincipal(p Principal) error {
	if strings.TrimSpace(p.OrganizationID) == "" || strings.TrimSpace(p.PriceProfileID) == "" || strings.TrimSpace(p.PriceProfileSHA256) == "" || strings.TrimSpace(p.RatePerSecond) == "" || strings.TrimSpace(p.Currency) == "" {
		return ErrAuthentication
	}
	_, err := quote(p.RatePerSecond, 1)
	return err
}
func quote(rate string, seconds int) (string, error) {
	r, ok := new(big.Rat).SetString(strings.TrimSpace(rate))
	if !ok || r.Sign() < 0 {
		return "", ErrInvalid
	}
	r.Mul(r, big.NewRat(int64(seconds), 1))
	return r.FloatString(decimalPlaces(rate)), nil
}
func decimalPlaces(v string) int {
	if i := strings.IndexByte(v, '.'); i >= 0 {
		return len(strings.TrimRight(v[i+1:], "0"))
	}
	return 0
}
