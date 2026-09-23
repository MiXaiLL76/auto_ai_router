package video

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type DB interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

type PostgresStore struct{ db DB }

func NewPostgresStore(db DB) *PostgresStore {
	if db == nil {
		return nil
	}
	return &PostgresStore{db: db}
}

func (s *PostgresStore) CreateJob(ctx context.Context, principal Principal, idem string, req CreateRequest, hash string) (*Job, bool, error) {
	org := principal.OrganizationID
	if s == nil || org == "" || idem == "" || hash == "" {
		return nil, false, ErrInvalid
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, false, err
	}
	principalRaw, err := json.Marshal(principal)
	if err != nil {
		return nil, false, err
	}
	id := "vid_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	amount, err := quote(principal.RatePerSecond, req.DurationSeconds)
	if err != nil {
		return nil, false, err
	}
	created := false
	tag, err := s.db.Exec(ctx, `INSERT INTO air_video_jobs
	  (id,organization_id,idempotency_key,request_hash,request,principal,public_status,state,next_run_at,quoted_amount,currency)
		VALUES ($1,$2,$3,$4,$5,$6,'queued','reserving',now()+interval '30 seconds',$7,$8) ON CONFLICT (organization_id,idempotency_key) DO NOTHING`, id, org, idem, hash, raw, principalRaw, amount, principal.Currency)
	if err != nil {
		return nil, false, err
	}
	created = tag.RowsAffected() == 1
	job, err := s.getByIdempotency(ctx, org, idem)
	if err != nil {
		return nil, false, err
	}
	if job.RequestHash != hash {
		return nil, false, ErrConflict
	}
	return job, !created, nil
}

func (s *PostgresStore) ActivateJob(ctx context.Context, job *Job, r Reservation) (*Job, error) {
	tag, err := s.db.Exec(ctx, `UPDATE air_video_jobs SET reservation_id=$1,reservation_handle=$2,quoted_amount=$3,currency=$4,state='queued',public_status='queued',next_run_at=now(),version=version+1,lease_owner='',lease_expires_at=NULL,updated_at=now() WHERE organization_id=$5 AND id=$6 AND version=$7 AND state='reserving'`, r.ID, r.Handle, r.Amount, r.Currency, job.OrganizationID, job.ID, job.Version)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() != 1 {
		return nil, ErrStateChanged
	}
	return s.GetJob(ctx, job.OrganizationID, job.ID)
}

func (s *PostgresStore) RejectJob(ctx context.Context, job *Job, code, message string) (*Job, error) {
	tag, err := s.db.Exec(ctx, `UPDATE air_video_jobs SET state='failed',public_status='failed',error_code=$1,error_message=$2,version=version+1,updated_at=now() WHERE organization_id=$3 AND id=$4 AND version=$5 AND state='reserving'`, code, message, job.OrganizationID, job.ID, job.Version)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() != 1 {
		return nil, ErrStateChanged
	}
	return s.GetJob(ctx, job.OrganizationID, job.ID)
}

func (s *PostgresStore) getByIdempotency(ctx context.Context, org, idem string) (*Job, error) {
	return scanJob(s.db.QueryRow(ctx, jobSelect+` WHERE organization_id=$1 AND idempotency_key=$2`, org, idem))
}

func (s *PostgresStore) GetJob(ctx context.Context, org, id string) (*Job, error) {
	job, err := scanJob(s.db.QueryRow(ctx, jobSelect+` WHERE organization_id=$1 AND id=$2`, org, id))
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	return job, err
}

func (s *PostgresStore) RequestCancel(ctx context.Context, org, id string) (*Job, error) {
	job, err := s.GetJob(ctx, org, id)
	if err != nil {
		return nil, err
	}
	if IsTerminal(job.State) || job.State == StateCancelRequested || job.State == StateCancelling {
		return job, nil
	}
	if job.LeaseOwner != "" && job.LeaseExpiresAt != nil && job.LeaseExpiresAt.After(time.Now().UTC()) {
		tag, updateErr := s.db.Exec(ctx, `UPDATE air_video_jobs SET cancel_requested=true,updated_at=now() WHERE organization_id=$1 AND id=$2 AND version=$3`, org, id, job.Version)
		err = updateErr
		if err != nil {
			return nil, err
		}
		if tag.RowsAffected() != 1 {
			return s.RequestCancel(ctx, org, id)
		}
		job.CancelRequested = true
		return job, nil
	}
	if job.State == StateSubmitting {
		tag, updateErr := s.db.Exec(ctx, `UPDATE air_video_jobs SET cancel_requested=true,updated_at=now() WHERE organization_id=$1 AND id=$2 AND version=$3 AND state='submitting'`, org, id, job.Version)
		err = updateErr
		if err != nil {
			return nil, err
		}
		if tag.RowsAffected() != 1 {
			return s.RequestCancel(ctx, org, id)
		}
		job.CancelRequested = true
		return job, nil
	}
	if job.State == StateStoring || job.State == StateSettling || job.State == StateReleasing {
		return job, nil
	}
	tag, err := s.db.Exec(ctx, `UPDATE air_video_jobs SET state='cancel_requested',public_status='in_progress',version=version+1,
      lease_owner='',lease_expires_at=NULL,next_run_at=now(),updated_at=now()
	  WHERE organization_id=$1 AND id=$2 AND version=$3 AND state IN ('reserving','queued','ready_to_submit','submitting','submitted','processing')`, org, id, job.Version)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() != 1 {
		return s.RequestCancel(ctx, org, id)
	}
	return s.GetJob(ctx, org, id)
}

func (s *PostgresStore) ClaimJob(ctx context.Context, owner string, now time.Time, lease time.Duration) (*Job, error) {
	if owner == "" || lease <= 0 {
		return nil, ErrInvalid
	}
	row := s.db.QueryRow(ctx, `WITH candidate AS (
      SELECT id FROM air_video_jobs WHERE state NOT IN ('completed','failed','cancelled','submission_unknown')
      AND next_run_at <= $2 AND (lease_expires_at IS NULL OR lease_expires_at < $2)
      ORDER BY next_run_at,created_at FOR UPDATE SKIP LOCKED LIMIT 1
    ) UPDATE air_video_jobs j SET lease_owner=$1,lease_expires_at=$2+$3::interval,version=version+1,updated_at=$2
      FROM candidate c WHERE j.id=c.id RETURNING `+jobColumns, owner, now, fmt.Sprintf("%f seconds", lease.Seconds()))
	job, err := scanJob(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	return job, err
}

func (s *PostgresStore) Transition(ctx context.Context, job *Job, to State, u JobUpdate) (*Job, error) {
	if job == nil || !CanTransition(job.State, to) {
		return nil, ErrStateChanged
	}
	next := time.Now().UTC()
	if u.NextRunAt != nil {
		next = *u.NextRunAt
	}
	values := func(p *string, old string) string {
		if p != nil {
			return *p
		}
		return old
	}
	resultSize := job.ResultSize
	if u.ResultSize != nil {
		resultSize = *u.ResultSize
	}
	tag, err := s.db.Exec(ctx, `UPDATE air_video_jobs SET state=CASE WHEN $1='submitted' AND cancel_requested THEN 'cancel_requested' ELSE $1 END,public_status=$2,version=version+1,
      provider_job_id=$3,result_url=$4,result_content_type=$5,result_size=$6,result_etag=$7,
      reservation_id=$8,reservation_handle=$9,quoted_amount=$10,currency=$11,error_code=$12,error_message=$13,
	  next_run_at=$14,attempts=CASE WHEN $20 THEN attempts+1 WHEN state<>$1 THEN 0 ELSE attempts END,
	  lease_owner=CASE WHEN $21 THEN lease_owner ELSE '' END,
	  lease_expires_at=CASE WHEN $21 THEN lease_expires_at ELSE NULL END,updated_at=now()
      WHERE organization_id=$15 AND id=$16 AND version=$17 AND state=$18 AND lease_owner=$19`,
		to, publicStatus(to), values(u.ProviderJobID, job.ProviderJobID), values(u.ResultURL, job.ResultURL),
		values(u.ResultContentType, job.ResultContentType), resultSize, values(u.ResultETag, job.ResultETag),
		values(u.ReservationID, job.ReservationID), values(u.ReservationHandle, job.ReservationHandle), values(u.QuotedAmount, job.QuotedAmount), values(u.Currency, job.Currency),
		values(u.ErrorCode, job.ErrorCode), values(u.ErrorMessage, job.ErrorMessage), next,
		job.OrganizationID, job.ID, job.Version, job.State, job.LeaseOwner, u.IncrementAttempts, u.RetainLease)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() != 1 {
		return nil, ErrStateChanged
	}
	return s.GetJob(ctx, job.OrganizationID, job.ID)
}

func (s *PostgresStore) CreateUpload(ctx context.Context, org string, req UploadRequest, now time.Time, ttl time.Duration) (*Upload, error) {
	if org == "" || ttl <= 0 || req.validate() != nil {
		return nil, ErrInvalid
	}
	id := "upl_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	key := safePart(org) + "/uploads/" + id + "/object"
	_, err := s.db.Exec(ctx, `INSERT INTO air_video_uploads
      (id,organization_id,purpose,mime,size_bytes,sha256,object_key,state,created_at,expires_at)
      VALUES ($1,$2,$3,$4,$5,$6,$7,'created',$8,$9)`, id, org, req.Purpose, strings.ToLower(req.MIME), req.SizeBytes, strings.ToLower(req.SHA256), key, now, now.Add(ttl))
	if err != nil {
		return nil, err
	}
	return s.GetUpload(ctx, org, id)
}

func (s *PostgresStore) GetUpload(ctx context.Context, org, id string) (*Upload, error) {
	u, err := scanUpload(s.db.QueryRow(ctx, uploadSelect+` WHERE organization_id=$1 AND id=$2`, org, id))
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	return u, err
}

func (s *PostgresStore) CompleteUpload(ctx context.Context, org, id string) (*Upload, error) {
	tag, err := s.db.Exec(ctx, `UPDATE air_video_uploads SET state='completed' WHERE organization_id=$1 AND id=$2 AND state='created' AND expires_at>now()`, org, id)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() == 0 {
		u, getErr := s.GetUpload(ctx, org, id)
		if getErr != nil {
			return nil, getErr
		}
		if u.State != "completed" {
			return nil, ErrNotReady
		}
		return u, nil
	}
	return s.GetUpload(ctx, org, id)
}

func (s *PostgresStore) ClaimExpiredUploads(ctx context.Context, now time.Time, limit int) ([]Upload, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.db.Query(ctx, `UPDATE air_video_uploads SET state='expiring' WHERE id IN (
	  SELECT id FROM air_video_uploads WHERE state IN ('created','expiring') AND expires_at<$1 ORDER BY expires_at LIMIT $2 FOR UPDATE SKIP LOCKED)
      RETURNING id,organization_id,purpose,mime,size_bytes,sha256,object_key,state,created_at,expires_at`, now, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Upload
	for rows.Next() {
		u, scanErr := scanUpload(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		out = append(out, *u)
	}
	return out, rows.Err()
}

func (s *PostgresStore) ExpireUpload(ctx context.Context, org, id string) error {
	_, err := s.db.Exec(ctx, `UPDATE air_video_uploads SET state='expired' WHERE organization_id=$1 AND id=$2 AND state IN ('created','expiring')`, org, id)
	return err
}

const jobColumns = `j.id,j.organization_id,j.idempotency_key,j.request_hash,j.request,j.principal,j.public_status,j.state,j.version,
 j.provider_job_id,j.cancel_requested,j.result_url,j.result_content_type,j.result_size,j.result_etag,j.reservation_id,j.reservation_handle,j.quoted_amount,j.currency,
 j.error_code,j.error_message,j.lease_owner,j.lease_expires_at,j.next_run_at,j.attempts,j.created_at,j.updated_at`
const jobSelect = `SELECT ` + jobColumns + ` FROM air_video_jobs j`
const uploadSelect = `SELECT id,organization_id,purpose,mime,size_bytes,sha256,object_key,state,created_at,expires_at FROM air_video_uploads`

type rowScanner interface{ Scan(...any) error }

func scanJob(row rowScanner) (*Job, error) {
	j := new(Job)
	var raw, principalRaw []byte
	err := row.Scan(&j.ID, &j.OrganizationID, &j.IdempotencyKey, &j.RequestHash, &raw, &principalRaw, &j.Status, &j.State, &j.Version,
		&j.ProviderJobID, &j.CancelRequested, &j.ResultURL, &j.ResultContentType, &j.ResultSize, &j.ResultETag, &j.ReservationID, &j.ReservationHandle, &j.QuotedAmount, &j.Currency,
		&j.ErrorCode, &j.ErrorMessage, &j.LeaseOwner, &j.LeaseExpiresAt, &j.NextRunAt, &j.Attempts, &j.CreatedAt, &j.UpdatedAt)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &j.Request); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(principalRaw, &j.Principal); err != nil {
		return nil, err
	}
	return j, nil
}

func scanUpload(row rowScanner) (*Upload, error) {
	u := new(Upload)
	err := row.Scan(&u.ID, &u.OrganizationID, &u.Purpose, &u.MIME, &u.SizeBytes, &u.SHA256, &u.ObjectKey, &u.State, &u.CreatedAt, &u.ExpiresAt)
	return u, err
}

func safePart(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}
