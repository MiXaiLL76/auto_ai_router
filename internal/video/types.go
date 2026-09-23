package video

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var (
	ErrInvalid             = errors.New("invalid_request")
	ErrNotFound            = errors.New("not_found")
	ErrConflict            = errors.New("idempotency_conflict")
	ErrNotReady            = errors.New("not_ready")
	ErrStateChanged        = errors.New("state_changed")
	ErrUnsupportedModel    = errors.New("unsupported_model")
	ErrQuota               = errors.New("insufficient_quota")
	ErrAuthentication      = errors.New("authentication_failed")
	ErrProviderUnavailable = errors.New("provider_unavailable")
)

type Status string

const (
	StatusQueued     Status = "queued"
	StatusInProgress Status = "in_progress"
	StatusCompleted  Status = "completed"
	StatusFailed     Status = "failed"
	StatusCancelled  Status = "cancelled"
)

type State string

const (
	StateQueued            State = "queued"
	StateReserving         State = "reserving"
	StateReadyToSubmit     State = "ready_to_submit"
	StateSubmitting        State = "submitting"
	StateSubmissionUnknown State = "submission_unknown"
	StateSubmitted         State = "submitted"
	StateProcessing        State = "processing"
	StateStoring           State = "storing"
	StateSettling          State = "settling"
	StateCancelRequested   State = "cancel_requested"
	StateCancelling        State = "cancelling"
	StateReleasing         State = "releasing"
	StateCompleted         State = "completed"
	StateFailed            State = "failed"
	StateCancelled         State = "cancelled"
)

type CreateRequest struct {
	Model           string `json:"model"`
	Prompt          string `json:"prompt"`
	InputReference  any    `json:"input_reference,omitempty"`
	InputImageID    string `json:"input_image_id,omitempty"`
	InputImageURL   string `json:"input_image_url,omitempty"`
	DurationSeconds int    `json:"duration_seconds,omitempty"`
	Seconds         string `json:"seconds,omitempty"`
	Size            string `json:"size,omitempty"`
	Quality         string `json:"quality,omitempty"`
	AspectRatio     string `json:"aspect_ratio,omitempty"`
	Seed            *int64 `json:"seed,omitempty"`
	NegativePrompt  string `json:"negative_prompt,omitempty"`
}

type Job struct {
	ID                string        `json:"id"`
	OrganizationID    string        `json:"-"`
	IdempotencyKey    string        `json:"-"`
	RequestHash       string        `json:"-"`
	Request           CreateRequest `json:"-"`
	Principal         Principal     `json:"-"`
	Status            Status        `json:"status"`
	State             State         `json:"-"`
	Version           int64         `json:"-"`
	ProviderJobID     string        `json:"-"`
	CancelRequested   bool          `json:"-"`
	ResultURL         string        `json:"-"`
	ResultContentType string        `json:"-"`
	ResultSize        int64         `json:"-"`
	ResultETag        string        `json:"-"`
	ReservationID     string        `json:"-"`
	ReservationHandle string        `json:"-"`
	QuotedAmount      string        `json:"-"`
	Currency          string        `json:"-"`
	ErrorCode         string        `json:"error_code,omitempty"`
	ErrorMessage      string        `json:"error_message,omitempty"`
	LeaseOwner        string        `json:"-"`
	LeaseExpiresAt    *time.Time    `json:"-"`
	NextRunAt         time.Time     `json:"-"`
	Attempts          int           `json:"-"`
	CreatedAt         time.Time     `json:"created_at"`
	UpdatedAt         time.Time     `json:"updated_at"`
}

type UploadRequest struct {
	Purpose   string `json:"purpose"`
	MIME      string `json:"mime"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

type Upload struct {
	ID             string    `json:"id"`
	OrganizationID string    `json:"-"`
	Purpose        string    `json:"purpose"`
	MIME           string    `json:"mime"`
	SizeBytes      int64     `json:"size_bytes"`
	SHA256         string    `json:"sha256"`
	ObjectKey      string    `json:"-"`
	State          string    `json:"status"`
	CreatedAt      time.Time `json:"created_at"`
	ExpiresAt      time.Time `json:"expires_at"`
}

const (
	MaxRequestBytes      = 64 << 10
	MaxImageDataURIBytes = 5_000_000
	MaxImageBytes        = 3_749_982
	MaxVideoBytes        = 512 << 20
	MaxArtifactBytes     = 1 << 30
)

var digestPattern = regexp.MustCompile(`^[a-fA-F0-9]{64}$`)

func (r CreateRequest) normalized() (CreateRequest, error) {
	r.Model = strings.TrimSpace(r.Model)
	r.Prompt = strings.TrimSpace(r.Prompt)
	r.InputImageID = strings.TrimSpace(r.InputImageID)
	r.InputImageURL = strings.TrimSpace(r.InputImageURL)
	r.Seconds = strings.TrimSpace(r.Seconds)
	r.Size = strings.TrimSpace(r.Size)
	r.Quality = strings.TrimSpace(r.Quality)
	r.AspectRatio = strings.TrimSpace(r.AspectRatio)
	r.NegativePrompt = strings.TrimSpace(r.NegativePrompt)
	if r.InputImageID != "" && r.InputImageURL != "" {
		return r, ErrInvalid
	}
	if r.InputImageID != "" && !strings.HasPrefix(r.InputImageID, "upl_") {
		return r, ErrInvalid
	}
	if r.InputImageURL != "" {
		parsed, err := url.Parse(r.InputImageURL)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
			return r, ErrInvalid
		}
	}
	if r.DurationSeconds == 0 && r.Seconds != "" {
		var n int
		for _, c := range r.Seconds {
			if c < '0' || c > '9' {
				return r, ErrInvalid
			}
			n = n*10 + int(c-'0')
		}
		r.DurationSeconds = n
	}
	if r.DurationSeconds == 0 {
		r.DurationSeconds = 5
	}
	r.Seconds = ""
	if r.Model != "runway/gen4.5" && r.Model != "runway/gen4_turbo" {
		return r, ErrUnsupportedModel
	}
	if r.Model == "runway/gen4_turbo" && r.InputImageID == "" && r.InputImageURL == "" {
		return r, ErrInvalid
	}
	if r.Prompt == "" {
		return r, ErrInvalid
	}
	if r.DurationSeconds < 2 || r.DurationSeconds > 10 {
		return r, ErrInvalid
	}
	if r.AspectRatio == "" && r.Size == "" {
		r.AspectRatio = "16:9"
	}
	if r.AspectRatio != "" {
		switch r.AspectRatio {
		case "16:9", "9:16", "1:1", "4:3", "3:4", "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672":
		default:
			return r, ErrInvalid
		}
	}
	if r.Size != "" {
		switch r.Size {
		case "720p", "1080p", "1024x1792", "1792x1024", "1280x720", "720x1280", "1920x1080", "1080x1920":
		default:
			return r, ErrInvalid
		}
	}
	if r.Quality != "" {
		switch r.Quality {
		case "standard", "high", "low", "medium", "auto":
		default:
			return r, ErrInvalid
		}
	}
	ratio := runwayRatio(r.AspectRatio)
	if ratio == "" {
		ratio = runwayRatioFromSize(r.Size)
	}
	hasImage := r.InputImageID != "" || r.InputImageURL != ""
	if !validVideoRatio(hasImage, ratio) {
		return r, ErrInvalid
	}
	return r, nil
}

func validVideoRatio(hasImage bool, ratio string) bool {
	if !hasImage {
		return ratio == "1280:720" || ratio == "720:1280"
	}
	switch ratio {
	case "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672":
		return true
	default:
		return false
	}
}

func (r UploadRequest) validate() error {
	r.Purpose = strings.TrimSpace(r.Purpose)
	r.MIME = strings.ToLower(strings.TrimSpace(r.MIME))
	if !digestPattern.MatchString(strings.TrimSpace(r.SHA256)) {
		return ErrInvalid
	}
	switch r.Purpose {
	case "video_input_image":
		if r.MIME != "image/png" && r.MIME != "image/jpeg" && r.MIME != "image/webp" {
			return ErrInvalid
		}
		if r.SizeBytes <= 0 || r.SizeBytes > MaxImageBytes {
			return ErrInvalid
		}
	case "video_input_video":
		if r.MIME != "video/mp4" && r.MIME != "video/quicktime" && r.MIME != "video/webm" {
			return ErrInvalid
		}
		if r.SizeBytes <= 0 || r.SizeBytes > MaxVideoBytes {
			return ErrInvalid
		}
	default:
		return ErrInvalid
	}
	return nil
}

type Store interface {
	CreateJob(context.Context, Principal, string, CreateRequest, string) (*Job, bool, error)
	ActivateJob(context.Context, *Job, Reservation) (*Job, error)
	RejectJob(context.Context, *Job, string, string) (*Job, error)
	GetJob(context.Context, string, string) (*Job, error)
	RequestCancel(context.Context, string, string) (*Job, error)
	ClaimJob(context.Context, string, time.Time, time.Duration) (*Job, error)
	Transition(context.Context, *Job, State, JobUpdate) (*Job, error)
	CreateUpload(context.Context, string, UploadRequest, time.Time, time.Duration) (*Upload, error)
	GetUpload(context.Context, string, string) (*Upload, error)
	CompleteUpload(context.Context, string, string) (*Upload, error)
	ClaimExpiredUploads(context.Context, time.Time, int) ([]Upload, error)
	ExpireUpload(context.Context, string, string) error
}

type JobUpdate struct {
	ProviderJobID     *string
	ResultURL         *string
	ResultContentType *string
	ResultSize        *int64
	ResultETag        *string
	ReservationID     *string
	ReservationHandle *string
	QuotedAmount      *string
	Currency          *string
	ErrorCode         *string
	ErrorMessage      *string
	NextRunAt         *time.Time
	IncrementAttempts bool
	RetainLease       bool
}

type ProviderResult struct {
	ID           string
	Status       Status
	ResultURL    string
	ErrorCode    string
	ErrorMessage string
}

type Provider interface {
	Submit(context.Context, *Job, string) (string, error)
	Poll(context.Context, string) (ProviderResult, error)
	Cancel(context.Context, string) error
}

type Reservation struct{ ID, Handle, Amount, Currency string }

type Billing interface {
	// Every method must be idempotent for operationKey across process crashes.
	Reserve(context.Context, *Job, string) (Reservation, error)
	Settle(context.Context, *Job, string) error
	Release(context.Context, *Job, string) error
}

// NoopBilling is an explicit choice for deployments that account outside AIR.
type NoopBilling struct{}

func (NoopBilling) Reserve(_ context.Context, j *Job, _ string) (Reservation, error) {
	return Reservation{Amount: j.QuotedAmount, Currency: j.Currency}, nil
}
func (NoopBilling) Settle(context.Context, *Job, string) error  { return nil }
func (NoopBilling) Release(context.Context, *Job, string) error { return nil }

type Object struct {
	Body        io.ReadCloser
	ContentType string
	Size        int64
	ETag        string
}

type ObjectStore interface {
	PutUpload(context.Context, *Upload, string, io.Reader) error
	VerifyUpload(context.Context, *Upload) error
	OpenUpload(context.Context, *Upload) (*Object, error)
	DeleteUpload(context.Context, *Upload) error
	FetchResult(context.Context, *Job, string) (Object, error)
	OpenResult(context.Context, *Job) (*Object, error)
}

type Principal struct {
	OrganizationID     string `json:"organization_id"`
	APIKeyHash         string `json:"api_key_hash,omitempty"`
	UserID             string `json:"user_id,omitempty"`
	TeamID             string `json:"team_id,omitempty"`
	BillingTeamID      string `json:"billing_team_id,omitempty"`
	PriceProfileID     string `json:"price_profile_id"`
	PriceProfileSHA256 string `json:"price_profile_sha256"`
	RatePerSecond      string `json:"rate_per_second"`
	Currency           string `json:"currency"`
}

type PrincipalResolver interface {
	ResolvePrincipal(http.ResponseWriter, *http.Request, string) (Principal, error)
}

func IsTerminal(s State) bool {
	return s == StateCompleted || s == StateFailed || s == StateCancelled || s == StateSubmissionUnknown
}

func publicStatus(s State) Status {
	switch s {
	case StateCompleted:
		return StatusCompleted
	case StateFailed, StateSubmissionUnknown:
		return StatusFailed
	case StateCancelled:
		return StatusCancelled
	case StateQueued, StateReserving, StateReadyToSubmit:
		return StatusQueued
	default:
		return StatusInProgress
	}
}
