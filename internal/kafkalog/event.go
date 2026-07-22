package kafkalog

import "time"

// SpendEvent is the flat JSON event published to the "air.spend_logs" Kafka
// topic for every logged request. The struct is a superset of
// litellmdb/models.SpendLogEntry: credential/server metadata is broken out
// into typed fields instead of one JSON blob, so ClickHouse can ingest it
// with JSONEachRow and query it with plain GROUP BY (no Nested/Tuple types).
//
// Request/response body capture is intentionally out of scope here (separate
// ТЗ/PR) — the Body* fields are always zero-value placeholders.
type SpendEvent struct {
	RequestID string    `json:"request_id"`
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	// CompletionStartTime is the time-to-first-token (TTFT) timestamp, nil
	// when the request wasn't streamed or no chunk was ever written.
	CompletionStartTime *time.Time `json:"completion_start_time,omitempty"`
	DurationMs          int64      `json:"duration_ms"`
	TTFTMs              *int64     `json:"ttft_ms,omitempty"`

	CallType     string `json:"call_type"`
	APIBase      string `json:"api_base"`
	Status       string `json:"status"` // "success" | "failure"
	HTTPStatus   int    `json:"http_status"`
	ErrorMessage string `json:"error_message,omitempty"`
	ErrorClass   string `json:"error_class,omitempty"`

	Model      string `json:"model"`      // Model alias, as requested by the client
	RealModel  string `json:"real_model"` // Real upstream model name (price lookup key)
	ModelID    string `json:"model_id"`   // "credential_name:model_name"
	ModelGroup string `json:"model_group"`

	CredentialName                 string `json:"credential_name"`
	CredentialType                 string `json:"credential_type"`
	CredentialBaseURL              string `json:"credential_base_url"`
	CredentialIsProxyRequest       bool   `json:"credential_is_proxy_request"`
	CredentialActualCredentialName string `json:"credential_actual_credential_name,omitempty"`

	ServerRouterID string `json:"server_router_id"`
	ServerVersion  string `json:"server_version"`
	ServerCommit   string `json:"server_commit"`

	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	AudioInputTokens         int `json:"audio_input_tokens"`
	AudioOutputTokens        int `json:"audio_output_tokens"`
	CachedInputTokens        int `json:"cached_input_tokens"`
	CacheCreationTokens      int `json:"cache_creation_tokens"`
	CachedOutputTokens       int `json:"cached_output_tokens"`
	ReasoningTokens          int `json:"reasoning_tokens"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	ImageCount               int `json:"image_count"`
	ImageTokens              int `json:"image_tokens"`
	OutputImageTokens        int `json:"output_image_tokens"`

	InputCost         float64 `json:"input_cost"`
	OutputCost        float64 `json:"output_cost"`
	AudioInputCost    float64 `json:"audio_input_cost"`
	AudioOutputCost   float64 `json:"audio_output_cost"`
	ReasoningCost     float64 `json:"reasoning_cost"`
	CachedInputCost   float64 `json:"cached_input_cost"`
	CacheCreationCost float64 `json:"cache_creation_cost"`
	CachedOutputCost  float64 `json:"cached_output_cost"`
	PredictionCost    float64 `json:"prediction_cost"`
	ImageCost         float64 `json:"image_cost"`
	TotalCost         float64 `json:"total_cost"`

	APIKeyHash     string `json:"api_key_hash"`
	UserID         string `json:"user_id"`
	TeamID         string `json:"team_id"`
	OrganizationID string `json:"organization_id"`
	EndUser        string `json:"end_user"`
	KeyAlias       string `json:"key_alias,omitempty"`
	UserAlias      string `json:"user_alias,omitempty"`
	TeamAlias      string `json:"team_alias,omitempty"`

	RequesterIP string  `json:"requester_ip"`
	SessionID   string  `json:"session_id"`
	OverheadMs  float64 `json:"overhead_ms"`

	// Placeholder fields for the deferred request/response body capture PR.
	// Always false/0 — no capture logic exists yet.
	BodyCaptured      bool `json:"body_captured"`
	BodyRequestBytes  int  `json:"body_request_bytes"`
	BodyResponseBytes int  `json:"body_response_bytes"`
}

// Key returns the Kafka record key (request_id), guaranteeing all events for
// the same request land on the same partition.
func (e *SpendEvent) Key() []byte {
	if e == nil {
		return nil
	}
	return []byte(e.RequestID)
}
