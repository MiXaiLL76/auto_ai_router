package proxy

import (
	"net/http"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// contentFilterProviderBody is the body an OpenRouter-style aggregator
// returned for a content-policy refusal, reproduced verbatim from production
// (request b34d69ffef0fc98e5c970be5eb657295, 2026-11-09, model
// qwen/qwen3.7-plus, provider "Alibaba"):
//
//	HTTP 200, choices: [], and the failure inside "error" -- with the upstream
//	provider's own NUMERIC code (502) and its own error_type
//	("provider_unavailable"), while the message names the real cause.
//
// Before the fix this produced a 502 APIConnectionError: the numeric "code"
// made the strict-string decode of the error object fail as a whole, so the
// message never reached the classifier, and the refusal was then reported as
// a generic bad gateway -- retried across every credential (4 attempts, 55s
// in production) for a request every credential refuses identically.
const contentFilterProviderBody = `{"id":"gen-1794222366-WuU4rEXib8vOi4gfIHLy","object":"chat.completion.chunk",` +
	`"created":1794222366,"model":"qwen/qwen3.7-plus","provider":"Alibaba","choices":[],` +
	`"error":{"code":502,"message":"Upstream error from Alibaba: Output data may contain inappropriate content.",` +
	`"metadata":{"error_type":"provider_unavailable"}}}`

// TestProviderErrorSignalsSurvivesNumericCode pins that a numeric "code" in
// a provider's error object does not discard the rest of the object. The
// message is the only place the real cause appears, so losing it loses the
// diagnosis.
func TestProviderErrorSignalsSurvivesNumericCode(t *testing.T) {
	signals, ok := providerErrorSignals([]byte(
		`{"code":502,"message":"Output data may contain inappropriate content.","metadata":{"error_type":"provider_unavailable"}}`))
	require.True(t, ok, "the error object must yield signals")
	assert.Contains(t, signals, "Output data may contain inappropriate content.",
		"the message must survive a numeric code")
	assert.Contains(t, signals, "provider_unavailable",
		"the metadata error_type must be read too")
}

// TestStatusCodeFromErrorSignalsContentFilter pins that a content-policy
// refusal is a 400, and that it wins over the provider-side signals that
// accompany it.
func TestStatusCodeFromErrorSignalsContentFilter(t *testing.T) {
	tests := []struct {
		name    string
		signals []string
		want    int
	}{
		{
			name:    "content filter message with a provider-side error type",
			signals: []string{"Output data may contain inappropriate content.", "provider_unavailable"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "content policy wording",
			signals: []string{"content_policy_violation"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "content policy in prose",
			signals: []string{"The prompt was blocked by the provider's content policy"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "content filter error type",
			signals: []string{"content_filter"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "safety violation",
			signals: []string{"safety_violation"},
			want:    http.StatusBadRequest,
		},
		// A genuine provider failure keeps its 5xx: the content-filter
		// branch must not swallow every unclassified error.
		{
			name:    "provider failure without content-filter wording",
			signals: []string{"Upstream error from Darkbloom: inference generation failed", "provider_unavailable"},
			want:    http.StatusBadGateway,
		},
		{
			name:    "explicit overload still maps to 503",
			signals: []string{"model_overloaded"},
			want:    http.StatusServiceUnavailable,
		},
		{
			name:    "rate limit still maps to 429",
			signals: []string{"rate_limit_exceeded"},
			want:    http.StatusTooManyRequests,
		},
		{
			name:    "unclassified still maps to 502",
			signals: []string{"something else entirely"},
			want:    http.StatusBadGateway,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, statusCodeFromErrorSignals(tt.signals))
		})
	}
}

// TestStatusCodeFromProviderBodyErrorContentFilter pins the end-to-end
// classification of the production body: a 200 response whose body carries an
// error must be reported as a 400, not as a 200 (error ignored) or a 502.
func TestStatusCodeFromProviderBodyErrorContentFilter(t *testing.T) {
	status, ok := statusCodeFromProviderBodyError(http.StatusOK, []byte(contentFilterProviderBody))
	require.True(t, ok, "the error inside a 200 body must be recognized")
	assert.Equal(t, http.StatusBadRequest, status)
}

// TestStatusCodeFromProviderStreamErrorContentFilter is the streaming
// counterpart: the same body arriving as a terminal stream error event.
func TestStatusCodeFromProviderStreamErrorContentFilter(t *testing.T) {
	assert.Equal(t, http.StatusBadRequest, statusCodeFromProviderStreamError(contentFilterProviderBody))
}

// TestProviderErrorSignalsKeepsNonContentProviderFailure guards the other
// direction: a genuine upstream failure with the same numeric code and the
// same "provider_unavailable" error type must NOT be classified as a content
// refusal.
func TestProviderErrorSignalsKeepsNonContentProviderFailure(t *testing.T) {
	body := []byte(`{"error":{"code":502,"message":"Upstream error from Darkbloom: inference generation failed",` +
		`"metadata":{"error_type":"provider_unavailable"}}}`)
	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadGateway, status,
		"a provider failure must stay a 5xx, not become a 400")
}

// TestContentFilterIsNotRetried pins that a content-policy refusal stops the
// retry cascade: every credential refuses the same content, so replaying it
// only multiplies the delay (measured in production: 4 attempts, 55s).
func TestContentFilterIsNotRetried(t *testing.T) {
	policy := config.DefaultRetryConfig()
	require.Contains(t, policy.NonRetryableMarkers, "inappropriate content",
		"the refusal wording the aggregator forwarded must be a non-retryable marker")

	retry, _ := ShouldRetryWithFallback(http.StatusBadRequest, []byte(contentFilterProviderBody))
	assert.False(t, retry, "a content-policy refusal must not be retried")
}

// TestProviderUnavailableAloneStaysRetryable is the counterpart: a provider
// that is merely unavailable must remain retryable -- that is exactly what
// moving to another credential fixes.
func TestProviderUnavailableAloneStaysRetryable(t *testing.T) {
	body := []byte(`{"error":{"code":502,"message":"Upstream error from Darkbloom: inference generation failed",` +
		`"metadata":{"error_type":"provider_unavailable"}}}`)
	retry, _ := ShouldRetryWithFallback(http.StatusBadGateway, body)
	assert.True(t, retry, "a provider outage must stay retryable")
}

// TestStatusCodeFromErrorSignalsPriority pins the priority order between a
// content-policy refusal and the other signals it can arrive alongside.
//
// The refusal is checked in its own pass, so it outranks the PROVIDER-side
// signals regardless of the order they appear in the object: an aggregator
// reports the refusal with the upstream's own code and error type, and that
// must not turn a request fault into a 5xx.
//
// It does NOT outrank the request-side ones. A rate limit or a timeout is a
// property of the account's state at that moment, not of the content:
// reporting those as 400 would tell the client "your prompt was refused" when
// the truth is "slow down", and would stop a retry that would have
// succeeded.
func TestStatusCodeFromErrorSignalsPriority(t *testing.T) {
	tests := []struct {
		name    string
		signals []string
		want    int
	}{
		{
			name:    "refusal outranks the provider's own error type",
			signals: []string{"Output data may contain inappropriate content.", "provider_unavailable"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "refusal outranks a provider error even when it comes second",
			signals: []string{"server_error", "Output data may contain inappropriate content."},
			want:    http.StatusBadRequest,
		},
		{
			name:    "refusal outranks an internal error",
			signals: []string{"internal_error", "content_filter"},
			want:    http.StatusBadRequest,
		},
		{
			name:    "a rate limit outranks the refusal",
			signals: []string{"rate_limit_exceeded", "content_filter"},
			want:    http.StatusTooManyRequests,
		},
		{
			name:    "a timeout outranks the refusal",
			signals: []string{"timeout", "content_filter"},
			want:    http.StatusRequestTimeout,
		},
		{
			name:    "a rate limit alone",
			signals: []string{"rate_limit_exceeded"},
			want:    http.StatusTooManyRequests,
		},
		{
			name:    "a provider failure alone",
			signals: []string{"provider_unavailable", "inference generation failed"},
			want:    http.StatusBadGateway,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, statusCodeFromErrorSignals(tt.signals))
		})
	}
}

// TestDataInspectionFailedIsContentPolicy pins Alibaba's own code for a
// content refusal.
//
// It is the most frequent content-refusal marker in production by a wide
// margin (~164k of the refusals logged in a month), and it arrives in the
// "code"/"type" field rather than the message -- with a message that may
// name no content at all ("Provider returned error"). Without the marker a
// refusal like that is classified as a generic error, and the request stays
// retryable across the pool.
func TestDataInspectionFailedIsContentPolicy(t *testing.T) {
	body := []byte(`{"error":{"code":"data_inspection_failed","param":null,` +
		`"message":"Output data may contain inappropriate content.","type":"data_inspection_failed"},` +
		`"id":"chatcmpl-1"}`)

	// A provider that reports the refusal inside a 200 body: the status has
	// to be derived from the body.
	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, status)

	// The same code with a message that names no content at all.
	body2 := []byte(`{"error":{"code":"data_inspection_failed","param":null,` +
		`"message":"Provider returned error","type":"data_inspection_failed"},"id":"chatcmpl-2"}`)
	status2, ok2 := statusCodeFromProviderBodyError(http.StatusOK, body2)
	require.True(t, ok2)
	assert.Equal(t, http.StatusBadRequest, status2,
		"the code alone identifies the refusal, even when the message does not")

	// And it must not be retried: every credential refuses the same content.
	retry, _ := ShouldRetryWithFallback(http.StatusBadRequest, body2)
	assert.False(t, retry, "a data_inspection_failed refusal must not be retried")
}

// TestStatusCodeFromProviderBodyErrorSurvivesRootNumericCode pins that a
// provider's NUMERIC top-level "code" does not take the whole document with
// it.
//
// The top-level fields used to be decoded through a struct with string
// fields, so a numeric "code" failed the decode outright ("cannot unmarshal
// number into Go struct field") and the function returned "no error here" --
// losing the "error" object nested BESIDE it. A provider that reports a
// content refusal that way would have been served as a success, with the
// refusal body passed through to the client.
//
// The decode is a map now: a non-string value is simply not a string, and
// every other key is read normally.
func TestStatusCodeFromProviderBodyErrorSurvivesRootNumericCode(t *testing.T) {
	body := []byte(`{"code":502,"error":{"message":"Output data may contain inappropriate content."},"id":"x"}`)

	status, ok := statusCodeFromProviderBodyError(http.StatusOK, body)
	require.True(t, ok, "the nested error must still be recognized")
	assert.Equal(t, http.StatusBadRequest, status,
		"the refusal nested beside a numeric code must still be classified as a 400")
}

// TestStatusCodeFromProviderBodyErrorReadsTopLevelFields pins that the
// top-level fields are still read as before -- the map decode must not have
// changed what a normal body yields.
func TestStatusCodeFromProviderBodyErrorReadsTopLevelFields(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   int
		wantOK bool
	}{
		{
			name:   "top-level rate limit error",
			status: http.StatusBadRequest,
			body:   `{"type":"error","code":"rate_limit_exceeded","message":"slow down"}`,
			want:   http.StatusTooManyRequests,
			wantOK: true,
		},
		{
			name:   "failed responses object",
			status: http.StatusOK,
			body:   `{"response":{"status":"failed","error":{"message":"boom"}}}`,
			want:   http.StatusBadGateway,
			wantOK: true,
		},
		{
			name:   "nested failed response event body",
			status: http.StatusOK,
			body:   `{"type":"response.failed","response":{"status":"failed","error":{"code":"rate_limit_exceeded"}}}`,
			want:   http.StatusTooManyRequests,
			wantOK: true,
		},
		{
			name:   "a plain success body",
			status: http.StatusOK,
			body:   `{"id":"x","choices":[]}`,
			want:   0,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, ok := statusCodeFromProviderBodyError(tt.status, []byte(tt.body))
			assert.Equal(t, tt.wantOK, ok)
			if tt.wantOK {
				assert.Equal(t, tt.want, status)
			}
		})
	}
}

// TestStatusCodeFromProviderBodyErrorNumericCode pins that a provider which
// states its status as a JSON NUMBER is classified by that status.
//
// OpenRouter relays Alibaba's error that way: an HTTP 200 whose body carries
// the status as a number. The number used to fail the unmarshal of the whole
// error object, so the message and the type were lost with it and the client
// got a blanket 502 instead of the 400 the provider reported.
//
// The number is a FALLBACK, not a first-class signal: it is used only when
// nothing else says what went wrong (see statusCodeFromErrorSignals). An
// aggregator's forwarding of the UPSTREAM's code ({"error":{"code":502,
// "message":"Output data may contain inappropriate content."}}) must still be
// classified by its message -- as the 400 the refusal is -- rather than
// inherited as a 502.
func TestStatusCodeFromProviderBodyErrorNumericCode(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantOK     bool
	}{
		{
			name:       "numeric code in error object",
			body:       `{"id":"gen-1","object":"chat.completion.chunk","provider":"Alibaba","choices":[],"error":{"code":400,"message":"Video inputs are not supported by this model.","metadata":{"error_type":"invalid_request"}}}`,
			wantStatus: http.StatusBadRequest,
			wantOK:     true,
		},
		{
			name:       "numeric code 5xx in error object",
			body:       `{"id":"gen-2","object":"chat.completion.chunk","provider":"Alibaba","choices":[],"error":{"code":502,"message":"Upstream error from Alibaba: inference generation failed","metadata":{"error_type":"provider_unavailable"}}}`,
			wantStatus: http.StatusBadGateway,
			wantOK:     true,
		},
		{
			name:       "error object is a bare number",
			body:       `{"error":400,"message":"bad request"}`,
			wantStatus: http.StatusBadRequest,
			wantOK:     true,
		},
		{
			// A number that is not an HTTP status says nothing about the
			// failure and must not become a status of its own: the generic
			// 502 stays.
			name:       "numeric code that is not an http status",
			body:       `{"error":{"code":42,"message":"unclear"}}`,
			wantStatus: http.StatusBadGateway,
			wantOK:     true,
		},
		{
			// The refusal wins over the number beside it: the code is the
			// upstream's, the message is the reason.
			name:       "a refusal outranks the numeric code beside it",
			body:       `{"error":{"code":502,"message":"Output data may contain inappropriate content.","metadata":{"error_type":"provider_unavailable"}}}`,
			wantStatus: http.StatusBadRequest,
			wantOK:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, ok := statusCodeFromProviderBodyError(http.StatusOK, []byte(tt.body))
			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.wantStatus, status)
		})
	}
}

// TestStatusCodeFromProviderStreamErrorNumericCode is the streaming
// counterpart: the same numbers, in a terminal stream error event.
func TestStatusCodeFromProviderStreamErrorNumericCode(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    int
	}{
		{
			name:    "numeric code in error object",
			payload: `{"type":"error","error":{"code":400,"message":"Video inputs are not supported by this model."}}`,
			want:    http.StatusBadRequest,
		},
		{
			name:    "numeric code in response error object",
			payload: `{"type":"response.failed","response":{"status":"failed","error":{"code":400,"message":"bad request"}}}`,
			want:    http.StatusBadRequest,
		},
		{
			name:    "numeric code 429",
			payload: `{"type":"error","error":{"code":429,"message":"too many requests"}}`,
			want:    http.StatusTooManyRequests,
		},
		{
			name:    "string code still classifies by keyword",
			payload: `{"type":"error","error":{"code":"rate_limit_exceeded","message":"slow down"}}`,
			want:    http.StatusTooManyRequests,
		},
		{
			name:    "a refusal outranks the numeric code beside it",
			payload: `{"type":"error","error":{"code":502,"message":"Output data may contain inappropriate content."}}`,
			want:    http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, statusCodeFromProviderStreamError(tt.payload))
		})
	}
}
