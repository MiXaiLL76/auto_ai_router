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
