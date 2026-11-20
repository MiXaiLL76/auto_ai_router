package upstreamerror

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExtractErrorSignalsSurvivesNumericCode pins that a numeric "code" in a
// provider's error object does not discard the rest of the object. The
// message is the only place the real cause appears, so losing it loses the
// diagnosis. (Reproduces production request b34d69ffef0fc98e5c970be5eb657295,
// 2026-11-09, model qwen/qwen3.7-plus, provider "Alibaba".)
func TestExtractErrorSignalsSurvivesNumericCode(t *testing.T) {
	signals, ok := ExtractErrorSignals([]byte(
		`{"code":502,"message":"Output data may contain inappropriate content.","metadata":{"error_type":"provider_unavailable"}}`))
	require.True(t, ok, "the error object must yield signals")
	assert.Contains(t, signals, "Output data may contain inappropriate content.",
		"the message must survive a numeric code")
	assert.Contains(t, signals, "provider_unavailable",
		"the metadata error_type must be read too")
}

// TestClassifyProviderErrorSignalsContentFilter pins that a content-policy
// refusal is a 400, and that it wins over the provider-side signals that
// accompany it.
func TestClassifyProviderErrorSignalsContentFilter(t *testing.T) {
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
			assert.Equal(t, tt.want, ClassifyProviderErrorSignals(tt.signals))
		})
	}
}

// TestClassifyProviderErrorSignalsPriority pins the priority order between a
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
func TestClassifyProviderErrorSignalsPriority(t *testing.T) {
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
			assert.Equal(t, tt.want, ClassifyProviderErrorSignals(tt.signals))
		})
	}
}
