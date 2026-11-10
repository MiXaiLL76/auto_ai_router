package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTransformContentFilterIsBadRequest pins that a content-policy refusal
// carried inside a 200 response's body is reported as a 400, not as the
// upstream code it happens to carry.
//
// The production case: an aggregator forwarded the upstream provider's own
// numeric code verbatim (502) alongside the refusal message, so the status
// was taken from that number and the client saw a bad gateway for a request
// that was simply refused. A content refusal is a fault of the request.
func TestTransformContentFilterIsBadRequest(t *testing.T) {
	body := `{"id":"gen-1","object":"chat.completion.chunk","model":"qwen/qwen3.7-plus","provider":"Alibaba","choices":[],` +
		`"error":{"code":502,"message":"Upstream error from Alibaba: Output data may contain inappropriate content.",` +
		`"metadata":{"error_type":"provider_unavailable"}}}`

	result := New().Transform(Context{Endpoint: "/v1/chat/completions"}, Response{
		StatusCode: http.StatusOK,
		Headers:    http.Header{"Content-Type": {"application/json"}},
		Body:       []byte(body),
	})

	assert.Equal(t, http.StatusBadRequest, result.StatusCode,
		"a content-policy refusal must not inherit the upstream's numeric code")
	var decoded map[string]any
	require.NoError(t, json.Unmarshal(result.Body, &decoded))
	errorBody, ok := decoded["error"].(map[string]any)
	require.True(t, ok, "the response must carry an error object")
	assert.Equal(t, "The prompt was blocked by the provider's content policy", errorBody["message"])
	assert.Equal(t, "content_policy_violation", errorBody["code"])
	assert.NotContains(t, string(result.Body), "Alibaba",
		"the provider's own text must not reach the client")
}

// TestTransformProviderFailureKeepsUpstreamStatus is the counterpart: a
// genuine provider failure (no content-refusal wording) must keep the status
// the provider reported, not be turned into a 400.
func TestTransformProviderFailureKeepsUpstreamStatus(t *testing.T) {
	body := `{"id":"gen-2","object":"chat.completion.chunk","model":"openai/gpt-oss-20b","provider":"Darkbloom","choices":[],` +
		`"error":{"code":502,"message":"Upstream error from Darkbloom: inference generation failed",` +
		`"metadata":{"error_type":"provider_unavailable"}}}`

	result := New().Transform(Context{Endpoint: "/v1/chat/completions"}, Response{
		StatusCode: http.StatusOK,
		Headers:    http.Header{"Content-Type": {"application/json"}},
		Body:       []byte(body),
	})

	assert.Equal(t, http.StatusBadGateway, result.StatusCode,
		"a provider failure must stay a 5xx")
}

// TestErrorStatusContentFilter covers the status mapping directly, including
// the wording variants an aggregator forwards verbatim.
func TestErrorStatusContentFilter(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
		want int
	}{
		{
			name: "inappropriate content message",
			body: map[string]any{"code": float64(502), "message": "Output data may contain inappropriate content."},
			want: http.StatusBadRequest,
		},
		{
			name: "content filter type",
			body: map[string]any{"type": "content_filter", "message": "refused"},
			want: http.StatusBadRequest,
		},
		{
			name: "metadata error type",
			body: map[string]any{"message": "refused", "metadata": map[string]any{"error_type": "content_policy_violation"}},
			want: http.StatusBadRequest,
		},
		{
			name: "provider failure keeps its code",
			body: map[string]any{"code": float64(502), "message": "inference generation failed"},
			want: http.StatusBadGateway,
		},
		{
			// Before errorStatus delegated to the shared classifier, a
			// message that matched no keyword fell to this function's own
			// 422 default. Now ClassifyProviderErrorSignals runs on it like
			// any other unclassified provider-side message, and its default
			// is 502 -- matching the native path exactly, which is the point
			// of sharing one implementation. A body with no recognized field
			// at all still gets 422 (see the "truly no signal at all" case
			// below).
			name: "unrecognized message now maps to 502, like the native path",
			body: map[string]any{"message": "something"},
			want: http.StatusBadGateway,
		},
		{
			// No recognized field at all (not even "message"): ExtractErrorSignals
			// reports no signal, and errorStatus keeps its own 422 default --
			// unchanged from before the delegation.
			name: "truly no signal at all stays 422",
			body: map[string]any{"unrelated_field": "something"},
			want: http.StatusUnprocessableEntity,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, errorStatus(tt.body))
		})
	}
}

// TestErrorStatusReadsTheSameFieldsAsProxy pins that the compatibility
// layer's status mapping reads the same fields the proxy's own
// classification reads -- both now go through the same
// upstreamerror.ExtractErrorSignals/ClassifyProviderErrorSignals, so they
// cannot read a different field set again the way they once did (the proxy
// read "code" and "status" as well, while this layer read only the
// message-bearing fields; a refusal phrased in "status": "content_filter"
// was recognized by the proxy and passed through as a 5xx here).
func TestErrorStatusReadsTheSameFieldsAsProxy(t *testing.T) {
	objects := []map[string]any{
		{"message": "Output data may contain inappropriate content."},
		{"type": "content_filter", "message": "refused"},
		{"status": "content_filter", "message": "refused"},
		{"code": "content_filter", "message": "refused"},
		{"reason": "content_filter", "message": "refused"},
		{"message": "refused", "metadata": map[string]any{"error_type": "content_filter"}},
		{"message": "refused", "metadata": map[string]any{"type": "content_filter"}},
		{"message": "refused", "metadata": map[string]any{"reason": "content_filter"}},
		{"code": "data_inspection_failed", "type": "data_inspection_failed", "message": "Provider returned error"},
	}

	for _, obj := range objects {
		t.Run(strings.Join([]string{
			fmt.Sprint(obj["message"]), fmt.Sprint(obj["type"]),
			fmt.Sprint(obj["status"]), fmt.Sprint(obj["code"]),
		}, "|"), func(t *testing.T) {
			assert.Equal(t, http.StatusBadRequest, errorStatus(obj),
				"a refusal must be recognized wherever the marker sits: %v", obj)
		})
	}
}

// TestErrorStatusIgnoresNonRefusals is the counterpart: a provider failure
// must not be mistaken for a refusal, whatever it carries.
func TestErrorStatusIgnoresNonRefusals(t *testing.T) {
	for _, obj := range []map[string]any{
		{"code": float64(502), "message": "inference generation failed"},
		{"message": "Rate limit exceeded"},
		{"message": "Your account has been blocked by the provider for suspected billing fraud."},
		{"message": "This request was blocked by the provider due to an IP/region restriction."},
		{"message": "safety_identifier must be a string"},
		{"code": float64(500), "message": "internal server error"},
	} {
		assert.NotEqual(t, http.StatusBadRequest, errorStatus(obj),
			"a provider failure must not be reported as a content refusal: %v", obj)
	}
}
