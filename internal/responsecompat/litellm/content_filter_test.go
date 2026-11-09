package litellm

import (
	"encoding/json"
	"net/http"
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
			name: "no status info at all",
			body: map[string]any{"message": "something"},
			want: http.StatusUnprocessableEntity,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, errorStatus(tt.body))
		})
	}
}
