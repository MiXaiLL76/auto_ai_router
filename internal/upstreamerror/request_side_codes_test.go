package upstreamerror

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClassifyProviderErrorSignals_RequestSideCodes covers the provider
// codes that mean the REQUEST is at fault.
//
// Some providers answer with an HTTP 200 and put the failure in the
// body, so the router has to read the code out of the body to classify
// the failure. Without that, the failure came out
// as a generic 502 "Request failed" -- indistinguishable from the provider
// being down, and the credential was blamed for a fault every credential
// would reproduce identically.
func TestClassifyProviderErrorSignals_RequestSideCodes(t *testing.T) {
	for _, tt := range []struct {
		name string
		body string
	}{
		{"provider invalid_parameter_error", `{"error":{"code":"invalid_parameter_error","param":null,"message":"Model output became abnormal while generating a JSON response for response_format."}}`},
		{"provider invalid_parameter_error with type", `{"error":{"code":"invalid_parameter_error","param":null,"message":"Range of input length should be [1, 262144]","type":"invalid_request_error"}}`},
		{"openai invalid_request_error type", `{"error":{"type":"invalid_request_error","message":"bad request"}}`},
		{"invalid_parameter code", `{"error":{"code":"invalid_parameter","message":"bad parameter"}}`},
		{"invalid_request code", `{"error":{"code":"invalid_request","message":"bad request"}}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var doc map[string]json.RawMessage
			require.NoError(t, json.Unmarshal([]byte(tt.body), &doc))

			signals, ok := ExtractErrorSignals(doc["error"])
			require.True(t, ok, "the error object must yield signals")

			assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorSignals(signals),
				"a request-side provider code must classify as 400")
		})
	}
}

// TestClassifyProviderErrorSignals_ProviderSideUnchanged makes sure the
// request-side codes do not swallow the statuses that describe the
// provider's or the account's state: those keep their own status, and a
// client must be told to slow down rather than that its request was
// refused.
func TestClassifyProviderErrorSignals_ProviderSideUnchanged(t *testing.T) {
	for _, tt := range []struct {
		name   string
		signal string
		want   int
	}{
		{"rate limit", "rate_limit_exceeded", http.StatusTooManyRequests},
		{"too many requests", "Too many requests", http.StatusTooManyRequests},
		{"timeout", "Request timed out", http.StatusRequestTimeout},
		{"overloaded", "Service overloaded", http.StatusServiceUnavailable},
		{"server error", "server error", http.StatusInternalServerError},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ClassifyProviderErrorSignals([]string{tt.signal}))
		})
	}
}

// TestClassifyProviderErrorSignals_RequestSideCodeDoesNotOverrideRateLimit
// covers the ordering: a body that carries both a rate-limit code and a
// request-side code keeps the rate limit, because a rate limit describes the
// account's state and a retry after it may well succeed.
func TestClassifyProviderErrorSignals_RequestSideCodeDoesNotOverrideRateLimit(t *testing.T) {
	signals := []string{"invalid_parameter_error", "rate_limit_exceeded"}
	assert.Equal(t, http.StatusTooManyRequests, ClassifyProviderErrorSignals(signals))
}

// TestClassifyProviderErrorSignals_UnknownCodeStays502 keeps the default: a
// code the router does not know says nothing about the failure, so it stays
// a 502 rather than being guessed into a 400.
func TestClassifyProviderErrorSignals_UnknownCodeStays502(t *testing.T) {
	assert.Equal(t, http.StatusBadGateway,
		ClassifyProviderErrorSignals([]string{"some_unknown_code", "something went wrong"}))
}
