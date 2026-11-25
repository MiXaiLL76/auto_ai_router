package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_ProviderErrorIn200Body covers a provider that answers
// with an HTTP 200 and puts the failure in the body -- some do this for a
// model that cannot satisfy the requested response_format.
//
// The failure used to reach the client as a generic 502 "Request failed":
// indistinguishable from the provider being down, and the credential was
// blamed for a fault every credential would reproduce identically.
func TestProxyRequest_ProviderErrorIn200Body(t *testing.T) {
	providerBody := `{"error":{"code":"invalid_parameter_error","param":null,"message":"Model output became abnormal while generating a JSON response for response_format. The generation was aborted because the partial output may be incomplete or invalid JSON."}}`

	var calls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(providerBody))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", "openai", upstream.URL, "upstream-key").
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"test-model",
		"max_tokens":64,
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// The client's request is at fault, so the status says so.
	require.Equal(t, http.StatusBadRequest, w.Code, "body: %s", w.Body.String())

	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.NotEqual(t, "Request failed", resp.Error.Message,
		"the client must not get the generic message")
	assert.NotEmpty(t, resp.Error.Message)
	assert.Equal(t, "invalid_request_error", resp.Error.Type)
	require.NotNil(t, resp.Error.Code)
	assert.NotEqual(t, "api_error", *resp.Error.Code,
		"the generic api_error code is what a 502 carried")
}

// TestProxyRequest_ProviderErrorIn200BodyIsNotRetried checks that the
// request-side failure is not replayed on another credential: every
// credential would refuse it identically.
func TestProxyRequest_ProviderErrorIn200BodyIsNotRetried(t *testing.T) {
	providerBody := `{"error":{"code":"invalid_parameter_error","param":null,"message":"Model output became abnormal while generating a JSON response for response_format."}}`

	var calls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(providerBody))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", "openai", upstream.URL, "upstream-key").
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"test-model","max_tokens":64,
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code, "body: %s", w.Body.String())
	assert.LessOrEqual(t, atomic.LoadInt32(&calls), int32(2),
		"a request-side failure must not be replayed across the credential pool")
}
