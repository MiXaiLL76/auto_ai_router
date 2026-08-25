package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
)

// TestProxyRequest_StreamingBodyErrorMetricUsesRemappedStatus is a regression
// test for two related bugs found while reviewing PR #149:
//
//  1. RecordRequest for a streaming proxy-credential response fired with
//     proxyResp.StatusCode (the original, pre-remap upstream status — usually
//     200), before the body/SSE terminal-error remap (applied deep inside
//     writeProxyStreamingResponseWithTokens/streamToClient) had happened. So a
//     request the client correctly received as 429 would still count as 200 in
//     RequestsTotal/RequestDuration.
//  2. Separately, even once the remap had already set logCtx.Status/HTTPStatus
//     correctly (via markProxyProviderStreamError), the unconditional
//     `logCtx.Status = "success"` / `logCtx.HTTPStatus = proxyResp.StatusCode`
//     in ProxyRequest's shared "Log proxy response" section ran afterward and
//     clobbered it back to "success"/200 — corrupting the persisted spend log,
//     not just the metric.
//
// This exercises the full ProxyRequest path (same trigger as
// TestProxyRequest_StreamingImmediateRateLimitEventReturns429) with metrics
// enabled, and asserts RequestsTotal was incremented under the client-visible
// 429 label, not 200.
func TestProxyRequest_StreamingBodyErrorMetricUsesRemappedStatus(t *testing.T) {
	monitoring.RequestsTotal.Reset()

	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"rate_limit_exceeded","message":"Request failed"}}}`+"\n\n")
	}))
	defer mockServer.Close()

	builder := NewTestProxyBuilder().
		WithSingleCredential("metrics-test-cred", config.ProviderTypeProxy, mockServer.URL, "upstream-key-1")
	builder.config.Metrics = monitoring.New(true)
	prx := builder.Build()

	reqBody := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusTooManyRequests, w.Code)

	rateLimited := testutil.ToFloat64(monitoring.RequestsTotal.WithLabelValues(
		"metrics-test-cred", "gpt-4", "/v1/chat/completions", "429",
	))
	assert.Equal(t, float64(1), rateLimited, "RequestsTotal should record the client-visible 429, not the pre-remap upstream 200")

	falseSuccess := testutil.ToFloat64(monitoring.RequestsTotal.WithLabelValues(
		"metrics-test-cred", "gpt-4", "/v1/chat/completions", "200",
	))
	assert.Equal(t, float64(0), falseSuccess, "RequestsTotal must not also record the pre-remap 200 for this request")

	// Give the async spend-log defer safety-net a moment to run so we're not
	// racing it (it doesn't touch these metrics, but keeps the test from
	// exiting mid-goroutine on a slow CI box).
	time.Sleep(10 * time.Millisecond)
}

// TestExecuteProxyRequest_NonOKBodyErrorStatusRecordsAttemptErrorOnce is a
// regression test: the pre-existing "resp.StatusCode != http.StatusOK" guard
// and the new statusCodeFromProviderBodyError remap both call
// RecordCredentialAttemptError independently. For a status already != 200
// (e.g. a hypothetical 201 with an error-shaped body, in [200,300) so the
// body-error check also fires) that double-counts a single failed attempt.
func TestExecuteProxyRequest_NonOKBodyErrorStatusRecordsAttemptErrorOnce(t *testing.T) {
	monitoring.CredentialErrorsTotal.Reset()

	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated) // 201: != 200, but still in [200,300)
		_, _ = io.WriteString(w, `{"error":{"message":"provider exploded","type":"server_error"}}`)
	}))
	defer mockServer.Close()

	cred := &config.CredentialConfig{
		Name:    "double-count-cred",
		Type:    config.ProviderTypeProxy,
		BaseURL: mockServer.URL,
		APIKey:  "upstream-key-1",
	}
	builder := NewTestProxyBuilder().WithCredentials(*cred)
	builder.config.Metrics = monitoring.New(true)
	prx := builder.Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	_, err := prx.executeProxyRequest(req, cred, "gpt-4", []byte(`{}`), time.Now())
	assert.NoError(t, err)

	count := testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("double-count-cred"))
	assert.Equal(t, float64(1), count, "a single failed attempt must record exactly one CredentialErrorsTotal, not two")
}
