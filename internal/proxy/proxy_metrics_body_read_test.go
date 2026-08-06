package proxy

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_DirectCredential_BodyTooLarge_RecordsFinalMetric is a
// regression test: when the response body from a direct (non-proxy)
// credential exceeds the configured size limit, the client gets a 502 via an
// early return in the retry loop — but that return path used to skip
// RecordRequest entirely, so the 502 never showed up in RequestsTotal. It
// also used to skip RecordCredentialAttemptError, since the earlier
// `resp.StatusCode != http.StatusOK` check can't see a failure that only
// happens while reading the (200 OK) body. Both must be recorded exactly
// once for this attempt.
//
// ErrResponseBodyTooLarge is used here only because it's a deterministic way
// to trigger the shared `readErr != nil` branch in the retry loop — the fix
// applies identically to any body-read failure (e.g. a connection dropped
// mid-body), not just an oversized one.
func TestProxyRequest_DirectCredential_BodyTooLarge_RecordsFinalMetric(t *testing.T) {
	monitoring.RequestsTotal.Reset()

	oversized := bytes.Repeat([]byte("x"), 2*1024*1024) // 2MB, over the 1MB cap below
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(oversized)
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMaxBodySizeMB(1).
		WithResponseBodyMultiplier(1). // 1MB * 1 = 1MB cap
		Build()
	prx.metrics = monitoring.New(true)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadGateway, w.Code)

	assert.Equal(t, 1.0,
		testutil.ToFloat64(monitoring.RequestsTotal.WithLabelValues("provider", "gpt-4", "/v1/chat/completions", "502")),
		"the 502 written to the client for an oversized body must be recorded in RequestsTotal")
	assert.Equal(t, 1.0,
		testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("provider")),
		"a body-read failure is a genuine attempt failure and must show up in credential-error metrics")
}

// TestExecuteProxyRequest_BodyTooLarge_RecordsCredentialError is the
// forwardToProxy/executeProxyRequest analog of the test above (the "relay"
// path used by proxy-like credentials, per the review note that the same gap
// exists there too).
func TestExecuteProxyRequest_BodyTooLarge_RecordsCredentialError(t *testing.T) {
	oversized := bytes.Repeat([]byte("x"), 2*1024*1024)
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(oversized)
	}))
	defer upstream.Close()

	cred := &config.CredentialConfig{
		Name:    "gateway",
		Type:    config.ProviderTypeProxy,
		BaseURL: upstream.URL,
		APIKey:  "key",
	}

	prx := NewTestProxyBuilder().
		WithSingleCredential("gateway", config.ProviderTypeProxy, upstream.URL, "key").
		WithMaxBodySizeMB(1).
		WithResponseBodyMultiplier(1).
		Build()
	prx.metrics = monitoring.New(true)
	monitoring.CredentialErrorsTotal.Reset()

	upstreamReq := httptest.NewRequest("POST", "/v1/test", strings.NewReader("body"))
	upstreamReq.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()

	proxyResp, err := prx.forwardToProxy(w, upstreamReq, "test-model", cred, []byte("body"), time.Now().UTC())

	require.Error(t, err)
	require.Nil(t, proxyResp)
	assert.Equal(t, 1.0,
		testutil.ToFloat64(monitoring.CredentialErrorsTotal.WithLabelValues("gateway")),
		"a body-read failure in executeProxyRequest is a genuine attempt failure and must show up in credential-error metrics")
}
