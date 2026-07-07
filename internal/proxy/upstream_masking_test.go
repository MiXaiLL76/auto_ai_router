package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMaskedUpstreamError_DirectImageErrorDoesNotLeakProviderBody(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"Sasana quota exhausted","type":"sasana_rate_limit","code":"SASANA_429"},"provider":"Sasana"}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "sosana-art",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"gpt-image-1","prompt":"cat"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusTooManyRequests, w.Code)
	assert.NotContains(t, w.Body.String(), "Sasana")
	assert.NotContains(t, w.Body.String(), "SASANA_429")

	var got APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	assert.Equal(t, "Upstream provider error", got.Error.Message)
	assert.Equal(t, "rate_limit_error", got.Error.Type)
	require.NotNil(t, got.Error.Code)
	assert.Equal(t, "upstream_rate_limit", *got.Error.Code)
}

func TestMaskedUpstreamError_ProxyChainActualCredentialDoesNotLeakProviderBody(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Credential-Name", "sosana-art-primary")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"Sasana internal failure","code":"SASANA_500"}}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("proxy-hop", config.ProviderTypeProxy, upstream.URL, "proxy-key").
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code)
	assert.NotContains(t, w.Body.String(), "Sasana")
	assert.NotContains(t, w.Body.String(), "SASANA_500")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
}

func TestMaskedUpstreamError_ProxyChainStreamingErrorDoesNotLeakProviderBody(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Credential-Name", "sosana-art-primary")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("data: {\"error\":\"Sasana stream failed\"}\n\n"))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("proxy-hop", config.ProviderTypeProxy, upstream.URL, "proxy-key").
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.NotContains(t, w.Body.String(), "Sasana")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
}
