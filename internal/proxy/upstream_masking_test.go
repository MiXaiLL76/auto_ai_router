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
			Name:               "sosana-art",
			Type:               config.ProviderTypeOpenAI,
			BaseURL:            upstream.URL,
			APIKey:             "upstream-key",
			RPM:                100,
			TPM:                10000,
			MaskUpstreamErrors: true,
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

func TestMaskedUpstreamImageSuccess_IsNormalizedToOpenAIShape(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"created":123,
			"provider":"Sasana",
			"background":"transparent",
			"output_format":"png",
			"quality":"high",
			"size":"1024x1024",
			"error":{"message":"Sasana warning that should not leak"},
			"data":[{"b64_json":"aW1hZ2U=","sasana_id":"vendor-image-id"}]
		}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:               "sosana-art",
			Type:               config.ProviderTypeOpenAI,
			BaseURL:            upstream.URL,
			APIKey:             "upstream-key",
			RPM:                100,
			TPM:                10000,
			MaskUpstreamErrors: true,
		}).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"gpt-image-1","prompt":"cat"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.NotContains(t, w.Body.String(), "Sasana")
	assert.NotContains(t, w.Body.String(), "sasana_id")
	assert.NotContains(t, w.Body.String(), "warning")

	var got struct {
		Created      int64  `json:"created"`
		Background   string `json:"background"`
		OutputFormat string `json:"output_format"`
		Quality      string `json:"quality"`
		Size         string `json:"size"`
		Data         []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	assert.Equal(t, int64(123), got.Created)
	assert.Equal(t, "transparent", got.Background)
	assert.Equal(t, "png", got.OutputFormat)
	assert.Equal(t, "high", got.Quality)
	assert.Equal(t, "1024x1024", got.Size)
	require.Len(t, got.Data, 1)
	assert.Equal(t, "aW1hZ2U=", got.Data[0].B64JSON)
}

func TestMaskedUpstreamImageSuccess_WithErrorEnvelopeBecomesRouterError(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"error":{"message":"Sasana moderation failed","code":"SASANA_POLICY"}}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:               "sosana-art",
			Type:               config.ProviderTypeOpenAI,
			BaseURL:            upstream.URL,
			APIKey:             "upstream-key",
			RPM:                100,
			TPM:                10000,
			MaskUpstreamErrors: true,
		}).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"gpt-image-1","prompt":"cat"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadGateway, w.Code)
	assert.NotContains(t, w.Body.String(), "Sasana")
	assert.NotContains(t, w.Body.String(), "SASANA_POLICY")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
}

func TestMaskedUpstreamSosanaCompletedResponse_IsConvertedToOpenAIShape(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"uid":"01977614-95b0-7000-8000-example",
			"status":"COMPLETED",
			"created_at":"2026-06-26T12:34:56Z",
			"prompt":"cat",
			"optimized_prompt":"A detailed cat illustration",
			"result_file_url":"https://cdn.sosana.art/results/cat.png",
			"elapsed":12.3,
			"provider":"Sosana"
		}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:               "vsellm-sosana-art",
			Type:               config.ProviderTypeOpenAI,
			BaseURL:            upstream.URL,
			APIKey:             "upstream-key",
			RPM:                100,
			TPM:                10000,
			MaskUpstreamErrors: true,
		}).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"gpt-image-2","prompt":"cat"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.NotContains(t, w.Body.String(), "Sosana")
	assert.NotContains(t, w.Body.String(), "uid")
	assert.NotContains(t, w.Body.String(), "result_file_url")
	assert.NotContains(t, w.Body.String(), "status")

	var got struct {
		Created int64 `json:"created"`
		Data    []struct {
			URL           string `json:"url"`
			RevisedPrompt string `json:"revised_prompt"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	assert.Equal(t, int64(1782477296), got.Created)
	require.Len(t, got.Data, 1)
	assert.Equal(t, "https://cdn.sosana.art/results/cat.png", got.Data[0].URL)
	assert.Equal(t, "A detailed cat illustration", got.Data[0].RevisedPrompt)
}

func TestMaskedUpstreamSosanaProcessingResponse_BecomesRouterError(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"uid":"01977614-95b0-7000-8000-example",
			"status":"PROCESSING",
			"created_at":"2026-06-26T12:34:56Z",
			"prompt":"cat",
			"provider":"Sosana"
		}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:               "vsellm-sosana-art",
			Type:               config.ProviderTypeOpenAI,
			BaseURL:            upstream.URL,
			APIKey:             "upstream-key",
			RPM:                100,
			TPM:                10000,
			MaskUpstreamErrors: true,
		}).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"gpt-image-2","prompt":"cat"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadGateway, w.Code)
	assert.NotContains(t, w.Body.String(), "Sosana")
	assert.NotContains(t, w.Body.String(), "PROCESSING")
	assert.NotContains(t, w.Body.String(), "uid")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
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

func TestNormalizeOpenAIImageResponseBodyRejectsInvalidSuccess(t *testing.T) {
	_, err := normalizeOpenAIImageResponseBody([]byte(`{"created":123,"data":[{"revised_prompt":"only text"}]}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "neither b64_json nor url")
}
