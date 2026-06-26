package proxy

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_SosanaImageGenerationSuccess(t *testing.T) {
	var createSeen, pollSeen bool
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer sosana-key", r.Header.Get("Authorization"))
		switch r.URL.Path {
		case "/api/banana/create-async":
			createSeen = true
			var req map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			assert.Equal(t, "draw a fox", req["prompt"])
			assert.Equal(t, "nano-banana", req["model"])
			assert.Equal(t, "1:1", req["aspect_ratio"])
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw a fox"}`))
		case "/api/banana/task-1":
			pollSeen = true
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"COMPLETED","created_at":"2026-01-01T00:00:00Z","prompt":"draw a fox","result_file_url":"https://cdn.sosana.art/fox.png"}`))
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw a fox","size":"1024x1024","n":1,"response_format":"b64_json"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 1)
	assert.Equal(t, "https://cdn.sosana.art/fox.png", resp.Data[0].URL)
	assert.Empty(t, resp.Data[0].B64JSON)
	assert.True(t, createSeen)
	assert.True(t, pollSeen)
}

func TestProxyRequest_SosanaRejectsNonImageEndpoint(t *testing.T) {
	called := false
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"nano-banana","messages":[]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "provider supports only image generation")
	assert.False(t, called)
}

func TestProxyRequest_SosanaCreateHTTPErrorMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"detail":"sosana balance secret marker"}`))
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusPaymentRequired, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "balance secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.NotContains(t, logBuf.String(), "balance secret")
}

func TestProxyRequest_SosanaRetriesCreateWithNextCredential(t *testing.T) {
	var createAuths []string
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			createAuths = append(createAuths, r.Header.Get("Authorization"))
			if r.Header.Get("Authorization") == "Bearer sosana-key-a" {
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"detail":"first credential rate limited"}`))
				return
			}
			assert.Equal(t, "Bearer sosana-key-b", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"uid":"task-2","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-2":
			assert.Equal(t, "Bearer sosana-key-b", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"uid":"task-2","status":"COMPLETED","prompt":"draw","result_file_url":"https://cdn.sosana.art/retry.png"}`))
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "sosana-a", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-a", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "sosana-b", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-b", RPM: 100, TPM: 10000},
		).
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, []string{"Bearer sosana-key-a", "Bearer sosana-key-b"}, createAuths)
	assert.Contains(t, w.Body.String(), "https://cdn.sosana.art/retry.png")
}

func TestProxyRequest_SosanaPollHTTPErrorMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"detail":"poll secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "poll secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.NotContains(t, logBuf.String(), "poll secret")
}

func TestProxyRequest_SosanaDoesNotRetryAfterTaskCreated(t *testing.T) {
	createCalls := 0
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			createCalls++
			assert.Equal(t, "Bearer sosana-key-a", r.Header.Get("Authorization"))
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","prompt":"draw"}`))
		case "/api/banana/task-1":
			assert.Equal(t, "Bearer sosana-key-a", r.Header.Get("Authorization"))
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"detail":"poll failed after task was created"}`))
		default:
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "sosana-a", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-a", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "sosana-b", Type: config.ProviderTypeSosana, BaseURL: upstream.URL, APIKey: "sosana-key-b", RPM: 100, TPM: 10000},
		).
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, 1, createCalls)
	assert.NotContains(t, w.Body.String(), "poll failed")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
}

func TestProxyRequest_SosanaTaskFailedMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"FAILED","created_at":"2026-01-01T00:00:00Z","prompt":"draw","error":"failed secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.NotContains(t, w.Body.String(), "failed secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.NotContains(t, logBuf.String(), "failed secret")
}

func TestProxyRequest_SosanaTaskModeratedMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/banana/create-async":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
		case "/api/banana/task-1":
			_, _ = w.Write([]byte(`{"uid":"task-1","status":"MODERATED","created_at":"2026-01-01T00:00:00Z","prompt":"draw","error":"moderation secret marker"}`))
		}
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "content_policy_violation")
	assert.NotContains(t, w.Body.String(), "moderation secret")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
	assert.NotContains(t, logBuf.String(), "moderation secret")
}

func TestProxyRequest_SosanaTimeoutMasked(t *testing.T) {
	var logBuf bytes.Buffer
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"uid":"task-1","status":"PROCESSING","created_at":"2026-01-01T00:00:00Z","prompt":"draw"}`))
	}))
	defer upstream.Close()

	prx := newSosanaTestProxy(upstream.URL, &logBuf)
	prx.requestTimeout = 5 * time.Millisecond
	req := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"model":"nano-banana","prompt":"draw","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusRequestTimeout, w.Code)
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.Contains(t, logBuf.String(), "response_body_masked=true")
}

func newSosanaTestProxy(baseURL string, logBuf *bytes.Buffer) *Proxy {
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	if logBuf != nil {
		logger = slog.New(slog.NewTextHandler(logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}
	return NewTestProxyBuilder().
		WithSingleCredential("sosana", config.ProviderTypeSosana, baseURL, "sosana-key").
		WithRequestTimeout(30 * time.Second).
		withLogger(logger).
		Build()
}

func (b *TestProxyBuilder) withLogger(logger *slog.Logger) *TestProxyBuilder {
	b.config.Logger = logger
	b.config.TokenManager = createTestTokenManager(logger)
	b.config.ModelManager = createTestModelManager(logger)
	return b
}
