package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	logger := createTestLogger()
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()

	tm := createTestTokenManager(logger)
	mm := createTestModelManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-master-key", rl, tm, mm, "test-version", "test-commit")

	assert.NotNil(t, prx)
	assert.Equal(t, "test-master-key", prx.masterKey)
	assert.Equal(t, 10, prx.maxBodySizeMB)
	assert.Equal(t, 30*time.Second, prx.requestTimeout)
	assert.NotNil(t, prx.client)
}

func TestProxyRequest_MissingAuthorization(t *testing.T) {
	logger := createTestLogger()
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "Missing Authorization header")
}

func TestProxyRequest_InvalidAuthorizationFormat(t *testing.T) {
	logger := createTestLogger()
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "InvalidFormat")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "Invalid Authorization header format")
}

func TestProxyRequest_InvalidMasterKey(t *testing.T) {
	logger := createTestLogger()
	bal, rl := createTestBalancer("http://test.com")
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "correct-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "Invalid master key")
}

func TestProxyRequest_ValidRequest(t *testing.T) {
	// Create mock upstream server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify upstream receives correct Authorization header
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer upstream-key-")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "success"})
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "success")
}

func TestProxyRequest_WithModel(t *testing.T) {
	// Create mock upstream server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "ok"})
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestProxyRequest_NoCredentialsAvailable(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(1, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test.com", RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)

	// Ban the only credential
	f2b.RecordResponse("test1", 500)

	metrics := monitoring.New(false)
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), "Service Unavailable")
}

func TestProxyRequest_RateLimitExceeded(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test.com", RPM: 1}, // Very low RPM
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	// Manually trigger rate limiter to exhaust the limit
	rl.Allow("test1")

	// Next request should fail due to rate limit
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusTooManyRequests, w.Code)
}

func TestProxyRequest_UpstreamError(t *testing.T) {
	// Create mock server that returns error
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": "upstream error"}`))
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
}

func TestProxyRequest_Streaming(t *testing.T) {
	// Create mock server that returns streaming response
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"chunk\": 1}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"chunk\": 2}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model": "gpt-4", "stream": true}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "chunk")
}

func TestHealthCheck_Healthy(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer("http://test.com")
	metrics := monitoring.New(false)
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.True(t, healthy)
	assert.Equal(t, "healthy", status.Status)
	assert.Equal(t, 2, status.TotalCredentials)
	assert.Equal(t, 2, status.CredentialsAvailable)
	assert.Equal(t, 0, status.CredentialsBanned)

	// Check credentials info is present
	assert.NotNil(t, status.Credentials)
	assert.Len(t, status.Credentials, 2)

	// Check models info is present (even if empty)
	assert.NotNil(t, status.Models)
}

func TestHealthCheck_Unhealthy(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(1, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test.com", RPM: 100},
		{Name: "test2", APIKey: "key2", BaseURL: "http://test.com", RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)

	// Ban all credentials
	f2b.RecordResponse("test1", 500)
	f2b.RecordResponse("test2", 500)

	metrics := monitoring.New(false)
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.False(t, healthy)
	assert.Equal(t, "unhealthy", status.Status)
	assert.Equal(t, 2, status.TotalCredentials)
	assert.Equal(t, 0, status.CredentialsAvailable)
	assert.Equal(t, 2, status.CredentialsBanned)

	// Check credentials info is present even when unhealthy
	assert.NotNil(t, status.Credentials)
	assert.Len(t, status.Credentials, 2)

	// Check models info is present
	assert.NotNil(t, status.Models)
}

func TestHealthCheck_WithModels(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test.com", RPM: 100, TPM: 100000},
		{Name: "test2", APIKey: "key2", BaseURL: "http://test.com", RPM: 50, TPM: 50000},
	}

	// Create balancer (it will add credentials to rate limiter)
	bal := balancer.New(credentials, f2b, rl)

	// Create model manager with test models
	testModels := []config.ModelRPMConfig{
		{Name: "gpt-4", RPM: 10, TPM: 30000},
		{Name: "gpt-3.5-turbo", RPM: 20, TPM: 40000},
	}
	mm := models.New(logger, 50, testModels)
	mm.LoadModelsFromConfig(credentials)

	// Add models to rate limiter
	rl.AddModelWithTPM("test1", "gpt-4", 10, 30000)
	rl.AddModelWithTPM("test1", "gpt-3.5-turbo", 20, 40000)
	rl.AddModelWithTPM("test2", "gpt-4", 5, 15000)
	rl.AddModelWithTPM("test2", "gpt-3.5-turbo", 15, 35000)

	// Simulate some usage
	rl.Allow("test1")
	rl.Allow("test1")
	rl.ConsumeTokens("test1", 5000)

	rl.AllowModel("test1", "gpt-4")
	rl.ConsumeModelTokens("test1", "gpt-4", 2000)

	metrics := monitoring.New(false)
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.True(t, healthy)
	assert.Equal(t, "healthy", status.Status)

	// Check credentials info
	assert.NotNil(t, status.Credentials)
	assert.Len(t, status.Credentials, 2)

	// Check test1 credential details
	test1Info, ok := status.Credentials["test1"]
	assert.True(t, ok)
	assert.Equal(t, 2, test1Info.CurrentRPM)
	assert.Equal(t, 5000, test1Info.CurrentTPM)
	assert.Equal(t, 100, test1Info.LimitRPM)
	assert.Equal(t, 100000, test1Info.LimitTPM)

	// Check models info
	assert.NotNil(t, status.Models)
	assert.Len(t, status.Models, 4) // 2 models × 2 credentials = 4 entries

	// Check test1:gpt-4 model details
	gpt4Info, ok := status.Models["test1:gpt-4"]
	assert.True(t, ok)
	assert.Equal(t, "test1", gpt4Info.Credential)
	assert.Equal(t, "gpt-4", gpt4Info.Model)
	assert.Equal(t, 1, gpt4Info.CurrentRPM)    // 1 request made
	assert.Equal(t, 2000, gpt4Info.CurrentTPM) // 2000 tokens consumed
	assert.Equal(t, 10, gpt4Info.LimitRPM)     // RPM limit
	assert.Equal(t, 30000, gpt4Info.LimitTPM)  // TPM limit

	// Check test1:gpt-3.5-turbo model details
	gpt35Info, ok := status.Models["test1:gpt-3.5-turbo"]
	assert.True(t, ok)
	assert.Equal(t, "test1", gpt35Info.Credential)
	assert.Equal(t, "gpt-3.5-turbo", gpt35Info.Model)
	assert.Equal(t, 20, gpt35Info.LimitRPM)    // RPM limit
	assert.Equal(t, 40000, gpt35Info.LimitTPM) // TPM limit
}

func TestExtractModelFromBody(t *testing.T) {
	tests := []struct {
		name              string
		body              string
		expectedModel     string
		expectedStream    bool
		checkModifiedBody bool
	}{
		{
			name:              "valid json with model",
			body:              `{"model": "gpt-4", "messages": []}`,
			expectedModel:     "gpt-4",
			expectedStream:    false,
			checkModifiedBody: false,
		},
		{
			name:              "valid json without model",
			body:              `{"messages": []}`,
			expectedModel:     "",
			expectedStream:    false,
			checkModifiedBody: false,
		},
		{
			name:              "empty body",
			body:              "",
			expectedModel:     "",
			expectedStream:    false,
			checkModifiedBody: false,
		},
		{
			name:              "invalid json",
			body:              `{invalid json}`,
			expectedModel:     "",
			expectedStream:    false,
			checkModifiedBody: false,
		},
		{
			name:              "model is empty string",
			body:              `{"model": "", "messages": []}`,
			expectedModel:     "",
			expectedStream:    false,
			checkModifiedBody: false,
		},
		{
			name:              "streaming request without stream_options",
			body:              `{"model": "gpt-4", "stream": true, "messages": []}`,
			expectedModel:     "gpt-4",
			expectedStream:    true,
			checkModifiedBody: true,
		},
		{
			name:              "streaming request with empty stream_options",
			body:              `{"model": "gpt-4", "stream": true, "stream_options": {}, "messages": []}`,
			expectedModel:     "gpt-4",
			expectedStream:    true,
			checkModifiedBody: true,
		},
		{
			name:              "streaming request with include_usage false",
			body:              `{"model": "gpt-4", "stream": true, "stream_options": {"include_usage": false}, "messages": []}`,
			expectedModel:     "gpt-4",
			expectedStream:    true,
			checkModifiedBody: true,
		},
		{
			name:              "non-streaming request",
			body:              `{"model": "gpt-4", "stream": false, "messages": []}`,
			expectedModel:     "gpt-4",
			expectedStream:    false,
			checkModifiedBody: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, stream, modifiedBody := extractMetadataFromBody([]byte(tt.body))
			assert.Equal(t, tt.expectedModel, model)
			assert.Equal(t, tt.expectedStream, stream)

			if tt.checkModifiedBody && stream {
				// For streaming requests, verify stream_options.include_usage is set to true
				var bodyMap map[string]interface{}
				err := json.Unmarshal(modifiedBody, &bodyMap)
				assert.NoError(t, err)

				streamOptions, ok := bodyMap["stream_options"].(map[string]interface{})
				assert.True(t, ok, "stream_options should be present")
				assert.Equal(t, true, streamOptions["include_usage"], "include_usage should be true")
			}
		})
	}
}

func TestIsStreamingResponse(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		expected    bool
	}{
		{
			name:        "text/event-stream",
			contentType: "text/event-stream",
			expected:    true,
		},
		{
			name:        "application/stream+json",
			contentType: "application/stream+json",
			expected:    true,
		},
		{
			name:        "text/event-stream with charset",
			contentType: "text/event-stream; charset=utf-8",
			expected:    true,
		},
		{
			name:        "application/json",
			contentType: "application/json",
			expected:    false,
		},
		{
			name:        "text/plain",
			contentType: "text/plain",
			expected:    false,
		},
		{
			name:        "empty content type",
			contentType: "",
			expected:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{
				Header: http.Header{},
			}
			resp.Header.Set("Content-Type", tt.contentType)

			result := IsStreamingResponse(resp)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDecodeResponseBody(t *testing.T) {
	tests := []struct {
		name        string
		body        []byte
		encoding    string
		expected    string
		shouldMatch bool
	}{
		{
			name:        "plain text",
			body:        []byte("plain text response"),
			encoding:    "",
			expected:    "plain text response",
			shouldMatch: true,
		},
		{
			name:        "gzip encoded",
			body:        createGzipBody("gzip compressed text"),
			encoding:    "gzip",
			expected:    "gzip compressed text",
			shouldMatch: true,
		},
		{
			name:        "gzip in content-encoding with case",
			body:        createGzipBody("test data"),
			encoding:    "Gzip",
			expected:    "test data",
			shouldMatch: true,
		},
		{
			name:        "non-gzip encoding",
			body:        []byte("test"),
			encoding:    "deflate",
			expected:    "test",
			shouldMatch: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := decodeResponseBody(tt.body, tt.encoding)
			if tt.shouldMatch {
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestMaskKey(t *testing.T) {
	tests := []struct {
		name     string
		key      string
		expected string
	}{
		{
			name:     "long key",
			key:      "sk-proj-1234567890abcdef",
			expected: "sk-proj...",
		},
		{
			name:     "short key",
			key:      "short",
			expected: "***",
		},
		{
			name:     "exactly 7 chars",
			key:      "1234567",
			expected: "***",
		},
		{
			name:     "8 chars",
			key:      "12345678",
			expected: "1234567...",
		},
		{
			name:     "empty key",
			key:      "",
			expected: "***",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := maskKey(tt.key)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestProxyRequest_HeadersForwarding(t *testing.T) {
	// Create mock server to verify headers
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify custom headers are forwarded
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "custom-value", r.Header.Get("X-Custom-Header"))

		// Verify Authorization is replaced with upstream key
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer upstream-key-")
		assert.NotContains(t, r.Header.Get("Authorization"), "master-key")

		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "ok"})
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Custom-Header", "custom-value")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestProxyRequest_QueryParameters(t *testing.T) {
	// Create mock server to verify query params
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "value1", r.URL.Query().Get("param1"))
		assert.Equal(t, "value2", r.URL.Query().Get("param2"))

		w.WriteHeader(http.StatusOK)
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions?param1=value1&param2=value2", strings.NewReader(`{"model": "gpt-4"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestVisualHealthCheck(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer mockServer.Close()

	logger := createTestLogger()
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := createTestProxyMetrics()
	tm := createTestTokenManager(logger)
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, createTestModelManager(logger), "test-version", "test-commit")

	req := httptest.NewRequest("GET", "/vhealth", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
	assert.NotEmpty(t, w.Body.String())
	assert.Contains(t, w.Body.String(), "html")
}

// Helper function to create gzip-compressed body
func createGzipBody(content string) []byte {
	var buf bytes.Buffer
	gzipWriter := gzip.NewWriter(&buf)
	_, _ = gzipWriter.Write([]byte(content))
	_ = gzipWriter.Close()
	return buf.Bytes()
}

func TestExtractTokensFromResponse(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		credType config.ProviderType
		expected int
	}{
		{
			name:     "OpenAI format with tokens",
			body:     `{"usage": {"total_tokens": 150}}`,
			credType: config.ProviderTypeOpenAI,
			expected: 150,
		},
		{
			name:     "Vertex AI format with tokens",
			body:     `{"usageMetadata": {"totalTokenCount": 200}}`,
			credType: config.ProviderTypeVertexAI,
			expected: 200,
		},
		{
			name:     "empty body",
			body:     "",
			credType: config.ProviderTypeOpenAI,
			expected: 0,
		},
		{
			name:     "invalid json",
			body:     `{invalid}`,
			credType: config.ProviderTypeOpenAI,
			expected: 0,
		},
		{
			name:     "no usage field",
			body:     `{"result": "ok"}`,
			credType: config.ProviderTypeOpenAI,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := extractTokensFromResponse(tt.body, tt.credType)
			assert.Equal(t, tt.expected, tokens)
		})
	}
}

func TestExtractTokensFromStreamingChunk(t *testing.T) {
	tests := []struct {
		name     string
		chunk    string
		expected int
	}{
		{
			name:     "chunk with usage",
			chunk:    "data: {\"usage\": {\"total_tokens\": 100}}\n\n",
			expected: 100,
		},
		{
			name:     "chunk without usage",
			chunk:    "data: {\"choices\": [{\"delta\": {\"content\": \"test\"}}]}\n\n",
			expected: 0,
		},
		{
			name:     "done chunk",
			chunk:    "data: [DONE]\n\n",
			expected: 0,
		},
		{
			name:     "multiple chunks",
			chunk:    "data: {\"choices\": []}\n\ndata: {\"usage\": {\"total_tokens\": 50}}\n\n",
			expected: 50,
		},
		{
			name:     "invalid json",
			chunk:    "data: {invalid}\n\n",
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := extractTokensFromStreamingChunk(tt.chunk)
			assert.Equal(t, tt.expected, tokens)
		})
	}
}
