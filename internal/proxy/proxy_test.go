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

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

func createTestBalancer(mockServerURL string) (*balancer.RoundRobin, *ratelimit.RPMLimiter) {
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "upstream-key-1", BaseURL: mockServerURL, RPM: 100},
		{Name: "test2", APIKey: "upstream-key-2", BaseURL: mockServerURL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	return bal, rl
}

func createTestTokenManager() *auth.VertexTokenManager {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	return auth.NewVertexTokenManager(logger)
}

func TestNew(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer("http://test.com")
	metrics := monitoring.New(false)

	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "test-master-key", rl, tm, "test-version", "test-commit")

	assert.NotNil(t, prx)
	assert.Equal(t, "test-master-key", prx.masterKey)
	assert.Equal(t, 10, prx.maxBodySizeMB)
	assert.Equal(t, 30*time.Second, prx.requestTimeout)
	assert.NotNil(t, prx.client)
}

func TestProxyRequest_MissingAuthorization(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer("http://test.com")
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "Missing Authorization header")
}

func TestProxyRequest_InvalidAuthorizationFormat(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer("http://test.com")
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "InvalidFormat")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "Invalid Authorization header format")
}

func TestProxyRequest_InvalidMasterKey(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer("http://test.com")
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "correct-key", rl, tm, "test-version", "test-commit")

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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

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
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
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
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	// Manually trigger rate limiter to exhaust the limit
	rl.Allow("test1")

	// Next request should fail due to rate limit
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"stream": true}`))
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
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.True(t, healthy)
	assert.Equal(t, "healthy", status["status"])
	assert.Equal(t, 2, status["total_credentials"])
	assert.Equal(t, 2, status["credentials_available"])
	assert.Equal(t, 0, status["credentials_banned"])

	// Check credentials info is present
	assert.Contains(t, status, "credentials")
	credentialsInfo, ok := status["credentials"].(map[string]interface{})
	assert.True(t, ok)
	assert.Len(t, credentialsInfo, 2)

	// Check models info is present (even if empty)
	assert.Contains(t, status, "models")
	modelsInfo, ok := status["models"].(map[string]interface{})
	assert.True(t, ok)
	assert.NotNil(t, modelsInfo)
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
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.False(t, healthy)
	assert.Equal(t, "unhealthy", status["status"])
	assert.Equal(t, 2, status["total_credentials"])
	assert.Equal(t, 0, status["credentials_available"])
	assert.Equal(t, 2, status["credentials_banned"])

	// Check credentials info is present even when unhealthy
	assert.Contains(t, status, "credentials")
	credentialsInfo, ok := status["credentials"].(map[string]interface{})
	assert.True(t, ok)
	assert.Len(t, credentialsInfo, 2)

	// Check models info is present
	assert.Contains(t, status, "models")
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

	// Add some models with limits
	rl.AddModelWithTPM("test1", "gpt-4", 10, 30000)
	rl.AddModelWithTPM("test1", "gpt-3.5-turbo", 20, 40000)
	rl.AddModelWithTPM("test2", "gpt-4", 5, 15000)

	// Simulate some usage
	rl.Allow("test1")
	rl.Allow("test1")
	rl.ConsumeTokens("test1", 5000)

	rl.AllowModel("test1", "gpt-4")
	rl.ConsumeModelTokens("test1", "gpt-4", 2000)

	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	assert.True(t, healthy)
	assert.Equal(t, "healthy", status["status"])

	// Check credentials info
	credentialsInfo, ok := status["credentials"].(map[string]interface{})
	assert.True(t, ok)
	assert.Len(t, credentialsInfo, 2)

	// Check test1 credential details
	test1Info, ok := credentialsInfo["test1"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, 2, test1Info["current_rpm"])
	assert.Equal(t, 5000, test1Info["current_tpm"])
	assert.Equal(t, 100, test1Info["limit_rpm"])
	assert.Equal(t, 100000, test1Info["limit_tpm"])

	// Check models info
	modelsInfo, ok := status["models"].(map[string]interface{})
	assert.True(t, ok)
	assert.Len(t, modelsInfo, 3) // 3 models added

	// Check test1:gpt-4 model details
	gpt4Info, ok := modelsInfo["test1:gpt-4"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "test1", gpt4Info["credential"])
	assert.Equal(t, "gpt-4", gpt4Info["model"])
	assert.Equal(t, 1, gpt4Info["current_rpm"])    // 1 request made
	assert.Equal(t, 2000, gpt4Info["current_tpm"]) // 2000 tokens consumed
	assert.Equal(t, 10, gpt4Info["limit_rpm"])     // RPM limit
	assert.Equal(t, 30000, gpt4Info["limit_tpm"])  // TPM limit

	// Check test1:gpt-3.5-turbo model details
	gpt35Info, ok := modelsInfo["test1:gpt-3.5-turbo"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "test1", gpt35Info["credential"])
	assert.Equal(t, "gpt-3.5-turbo", gpt35Info["model"])
	assert.Equal(t, 20, gpt35Info["limit_rpm"])    // RPM limit
	assert.Equal(t, 40000, gpt35Info["limit_tpm"]) // TPM limit
}

func TestExtractModelFromBody(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected string
	}{
		{
			name:     "valid json with model",
			body:     `{"model": "gpt-4", "messages": []}`,
			expected: "gpt-4",
		},
		{
			name:     "valid json without model",
			body:     `{"messages": []}`,
			expected: "",
		},
		{
			name:     "empty body",
			body:     "",
			expected: "",
		},
		{
			name:     "invalid json",
			body:     `{invalid json}`,
			expected: "",
		},
		{
			name:     "model is empty string",
			body:     `{"model": "", "messages": []}`,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractModelFromBody([]byte(tt.body))
			assert.Equal(t, tt.expected, result)
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

			result := isStreamingResponse(resp)
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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

	req := httptest.NewRequest("GET", "/v1/models?param1=value1&param2=value2", nil)
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

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	bal, rl := createTestBalancer(mockServer.URL)
	metrics := monitoring.New(false)
	tm := createTestTokenManager()
	prx := New(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, "test-version", "test-commit")

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
