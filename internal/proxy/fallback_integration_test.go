package proxy

// Integration Tests for Fallback Proxy Chain Mechanism
//
// This test file validates the fallback proxy mechanism when primary credential fails.
// It tests scenarios critical for proxy-chain reliability:
// 1. Primary returns 429 (rate limit) -> fallback should handle
// 2. Primary returns 5xx (server error) -> fallback should handle
// 3. Both primary and fallback fail -> original error returned
// 4. Request body and headers are preserved during fallback
//
// CURRENT STATUS: Tests reveal a critical gap in the implementation:
// - Fallback retry WORKS for non-proxy credentials (Vertex, Anthropic, OpenAI)
// - Fallback retry FAILS for proxy-type credentials (Type: "proxy")
//
// Root Cause: In proxy.go:309-311, when credential type is ProviderTypeProxy,
// the proxy.forwardToProxy() function is called and immediately returns WITHOUT
// checking for fallback retry. The fallback retry logic (lines 493-513) is only
// implemented for the non-proxy credential path.
//
// Impact: Proxy-chain deployments with fallback proxies won't actually fallback
// when primary returns errors, compromising high-availability design.
//
// See fallback_retry_analysis.md for detailed analysis and implementation plan.

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFallbackPath_PrimaryReturns429 tests that when primary credential returns 429,
// the request is retried with fallback credential and response is returned correctly.
//
// NOTE: This test is currently EXPECTED TO FAIL - it reveals a critical gap in the
// implementation. Proxy-type credentials don't support fallback retry, even though
// the ShouldRetryWithFallback() and TryFallbackProxy() logic exists.
// See fallback_retry_analysis.md for details.
//
// The issue is in proxy.go:309-311 where proxy credentials are handled with
// immediate return without fallback retry logic. This needs to be fixed to match
// the non-proxy credential path (lines 493-513).
func TestFallbackPath_PrimaryReturns429(t *testing.T) {
	// Track which server was called and how many times
	var primaryCalls, fallbackCalls int32

	// Create primary mock server (returns 429)
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	// Create fallback mock server (returns 200 with success)
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "Hello from fallback"}}},
		})
	}))
	defer fallbackServer.Close()

	// Setup logger and balancer
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	// Create credentials: primary (normal proxy) and fallback (proxy with IsFallback=true)
	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	// Add credentials to rate limiter
	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	// Create balancer
	bal := balancer.New(credentials, f2b, rl)

	// Create proxy
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	// Make request
	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	// Execute proxy request
	prx.ProxyRequest(w, req)

	// Verify response
	assert.Equal(t, http.StatusOK, w.Code, "Expected 200 OK from fallback")
	assert.Contains(t, w.Body.String(), "Hello from fallback", "Expected fallback response content")

	// Verify that primary was called first, then fallback
	assert.Equal(t, int32(1), primaryCalls, "Expected primary to be called once")
	assert.Equal(t, int32(1), fallbackCalls, "Expected fallback to be called once")
}

// TestFallbackPath_PrimaryReturns500 tests that when primary credential returns 500,
// the request is retried with fallback credential.
func TestFallbackPath_PrimaryReturns500(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "internal_error"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-456",
			"choices": []map[string]interface{}{
				{"message": map[string]string{"role": "assistant", "content": "Response from fallback"}},
			},
		})
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code, "Expected 200 OK from fallback")
	assert.Contains(t, w.Body.String(), "Response from fallback")
	assert.Equal(t, int32(1), primaryCalls)
	assert.Equal(t, int32(1), fallbackCalls)
}

// TestFallbackPath_NoFallbackAvailable tests that when no fallback is available,
// the original error is returned to the client.
func TestFallbackPath_NoFallbackAvailable(t *testing.T) {
	var primaryCalls int32

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	// Only primary credential, no fallback
	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Original error should be returned
	assert.Equal(t, http.StatusTooManyRequests, w.Code)
	assert.Equal(t, int32(1), primaryCalls, "Expected only primary to be called")
}

// TestFallbackPath_FallbackAlsoFails tests behavior when both primary and fallback fail.
func TestFallbackPath_FallbackAlsoFails(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "fallback_error"})
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Fallback error should be returned (it was attempted)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, int32(1), primaryCalls)
	assert.Equal(t, int32(1), fallbackCalls)
}

// TestFallbackPath_NonRetryableError tests that errors like "model not found" are NOT retried
// even if other conditions would trigger fallback.
func TestFallbackPath_NonRetryableError(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "model does not exist"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Original error returned (no retry attempted)
	assert.Equal(t, http.StatusNotFound, w.Code)
	assert.Equal(t, int32(1), primaryCalls)
	assert.Equal(t, int32(0), fallbackCalls, "Fallback should NOT be called for non-retryable error")
}

// TestFallbackPath_Streaming_NotSupported tests the current limitation that
// streaming requests don't support fallback retry when primary fails.
//
// NOTE: This test documents current behavior. Streaming fallback support
// would require architectural changes to buffer streaming data and retry.
func TestFallbackPath_Streaming_NotSupported(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	// Primary returns 500 during streaming
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("data: {\"error\": \"server error\"}\n\n"))
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\": [...]}\n\n"))
		flusher.Flush()
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}], "stream": true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// CURRENT BEHAVIOR: Streaming error is returned as-is, no fallback retry
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Equal(t, int32(1), primaryCalls)
	assert.Equal(t, int32(0), fallbackCalls, "Fallback NOT called for streaming (current limitation)")

	// TODO: When streaming fallback support is implemented, this should be changed to:
	// - Expect fallback to be called
	// - Expect 200 response from fallback
}

// TestFallbackPath_RequestBodyIntegrity tests that request body is preserved
// when retrying with fallback proxy.
func TestFallbackPath_RequestBodyIntegrity(t *testing.T) {
	var primaryBody, fallbackBody []byte

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		primaryBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		fallbackBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-789",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "OK"}}},
		})
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	testBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(testBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	// Verify request body is identical in both calls
	assert.Equal(t, primaryBody, fallbackBody, "Request body should be identical for both primary and fallback")
	assert.Equal(t, testBody, string(primaryBody), "Request body should match original request")
}

// TestFallbackPath_HeadersPreserved tests that request headers are correctly forwarded to fallback.
func TestFallbackPath_HeadersPreserved(t *testing.T) {
	var fallbackHeaders http.Header

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackHeaders = r.Header.Clone()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-999",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "Success"}}},
		})
	}))
	defer fallbackServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "primary-key",
			BaseURL:    primaryServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "fallback-key",
			BaseURL:    fallbackServer.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "master-key", rl, tm, mm, "test-version", "test-commit")

	reqBody := `{"model": "gpt-4", "messages": []}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Custom-Header", "test-value")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	// Custom header should be forwarded to fallback
	assert.Equal(t, "test-value", fallbackHeaders.Get("X-Custom-Header"))
	// Content-Type should be preserved
	assert.Equal(t, "application/json", fallbackHeaders.Get("Content-Type"))
}
