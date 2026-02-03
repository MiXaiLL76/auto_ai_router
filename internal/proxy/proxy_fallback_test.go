package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyFallbackOn429 tests the fallback-on-429 scenario:
// 1. Primary credential (upstream proxy) returns 429 with retryable body
// 2. Proxy.ProxyRequest should attempt fallback on the fallback proxy
// 3. Client receives 200 OK response from fallback
//
// This test validates that the full proxy chain fallback mechanism works correctly
// for proxy-type credentials, ensuring high-availability deployments function as designed.
func TestProxyFallbackOn429(t *testing.T) {
	// Track number of calls to each server
	var primaryCalls, fallbackCalls int32

	// Create primary server mock that returns 429 with retryable body
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "rate_limit_exceeded",
			"message": "rate limited",
		})
	}))
	defer primaryServer.Close()

	// Create fallback server mock that returns 200 OK
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		_ = NewResponseBuilder().
			WithStatus(http.StatusOK).
			WithJSONBody(createMockChatCompletionResponse(
				"chatcmpl-test-429-fallback",
				"gpt-4",
				"OK from fallback",
			)).
			Write(w)
	}))
	defer fallbackServer.Close()

	// Build proxy with primary + fallback credentials
	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback(primaryServer.URL, fallbackServer.URL).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Test message for fallback",
			},
		},
	}
	bodyBytes, err := json.Marshal(requestBody)
	require.NoError(t, err, "Failed to marshal request body")

	// Create HTTP request to proxy endpoint
	req := httptest.NewRequest(
		"POST",
		"/v1/chat/completions",
		strings.NewReader(string(bodyBytes)),
	)

	// Set required headers
	req.Header.Set("Authorization", "Bearer sk-master")
	req.Header.Set("Content-Type", "application/json")

	// Execute proxy request
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	// --- ASSERTIONS ---

	// Assert: Response code should be 200 OK from fallback
	assert.Equal(t, http.StatusOK, w.Code, "Expected HTTP 200 from fallback proxy")

	// Read and parse response body
	responseBody, err := io.ReadAll(w.Body)
	require.NoError(t, err, "Failed to read response body")

	var respData map[string]interface{}
	err = json.Unmarshal(responseBody, &respData)
	require.NoError(t, err, "Failed to parse response JSON")

	// Assert: Response body should contain content from fallback
	assert.Contains(t, string(responseBody), "OK from fallback", "Expected fallback response content")

	// Assert: Response ID should indicate fallback response
	respID, ok := respData["id"].(string)
	require.True(t, ok, "Response should contain 'id' field")
	assert.Equal(t, "chatcmpl-test-429-fallback", respID, "Response ID should be from fallback")

	// Assert: Primary server should have been called exactly once
	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "Primary should be called once")

	// Assert: Fallback server should have been called exactly once
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallbackCalls), "Fallback should be called once")

	// Assert: Response should have correct structure
	_, hasChoices := respData["choices"]
	assert.True(t, hasChoices, "Response should contain 'choices' field")

	_, hasUsage := respData["usage"]
	assert.True(t, hasUsage, "Response should contain 'usage' field for token tracking")
}

// TestProxyFallbackOn429_NonRetryableBody tests that when primary returns 429
// but with a non-retryable body, fallback is NOT attempted.
func TestProxyFallbackOn429_NonRetryableBody(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	// Primary returns 429 with content policy violation (non-retryable)
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "content_policy_violation",
			"message": "This request violates our content policy",
		})
	}))
	defer primaryServer.Close()

	// Fallback server (should NOT be called)
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "ok"})
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback(primaryServer.URL, fallbackServer.URL).
		Build()

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer sk-master")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Primary should be called
	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "Primary should be called")
	// Fallback should NOT be called because body contains "content_policy"
	assert.Equal(t, int32(0), atomic.LoadInt32(&fallbackCalls), "Fallback should NOT be called for non-retryable content")
	// Original error (429) should be returned
	assert.Equal(t, http.StatusTooManyRequests, w.Code, "Original error code should be returned")
}

// TestProxyFallbackOn429_RequestBodyPreserved tests that the exact request body
// is forwarded to both primary and fallback proxies.
func TestProxyFallbackOn429_RequestBodyPreserved(t *testing.T) {
	var primaryReceivedBody, fallbackReceivedBody []byte

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		primaryReceivedBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		fallbackReceivedBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-bodytest",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback(primaryServer.URL, fallbackServer.URL).
		Build()

	// Test with specific request body including various fields
	testRequestBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7, "max_tokens": 100}`

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(testRequestBody))
	req.Header.Set("Authorization", "Bearer sk-master")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Verify both servers received identical request body
	assert.Equal(t, testRequestBody, string(primaryReceivedBody), "Primary should receive exact request body")
	assert.Equal(t, testRequestBody, string(fallbackReceivedBody), "Fallback should receive exact request body")
	assert.Equal(t, primaryReceivedBody, fallbackReceivedBody, "Request bodies should be identical")
	assert.Equal(t, http.StatusOK, w.Code, "Should return OK from fallback")
}

// TestProxyFallbackOn429_CredentialAPIKeysPreserved tests that credential API keys
// are correctly applied when forwarding to primary and fallback.
func TestProxyFallbackOn429_CredentialAPIKeysPreserved(t *testing.T) {
	var primaryAuth, fallbackAuth string

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-authtest",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-api-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-api-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer sk-master")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Verify primary received its API key
	assert.Equal(t, "Bearer primary-api-key", primaryAuth, "Primary should use its own API key")
	// Verify fallback received its API key
	assert.Equal(t, "Bearer fallback-api-key", fallbackAuth, "Fallback should use its own API key")
	assert.Equal(t, http.StatusOK, w.Code)
}

// TestProxyFallbackOn429_MultipleRetries tests fallback behavior with
// multiple fallback credentials (only first available fallback should be used).
func TestProxyFallbackOn429_MultipleRetries(t *testing.T) {
	var primaryCalls, fallback1Calls, fallback2Calls int32

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallback1Server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallback1Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-fb1",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer fallback1Server.Close()

	fallback2Server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallback2Calls, 1)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-fb2",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer fallback2Server.Close()

	prx := NewTestProxyBuilder().
		WithMultipleFallbacks(
			primaryServer.URL,
			fallback1Server.URL,
			fallback2Server.URL,
		).
		Build()

	reqBody := `{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer sk-master")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Only first fallback should be used
	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "Primary should be called once")
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallback1Calls), "First fallback should be called")
	assert.Equal(t, int32(0), atomic.LoadInt32(&fallback2Calls), "Second fallback should NOT be called")
	assert.Equal(t, http.StatusOK, w.Code)
}
