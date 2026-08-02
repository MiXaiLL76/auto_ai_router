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
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/prometheus/client_golang/prometheus/testutil"
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
	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		_ = testhelpers.NewResponseBuilder().
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
	req.Header.Set("Authorization", "Bearer master-key")
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

// TestProxyFallbackOn429_ModelNotFoundBody tests that when primary returns 429
// with a model-not-found body, fallback is NOT attempted (model errors are non-retryable).
func TestProxyFallbackOn429_ModelNotFoundBody(t *testing.T) {
	var primaryCalls, fallbackCalls int32

	// Primary returns 429 with model not found (non-retryable)
	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "model_not_found",
			"message": "model not found: gpt-4",
		})
	}))
	defer primaryServer.Close()

	// Fallback server (should NOT be called)
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "Primary should be called")
	assert.Equal(t, int32(0), atomic.LoadInt32(&fallbackCalls), "Fallback should NOT be called for model-not-found")
	assert.Equal(t, http.StatusTooManyRequests, w.Code, "Original error code should be returned")
}

// TestProxyFallbackOn429_RequestBodyPreserved tests that the exact request body
// is forwarded to both primary and fallback proxies.
func TestProxyFallbackOn429_RequestBodyPreserved(t *testing.T) {
	var primaryReceivedBody, fallbackReceivedBody []byte

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		primaryReceivedBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	req.Header.Set("Authorization", "Bearer master-key")
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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	req.Header.Set("Authorization", "Bearer master-key")
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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallback1Server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallback1Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-fb1",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer fallback1Server.Close()

	fallback2Server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Only first fallback should be used
	assert.Equal(t, int32(1), atomic.LoadInt32(&primaryCalls), "Primary should be called once")
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallback1Calls), "First fallback should be called")
	assert.Equal(t, int32(0), atomic.LoadInt32(&fallback2Calls), "Second fallback should NOT be called")
	assert.Equal(t, http.StatusOK, w.Code)
}

// TestProxy_429PreservedOverNetworkError is a regression test for the bug where a 429
// returned by the first proxy credential was replaced by a 502 when the retry attempt
// against a second credential produced a network error.
//
// Before the fix: proxyResp was overwritten to nil on network error → 502.
// After the fix:  proxyResp is only updated on a successful HTTP response, so the
//
//	saved 429 survives and is returned to the client.
func TestProxy_429PreservedOverNetworkError(t *testing.T) {
	var cred1Calls int32

	// cred1 returns 429 (rate-limited).
	cred1Server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&cred1Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "rate_limit_exceeded",
			"message": "rate limited",
		})
	}))
	defer cred1Server.Close()

	// cred2 is dead (connection refused): create a server and immediately close it.
	deadServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := deadServer.URL
	deadServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "cred1",
				Type:       config.ProviderTypeProxy,
				APIKey:     "key1",
				BaseURL:    cred1Server.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "cred2",
				Type:       config.ProviderTypeProxy,
				APIKey:     "key2",
				BaseURL:    deadURL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
		).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	// The 429 from cred1 must be returned — not a 502 from the network error on cred2.
	assert.Equal(t, http.StatusTooManyRequests, w.Code,
		"client must receive 429 from first credential, not 502 from the network error on retry")
	assert.Equal(t, int32(1), atomic.LoadInt32(&cred1Calls), "cred1 should be called exactly once")

	// Verify the response body still contains the original rate-limit error JSON.
	var respBody map[string]interface{}
	require.NoError(t, json.NewDecoder(w.Body).Decode(&respBody))
	assert.Equal(t, "rate_limit_exceeded", respBody["error"])
}

// TestProxy_429PreservedWhenNoFallbackAndNetworkError is similar but without any
// fallback credential configured, ensuring the same invariant holds in that simpler topology.
func TestProxy_429PreservedWhenNoFallbackAndNetworkError(t *testing.T) {
	var cred1Calls int32

	cred1Server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&cred1Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "rate_limit_exceeded",
			"message": "too many requests",
		})
	}))
	defer cred1Server.Close()

	deadServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := deadServer.URL
	deadServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "key",
				BaseURL:    cred1Server.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "secondary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "key2",
				BaseURL:    deadURL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
		).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusTooManyRequests, w.Code,
		"a single-credential 429 followed by a network error must still return 429, not 502")
	assert.Equal(t, int32(1), atomic.LoadInt32(&cred1Calls))
}

// TestProxy_MetricsAttributeToRespondingCredentialNotLastAttempted is a
// regression test for a metrics attribution bug: when attempt A (credential
// "primary") returns a retryable 429 and the subsequent retry against
// credential "secondary" fails with a transport error (no fallback
// available), the *response* written to the client is still primary's 429 —
// but the loop variable `cred` has already moved on to "secondary" by the
// time the final RecordRequest call ran. Without restoring the credential
// that actually produced the response, RequestsTotal/RequestDuration would be
// mislabeled under "secondary" for a response "primary" produced, defeating
// this PR's central goal of correct per-credential attribution.
func TestProxy_MetricsAttributeToRespondingCredentialNotLastAttempted(t *testing.T) {
	monitoring.RequestsTotal.Reset()

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	deadServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := deadServer.URL
	deadServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name: "primary", Type: config.ProviderTypeProxy, APIKey: "key1",
				BaseURL: primaryServer.URL, RPM: 100, TPM: 10000, IsFallback: false,
			},
			config.CredentialConfig{
				Name: "secondary", Type: config.ProviderTypeProxy, APIKey: "key2",
				BaseURL: deadURL, RPM: 100, TPM: 10000, IsFallback: false,
			},
		).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()
	prx.metrics = monitoring.New(true)

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusTooManyRequests, w.Code)

	assert.Equal(t, 1.0,
		testutil.ToFloat64(monitoring.RequestsTotal.WithLabelValues("primary", "gpt-4", "/v1/chat/completions", "429")),
		"the 429 response actually written to the client came from 'primary' and must be recorded under its name")
	assert.Equal(t, 0.0,
		testutil.ToFloat64(monitoring.RequestsTotal.WithLabelValues("secondary", "gpt-4", "/v1/chat/completions", "429")),
		"'secondary' never produced a response — it must not be attributed a 429 that belongs to 'primary'")
}
