package proxy

// Integration Tests for Fallback Proxy Chain Mechanism
//
// This test file validates the fallback proxy mechanism when primary credential fails.
// It tests scenarios critical for proxy-chain reliability:
// 1. Primary returns 429 (rate limit) -> fallback should handle
// 2. Primary returns 5xx (server error) -> fallback should handle
// 3. Both primary and fallback fail -> original error returned
// 4. Request body and headers are preserved during fallback
// 5. Proxy chain: router01 → router02 (primary) fails → router03 (fallback) succeeds,
//    router02 is called exactly once (not re-selected by the same-type retry loop)

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newIPv4Server(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()

	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("tcp4 listener unavailable in test environment: %v", err)
	}
	server := &httptest.Server{
		Listener: listener,
		Config:   &http.Server{Handler: handler},
	}
	server.Start()
	return server
}

// TestFallbackPath_PrimaryReturns429 tests that when primary credential returns 429,
// the request is retried with fallback credential and response is returned correctly.
func TestFallbackPath_PrimaryReturns429(t *testing.T) {
	// Track which server was called and how many times
	var primaryCalls, fallbackCalls int32

	// Create primary mock server (returns 429)
	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	// Create fallback mock server (returns 200 with success)
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "internal_error"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "fallback_error"})
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "model does not exist"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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
	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryCalls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("data: {\"error\": \"server error\"}\n\n"))
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\": [...]}\n\n"))
		flusher.Flush()
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		primaryBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

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

	primaryServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primaryServer.Close()

	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackHeaders = r.Header.Clone()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-999",
			"choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "Success"}}},
		})
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "primary-key",
				BaseURL:    primaryServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "fallback",
				Type:       config.ProviderTypeProxy,
				APIKey:     "fallback-key",
				BaseURL:    fallbackServer.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		Build()

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}`
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

// callRecorder is a thread-safe ordered log of which upstream was hit.
// Used in proxy chain tests to assert both the call count AND the exact sequence.
type callRecorder struct {
	mu  sync.Mutex
	log []string
}

func (c *callRecorder) record(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.log = append(c.log, name)
}

func (c *callRecorder) snapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, len(c.log))
	copy(out, c.log)
	return out
}

// buildProxyChain returns (primary httptest.Server, fallback httptest.Server, *Proxy).
// primary always returns primaryStatus; fallback always returns 200 with a JSON body.
// maxRetries is passed directly to WithMaxProviderRetries so the test controls
// how many same-type retry attempts are made.
func buildProxyChain(t *testing.T, rec *callRecorder, primaryStatus int, maxRetries int) (*httptest.Server, *httptest.Server, *Proxy) {
	t.Helper()

	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record("router02")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(primaryStatus)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "upstream_error"})
	}))

	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record("router03")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-router03",
			"object": "chat.completion",
			"choices": []map[string]interface{}{
				{"message": map[string]string{"role": "assistant", "content": "Response from router03"}},
			},
		})
	}))

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "router02",
				Type:       config.ProviderTypeProxy,
				APIKey:     "router02-key",
				BaseURL:    primary.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: false,
			},
			config.CredentialConfig{
				Name:       "router03",
				Type:       config.ProviderTypeProxy,
				APIKey:     "router03-key",
				BaseURL:    fallback.URL,
				RPM:        100,
				TPM:        10000,
				IsFallback: true,
			},
		).
		WithMaxProviderRetries(maxRetries).
		Build()

	return primary, fallback, prx
}

// doChainRequest fires a single POST /v1/chat/completions at prx and returns the recorder.
func doChainRequest(prx *Proxy) *httptest.ResponseRecorder {
	body := `{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// TestProxyChain_PrimaryFailsFallbackSucceeds_429 tests the distributed proxy chain:
//
//	router01 receives request
//	  └─► router02 (primary proxy) → 429 rate-limit
//	      └─► router01 detects retryable error
//	          └─► router03 (fallback proxy) → 200 success
//
// Verified invariants:
//  1. Final response is 200 from router03.
//  2. The exact call sequence is ["router02", "router03"] — router02 is called
//     once before router03, never a second time.
//
// Without the fix (missing triedCreds[cred.Name]=true at loop start) and with
// maxProviderRetries=2, the balancer re-selects router02 on attempt=1 because
// triedCreds was empty. The recorded sequence would be
// ["router02", "router02", "router03"], causing this test to fail.
func TestProxyChain_PrimaryFailsFallbackSucceeds_429(t *testing.T) {
	rec := &callRecorder{}
	primary, fallback, prx := buildProxyChain(t, rec, http.StatusTooManyRequests, 2)
	defer primary.Close()
	defer fallback.Close()

	w := doChainRequest(prx)

	require.Equal(t, http.StatusOK, w.Code, "final response must be 200 from router03")
	require.Contains(t, w.Body.String(), "router03")

	calls := rec.snapshot()
	require.Equal(t, []string{"router02", "router03"}, calls,
		"expected exactly [router02 → router03]; got %v\n"+
			"if router02 appears twice the triedCreds fix is missing", calls)
}

// TestProxyChain_PrimaryFailsFallbackSucceeds_500 tests the same chain
// with a 5xx server error from the primary proxy.
//
// Same invariants as the 429 test; only the triggering status code differs.
func TestProxyChain_PrimaryFailsFallbackSucceeds_500(t *testing.T) {
	rec := &callRecorder{}
	primary, fallback, prx := buildProxyChain(t, rec, http.StatusInternalServerError, 2)
	defer primary.Close()
	defer fallback.Close()

	w := doChainRequest(prx)

	require.Equal(t, http.StatusOK, w.Code, "final response must be 200 from router03")
	require.Contains(t, w.Body.String(), "router03")

	calls := rec.snapshot()
	require.Equal(t, []string{"router02", "router03"}, calls,
		"expected exactly [router02 → router03]; got %v", calls)
}

// TestProxyChain_NoFallback_ErrorPropagated tests that when there is no fallback
// configured, the original error is returned to the client.
//
// With maxProviderRetries=2 and only one primary credential:
//   - With the fix:    router02 is added to triedCreds immediately → tried once → error returned.
//   - Without the fix: triedCreds is empty on attempt=1, so the balancer re-selects router02
//     → tried twice before the loop exits — the call log would be ["router02", "router02"].
func TestProxyChain_NoFallback_ErrorPropagated(t *testing.T) {
	rec := &callRecorder{}

	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record("router02")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit_exceeded"})
	}))
	defer primary.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:       "router02",
			Type:       config.ProviderTypeProxy,
			APIKey:     "router02-key",
			BaseURL:    primary.URL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		}).
		WithMaxProviderRetries(2).
		Build()

	w := doChainRequest(prx)

	require.Equal(t, http.StatusTooManyRequests, w.Code, "original error must be propagated")

	calls := rec.snapshot()
	require.Equal(t, []string{"router02"}, calls,
		"router02 must be called exactly once; got %v\n"+
			"if router02 appears twice the triedCreds fix is missing", calls)
}
