package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTryFallbackProxy_NoFallback tests the scenario where no fallback credential is available.
//
// Test Case A: No available fallback
// - Balancer is configured with a single primary credential (IsFallback=false)
// - When TryFallbackProxy is called, balancer.NextFallbackForModel returns an error
// - Function should return (false, "no_fallback_available")
// - HTTP response should not be modified (response writer should remain empty)
//
// This test validates that TryFallbackProxy correctly handles the case where no fallback
// credential is configured for the requested model.
func TestTryFallbackProxy_NoFallback(t *testing.T) {
	// Build proxy with only primary credential (no fallback)
	prx := NewTestProxyBuilder().
		WithSingleCredential(
			"primary",
			config.ProviderTypeProxy,
			"http://primary.example.com",
			"primary-api-key",
		).
		Build()

	// === PREPARE REQUEST

	// Create a minimal request body with model specification
	requestBody := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Test message",
			},
		},
	}
	bodyBytes, err := json.Marshal(requestBody)
	require.NoError(t, err, "Failed to marshal request body")

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	// Create response recorder to capture the response
	w := httptest.NewRecorder()

	// === CALL FUNCTION

	// Call TryFallbackProxy with specific parameters:
	// - modelID: "gpt-4" (the model requested)
	// - originalCredName: "primary" (the credential that failed)
	// - originalStatus: http.StatusTooManyRequests (429 - why we're attempting fallback)
	// - retryReason: RetryReasonRateLimit (rate limit exceeded)
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",                    // modelID
		"primary",                  // originalCredName
		http.StatusTooManyRequests, // originalStatus
		RetryReasonRateLimit,       // originalReason
		bodyBytes,                  // request body to retry
		time.Now().UTC(),           // start time
		nil,                        // logCtx
	)

	// === ASSERTIONS

	// ✓ Assert: Function should return success=false (no fallback was found)
	assert.False(t, success, "TryFallbackProxy should return success=false when no fallback available")

	// ✓ Assert: Reason should indicate no fallback was available
	assert.Equal(t, "no_fallback_available", reason,
		"Should return reason='no_fallback_available' when NextFallbackForModel fails")

	// ✓ Assert: Response writer should be empty (no response was written)
	// Since no fallback was found, TryFallbackProxy returns early and doesn't write response
	assert.Empty(t, w.Body.String(),
		"Response body should be empty when no fallback is available (function returns before writing)")

	// ✓ Assert: WriteHeader was not called by TryFallbackProxy
	// The response recorder has default status code 200 when nothing is written,
	// but we verify through Body.Len() == 0 that TryFallbackProxy didn't process anything
	assert.Equal(t, 0, w.Body.Len(),
		"Response body length should be 0 when no fallback available")
}

// TestTryFallbackProxy_SameCredential tests the scenario where the only fallback credential
// has the same name as the original credential (circular reference protection).
//
// Test Case B: Fallback credential is same as original
// - The original credential is pre-emptively excluded from the fallback candidate pool
// - NextFallbackProxyForModelExcluding finds no remaining candidates
// - Function should return (false, "no_fallback_available") immediately
// - Function should NOT attempt to forward the request to avoid infinite loops
//
// This test validates that TryFallbackProxy pre-emptively excludes the original credential
// from the fallback candidate pool, preventing circular retries.
// Previously the function would return "fallback_is_same_credential" after finding the candidate;
// now the candidate is never found because it is excluded up front.
func TestTryFallbackProxy_SameCredential(t *testing.T) {
	// Build proxy with single credential marked as both primary and fallback (edge case)
	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "api-key",
				BaseURL:    "http://primary.example.com",
				RPM:        100,
				TPM:        10000,
				IsFallback: true, // Edge case: marked as fallback
			},
		).
		Build()

	// === PREPARE REQUEST

	// Create minimal request body
	requestBody := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Test message",
			},
		},
	}
	bodyBytes, err := json.Marshal(requestBody)
	require.NoError(t, err, "Failed to marshal request body")

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	// Create response recorder to capture the response
	w := httptest.NewRecorder()

	// === CALL FUNCTION

	// Call TryFallbackProxy:
	// - originalCredName: "primary" (the credential that failed)
	// - balancer.NextFallbackForModel will return the only credential, which is also named "primary"
	// - This should trigger the circular reference check
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",                    // modelID
		"primary",                  // originalCredName (same as fallback that will be returned)
		http.StatusTooManyRequests, // originalStatus
		RetryReasonRateLimit,       // originalReason
		bodyBytes,                  // request body to retry
		time.Now().UTC(),           // start time
		nil,                        // logCtx
	)

	// === ASSERTIONS

	// ✓ Assert: Function should return success=false (same credential is not a valid fallback)
	assert.False(t, success,
		"TryFallbackProxy should return success=false when fallback is same credential")

	// ✓ Assert: Reason should indicate no fallback available
	// (original credential is excluded from the candidate pool, leaving no candidates)
	assert.Equal(t, "no_fallback_available", reason,
		"Should return reason='no_fallback_available' when original credential is the only fallback")

	// ✓ Assert: Response writer should be empty (request was never forwarded)
	// Since the function detected the same credential, it returns early without
	// forwarding the request to prevent infinite loops.
	assert.Empty(t, w.Body.String(),
		"Response body should be empty (function returns early before forwarding)")

	// ✓ Assert: WriteHeader was not called by TryFallbackProxy
	// The function detects the circular reference and returns early without writing anything.
	assert.Equal(t, 0, w.Body.Len(),
		"Response body length should be 0 when fallback is same credential")

	// ✓ Assert: Verify that the request was NOT forwarded
	// If TryFallbackProxy properly detected the circular reference, no HTTP request
	// should have been made to the upstream server.
	// (In this test, there's no upstream server to verify, but the empty response proves it)
}
