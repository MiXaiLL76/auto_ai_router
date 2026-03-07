package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestShouldRetryWithFallback_RateLimitError(t *testing.T) {
	shouldRetry, reason := ShouldRetryWithFallback(http.StatusTooManyRequests, []byte("rate limited"))

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonRateLimit, reason)
}

func TestShouldRetryWithFallback_ServerErrors(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"500 Internal Server Error", http.StatusInternalServerError},
		{"501 Not Implemented", http.StatusNotImplemented},
		{"502 Bad Gateway", http.StatusBadGateway},
		{"503 Service Unavailable", http.StatusServiceUnavailable},
		{"504 Gateway Timeout", http.StatusGatewayTimeout},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("server error"))

			assert.True(t, shouldRetry)
			assert.Equal(t, RetryReasonServerErr, reason)
		})
	}
}

func TestShouldRetryWithFallback_AuthErrors(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"401 Unauthorized", http.StatusUnauthorized},
		{"403 Forbidden", http.StatusForbidden},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("unauthorized"))

			assert.True(t, shouldRetry)
			assert.Equal(t, RetryReasonAuthErr, reason)
		})
	}
}

func TestShouldRetryWithFallback_NonRetryableStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"200 OK", http.StatusOK},
		{"201 Created", http.StatusCreated},
		{"400 Bad Request", http.StatusBadRequest},
		{"404 Not Found", http.StatusNotFound},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("test"))

			assert.False(t, shouldRetry)
			assert.Equal(t, RetryReason(""), reason)
		})
	}
}

func TestShouldRetryWithFallback_ContentPolicyViolation(t *testing.T) {
	// Even with 500 status, content policy violation should not be retried
	tests := []struct {
		name     string
		respBody string
	}{
		{"content policy violation", "content policy violation"},
		{"Content Policy violation uppercase", "Content Policy violation"},
		{"CONTENT POLICY", "CONTENT POLICY"},
		{"content management policy", "content management policy"},
		{"policy violation", "policy violation"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(
				http.StatusInternalServerError,
				[]byte(tt.respBody),
			)

			assert.False(t, shouldRetry)
			assert.Equal(t, RetryReason(""), reason)
		})
	}
}

func TestShouldRetryWithFallback_ModelNotFound(t *testing.T) {
	// Model-specific errors should not be retried
	tests := []struct {
		name     string
		respBody string
	}{
		{"model not found", "model not found"},
		{"Model Not Found uppercase", "Model Not Found"},
		{"model does not exist", "model does not exist"},
		{"Model Does Not Exist", "Model Does Not Exist"},
		{"unsupported model", "unsupported model"},
		{"UNSUPPORTED MODEL", "UNSUPPORTED MODEL"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(
				http.StatusInternalServerError,
				[]byte(tt.respBody),
			)

			assert.False(t, shouldRetry)
			assert.Equal(t, RetryReason(""), reason)
		})
	}
}

func TestShouldRetryWithFallback_RetryableInfrastructureError(t *testing.T) {
	// Regular infrastructure errors should be retried
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusServiceUnavailable,
		[]byte("service temporarily unavailable"),
	)

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestShouldRetryWithFallback_RateLimitWithContentPolicy(t *testing.T) {
	// If response contains both rate limit AND content policy, content policy wins
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusTooManyRequests,
		[]byte("content policy violation during rate limit"),
	)

	assert.False(t, shouldRetry)
	assert.Equal(t, RetryReason(""), reason)
}

func TestShouldRetryWithFallback_EmptyResponseBody(t *testing.T) {
	// Empty body should be treated as retryable for retryable status codes
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusInternalServerError,
		[]byte(""),
	)

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestIsRetryableContent_ContentPolicyViolation(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{"content policy lowercase", "content policy violation", false},
		{"content policy uppercase", "CONTENT POLICY VIOLATION", false},
		{"content policy mixed", "Content Policy Violation", false},
		{"content management policy", "content management policy violation", false},
		{"policy violation", "policy violation detected", false},
		{"no violation", "server error", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableContent([]byte(tt.content))
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsRetryableContent_ModelErrors(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{"model not found", "model not found", false},
		{"Model Not Found uppercase", "MODEL NOT FOUND", false},
		{"model does not exist", "model does not exist", false},
		{"Model Does Not Exist", "MODEL DOES NOT EXIST", false},
		{"unsupported model", "unsupported model gpt-4", false},
		{"Unsupported Model", "UNSUPPORTED MODEL", false},
		{"other error", "validation error", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableContent([]byte(tt.content))
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsRetryableContent_CaseInsensitive(t *testing.T) {
	// Verify case-insensitive matching works across all patterns
	testCases := []string{
		"Model not Found",
		"MODEL NOT FOUND",
		"Content POLICY violation",
		"CONTENT management POLICY",
		"POLICY VIOLATION",
		"Unsupported MODEL",
		"MODEL DOES NOT EXIST",
	}

	for _, tc := range testCases {
		result := isRetryableContent([]byte(tc))
		assert.False(t, result, "should not be retryable for: %s", tc)
	}
}

func TestRetryReasonConstants(t *testing.T) {
	// Verify retry reason constants are defined
	assert.Equal(t, RetryReason("rate_limit"), RetryReasonRateLimit)
	assert.Equal(t, RetryReason("server_error"), RetryReasonServerErr)
	assert.Equal(t, RetryReason("auth_error"), RetryReasonAuthErr)
	assert.Equal(t, RetryReason("network_error"), RetryReasonNetErr)
}

func TestTryFallbackProxy_Success(t *testing.T) {
	// Track number of calls to fallback server
	var fallbackCalls int32

	// Create fallback server mock that returns 200 OK
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		_ = NewResponseBuilder().
			WithStatus(http.StatusOK).
			WithJSONBody(createMockChatCompletionResponse(
				"chatcmpl-test-fallback",
				"gpt-4",
				"fallback ok",
			)).
			Write(w)
	}))
	defer fallbackServer.Close()

	// Build proxy with primary + fallback credentials
	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.local", fallbackServer.URL).
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

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",              // modelID
		"primary",            // originalCredName
		http.StatusOK,        // originalStatus
		RetryReasonRateLimit, // originalReason
		bodyBytes,            // body
		time.Now().UTC(),     // start
		nil,                  // logCtx
	)

	// Assertions
	assert.True(t, success, "TryFallbackProxy should return success=true")
	assert.Empty(t, reason, "TryFallbackProxy should return empty reason on success")

	// Verify fallback server was called
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallbackCalls), "Fallback server should be called exactly once")

	// Check response recorder
	assert.Equal(t, http.StatusOK, w.Code, "Response status code should be 200 OK")
	assert.NotEmpty(t, w.Body.String(), "Response body should not be empty")

	// Verify response contains expected data
	var respData map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &respData)
	require.NoError(t, err, "Failed to unmarshal response")
	assert.Equal(t, "chatcmpl-test-fallback", respData["id"])
	assert.Equal(t, "gpt-4", respData["model"])
}

func TestTryFallbackProxy_NoFallbackAvailable(t *testing.T) {
	// Build proxy with only primary credential (no fallback)
	prx := NewTestProxyBuilder().
		WithSingleCredential(
			"primary",
			config.ProviderTypeProxy,
			"http://primary.local",
			"pkey",
		).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{
		"model": "gpt-4",
	}
	bodyBytes, _ := json.Marshal(requestBody)

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer sk-master")

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy (should fail to find fallback)
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",
		"primary",
		http.StatusOK,
		RetryReasonRateLimit,
		bodyBytes,
		time.Now().UTC(),
		nil,
	)

	// Assertions
	assert.False(t, success, "TryFallbackProxy should return success=false when no fallback available")
	assert.Equal(t, "no_fallback_available", reason, "Should return no_fallback_available reason")
}

func TestFormatTriedCreds(t *testing.T) {
	tests := []struct {
		name    string
		input   map[string]bool
		checkFn func(t *testing.T, result string)
	}{
		{
			name:  "nil slice returns none",
			input: nil,
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "none", result)
			},
		},
		{
			name:  "empty map returns none",
			input: map[string]bool{},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "none", result)
			},
		},
		{
			name:  "single entry",
			input: map[string]bool{"cred-a": true},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "[[cred-a]]", result)
			},
		},
		{
			name:  "multiple entries",
			input: map[string]bool{"cred-a": true, "cred-b": true},
			checkFn: func(t *testing.T, result string) {
				// Map iteration order is non-deterministic, so check both possibilities
				assert.True(t,
					result == "[[cred-a cred-b]]" || result == "[[cred-b cred-a]]",
					"unexpected result: %s", result)
			},
		},
		{
			name:  "entry with false value is excluded",
			input: map[string]bool{"cred-a": true, "cred-b": false},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "[[cred-a]]", result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatTriedCreds(tt.input)
			tt.checkFn(t, result)
		})
	}
}

func TestTryFallbackProxy_SameCredentialAsOriginal(t *testing.T) {
	// Build proxy with single credential marked as both primary and fallback (edge case)
	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "pkey",
				BaseURL:    "http://primary.local",
				RPM:        100,
				TPM:        10000,
				IsFallback: true, // Edge case: marked as fallback
			},
		).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{"model": "gpt-4"}
	bodyBytes, _ := json.Marshal(requestBody)

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy (should detect same credential)
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",
		"primary",
		http.StatusOK,
		RetryReasonRateLimit,
		bodyBytes,
		time.Now().UTC(),
		nil,
	)

	// Assertions
	assert.False(t, success, "TryFallbackProxy should return success=false when fallback is same credential")
	assert.Equal(t, "fallback_is_same_credential", reason, "Should return fallback_is_same_credential reason")
}
