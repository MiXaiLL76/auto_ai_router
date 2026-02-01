package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
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
