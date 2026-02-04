package proxy

import (
	"bytes"
	"fmt"
	"math/rand"
	"net/http"
	"time"
)

// RetryReason describes why a request is being retried
type RetryReason string

const (
	RetryReasonRateLimit RetryReason = "rate_limit"
	RetryReasonServerErr RetryReason = "server_error"
	RetryReasonAuthErr   RetryReason = "auth_error"
	RetryReasonNetErr    RetryReason = "network_error"
)

// ShouldRetryWithFallback determines if request should be retried based on status code and response body.
// Returns (shouldRetry, reason)
func ShouldRetryWithFallback(statusCode int, respBody []byte) (bool, RetryReason) {
	// Determine if status code is retryable
	var retryReason RetryReason
	switch {
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		retryReason = RetryReasonAuthErr
	case statusCode == http.StatusTooManyRequests:
		retryReason = RetryReasonRateLimit
	case statusCode >= 500 && statusCode < 600:
		retryReason = RetryReasonServerErr
	default:
		return false, ""
	}

	// Check if response body contains non-retryable errors
	if !isRetryableContent(respBody) {
		return false, ""
	}

	return true, retryReason
}

// isRetryableContent checks if response body contains errors that shouldn't be retried.
// This is a helper function extracted for DRY compliance.
func isRetryableContent(respBody []byte) bool {
	const maxRetryBodyScan = 8 * 1024
	if len(respBody) > maxRetryBodyScan {
		respBody = respBody[:maxRetryBodyScan]
	}
	bodyLower := bytes.ToLower(respBody)

	// Don't retry if content policy violation (provider-specific business logic error)
	if bytes.Contains(bodyLower, []byte("content policy")) ||
		bytes.Contains(bodyLower, []byte("content management policy")) ||
		bytes.Contains(bodyLower, []byte("policy violation")) {
		return false
	}

	// Don't retry if it's a model-specific error that won't be fixed by retrying
	if bytes.Contains(bodyLower, []byte("model not found")) ||
		bytes.Contains(bodyLower, []byte("model does not exist")) ||
		bytes.Contains(bodyLower, []byte("unsupported model")) {
		return false
	}

	// Otherwise, it's potentially retryable (infrastructure error, account issue, etc)
	return true
}

// TryFallbackProxy attempts to retry the request on a fallback proxy credential.
// Returns (success, fallbackReason) where fallbackReason explains why fallback wasn't attempted.
func (p *Proxy) TryFallbackProxy(
	w http.ResponseWriter,
	r *http.Request,
	modelID string,
	originalCredName string,
	originalStatus int,
	originalReason RetryReason,
	body []byte,
	start time.Time,
) (bool, string) {
	// Try to find a fallback proxy credential
	fallbackCred, err := p.balancer.NextFallbackForModel(modelID)
	if err != nil {
		p.logger.Debug("No fallback proxy available for retry",
			"original_credential", originalCredName,
			"model", modelID,
			"original_status", originalStatus,
			"reason", originalReason,
		)
		return false, "no_fallback_available"
	}

	// Safety check: don't retry with the same credential
	if fallbackCred.Name == originalCredName {
		p.logger.Warn("Fallback credential is the same as original, skipping retry",
			"credential", fallbackCred.Name,
			"model", modelID,
		)
		return false, "fallback_is_same_credential"
	}

	p.logger.Info("Retrying request on fallback proxy",
		"original_credential", originalCredName,
		"fallback_credential", fallbackCred.Name,
		"model", modelID,
		"original_status", originalStatus,
		"retry_reason", originalReason,
	)

	// Add jitter (0-50ms) to prevent thundering herd when multiple requests fail simultaneously
	jitter := time.Duration(rand.Intn(50)) * time.Millisecond
	time.Sleep(jitter)

	// Forward request to fallback proxy
	proxyResp, err := p.forwardToProxy(w, r, modelID, fallbackCred, body, start)
	if err != nil {
		p.logger.Error("Fallback proxy request failed",
			"fallback_credential", fallbackCred.Name,
			"error", err,
		)
		return false, "fallback_request_failed"
	}

	// Write response headers (skip hop-by-hop headers)
	for key, values := range proxyResp.Headers {
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length and Transfer-Encoding - we'll set them correctly
		if key == "Content-Length" || key == "Transfer-Encoding" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set correct Content-Length for the actual response body
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(proxyResp.Body)))

	w.WriteHeader(proxyResp.StatusCode)
	if _, err := w.Write(proxyResp.Body); err != nil {
		p.logger.Error("Failed to write fallback proxy response body", "error", err)
	}

	// Log that retry was completed
	p.logger.Debug("Fallback proxy retry completed",
		"fallback_credential", fallbackCred.Name,
		"duration", time.Since(start),
	)

	return true, ""
}
