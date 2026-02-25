package proxy

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// RetryReason describes why a request is being retried
type RetryReason string

const (
	RetryReasonRateLimit RetryReason = "rate_limit"
	RetryReasonServerErr RetryReason = "server_error"
	RetryReasonAuthErr   RetryReason = "auth_error"
	RetryReasonNetErr    RetryReason = "network_error"
)

// TriedCredentialsKey is the context key for tracking attempted credentials
// Prevents circular retries (proxy-a -> proxy-b -> proxy-a)
// Exported for use in other proxy package functions
type TriedCredentialsKey struct{}

// AttemptCountKey is the context key for tracking the number of credential attempts
type AttemptCountKey struct{}

// MaxRetryAttempts defines the maximum number of credential attempts per request
// Value: 2 = primary credential + 1 fallback
const MaxRetryAttempts = 2

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

// GetTried gets the set of tried credentials from context.
// Returns an empty map if not found (new request without context).
func GetTried(ctx context.Context) map[string]bool {
	if tried, ok := ctx.Value(TriedCredentialsKey{}).(map[string]bool); ok {
		return tried
	}
	return make(map[string]bool)
}

// SetTried stores the set of tried credentials in context.
func SetTried(ctx context.Context, tried map[string]bool) context.Context {
	return context.WithValue(ctx, TriedCredentialsKey{}, tried)
}

// incrementAttempts safely increments the attempt counter in context.
// Returns the updated attempt count.
func incrementAttempts(ctx context.Context) (int, context.Context) {
	// Get current count (key: "__attempt_count")
	currentCount := 0
	if count, ok := ctx.Value(AttemptCountKey{}).(int); ok {
		currentCount = count
	}
	newCount := currentCount + 1
	newCtx := context.WithValue(ctx, AttemptCountKey{}, newCount)
	return newCount, newCtx
}

// TryFallbackProxy attempts to retry the request on a fallback proxy credential.
// Returns (success, fallbackReason) where fallbackReason explains why fallback wasn't attempted.
//
// Protection against infinite loops:
// - Tracks attempted credentials in request context
// - Prevents circular retries (proxy-a -> proxy-b -> proxy-a)
// - Enforces max retry limit of 2 total attempts (primary + 1 fallback)
// - Validates that fallback differs from already-tried credentials
func (p *Proxy) TryFallbackProxy(
	w http.ResponseWriter,
	r *http.Request,
	modelID string,
	originalCredName string,
	originalStatus int,
	originalReason RetryReason,
	body []byte,
	start time.Time,
	logCtx *RequestLogContext,
) (bool, string) {
	ctx := r.Context()

	// Check attempt count - max 2 total attempts (primary + 1 fallback)
	attemptCount, ctx := incrementAttempts(ctx)
	if attemptCount >= MaxRetryAttempts {
		p.logger.Warn("Max retry attempts reached, not attempting additional fallback",
			"original_credential", originalCredName,
			"model", modelID,
			"attempt_count", attemptCount,
			"max_attempts", MaxRetryAttempts,
		)
		return false, "max_retry_attempts_exceeded"
	}

	// Get set of already-tried credentials from context
	triedCreds := GetTried(ctx)

	// Try to find a fallback proxy credential
	fallbackCred, err := p.balancer.NextFallbackProxyForModel(modelID)
	if err != nil {
		p.logger.Debug("No fallback proxy available for retry",
			"original_credential", originalCredName,
			"model", modelID,
			"original_status", originalStatus,
			"reason", originalReason,
		)
		return false, "no_fallback_available"
	}

	// Guard against nil credential (balancer returned no error but also no credential)
	if fallbackCred == nil {
		p.logger.Warn("Balancer returned nil credential without error",
			"model", modelID,
			"original_credential", originalCredName,
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

	// Check if fallback credential has already been tried in this request chain
	if triedCreds[fallbackCred.Name] {
		p.logger.Warn("Fallback credential already attempted, skipping to prevent circular retry",
			"fallback_credential", fallbackCred.Name,
			"tried_credentials", formatTriedCreds(triedCreds),
			"model", modelID,
		)
		return false, "credential_already_tried"
	}

	p.logger.Info("Retrying request on fallback proxy",
		"original_credential", originalCredName,
		"fallback_credential", fallbackCred.Name,
		"model", modelID,
		"original_status", originalStatus,
		"retry_reason", originalReason,
		"attempt_number", attemptCount+1,
		"max_attempts", MaxRetryAttempts,
	)

	// Add fallback credential to tried set
	triedCreds[fallbackCred.Name] = true
	ctx = SetTried(ctx, triedCreds)

	// Create new request with updated context for fallback attempt
	r = r.WithContext(ctx)

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

	if proxyResp.IsStreaming {
		totalTokens, err := p.writeProxyStreamingResponseWithTokens(w, proxyResp, r, fallbackCred.Name)
		if err != nil {
			p.logger.Error("Failed to write fallback streaming proxy response",
				"fallback_credential", fallbackCred.Name,
				"error", err,
			)
			return false, "fallback_stream_write_failed"
		}
		if totalTokens > 0 {
			p.rateLimiter.ConsumeTokens(fallbackCred.Name, totalTokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(fallbackCred.Name, modelID, totalTokens)
			}
			p.logger.Debug("Fallback proxy streaming token usage recorded",
				"fallback_credential", fallbackCred.Name,
				"model", modelID,
				"tokens", totalTokens,
			)
		}
	} else {
		p.writeProxyResponse(w, proxyResp, r)
		tokens := extractTokensFromResponse(string(proxyResp.Body), config.ProviderTypeOpenAI)
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(fallbackCred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(fallbackCred.Name, modelID, tokens)
			}
			p.logger.Debug("Fallback proxy token usage recorded",
				"fallback_credential", fallbackCred.Name,
				"model", modelID,
				"tokens", tokens,
			)
		}
	}

	// Log that retry was completed
	p.logger.Debug("Fallback proxy retry completed",
		"fallback_credential", fallbackCred.Name,
		"duration", time.Since(start),
	)

	// Log fallback response to LiteLLM DB
	if logCtx != nil && !logCtx.Logged {
		// Update logCtx with fallback credential info
		logCtx.Credential = fallbackCred
		logCtx.TargetURL = fallbackCred.BaseURL
		logCtx.Status = "success"
		if proxyResp.StatusCode >= 400 {
			logCtx.Status = "failure"
		}
		logCtx.HTTPStatus = proxyResp.StatusCode
		logCtx.Logged = true

		if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
			p.logger.Warn("Failed to queue fallback spend log",
				"error", err,
				"request_id", logCtx.RequestID,
				"fallback_credential", fallbackCred.Name,
			)
		}
	}

	return true, ""
}

// formatTriedCreds converts the tried credentials map to a readable string
func formatTriedCreds(tried map[string]bool) string {
	var creds []string
	for cred := range tried {
		if tried[cred] {
			creds = append(creds, cred)
		}
	}
	if len(creds) == 0 {
		return "none"
	}
	return fmt.Sprintf("[%v]", creds)
}
