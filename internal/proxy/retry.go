package proxy

import (
	"bytes"
	"context"
	"fmt"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// RetryReason describes why a request is being retried
type RetryReason string

const (
	RetryReasonRateLimit  RetryReason = "rate_limit"
	RetryReasonServerErr  RetryReason = "server_error"
	RetryReasonAuthErr    RetryReason = "auth_error"
	RetryReasonNetErr     RetryReason = "network_error"
	RetryReasonPaymentErr RetryReason = "payment_error"
)

// TriedCredentialsKey is the context key for tracking attempted credentials
// Prevents circular retries (proxy-a -> proxy-b -> proxy-a)
// Exported for use in other proxy package functions
type TriedCredentialsKey struct{}

// AttemptCountKey is the context key for tracking the number of credential attempts
type AttemptCountKey struct{}

// defaultMaxFallbackAttempts is the fallback value used when Proxy.maxFallbackAttempts is 0.
const defaultMaxFallbackAttempts = 5

// ShouldRetryWithFallback determines if request should be retried based on status code and response body.
// Returns (shouldRetry, reason)
func ShouldRetryWithFallback(statusCode int, respBody []byte) (bool, RetryReason) {
	// Determine if status code is retryable
	var retryReason RetryReason
	switch {
	case statusCode == http.StatusBadRequest || statusCode == http.StatusNotFound:
		retryReason = RetryReasonServerErr
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		retryReason = RetryReasonAuthErr
	case statusCode == http.StatusPaymentRequired:
		retryReason = RetryReasonPaymentErr
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

	return true
}

// setRetryAfterFromBan sets the Retry-After header for a 429 response to
// modelID's caller. It prefers the shortest remaining fail2ban ban among
// modelID's eligible credentials (a precise ETA); if none of them are
// currently banned it falls back to a configured or default duration via
// DefaultRetryAfterForModel, so a 429 reaching the client always carries a
// Retry-After header.
func (p *Proxy) setRetryAfterFromBan(w http.ResponseWriter, modelID string, exclude map[string]bool, visibility scope.Context) {
	remaining := p.balancer.DefaultRetryAfterForModel(modelID, exclude, visibility)
	seconds := int(remaining / time.Second)
	if remaining%time.Second != 0 {
		seconds++
	}
	if seconds < 1 {
		seconds = 1
	}
	w.Header().Set("Retry-After", fmt.Sprintf("%d", seconds))
}

// ensureRetryAfterOn429 sets a Retry-After header on the client response for
// a 429 that is about to be relayed WITHOUT going through writeFallbackResponse
// — specifically, when TryFallbackProxy found no fallback credential to even
// attempt (e.g. the model has none configured), so the original upstream 429
// falls straight through to the normal response-writing path instead. Same
// guarantee as writeFallbackResponse's inline check: never override a
// Retry-After the upstream already sent, and never omit one.
func (p *Proxy) ensureRetryAfterOn429(w http.ResponseWriter, statusCode int, upstreamHeaders http.Header, modelID string, visibility scope.Context) {
	if statusCode != http.StatusTooManyRequests {
		return
	}
	if upstreamHeaders != nil && upstreamHeaders.Get("Retry-After") != "" {
		return
	}
	p.setRetryAfterFromBan(w, modelID, nil, visibility)
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

// TryFallbackProxy attempts to retry the request on fallback proxy credentials.
// Returns (success, fallbackReason) where fallbackReason explains why all fallbacks failed.
//
// Protection against infinite loops:
// - Tracks attempted credentials in request context (triedCreds)
// - Prevents circular retries (proxy-a -> proxy-b -> proxy-a)
// - Enforces MaxFallbackAttempts as an upper bound
//
// When a fallback returns a retryable error (429, 5xx), the next configured fallback
// is tried automatically, exhausting the full chain before writing the final response.
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
	triedCreds := GetTried(ctx)

	maxAttempts := p.maxFallbackAttempts
	if maxAttempts <= 0 {
		maxAttempts = defaultMaxFallbackAttempts
	}

	exitReason := "no_fallback_available"
	var lastProxyResp *ProxyResponse
	var lastFallbackCred *config.CredentialConfig

	for attempt := 0; attempt < maxAttempts; attempt++ {
		visibility := scope.PublicContext()
		if logCtx != nil {
			visibility = logCtx.Scope
		}
		fallbackCred, err := p.balancer.NextFallbackProxyForModelExcludingScoped(modelID, triedCreds, visibility)
		if err != nil {
			if attempt == 0 {
				p.logger.DebugContext(r.Context(), "No fallback proxy available for retry",
					"original_credential", originalCredName,
					"model", modelID,
					"original_status", originalStatus,
					"reason", originalReason,
				)
			}
			break
		}

		if fallbackCred == nil {
			p.logger.WarnContext(r.Context(), "Balancer returned nil credential without error",
				"model", modelID,
				"original_credential", originalCredName,
			)
			break
		}

		if fallbackCred.Name == originalCredName {
			if attempt == 0 {
				p.logger.WarnContext(r.Context(), "Fallback credential is the same as original, skipping retry",
					"credential", fallbackCred.Name,
					"model", modelID,
				)
				exitReason = "fallback_is_same_credential"
			}
			break
		}

		if triedCreds[fallbackCred.Name] {
			p.logger.DebugContext(r.Context(), "All fallback proxies exhausted, stopping retry chain",
				"tried_credentials", formatTriedCreds(triedCreds),
				"model", modelID,
			)
			break
		}

		p.logger.InfoContext(r.Context(), "Retrying request on fallback proxy",
			"original_credential", originalCredName,
			"fallback_credential", fallbackCred.Name,
			"model", modelID,
			"original_status", originalStatus,
			"retry_reason", originalReason,
			"attempt_number", attempt+2,
			"max_attempts", maxAttempts+1,
		)

		triedCreds[fallbackCred.Name] = true
		ctx = SetTried(ctx, triedCreds)
		r = r.WithContext(ctx)

		// Add jitter (0-50ms) to prevent thundering herd when multiple requests fail simultaneously
		jitter := time.Duration(rand.IntN(50)) * time.Millisecond
		time.Sleep(jitter)
		proxyResp, fwdErr := p.forwardToProxy(w, r, modelID, fallbackCred, body, start)
		if fwdErr != nil {
			// Mid-chain failure — the next fallback is tried; the final outcome
			// is logged at ERROR when the response is written to the client.
			p.logger.WarnContext(r.Context(), "Fallback proxy request failed, trying next fallback",
				"fallback_credential", fallbackCred.Name,
				"model", modelID,
				"error", fwdErr,
			)
			continue
		}
		if proxyResp == nil {
			continue
		}
		lastFallbackCred = fallbackCred
		if logCtx != nil {
			// ActualCredentialName belongs to this exact response. Assigning the
			// empty value is intentional: a prior retry's nested credential must
			// not leak into a later response that did not report one.
			logCtx.ActualCredentialName = proxyResp.ActualCredentialName
		}
		lastProxyResp = proxyResp
		// Streaming responses cannot be retried — write immediately.
		if proxyResp.IsStreaming {
			return p.writeFallbackResponse(w, r, proxyResp, fallbackCred, modelID, originalCredName, logCtx, start)
		}

		shouldRetry, _ := ShouldRetryWithFallback(proxyResp.StatusCode, proxyResp.Body)
		if !shouldRetry {
			return p.writeFallbackResponse(w, r, proxyResp, fallbackCred, modelID, originalCredName, logCtx, start)
		}

		p.logger.WarnContext(r.Context(), "Fallback credential returned retryable error, trying next fallback",
			"fallback_credential", fallbackCred.Name,
			"status", proxyResp.StatusCode,
			"model", modelID,
		)
	}

	// All fallbacks exhausted — write last response if we have one.
	if lastProxyResp != nil && lastFallbackCred != nil {
		return p.writeFallbackResponse(w, r, lastProxyResp, lastFallbackCred, modelID, originalCredName, logCtx, start)
	}

	return false, exitReason
}

// writeFallbackResponse writes the proxy response to the client, records token usage,
// and updates logCtx. Called once when a fallback succeeds or all fallbacks are exhausted.
func (p *Proxy) writeFallbackResponse(
	w http.ResponseWriter,
	r *http.Request,
	proxyResp *ProxyResponse,
	fallbackCred *config.CredentialConfig,
	modelID string,
	originalCredName string,
	logCtx *RequestLogContext,
	start time.Time,
) (bool, string) {
	if logCtx != nil {
		logCtx.Credential = fallbackCred
		logCtx.TargetURL = fallbackCred.BaseURL
		logCtx.HTTPStatus = proxyResp.StatusCode
		logCtx.Status = "success"
		if proxyResp.StatusCode >= http.StatusBadRequest {
			logCtx.Status = "failure"
		}
	}

	// Client-facing outcome decided — this is the response actually written to
	// the client below, whether the fallback chain succeeded or exhausted all
	// attempts. Recorded exactly once here (not per fallback attempt) with
	// genuine end-to-end duration since the original client request arrived.
	p.metrics.RecordRequest(fallbackCred.Name, r.URL.Path, modelID, proxyResp.StatusCode, time.Since(start))

	if proxyResp.StatusCode >= 400 {
		// Final error returned to the client after the fallback chain —
		// single unified ERROR record (response_body is nil for streaming).
		requestID := ""
		if logCtx != nil {
			requestID = logCtx.RequestID
		}
		p.logUpstreamError(r.Context(), "Fallback proxy completed with error status", proxyResp.StatusCode, fallbackCred, modelID, proxyResp.Body,
			"url", fallbackCred.BaseURL,
			"streaming", proxyResp.IsStreaming,
			"original_credential", originalCredName,
			"request_id", requestID)
	}

	// Computed once: both branches below need the same audio-usage-contract
	// derived from the fallback upstream's response, regardless of streaming.
	usageOptions := tokenUsageExtractionOptionsForResponse(fallbackCred, proxyResp.Headers)

	// The fallback chain is exhausted and this 429 is a real upstream response
	// being relayed as-is — unlike the self-generated "no credentials available"
	// 429 in selectCredentialForModel, nothing here has computed a Retry-After
	// hint yet. Add one from the shortest active ban, but only if the upstream
	// didn't already send its own (never override a provider's own guidance).
	if proxyResp.StatusCode == http.StatusTooManyRequests && proxyResp.Headers.Get("Retry-After") == "" {
		visibility := scope.PublicContext()
		if logCtx != nil {
			visibility = logCtx.Scope
		}
		p.setRetryAfterFromBan(w, modelID, nil, visibility)
	}

	if proxyResp.IsStreaming {
		p.setCredentialResponseHeader(w, logCtx, "")
		streamUsage, err := p.writeProxyStreamingResponseWithTokens(
			w, proxyResp, r, fallbackCred, modelID, modelID, logCtx, usageOptions,
		)
		if err != nil {
			markStreamFailure(logCtx, err)
			p.logStreamHandlerError(r.Context(), "Failed to write fallback streaming proxy response", err,
				"fallback_credential", fallbackCred.Name,
				"model", modelID,
			)
			// WriteHeader was already sent by writeProxyStreamingResponseWithTokens before
			// the stream body failed — return true so the caller does not attempt another
			// WriteHeader call (which would produce a "superfluous WriteHeader" warning and
			// corrupt the response).
			// Still propagate partial token usage so the defer-logged spend entry isn't empty.
			if streamUsage != nil && logCtx != nil {
				if streamUsage.PromptTokens == 0 && logCtx.UsageSource != "provider" {
					if estimate := logCtx.promptTokensEstimate(); estimate > 0 {
						streamUsage.PromptTokens = estimate
					}
				}
				logCtx.TokenUsage = streamUsage
			}
			if logCtx != nil && !logCtx.Logged {
				logCtx.Logged = true
				if queueErr := p.logSpendToLiteLLMDB(logCtx); queueErr != nil {
					p.logger.WarnContext(r.Context(), "Failed to queue fallback stream failure spend log",
						"error", queueErr,
						"request_id", logCtx.RequestID,
						"fallback_credential", fallbackCred.Name,
					)
				}
			}
			return true, "fallback_stream_write_failed"
		}
		if streamUsage != nil && logCtx != nil {
			// Backfill PromptTokens from estimate when provider didn't include it.
			if streamUsage.PromptTokens == 0 && logCtx.UsageSource != "provider" {
				if estimate := logCtx.promptTokensEstimate(); estimate > 0 {
					streamUsage.PromptTokens = estimate
				}
			}
			logCtx.TokenUsage = streamUsage
			if proxyResp.StatusCode < 400 {
				p.metrics.RecordTokenUsage(fallbackCred.Name, modelID,
					streamUsage.PromptTokens, streamUsage.CompletionTokens,
					streamUsage.ReasoningTokens, streamUsage.CachedInputTokens)
				totalTokens := streamUsage.Total()
				if totalTokens > 0 {
					p.rateLimiter.ConsumeTokens(fallbackCred.Name, totalTokens)
					if modelID != "" {
						p.rateLimiter.ConsumeModelTokens(fallbackCred.Name, modelID, totalTokens)
					}
				}
				p.logger.DebugContext(r.Context(), "Fallback proxy streaming token usage recorded",
					"fallback_credential", fallbackCred.Name,
					"model", modelID,
					"prompt_tokens", streamUsage.PromptTokens,
					"completion_tokens", streamUsage.CompletionTokens,
				)
			}
		}
	} else {
		p.setCredentialResponseHeader(w, logCtx, "")
		// Single shared decode for both token-accounting consumers (plan item
		// G) instead of two independent full-body Unmarshals of the same
		// bytes: the rate-limiter's total-tokens count and the spend-logging
		// TokenUsage are genuinely different numbers computed off the same
		// decode, not derived from each other — see
		// converter.ExtractTotalTokensAndUsageWithOptions's doc comment.
		tokens, usage := extractOpenAITokensAndUsage(proxyResp.Body, usageOptions)
		if logCtx != nil {
			logCtx.TokenUsage = usage
		}
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(fallbackCred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(fallbackCred.Name, modelID, tokens)
			}
			p.logger.DebugContext(r.Context(), "Fallback proxy token usage recorded",
				"fallback_credential", fallbackCred.Name,
				"model", modelID,
				"tokens", tokens,
			)
		}
	}

	p.logger.DebugContext(r.Context(), "Fallback proxy retry completed",
		"fallback_credential", fallbackCred.Name,
		"duration", time.Since(start),
	)

	if logCtx != nil && !logCtx.Logged {
		if logCtx.Status == "failure" && !proxyResp.IsStreaming {
			logCtx.ErrorMsg = extractErrorMessage(proxyResp.Body)
		}
		if proxyResp.IsStreaming {
			logCtx.Logged = true
			if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
				p.logger.WarnContext(r.Context(), "Failed to queue fallback spend log",
					"error", err,
					"request_id", logCtx.RequestID,
					"fallback_credential", fallbackCred.Name,
				)
			}
		}
	}
	if !proxyResp.IsStreaming {
		p.writeProxyResponse(w, proxyResp, r, fallbackCred, modelID, logCtx, usageOptions)
	}
	if logCtx != nil && logCtx.Status == "success" {
		logCtx.RequestCompleted = true
		p.setSessionBinding(logCtx.SessionID, modelID, fallbackCred.Name)
		p.logger.DebugContext(r.Context(), "Session-sticky routing: updated session after failover",
			"session_id", logCtx.SessionID,
			"old_credential", originalCredName,
			"new_credential", fallbackCred.Name,
			"model", modelID,
		)
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
