package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"syscall"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
)

// ErrResponseBodyTooLarge is returned when a response body exceeds the configured size limit.
var ErrResponseBodyTooLarge = errors.New("response body too large")

// isTimeoutError checks if an error is a timeout error
func isTimeoutError(err error) bool {
	if err == nil {
		return false
	}

	// Check for context deadline exceeded
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	// Check for net.Error timeout
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	return false
}

// isClientDisconnectError checks if an error indicates the client disconnected
// (broken pipe, connection reset, context canceled). These are expected during
// normal operation and should be logged at lower severity.
func isClientDisconnectError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return true
	}
	if errors.Is(err, syscall.EPIPE) {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "write: broken pipe") ||
		strings.Contains(msg, "connection reset by peer")
}

func (p *Proxy) recordAbortedRequest(credential, endpoint, model string) {
	if p == nil || p.metrics == nil {
		return
	}
	p.metrics.RecordAbortedRequest(credential, endpoint, model)
}

func metricModelID(fallback string, logCtx *RequestLogContext) string {
	if logCtx != nil && logCtx.ModelID != "" {
		return logCtx.ModelID
	}
	return fallback
}

func endpointFromLogContext(logCtx *RequestLogContext) string {
	if logCtx == nil {
		return ""
	}
	return endpointFromRequest(logCtx.Request)
}

func endpointFromRequest(r *http.Request) string {
	if r != nil && r.URL != nil {
		return r.URL.Path
	}
	return ""
}

// extractErrorMessage returns the raw error response body as a string
// The HTTP status code is captured separately in error_code
func extractErrorMessage(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	// Return raw body (truncate if too large)
	const maxLen = 512
	if len(body) > maxLen {
		return string(body[:maxLen]) + "..."
	}
	return string(body)
}

// mapHTTPStatusToErrorClass maps HTTP status codes to LiteLLM exception class names
// Reference: https://docs.litellm.ai/docs/exception_mapping
func mapHTTPStatusToErrorClass(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "BadRequestError"
	case http.StatusUnauthorized:
		return "AuthenticationError"
	case http.StatusForbidden:
		return "PermissionDeniedError"
	case http.StatusNotFound:
		return "NotFoundError"
	case http.StatusRequestTimeout:
		return "Timeout"
	case http.StatusUnprocessableEntity:
		return "UnprocessableEntityError"
	case http.StatusTooManyRequests:
		return "RateLimitError"
	case http.StatusServiceUnavailable:
		return "ServiceUnavailableError"
	case http.StatusInternalServerError:
		return "InternalServerError"
	default:
		if statusCode >= 400 && statusCode < 500 {
			return "BadRequestError"
		} else if statusCode >= 500 {
			return "APIConnectionError"
		}
		return "APIError"
	}
}

func (logCtx *RequestLogContext) applyWebSearchUsageDefaults(status string) {
	if logCtx == nil || status != "success" {
		return
	}
	if logCtx.TokenUsage == nil {
		logCtx.TokenUsage = &converter.TokenUsage{}
	}
	if logCtx.WebSearchContextSize != "" {
		logCtx.TokenUsage.WebSearchContextSize = logCtx.WebSearchContextSize
	}
}

// buildMetadata builds metadata JSON with user/team alias, usage, cost, and optional error info
func buildMetadata(hashedToken string, tokenInfo *litellmdb.TokenInfo, errorMsg string, httpStatus int, usage *converter.TokenUsage, requesterIP string, costs *converter.TokenCosts, modelID string, overheadMs float64, kafkaFallbackReason string) string {
	var userID, teamID, organizationID string
	if tokenInfo != nil {
		userID = tokenInfo.UserID
		teamID = tokenInfo.TeamID
		organizationID = tokenInfo.OrganizationID
	}

	// Build usage_object and additional_usage_values
	promptTokensDetails := map[string]interface{}{
		"text_tokens":                  nil,
		"audio_tokens":                 0,
		"image_tokens":                 nil,
		"cached_tokens":                0,
		"cached_audio_tokens":          0,
		"cache_creation_tokens":        0,
		"cache_creation_token_details": nil,
	}
	completionTokensDetails := map[string]interface{}{
		"text_tokens":                nil,
		"audio_tokens":               0,
		"image_tokens":               nil,
		"reasoning_tokens":           0,
		"accepted_prediction_tokens": 0,
		"rejected_prediction_tokens": 0,
	}
	serverToolUse := map[string]interface{}{
		"web_search_requests":     0,
		"web_search_context_size": nil,
	}

	var usageObject interface{}
	if usage != nil {
		normalizedUsage := *usage
		usage = normalizedUsage.Normalize()
		promptTokensDetails["audio_tokens"] = usage.AudioInputTokens
		promptTokensDetails["image_tokens"] = usage.ImageTokens
		promptTokensDetails["cached_tokens"] = usage.CachedInputTokens
		promptTokensDetails["cached_audio_tokens"] = usage.CachedAudioInputTokens
		promptTokensDetails["cache_creation_tokens"] = usage.CacheCreationTokens
		if usage.CacheCreation5mTokens > 0 || usage.CacheCreation1hTokens > 0 {
			promptTokensDetails["cache_creation_token_details"] = map[string]interface{}{
				"ephemeral_5m_input_tokens": usage.CacheCreation5mTokens,
				"ephemeral_1h_input_tokens": usage.CacheCreation1hTokens,
			}
		}
		completionTokensDetails["audio_tokens"] = usage.AudioOutputTokens
		if usage.OutputTextTokens > 0 {
			completionTokensDetails["text_tokens"] = usage.OutputTextTokens
		}
		completionTokensDetails["image_tokens"] = usage.OutputImageTokens
		completionTokensDetails["reasoning_tokens"] = usage.ReasoningTokens
		completionTokensDetails["accepted_prediction_tokens"] = usage.AcceptedPredictionTokens
		completionTokensDetails["rejected_prediction_tokens"] = usage.RejectedPredictionTokens
		serverToolUse["web_search_requests"] = usage.WebSearchRequests
		if usage.WebSearchRequests > 0 {
			serverToolUse["web_search_context_size"] = usage.WebSearchContextSize
		}
		usageObject = map[string]interface{}{
			"total_tokens":              usage.Total(),
			"prompt_tokens":             usage.PromptTokens,
			"completion_tokens":         usage.CompletionTokens,
			"prompt_tokens_details":     promptTokensDetails,
			"completion_tokens_details": completionTokensDetails,
			"server_tool_use":           serverToolUse,
		}
	}

	additionalUsage := map[string]interface{}{
		"prompt_tokens_details": map[string]interface{}{
			"audio_tokens":                 promptTokensDetails["audio_tokens"],
			"image_tokens":                 promptTokensDetails["image_tokens"],
			"cached_tokens":                promptTokensDetails["cached_tokens"],
			"cached_audio_tokens":          promptTokensDetails["cached_audio_tokens"],
			"cache_creation_tokens":        promptTokensDetails["cache_creation_tokens"],
			"cache_creation_token_details": promptTokensDetails["cache_creation_token_details"],
		},
		"completion_tokens_details": map[string]interface{}{
			"text_tokens":                completionTokensDetails["text_tokens"],
			"audio_tokens":               completionTokensDetails["audio_tokens"],
			"image_tokens":               completionTokensDetails["image_tokens"],
			"reasoning_tokens":           completionTokensDetails["reasoning_tokens"],
			"accepted_prediction_tokens": completionTokensDetails["accepted_prediction_tokens"],
			"rejected_prediction_tokens": completionTokensDetails["rejected_prediction_tokens"],
		},
		"server_tool_use": serverToolUse,
	}

	// Build cost_breakdown
	var costBreakdown interface{}
	if costs != nil {
		costBreakdown = map[string]interface{}{
			"input_cost":          costs.InputCost,
			"output_cost":         costs.OutputCost,
			"reasoning_cost":      costs.ReasoningCost,
			"cached_input_cost":   costs.CachedInputCost,
			"cache_creation_cost": costs.CacheCreationCost,
			"total_cost":          costs.TotalCost,
			"original_cost":       costs.TotalCost,
			"margin_percent":      0.0,
			"discount_amount":     0.0,
			"tool_usage_cost":     costs.WebSearchCost,
			"web_search_cost":     costs.WebSearchCost,
			"discount_percent":    0.0,
			"margin_fixed_amount": 0.0,
			"margin_total_amount": 0.0,
		}
	}

	metadata := map[string]interface{}{
		"batch_models":                  nil,
		"usage_object":                  usageObject,
		"user_api_key":                  hashedToken,
		"cost_breakdown":                costBreakdown,
		"applied_guardrails":            []interface{}{},
		"user_api_key_org_id":           organizationID,
		"requester_ip_address":          requesterIP,
		"user_api_key_team_id":          teamID,
		"user_api_key_user_id":          userID,
		"guardrail_information":         nil,
		"model_map_information":         map[string]interface{}{"model_map_key": modelID, "model_map_value": nil},
		"mcp_tool_call_metadata":        nil,
		"additional_usage_values":       additionalUsage,
		"cold_storage_object_key":       nil,
		"litellm_overhead_time_ms":      overheadMs,
		"vector_store_request_metadata": nil,
		"status":                        "success",
	}

	if tokenInfo != nil {
		if tokenInfo.KeyAlias != "" {
			metadata["user_api_key_alias"] = tokenInfo.KeyAlias
		}
		if tokenInfo.UserAlias != "" {
			metadata["user_api_key_user_alias"] = tokenInfo.UserAlias
		}
		if tokenInfo.TeamAlias != "" {
			metadata["user_api_key_team_alias"] = tokenInfo.TeamAlias
		}
	}

	if errorMsg != "" {
		metadata["error_information"] = map[string]interface{}{
			"error_message": errorMsg,
			"error_code":    httpStatus,
			"error_class":   mapHTTPStatusToErrorClass(httpStatus),
		}
		metadata["status"] = "failure"
	}

	// kafkaFallbackReason is set by the caller when publishing this event's
	// Kafka copy failed (e.g. queue full after the 5s backpressure wait, see
	// kafkalog.ErrQueueFull). Flagging it here — in the row that's about to be
	// inserted anyway — lets it be found later via metadata->>'kafka_fallback'
	// (e.g. by a DBA script) and re-published to Kafka, instead of the event
	// being lost entirely when Kafka is degraded. AIR intentionally does not
	// run its own resend job — see internal/litellmdb.Manager.MarkSpendLogKafkaFallback.
	if kafkaFallbackReason != "" {
		metadata["kafka_fallback"] = true
		metadata["kafka_fallback_reason"] = kafkaFallbackReason
	}

	jsonBytes, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Sprintf(`{"user_api_key":"%s","user_api_key_org_id":"%s","user_api_key_team_id":"%s","user_api_key_user_id":"%s"}`, hashedToken, organizationID, teamID, userID)
	}
	return string(jsonBytes)
}

func addAIRSpendMetadata(metadata, eventID, providerResponseID string, isProxyRequest bool, modelID, publicModelID, priceModelID string) string {
	var doc map[string]interface{}
	if json.Unmarshal([]byte(metadata), &doc) != nil || doc == nil {
		// Unmarshal leaves doc nil for a decode error and also for the literal
		// "null"; either way a fresh map is needed before assigning into it.
		doc = make(map[string]interface{})
	}
	spendMetadata, _ := doc["spend_logs_metadata"].(map[string]interface{})
	if spendMetadata == nil {
		spendMetadata = make(map[string]interface{})
		doc["spend_logs_metadata"] = spendMetadata
	}
	spendMetadata["air_event_id"] = eventID
	spendMetadata["accounting_eligible"] = !isProxyRequest
	spendMetadata["is_proxy_request"] = isProxyRequest
	if providerResponseID != "" {
		spendMetadata["provider_response_id"] = providerResponseID
	}
	// Only set when the client used a public alias (e.g. a "-highlimits"
	// pricing tier) that differs from the model billing was actually
	// resolved against (priceModelID, from resolveBillingPrice — falls back
	// to modelID when billing was never resolved for this request), so
	// spend rows can be told apart specifically when the alias changed which
	// price row was used, not merely whenever an alias was present.
	// Deliberately kept out of model_map_information: in upstream LiteLLM
	// that field's value is a model_info/pricing object resolved via
	// litellm.model_cost, not an identifier string, and overloading it
	// risks a type mismatch for any future consumer that parses it against
	// that contract.
	billedAgainst := priceModelID
	if billedAgainst == "" {
		billedAgainst = modelID
	}
	// publicModelID != modelID is required here: without it, a request that never
	// used an alias (publicModelID == modelID) can still get flagged whenever price
	// lookup resolves against realModelID instead of modelID — GetPriceAny may match
	// realModelID's price entry when modelID itself has none, making billedAgainst
	// diverge from publicModelID for a reason unrelated to aliasing.
	if publicModelID != "" && publicModelID != modelID && publicModelID != billedAgainst {
		spendMetadata["public_model_name"] = publicModelID
	}
	encoded, err := json.Marshal(doc)
	if err != nil {
		return metadata
	}
	return string(encoded)
}

func addRequestSpendMetadata(metadata string, logCtx *RequestLogContext) string {
	if logCtx == nil {
		return metadata
	}
	var doc map[string]interface{}
	if json.Unmarshal([]byte(metadata), &doc) != nil || doc == nil {
		doc = make(map[string]interface{})
	}
	spendMetadata, _ := doc["spend_logs_metadata"].(map[string]interface{})
	if spendMetadata == nil {
		spendMetadata = make(map[string]interface{})
		doc["spend_logs_metadata"] = spendMetadata
	}
	spendMetadata["reasoning_requested"] = logCtx.ReasoningRequested
	if logCtx.ReasoningSource != "" {
		spendMetadata["reasoning_source"] = logCtx.ReasoningSource
	}
	if logCtx.RequestEndpoint != "" {
		spendMetadata["request_endpoint"] = logCtx.RequestEndpoint
	}
	encoded, err := json.Marshal(doc)
	if err != nil {
		return metadata
	}
	return string(encoded)
}

func addOrganizationPolicySpendMetadata(metadata string, logCtx *RequestLogContext) string {
	if logCtx == nil || logCtx.OrganizationPolicy == nil {
		return metadata
	}
	var doc map[string]interface{}
	if json.Unmarshal([]byte(metadata), &doc) != nil || doc == nil {
		// Unmarshal leaves doc nil for a decode error and also for the literal
		// "null"; either way a fresh map is needed before assigning into it.
		doc = make(map[string]interface{})
	}
	spendMetadata, _ := doc["spend_logs_metadata"].(map[string]interface{})
	if spendMetadata == nil {
		spendMetadata = make(map[string]interface{})
		doc["spend_logs_metadata"] = spendMetadata
	}
	spendMetadata["public_model_name"] = logCtx.PublicModelID
	spendMetadata["canonical_model_name"] = logCtx.CanonicalModelID
	spendMetadata["billing_profile_id"] = logCtx.BillingProfileID
	spendMetadata["billing_profile_sha256"] = logCtx.BillingProfileSHA256
	spendMetadata["billing_price_model_name"] = logCtx.PriceModelID
	spendMetadata["billing_organization_id"] = logCtx.BillingOrganizationID
	encoded, err := json.Marshal(doc)
	if err != nil {
		return metadata
	}
	return string(encoded)
}

// extractEndUser extracts end_user from request headers or body
func extractEndUser(r *http.Request) string {
	// Check X-End-User header first
	if endUser := r.Header.Get("X-End-User"); endUser != "" {
		return endUser
	}
	return ""
}

// getClientIP gets the client IP address
func getClientIP(r *http.Request) string {
	// X-Forwarded-For header (first IP)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	// X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	// RemoteAddr
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// extractVersionSuffix returns the version segment (e.g. "/v1", "/v4") from the
// end of a URL base path, or empty string if none found. Only matches /v followed
// by one or more digits at the very end.
func extractVersionSuffix(baseURL string) string {
	idx := strings.LastIndex(baseURL, "/")
	if idx < 0 {
		return ""
	}
	segment := baseURL[idx:] // e.g. "/v1"
	if len(segment) < 3 || segment[1] != 'v' {
		return ""
	}
	for _, c := range segment[2:] {
		if c < '0' || c > '9' {
			return ""
		}
	}
	return segment
}

// extractVersionPrefix returns the version segment (e.g. "/v1") from the
// beginning of a URL path, or empty string if none found.
func extractVersionPrefix(urlPath string) string {
	if len(urlPath) < 3 || urlPath[0] != '/' || urlPath[1] != 'v' {
		return ""
	}
	i := 2
	for i < len(urlPath) && urlPath[i] >= '0' && urlPath[i] <= '9' {
		i++
	}
	if i == 2 {
		return "" // no digits after /v
	}
	// Must end at string end or next slash
	if i < len(urlPath) && urlPath[i] != '/' {
		return ""
	}
	return urlPath[:i]
}
