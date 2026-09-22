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
	"unicode"
	"unicode/utf8"

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

// isClientContextCanceled reports whether r's own context is already
// canceled — i.e. the client itself gave up (closed the connection, hit its
// own request timeout) before AIR finished talking to any upstream. Checked
// directly against r.Context().Err() rather than pattern-matching the
// transport error returned by p.client.Do, so it can't be confused with
// isClientDisconnectError's EPIPE/ECONNRESET cases: for an *outbound* call
// (AIR -> provider) those mean the connection to the *provider* broke, a
// genuine upstream failure, not the inbound client having left. A canceled
// r.Context() is unambiguous either way: it's used as the parent context for
// every upstream request instead of the incoming request's own connection
// (see upstreamRequestContext), so it can only become Done via the client
// disconnecting or the handler itself returning.
func isClientContextCanceled(r *http.Request) bool {
	if r == nil {
		return false
	}
	return errors.Is(r.Context().Err(), context.Canceled)
}

// isClientCanceledTransportError reports whether attemptErr -- the error a
// specific credential attempt just failed with -- was actually *caused* by
// the client disconnecting, as opposed to r's context merely being canceled
// at some point during a longer retry sequence for an unrelated reason.
//
// The two checks answer different questions and neither alone is enough:
// isClientContextCanceled(r) alone would also fire for a credential attempt
// that failed for a genuine, unrelated reason (e.g. a real ECONNREFUSED)
// simply because the client *happened* to also give up around the same
// time -- plausible whenever AIR's retry/fallback sequence takes long enough
// that the client's own (often shorter) timeout elapses before AIR finishes
// working through a real outage. Misclassifying that as client_canceled
// would hide a genuine multi-credential outage from fail2ban/ERROR-level
// alerting exactly when it matters most. Conversely, checking only
// errors.Is(attemptErr, context.Canceled) without isClientContextCanceled(r)
// would fire on a transport that returns a bare context.Canceled for
// reasons unrelated to r's own context (see the "transport error" case in
// client_error_messages_test.go, which relies on exactly this not
// happening). Requiring both pins the classification to the one case that
// actually matters: THIS attempt failed specifically because the client's
// own context is what unblocked p.client.Do.
func isClientCanceledTransportError(r *http.Request, attemptErr error) bool {
	return errors.Is(attemptErr, context.Canceled) && isClientContextCanceled(r)
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

// maxErrorBodyRawBytes bounds RequestLogContext.ErrorBodyRaw so one
// pathological provider error (e.g. echoing back an oversized prompt in a
// validation message) can't inflate a single Kafka kafkalog.RawBodyEvent
// unreasonably.
// Larger than extractErrorMessage's 512-byte cap on purpose: this field
// exists specifically so operators can see a provider failure in full,
// where the short error_message got cut off.
const maxErrorBodyRawBytes = 16 * 1024

// extractErrorBodyRaw returns the raw upstream error response body,
// untruncated up to maxErrorBodyRawBytes. Only ever called from failure
// paths (see call sites) — never populated for a successful response.
func extractErrorBodyRaw(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	if len(body) > maxErrorBodyRawBytes {
		return string(body[:maxErrorBodyRawBytes]) + "..."
	}
	return string(body)
}

// ErrorOrigin names the specific code path that produced a failure outcome,
// independent of ErrorMsg's free text — a short, fixed, greppable/filterable
// tag rather than a message meant for a human to read once. It exists
// because several failure paths (most notably the ones that end in 502) have
// no upstream body to show at all (ErrorBodyRaw empty: the provider never
// responded), so RawBodyEvent.ResponseBody alone can't distinguish "every
// credential's connection attempt failed" from "the response was too big to
// read" from "a mid-stream error's text didn't match any known signal" — all
// three currently surface as an indistinguishable bare 502 unless the
// operator parses ErrorMsg's prose by hand. Set alongside ErrorMsg/HTTPStatus
// at each distinct failure call site; empty when a failure's cause is already
// self-evident from HTTPStatus/ErrorBodyRaw alone (e.g. a plain classified
// 4xx with the real provider text attached).
type ErrorOrigin string

const (
	// ErrorOriginAllAttemptsExhausted: every direct-provider credential (and
	// fallback) attempt failed at the transport level — no HTTP response was
	// ever received from anyone. See proxyRequest's "All provider attempts
	// failed" tail.
	ErrorOriginAllAttemptsExhausted ErrorOrigin = "all_attempts_exhausted"
	// ErrorOriginProxyForwardError: same as ErrorOriginAllAttemptsExhausted,
	// but for an AIR-to-AIR proxy-type credential chain (base_url pointing at
	// another AIR instance) — see proxyRequest's "Proxy forward error" tail.
	ErrorOriginProxyForwardError ErrorOrigin = "proxy_forward_error"
	// ErrorOriginResponseTooLarge: the upstream response body exceeded the
	// configured read-size limit (ErrResponseBodyTooLarge). Treated as fatal
	// — another credential's response would likely be just as large — so
	// this is a final outcome, never retried.
	ErrorOriginResponseTooLarge ErrorOrigin = "response_too_large"
	// ErrorOriginUnclassifiedStreamError: a provider streamed a terminal
	// error event whose embedded message/type/code didn't match any of
	// statusCodeFromErrorSignals' known keywords, so the status defaulted to
	// 502 as a catch-all rather than a genuine "bad gateway" diagnosis.
	ErrorOriginUnclassifiedStreamError ErrorOrigin = "unclassified_stream_error"
	// ErrorOriginWebSocketStreamError: a native Realtime WebSocket turn
	// ended with outcome "stream_error".
	ErrorOriginWebSocketStreamError ErrorOrigin = "websocket_stream_error"
	// ErrorOriginClientCanceled: the client disconnected (closed the
	// connection, or its own request timeout fired) before any credential
	// attempt produced a response — see isClientContextCanceled. Distinct
	// from ErrorOriginAllAttemptsExhausted/ErrorOriginProxyForwardError:
	// those name a genuine upstream transport failure, while this one means
	// no upstream failure occurred at all, AIR just gave up because the
	// original caller already left. Carries StatusClientClosedRequest (499),
	// not 502, and is deliberately excluded from fail2ban/credential-error
	// accounting (see the isClientContextCanceled checks in the retry loops)
	// since it reflects the client's behavior, not the credential's.
	ErrorOriginClientCanceled ErrorOrigin = "client_canceled"
)

// StatusClientClosedRequest is the nginx-convention status (not defined by
// net/http) used to record that a request ended because the client itself
// disconnected before any response was available — as opposed to 502, which
// would claim an upstream transport failure that never actually happened.
// Never meaningfully delivered to the client (which is already gone by the
// time this is decided); it exists for accurate logging/metrics/raw-body
// classification.
const StatusClientClosedRequest = 499

// sensitiveRequestBodyFields are the top-level JSON keys that carry the
// client's own prompt/conversation content, across the request shapes AIR
// accepts: messages (chat completions, Anthropic native), prompt (legacy
// completions), input (Responses API, embeddings), instructions (Responses
// API system prompt), contents (Gemini/Vertex native). Every one of these
// shapes carries the field at the top level -- never nested inside e.g. a
// tool's JSON-Schema parameters -- so redactSensitiveFields matches only at
// the top level, not by name anywhere in the tree. Everything else in the
// body -- model, tools, tool_choice, temperature, max_tokens, stream,
// response_format, ... -- is request shape/parameters, not content, and is
// left untouched, even if a tool parameter happens to share one of these
// names.
var sensitiveRequestBodyFields = map[string]struct{}{
	"messages":     {},
	"system":       {},
	"prompt":       {},
	"input":        {},
	"contents":     {},
	"instructions": {},
}

// redactRequestBodyForLogging returns body with sensitiveRequestBodyFields
// replaced by shape-preserving placeholders (role and count kept, actual
// text dropped), for the client-request-body opt-in
// (kafka.raw_bodies.store_raw_body). Fails closed: ("", false) when body
// isn't valid JSON (multipart, binary, malformed) rather than risk shipping
// unredacted content, since the whole point of this function is the safety
// gate on that opt-in.
func redactRequestBodyForLogging(body []byte) (string, bool) {
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", false
	}
	redactSensitiveFields(parsed)
	out, err := json.Marshal(parsed)
	if err != nil {
		return "", false
	}
	return string(out), true
}

// redactSensitiveFields replaces sensitiveRequestBodyFields found at the
// top level of parsed via redactFieldValueShape, in place. Every shape AIR
// accepts (chat completions, legacy completions, Responses API, Anthropic
// native, Gemini/Vertex native) carries these as top-level request fields,
// never nested inside e.g. a tool's JSON-Schema parameters -- so unlike an
// earlier version of this function, this does NOT recurse into unrelated
// keys (tools, tool_choice, response_format, ...) looking for name
// collisions. Those are request parameters that must survive untouched for
// error analysis, and a tool parameter happening to be named "input" or
// "messages" is not conversation content.
func redactSensitiveFields(parsed map[string]any) {
	for key, child := range parsed {
		if _, sensitive := sensitiveRequestBodyFields[key]; sensitive {
			parsed[key] = redactFieldValueShape(child)
		}
	}
}

// redactFieldValueShape blanks a sensitive field's actual content while
// keeping enough shape to debug with. For a messages-style array (each
// element an object with e.g. "role"/"type"), keeps those identifying keys
// per element and replaces the rest with a single "content": "[REDACTED]"
// placeholder -- preserving turn count and roles without the text. Known
// limitation: shapes that don't use "role"/"type"/"content" (e.g. Gemini's
// content.parts) aren't specially preserved and just collapse to the
// generic placeholder. Anything else (a plain string, or an array of plain
// strings as with embeddings' `input`) becomes a flat "[REDACTED]".
func redactFieldValueShape(value any) any {
	items, ok := value.([]any)
	if !ok {
		return "[REDACTED]"
	}
	result := make([]any, len(items))
	for i, item := range items {
		obj, ok := item.(map[string]any)
		if !ok {
			result[i] = "[REDACTED]"
			continue
		}
		redacted := map[string]any{"content": "[REDACTED]"}
		if role, ok := obj["role"]; ok {
			redacted["role"] = role
		}
		if typ, ok := obj["type"]; ok {
			redacted["type"] = typ
		}
		result[i] = redacted
	}
	return result
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
	case StatusClientClosedRequest:
		// Not a real provider/LiteLLM exception class (this status never
		// reaches a client) -- distinguishes a client-side cancellation from
		// both a genuine 4xx (BadRequestError) and a genuine upstream 5xx
		// (APIConnectionError), which the >=400/>=500 default below would
		// otherwise collapse it into.
		return "ClientDisconnected"
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
	if logCtx.Credential != nil {
		doc["credential_name"] = logCtx.Credential.Name
	}
	if logCtx.ActualCredentialName != "" {
		doc["credential_name"] = logCtx.ActualCredentialName
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
	if logCtx.ThinkingMode != "" {
		spendMetadata["thinking_mode"] = logCtx.ThinkingMode
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
	if logCtx.OrganizationPolicy.HasCustomPricing() {
		spendMetadata["billing_profile_id"] = logCtx.BillingProfileID
		spendMetadata["billing_profile_sha256"] = logCtx.BillingProfileSHA256
	}
	spendMetadata["billing_price_model_name"] = logCtx.PriceModelID
	spendMetadata["billing_organization_id"] = logCtx.BillingOrganizationID
	encoded, err := json.Marshal(doc)
	if err != nil {
		return metadata
	}
	return string(encoded)
}

// Identity headers, in priority order. They mirror LiteLLM's user_header_mappings for
// the callers this deployment fronts: the *-Email headers carry the end user
// (LiteLLM role "customer": LiteLLM_EndUserTable / DailyEndUserSpend), the *-Id
// headers carry the internal user id (LiteLLM role "internal_user": the SID recorded
// in SpendLogs and DailyUserSpend for an ownerless service key). X-End-User is AIR's
// original end-user header and stays supported.
var (
	endUserHeaders = []string{"X-AIR-User-Email", "X-End-User", "X-OpenWebUI-User-Email", "X-AirClaw-User-Email"}
	userIDHeaders  = []string{"X-AIR-User-Id", "X-OpenWebUI-User-Id", "X-AirClaw-User-Id"}
)

// maxIdentityHeaderLen bounds an identity value; real emails and Windows SIDs are far
// shorter, and the value ends up in indexed database columns.
const maxIdentityHeaderLen = 256

// firstIdentityHeader returns the first usable value among names. A value is unusable
// when it is empty after trimming, too long, invalid UTF-8 or contains control
// characters; such a header is skipped so a lower-priority one can still apply.
func firstIdentityHeader(r *http.Request, names []string) string {
	if r == nil {
		return ""
	}
	for _, name := range names {
		value := strings.TrimSpace(r.Header.Get(name))
		if value == "" || len(value) > maxIdentityHeaderLen || !utf8.ValidString(value) {
			continue
		}
		if strings.ContainsFunc(value, unicode.IsControl) {
			continue
		}
		return value
	}
	return ""
}

// extractEndUser returns the end user (email) the caller identified via headers.
// The key owner's email is never used as a fallback: see logSpend.
func extractEndUser(r *http.Request) string {
	return firstIdentityHeader(r, endUserHeaders)
}

// extractUserID returns the internal user id the caller identified via headers, or "".
func extractUserID(r *http.Request) string {
	return firstIdentityHeader(r, userIDHeaders)
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
