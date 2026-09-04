package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
)

// APIErrorResponse represents an OpenAI-compatible error response.
type APIErrorResponse struct {
	Error     APIError `json:"error"`
	RequestID string   `json:"request_id,omitempty"`
}

// APIError represents the error object inside an OpenAI-compatible error response.
type APIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// errorTypeForStatus maps HTTP status codes to OpenAI error type strings.
func errorTypeForStatus(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusPaymentRequired:
		return "insufficient_quota"
	case http.StatusForbidden:
		return "permission_denied"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusMethodNotAllowed:
		return "invalid_request_error"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return "timeout_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusBadGateway:
		return "api_error"
	default:
		if statusCode >= 500 {
			return "server_error"
		}
		return "invalid_request_error"
	}
}

// WriteJSONError writes an OpenAI-compatible JSON error response.
func WriteJSONError(w http.ResponseWriter, statusCode int, message, errorType string, param, code *string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	resp := APIErrorResponse{
		Error: APIError{
			Message: message,
			Type:    errorType,
			Param:   param,
			Code:    code,
		},
		RequestID: errorBodyRequestID(w.Header().Get(requestid.Header)),
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func errorBodyRequestID(id string) string {
	return requestid.Canonical(id)
}

func maskedUpstreamErrorBody(statusCode int, requestID string, rawBodies ...[]byte) []byte {
	message := "Request failed"
	code := "api_error"
	param := (*string)(nil)
	switch statusCode {
	case http.StatusBadRequest:
		detail := classifiedBadRequestError(rawBodies...)
		message = detail.Message
		code = *detail.Code
		param = detail.Param
	case http.StatusTooManyRequests:
		message = "Rate limit exceeded"
		code = "rate_limit_error"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		message = "Request timed out"
		code = "timeout_error"
	}

	resp := APIErrorResponse{
		Error: APIError{
			Message: message,
			Type:    errorTypeForStatus(statusCode),
			Param:   param,
			Code:    &code,
		},
	}
	resp.RequestID = errorBodyRequestID(requestID)
	body, err := json.Marshal(resp)
	if err != nil {
		return []byte(`{"error":{"message":"Request failed","type":"api_error","param":null,"code":"api_error"}}`)
	}
	return append(body, '\n')
}

func classifiedBadRequestError(rawBodies ...[]byte) APIError {
	errorCode := "invalid_request"
	result := APIError{
		Message: "Invalid request",
		Type:    errorTypeForStatus(http.StatusBadRequest),
		Param:   nil,
		Code:    &errorCode,
	}
	if len(rawBodies) == 0 || len(rawBodies[0]) == 0 {
		return result
	}

	signals, providerParam := providerErrorSignalsFromBody(rawBodies[0])
	joined := strings.ToLower(strings.Join(signals, " "))

	switch {
	case hasSignal(joined, "tool_choice", "tool choice", "toolchoice"):
		param := "tool_choice"
		code := "invalid_tool_choice"
		result.Message = "Invalid tool_choice"
		result.Param = &param
		result.Code = &code
	case hasSignal(joined, "max_completion_tokens", "max_output_tokens", "max_tokens", "max tokens", "maximum tokens", "output tokens"):
		param := inferBadRequestParam(joined, providerParam)
		if param == nil || !strings.Contains(*param, "token") {
			v := inferMaxTokensParam(joined)
			param = &v
		}
		code := "invalid_max_tokens"
		result.Message = "Invalid " + *param
		result.Param = param
		result.Code = &code
	case hasSignal(joined, "context length", "context window", "context limit", "too many tokens", "input too long", "prompt too long", "token limit"):
		code := "context_length_exceeded"
		result.Message = "Context length exceeded"
		result.Param = inferBadRequestParam(joined, providerParam)
		result.Code = &code
	case hasSignal(joined, "model group", "model not found", "model does not exist", "unsupported model", "invalid model", "model not supported"):
		param := "model"
		code := "invalid_model"
		result.Message = "Invalid model"
		result.Param = &param
		result.Code = &code
	case hasSignal(joined, "invalid argument", "invalid parameter", "invalidparameter", "invalid value", "unsupported parameter", "unknown parameter", "unrecognized parameter", "missing required", "required field", "must be", "should be", "does not support", "is not supported"):
		code := "invalid_parameter"
		result.Message = "Invalid request parameter"
		// Precedence: an explicit "param" from the provider's own JSON is
		// authoritative; next, a field path quoted directly in the message
		// ("Invalid 'output[1].type': 'input_file'. Supported values are:
		// ...") is precise even for dynamic/nested paths a fixed list can't
		// cover; only fall back to the generic keyword list last — it does
		// broad substring matching (e.g. "input") that a quoted path like
		// "input_file" would otherwise shadow.
		param := providerParam
		if param == nil {
			param = extractQuotedInvalidField(signals)
		}
		if param == nil {
			param = inferBadRequestParam(joined, nil)
		}
		result.Param = param
		result.Code = &code
	default:
		result.Param = providerParam
	}

	return result
}

const maxProviderErrorSignalBytes = 8 * 1024

func providerErrorSignalsFromBody(body []byte) ([]string, *string) {
	if len(body) > maxProviderErrorSignalBytes {
		body = body[:maxProviderErrorSignalBytes]
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil, nil
	}

	signals := make([]string, 0, 8)
	var param *string
	var value any
	if err := json.Unmarshal(trimmed, &value); err == nil {
		collectProviderErrorSignals(value, 0, &signals, &param)
	} else {
		signals = append(signals, string(trimmed))
	}
	return signals, param
}

func collectProviderErrorSignals(value any, depth int, signals *[]string, param **string) {
	if depth > 5 || len(*signals) >= 32 {
		return
	}
	switch v := value.(type) {
	case map[string]any:
		for _, key := range []string{"message", "type", "code", "status", "reason", "error"} {
			raw, exists := mapValueFold(v, key)
			if !exists {
				continue
			}
			text, ok := raw.(string)
			if ok {
				if s := cleanProviderErrorSignal(text); s != "" {
					*signals = append(*signals, s)
				}
			}
		}
		if *param == nil {
			for _, key := range []string{"param", "parameter", "field"} {
				raw, exists := mapValueFold(v, key)
				if !exists {
					continue
				}
				if text, ok := raw.(string); ok {
					if p := cleanProviderErrorParam(text); p != "" {
						*param = &p
						break
					}
				}
			}
		}
		for _, key := range []string{"error", "response", "details", "detail", "violations"} {
			raw, exists := mapValueFold(v, key)
			if exists {
				collectProviderErrorSignals(raw, depth+1, signals, param)
			}
		}
	case []any:
		for _, item := range v {
			collectProviderErrorSignals(item, depth+1, signals, param)
		}
	case string:
		if s := cleanProviderErrorSignal(v); s != "" {
			*signals = append(*signals, s)
		}
	}
}

func mapValueFold(values map[string]any, key string) (any, bool) {
	if raw, exists := values[key]; exists {
		return raw, true
	}
	for k, raw := range values {
		if strings.EqualFold(k, key) {
			return raw, true
		}
	}
	return nil, false
}

func cleanProviderErrorSignal(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if len(s) > 512 {
		s = s[:512]
	}
	return s
}

func cleanProviderErrorParam(s string) string {
	s = strings.TrimSpace(s)
	if s == "" || len(s) > 80 {
		return ""
	}
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			continue
		}
		switch r {
		case '_', '.', '-', '[', ']':
			continue
		default:
			return ""
		}
	}
	return s
}

func hasSignal(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func inferBadRequestParam(joined string, providerParam *string) *string {
	if providerParam != nil {
		return providerParam
	}
	for _, param := range []string{
		"max_completion_tokens",
		"max_output_tokens",
		"max_tokens",
		"tool_choice",
		"parallel_tool_calls",
		"response_format",
		"reasoning_effort",
		"service_tier",
		"stream_options",
		"temperature",
		"top_p",
		"logprobs",
		"messages",
		"input",
		"tools",
		"model",
		"stop",
		"metadata",
		"audio",
		"image",
		"prompt",
		"n",
	} {
		if strings.Contains(joined, param) {
			p := param
			return &p
		}
	}
	return nil
}

// extractQuotedInvalidField pulls the field path out of a provider message
// shaped like `Invalid 'output[1].type': 'input_file'. Supported values
// are: ...` — the quoted token right after "Invalid " is the offending
// field/parameter path. Fixed param-name lists can't cover these because
// the path is dynamic (array index, nested field), so this is checked as a
// fallback once the static list in inferBadRequestParam comes up empty.
// Tries each raw (pre-lowercased) signal in turn and returns the first hit.
func extractQuotedInvalidField(signals []string) *string {
	const marker = "invalid '"
	for _, s := range signals {
		idx := strings.Index(strings.ToLower(s), marker)
		if idx == -1 {
			continue
		}
		rest := s[idx+len(marker):]
		end := strings.IndexByte(rest, '\'')
		if end <= 0 {
			continue
		}
		field := rest[:end]
		return &field
	}
	return nil
}

func inferMaxTokensParam(joined string) string {
	switch {
	case strings.Contains(joined, "max_completion_tokens"):
		return "max_completion_tokens"
	case strings.Contains(joined, "max_output_tokens"):
		return "max_output_tokens"
	default:
		return "max_tokens"
	}
}

// WriteErrorBadRequest writes a 400 Bad Request JSON error.
func WriteErrorBadRequest(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusBadRequest, message, errorTypeForStatus(http.StatusBadRequest), nil, nil)
}

// WriteErrorUnauthorized writes a 401 Unauthorized JSON error.
func WriteErrorUnauthorized(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusUnauthorized, message, errorTypeForStatus(http.StatusUnauthorized), nil, nil)
}

// WriteErrorPaymentRequired writes a 402 Payment Required JSON error.
func WriteErrorPaymentRequired(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusPaymentRequired, message, errorTypeForStatus(http.StatusPaymentRequired), nil, nil)
}

// WriteErrorForbidden writes a 403 Forbidden JSON error.
func WriteErrorForbidden(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusForbidden, message, errorTypeForStatus(http.StatusForbidden), nil, nil)
}

// WriteErrorNotFound writes a 404 Not Found JSON error.
func WriteErrorNotFound(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusNotFound, message, errorTypeForStatus(http.StatusNotFound), nil, nil)
}

// WriteErrorTooLarge writes a 413 Request Entity Too Large JSON error.
func WriteErrorTooLarge(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusRequestEntityTooLarge, message, errorTypeForStatus(http.StatusRequestEntityTooLarge), nil, nil)
}

// statusForValidationError returns the HTTP status a converter-reported
// RequestValidationError should be surfaced as. Most validation errors don't
// specify one (StatusCode == 0) and default to 400, as they always have; a
// few (e.g. an inline base64 payload too large to forward) carry their own,
// more specific status.
func statusForValidationError(e *converterutil.RequestValidationError) int {
	if e.StatusCode != 0 {
		return e.StatusCode
	}
	return http.StatusBadRequest
}

// writeValidationError writes the client-facing response for a converter
// RequestValidationError, honoring its StatusCode when set instead of always
// answering 400. Generic over the status rather than special-casing 413:
// statusForValidationError is also what callers log as error_code/HTTPStatus,
// so any future non-413 StatusCode must reach the client as the same status
// it was logged under, not silently collapse to 400.
func writeValidationError(w http.ResponseWriter, e *converterutil.RequestValidationError, message string) {
	status := statusForValidationError(e)
	WriteJSONError(w, status, message, errorTypeForStatus(status), nil, nil)
}

// WriteErrorRateLimit writes a 429 Too Many Requests JSON error.
func WriteErrorRateLimit(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusTooManyRequests, message, errorTypeForStatus(http.StatusTooManyRequests), nil, nil)
}

// WriteErrorInternal writes a 500 Internal Server Error JSON error.
func WriteErrorInternal(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusInternalServerError, message, errorTypeForStatus(http.StatusInternalServerError), nil, nil)
}

// WriteErrorServiceUnavailable writes a 503 Service Unavailable JSON error.
func WriteErrorServiceUnavailable(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusServiceUnavailable, message, errorTypeForStatus(http.StatusServiceUnavailable), nil, nil)
}

// WriteErrorBadGateway writes a 502 Bad Gateway JSON error.
func WriteErrorBadGateway(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusBadGateway, message, errorTypeForStatus(http.StatusBadGateway), nil, nil)
}

// WriteErrorTimeout writes a 408 Request Timeout JSON error.
func WriteErrorTimeout(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusRequestTimeout, message, errorTypeForStatus(http.StatusRequestTimeout), nil, nil)
}
