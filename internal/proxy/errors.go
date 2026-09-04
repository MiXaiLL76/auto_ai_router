package proxy

import (
	"encoding/json"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/mixaill76/auto_ai_router/internal/upstreamerror"
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

// classifiedBadRequestError delegates to the shared upstreamerror classifier
// (also used by litellm compatibility mode's normalizeError) so both modes
// surface the same classification instead of one discarding it.
func classifiedBadRequestError(rawBodies ...[]byte) APIError {
	var raw []byte
	if len(rawBodies) > 0 {
		raw = rawBodies[0]
	}
	classified := upstreamerror.ClassifyBadRequest(raw)
	code := classified.Code
	return APIError{
		Message: classified.Message,
		Type:    errorTypeForStatus(http.StatusBadRequest),
		Param:   classified.Param,
		Code:    &code,
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
