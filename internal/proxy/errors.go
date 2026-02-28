package proxy

import (
	"encoding/json"
	"net/http"
)

// APIErrorResponse represents an OpenAI-compatible error response.
type APIErrorResponse struct {
	Error APIError `json:"error"`
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
	case http.StatusRequestTimeout:
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
	}
	_ = json.NewEncoder(w).Encode(resp)
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

// WriteErrorRateLimit writes a 429 Too Many Requests JSON error.
func WriteErrorRateLimit(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusTooManyRequests, message, errorTypeForStatus(http.StatusTooManyRequests), nil, nil)
}

// WriteErrorInternal writes a 500 Internal Server Error JSON error.
func WriteErrorInternal(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusInternalServerError, message, errorTypeForStatus(http.StatusInternalServerError), nil, nil)
}

// WriteErrorBadGateway writes a 502 Bad Gateway JSON error.
func WriteErrorBadGateway(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusBadGateway, message, errorTypeForStatus(http.StatusBadGateway), nil, nil)
}

// WriteErrorTimeout writes a 408 Request Timeout JSON error.
func WriteErrorTimeout(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusRequestTimeout, message, errorTypeForStatus(http.StatusRequestTimeout), nil, nil)
}
