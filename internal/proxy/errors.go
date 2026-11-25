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

// isRequestScopedStatus reports whether a provider's status describes the
// REQUEST rather than the CREDENTIAL that served it -- and may therefore be
// relayed to the client as-is when no response is left to relay.
//
// The distinction decides what the client is told after a cascade ends with
// a transport failure. A request-scoped status (a bad prompt, a
// content-policy refusal, a model that does not exist) is the same on every
// credential: replaying it only multiplies the delay, and a 502 would blame
// the provider for being broken. A credential-scoped status (401, 402, 403)
// says something about ONE credential -- its key is wrong, its account is
// exhausted, it is not allowed to serve this -- and the next credential may
// answer the very same request fine. Relaying such a status after an
// unrelated transport failure tells the client its own key was rejected,
// which it never was: the router's own key is a different thing entirely,
// and the client cannot act on that answer.
//
// 429 is excluded as well: it is credential-scoped too (the upstream
// ACCOUNT is over its limit), and it carries its own treatment -- a
// Retry-After, which a plain status relay does not give.
//
// 404 is kept: "the model does not exist" is a statement about the request,
// and the model named is the same for every credential.
func isRequestScopedStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusBadRequest, http.StatusNotFound:
		return true
	default:
		return false
	}
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
	var raw []byte
	if len(rawBodies) > 0 {
		raw = rawBodies[0]
	}
	resp := APIErrorResponse{Error: maskedUpstreamError(statusCode, raw)}
	resp.RequestID = errorBodyRequestID(requestID)
	body, err := json.Marshal(resp)
	if err != nil {
		return []byte(`{"error":{"message":"Request failed","type":"api_error","param":null,"code":"api_error"}}`)
	}
	return append(body, '\n')
}

// classifiedErrorMessage returns the same "message" maskedUpstreamErrorBody
// would put in the client-facing body for statusCode — used to keep
// logCtx.ErrorMsg (and, downstream, the spend-log/analytics
// metadata.error_information.error_message column) in sync with what the
// client actually saw, instead of the historical hardcoded "Request failed"
// placeholder (see internal/proxy/proxy.go's two logCtx.ErrorMsg call sites).
func classifiedErrorMessage(statusCode int, rawBody []byte) string {
	return maskedUpstreamError(statusCode, rawBody).Message
}

func maskedUpstreamError(statusCode int, rawBody []byte) APIError {
	classified, ok := upstreamerror.ClassifyStatus(statusCode, rawBody)
	if !ok {
		switch statusCode {
		case http.StatusTooManyRequests:
			classified.Message, classified.Code = "Rate limit exceeded", "rate_limit_error"
		case http.StatusRequestTimeout, http.StatusGatewayTimeout:
			classified.Message, classified.Code = "Request timed out", "timeout_error"
		case http.StatusRequestEntityTooLarge:
			classified.Message, classified.Code = "Request entity too large", "request_entity_too_large"
		default:
			classified.Message, classified.Code = "Request failed", "api_error"
		}
	}
	return APIError{
		Message: classified.Message,
		Type:    errorTypeForStatus(statusCode),
		Param:   classified.Param,
		Code:    &classified.Code,
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
	var param, code *string
	if e.Param != "" {
		param = &e.Param
	}
	if e.Code != "" {
		code = &e.Code
		message = e.Message
	}
	WriteJSONError(w, status, message, errorTypeForStatus(status), param, code)
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

// ErrorCodeOverloaded is the error code of a request refused at server.max_in_flight_requests.
const ErrorCodeOverloaded = "overloaded"

// WriteErrorOverloaded writes the 503 of a request the router refuses because it is at its in-flight
// cap. The code tells it apart from the router's other 503s.
func WriteErrorOverloaded(w http.ResponseWriter, message string) {
	code := ErrorCodeOverloaded
	WriteJSONError(w, http.StatusServiceUnavailable, message, errorTypeForStatus(http.StatusServiceUnavailable), nil, &code)
}

// WriteErrorBadGateway writes a 502 Bad Gateway JSON error.
func WriteErrorBadGateway(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusBadGateway, message, errorTypeForStatus(http.StatusBadGateway), nil, nil)
}

// WriteErrorTimeout writes a 408 Request Timeout JSON error.
func WriteErrorTimeout(w http.ResponseWriter, message string) {
	WriteJSONError(w, http.StatusRequestTimeout, message, errorTypeForStatus(http.StatusRequestTimeout), nil, nil)
}

// WriteErrorClientClosed writes a 499 Client Closed Request JSON error.
// Best-effort only: by the time this is reached the client's own connection
// is already gone (see StatusClientClosedRequest), so these bytes are never
// actually read by anyone -- kept for parity with the other WriteError*
// helpers and in case an intermediary in the chain is still listening.
func WriteErrorClientClosed(w http.ResponseWriter, message string) {
	WriteJSONError(w, StatusClientClosedRequest, message, "client_closed_request", nil, nil)
}
