package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
)

func TestWriteJSONError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantType   string
	}{
		{"400", http.StatusBadRequest, "invalid_request_error"},
		{"401", http.StatusUnauthorized, "authentication_error"},
		{"402", http.StatusPaymentRequired, "insufficient_quota"},
		{"403", http.StatusForbidden, "permission_denied"},
		{"404", http.StatusNotFound, "not_found_error"},
		{"405", http.StatusMethodNotAllowed, "invalid_request_error"},
		{"408", http.StatusRequestTimeout, "timeout_error"},
		{"413", http.StatusRequestEntityTooLarge, "invalid_request_error"},
		{"429", http.StatusTooManyRequests, "rate_limit_error"},
		{"500", http.StatusInternalServerError, "server_error"},
		{"502", http.StatusBadGateway, "api_error"},
		{"503_5xx_default", http.StatusServiceUnavailable, "server_error"},
		{"299_default", 299, "invalid_request_error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			WriteJSONError(recorder, tt.statusCode, "test message", errorTypeForStatus(tt.statusCode), nil, nil)
			testhelpers.AssertJSONErrorResponse(t, recorder, tt.statusCode, tt.wantType, "test message")
		})
	}
}

func TestWriteErrorConvenienceFunctions(t *testing.T) {
	tests := []struct {
		name       string
		fn         func(http.ResponseWriter, string)
		wantStatus int
		wantType   string
	}{
		{"BadRequest", WriteErrorBadRequest, 400, "invalid_request_error"},
		{"PaymentRequired", WriteErrorPaymentRequired, 402, "insufficient_quota"},
		{"Forbidden", WriteErrorForbidden, 403, "permission_denied"},
		{"TooLarge", WriteErrorTooLarge, 413, "invalid_request_error"},
		{"BadGateway", WriteErrorBadGateway, 502, "api_error"},
		{"Timeout", WriteErrorTimeout, 408, "timeout_error"},
		{"Unauthorized", WriteErrorUnauthorized, 401, "authentication_error"},
		{"NotFound", WriteErrorNotFound, 404, "not_found_error"},
		{"RateLimit", WriteErrorRateLimit, 429, "rate_limit_error"},
		{"Internal", WriteErrorInternal, 500, "server_error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			tt.fn(recorder, "error: "+tt.name)
			testhelpers.AssertJSONErrorResponse(t, recorder, tt.wantStatus, tt.wantType, "error: "+tt.name)
		})
	}
}
