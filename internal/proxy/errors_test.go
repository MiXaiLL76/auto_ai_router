package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
		{"504", http.StatusGatewayTimeout, "timeout_error"},
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

func TestWriteJSONErrorIncludesRequestIDFromHeader(t *testing.T) {
	const id = "0123456789abcdef0123456789abcdef"
	recorder := httptest.NewRecorder()
	recorder.Header().Set(requestid.Header, id)

	WriteJSONError(recorder, http.StatusTooManyRequests, "test message", errorTypeForStatus(http.StatusTooManyRequests), nil, nil)

	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
	assert.Equal(t, id, response.RequestID)
}

func TestMaskedUpstreamErrorBodyIncludesRequestID(t *testing.T) {
	const id = "0123456789abcdef0123456789abcdef"

	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(maskedUpstreamErrorBody(http.StatusTooManyRequests, id), &response))
	assert.Equal(t, id, response.RequestID)
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

func TestMaskedUpstreamErrorBodyClassifiesBadRequest(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantMessage string
		wantCode    string
		wantParam   *string
		notContains []string
	}{
		{
			name:        "invalid tool choice",
			body:        `{"error":{"message":"Provider rejected tool_choice. Debug trace abc123","type":"invalid_request_error","param":"tool_choice","code":"bad_request"}}`,
			wantMessage: "Invalid tool_choice",
			wantCode:    "invalid_tool_choice",
			wantParam:   stringPtr("tool_choice"),
			notContains: []string{"Provider rejected", "Debug trace"},
		},
		{
			name:        "max tokens",
			body:        `{"error":{"message":"max_completion_tokens must be less than 8192","param":"max_completion_tokens","code":"invalid_request_error"}}`,
			wantMessage: "Invalid max_completion_tokens",
			wantCode:    "invalid_max_tokens",
			wantParam:   stringPtr("max_completion_tokens"),
			notContains: []string{"8192"},
		},
		{
			name:        "context length",
			body:        `{"error":{"message":"Input is too long for the context window","code":"context_length_exceeded"}}`,
			wantMessage: "Context length exceeded",
			wantCode:    "context_length_exceeded",
			wantParam:   stringPtr("input"),
		},
		{
			name:        "invalid model",
			body:        `{"error":{"message":"litellm.BadRequestError: Received Model Group=secret/internal Available Model Group Fallbacks=None"}}`,
			wantMessage: "Invalid model",
			wantCode:    "invalid_model",
			wantParam:   stringPtr("model"),
			notContains: []string{"litellm", "secret/internal", "Fallbacks"},
		},
		{
			name:        "invalid parameter",
			body:        `{"error":{"message":"unsupported parameter response_format for this model","param":"response_format"}}`,
			wantMessage: "Invalid request parameter",
			wantCode:    "invalid_parameter",
			wantParam:   stringPtr("response_format"),
			notContains: []string{"unsupported parameter", "this model"},
		},
		{
			name:        "plain text fallback",
			body:        `vendor stack id 012345`,
			wantMessage: "Invalid request",
			wantCode:    "invalid_request",
			notContains: []string{"vendor stack"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := maskedUpstreamErrorBody(http.StatusBadRequest, "", []byte(tt.body))

			var resp APIErrorResponse
			err := json.Unmarshal(body, &resp)
			if err != nil {
				t.Fatalf("decode error: %v", err)
			}
			if resp.Error.Type != "invalid_request_error" {
				t.Fatalf("type = %q", resp.Error.Type)
			}
			if resp.Error.Message != tt.wantMessage {
				t.Fatalf("message = %q, want %q", resp.Error.Message, tt.wantMessage)
			}
			if resp.Error.Code == nil || *resp.Error.Code != tt.wantCode {
				t.Fatalf("code = %v, want %q", resp.Error.Code, tt.wantCode)
			}
			if !equalStringPtr(resp.Error.Param, tt.wantParam) {
				t.Fatalf("param = %v, want %v", resp.Error.Param, tt.wantParam)
			}
			for _, s := range tt.notContains {
				if strings.Contains(string(body), s) {
					t.Fatalf("body contains %q", s)
				}
			}
		})
	}
}

func TestMaskedUpstreamErrorBodyKeepsGenericNonBadRequest(t *testing.T) {
	body := maskedUpstreamErrorBody(http.StatusBadGateway, "", []byte(`{"error":{"message":"upstream failed"}}`))

	var resp APIErrorResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if resp.Error.Message != "Request failed" {
		t.Fatalf("message = %q", resp.Error.Message)
	}
	if resp.Error.Code == nil || *resp.Error.Code != "api_error" {
		t.Fatalf("code = %v", resp.Error.Code)
	}
}

// TestWriteValidationError_HonorsArbitraryStatusCode guards against
// writeValidationError silently collapsing a future non-413 StatusCode (e.g.
// 415) to 400 while statusForValidationError — what callers log as
// error_code/HTTPStatus — reports the real value. The client-facing status
// must match what got logged, for any StatusCode, not just 0/413.
func TestWriteValidationError_HonorsArbitraryStatusCode(t *testing.T) {
	e := &converterutil.RequestValidationError{
		Param:      "content_type",
		Message:    "unsupported media type",
		StatusCode: http.StatusUnsupportedMediaType,
	}
	require.Equal(t, http.StatusUnsupportedMediaType, statusForValidationError(e),
		"sanity check: this is what orchestrator/proxy log as error_code/HTTPStatus")

	w := httptest.NewRecorder()
	writeValidationError(w, e, e.Message)

	assert.Equal(t, http.StatusUnsupportedMediaType, w.Code,
		"client-facing status must match statusForValidationError, not silently fall back to 400")

	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "unsupported media type", resp.Error.Message)
}

func TestWriteValidationError_DefaultsToBadRequest(t *testing.T) {
	e := &converterutil.RequestValidationError{Param: "model", Message: "missing model"}

	w := httptest.NewRecorder()
	writeValidationError(w, e, e.Message)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestWriteValidationError_TooLarge(t *testing.T) {
	e := &converterutil.RequestValidationError{
		Message:    "payload too large",
		StatusCode: http.StatusRequestEntityTooLarge,
	}

	w := httptest.NewRecorder()
	writeValidationError(w, e, e.Message)

	assert.Equal(t, http.StatusRequestEntityTooLarge, w.Code)
}

func stringPtr(s string) *string {
	return &s
}

func equalStringPtr(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}
