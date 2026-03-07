package testhelpers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// APIErrorResponse mirrors proxy.APIErrorResponse for test assertions.
type APIErrorResponse struct {
	Error APIError `json:"error"`
}

// APIError mirrors proxy.APIError for test assertions.
type APIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// AssertJSONErrorResponse decodes the JSON response from the recorder and
// verifies the HTTP status, error type, and error message.
func AssertJSONErrorResponse(t *testing.T, recorder *httptest.ResponseRecorder, expectedStatus int, expectedType, expectedMsg string) {
	t.Helper()

	assert.Equal(t, expectedStatus, recorder.Code)
	assert.Equal(t, "application/json", recorder.Header().Get("Content-Type"))

	var resp APIErrorResponse
	err := json.NewDecoder(recorder.Body).Decode(&resp)
	require.NoError(t, err, "failed to decode JSON error response")

	assert.Equal(t, expectedType, resp.Error.Type)
	assert.Equal(t, expectedMsg, resp.Error.Message)
}

// NewTestRequest creates an *http.Request with a JSON body for testing.
func NewTestRequest(method, path string, body interface{}) *http.Request {
	var bodyReader *bytes.Reader
	if body != nil {
		data, _ := json.Marshal(body)
		bodyReader = bytes.NewReader(data)
	} else {
		bodyReader = bytes.NewReader(nil)
	}

	req := httptest.NewRequest(method, path, bodyReader)
	req.Header.Set("Content-Type", "application/json")
	return req
}

// NewTestRequestWithHeaders creates an *http.Request with a JSON body and custom headers.
func NewTestRequestWithHeaders(method, path string, body interface{}, headers map[string]string) *http.Request {
	req := NewTestRequest(method, path, body)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return req
}
