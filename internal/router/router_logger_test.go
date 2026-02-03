package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResponseCapture_WriteHeader(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	rc.WriteHeader(http.StatusCreated)

	assert.Equal(t, http.StatusCreated, rc.statusCode)
	assert.Equal(t, http.StatusCreated, w.Code)
}

func TestResponseCapture_Write(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	data := []byte(`{"result": "success"}`)
	n, err := rc.Write(data)

	assert.NoError(t, err)
	assert.Equal(t, len(data), n)
	assert.Equal(t, string(data), rc.body.String())
	assert.Equal(t, string(data), w.Body.String())
}

func TestResponseCapture_DefaultStatus(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	assert.Equal(t, http.StatusOK, rc.statusCode)
}

func TestResponseCapture_Flush(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	// Flush should not panic even if underlying ResponseWriter doesn't support it
	rc.Flush()
}

func TestIsErrorStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		expected   bool
	}{
		{"200 OK", http.StatusOK, false},
		{"201 Created", http.StatusCreated, false},
		{"300 Multiple Choices", http.StatusMultipleChoices, false},
		{"304 Not Modified", http.StatusNotModified, false},
		{"400 Bad Request", http.StatusBadRequest, true},
		{"401 Unauthorized", http.StatusUnauthorized, true},
		{"403 Forbidden", http.StatusForbidden, true},
		{"404 Not Found", http.StatusNotFound, true},
		{"500 Internal Server Error", http.StatusInternalServerError, true},
		{"502 Bad Gateway", http.StatusBadGateway, true},
		{"503 Service Unavailable", http.StatusServiceUnavailable, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isErrorStatus(tt.statusCode)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCaptureRequestBody_NoBody(t *testing.T) {
	req, err := http.NewRequest("GET", "/test", nil)
	assert.NoError(t, err)

	body, err := captureRequestBody(req)

	assert.NoError(t, err)
	assert.Equal(t, []byte{}, body)
}

func TestCaptureRequestBody_SimpleBody(t *testing.T) {
	data := []byte(`{"key": "value"}`)
	req, err := http.NewRequest("POST", "/test", io.NopCloser(bytes.NewReader(data)))
	assert.NoError(t, err)

	body, err := captureRequestBody(req)

	assert.NoError(t, err)
	assert.Equal(t, data, body)
	// Verify body is restored for further reading
	restoredBody, _ := io.ReadAll(req.Body)
	assert.Equal(t, data, restoredBody)
}

func TestCaptureRequestBody_WithStreamKeyword(t *testing.T) {
	data := []byte(`{"stream": true, "messages": []}`)
	req, err := http.NewRequest("POST", "/test", io.NopCloser(bytes.NewReader(data)))
	assert.NoError(t, err)

	body, err := captureRequestBody(req)

	assert.Equal(t, data, body)
	assert.NoError(t, err)
	// Body should be restored
	restoredBody, _ := io.ReadAll(req.Body)
	assert.Equal(t, data, restoredBody)
}

func TestCaptureRequestBody_LargeBody(t *testing.T) {
	data := bytes.Repeat([]byte("x"), 10*1024)
	req, err := http.NewRequest("POST", "/test", io.NopCloser(bytes.NewReader(data)))
	assert.NoError(t, err)

	body, err := captureRequestBody(req)

	assert.NoError(t, err)
	assert.Equal(t, data, body)
}

func TestLogErrorResponse_EmptyPath(t *testing.T) {
	req := httptest.NewRequest("POST", "/api/test", nil)
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	err := logErrorResponse("", req, rc, []byte{})

	assert.NoError(t, err)
}

func TestLogErrorResponse_Success(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	req := httptest.NewRequest("POST", "/api/test", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	rc := newResponseCapture(w)
	rc.WriteHeader(http.StatusBadRequest)
	_, _ = rc.Write([]byte("error message"))

	requestBody := []byte(`{"bad": "request"}`)
	err := logErrorResponse(logFile, req, rc, requestBody)

	assert.NoError(t, err)

	// Verify log file was created and contains JSON
	content, _ := os.ReadFile(logFile)
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	assert.Len(t, lines, 1)

	var entry ErrorLogEntry
	err = json.Unmarshal([]byte(lines[0]), &entry)
	assert.NoError(t, err)

	assert.Equal(t, "/api/test", entry.Path)
	assert.Equal(t, "POST", entry.Method)
	assert.Equal(t, http.StatusBadRequest, entry.Status)
	assert.Equal(t, "Bearer ***", entry.Request.Headers["Authorization"])
	assert.Equal(t, string(requestBody), entry.Request.Body)
	assert.Equal(t, "error message", entry.Response.Body)
}

func TestLogErrorResponse_MultipleEntries(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	// Log first error
	req1 := httptest.NewRequest("GET", "/api/first", nil)
	w1 := httptest.NewRecorder()
	rc1 := newResponseCapture(w1)
	rc1.WriteHeader(http.StatusNotFound)
	_, _ = rc1.Write([]byte("not found"))

	err := logErrorResponse(logFile, req1, rc1, []byte{})
	assert.NoError(t, err)

	// Log second error
	req2 := httptest.NewRequest("POST", "/api/second", nil)
	w2 := httptest.NewRecorder()
	rc2 := newResponseCapture(w2)
	rc2.WriteHeader(http.StatusInternalServerError)
	_, _ = rc2.Write([]byte("server error"))

	err = logErrorResponse(logFile, req2, rc2, []byte{})
	assert.NoError(t, err)

	// Verify both entries
	content, _ := os.ReadFile(logFile)
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	assert.Len(t, lines, 2)

	var entry1, entry2 ErrorLogEntry
	_ = json.Unmarshal([]byte(lines[0]), &entry1)
	_ = json.Unmarshal([]byte(lines[1]), &entry2)

	assert.Equal(t, "/api/first", entry1.Path)
	assert.Equal(t, http.StatusNotFound, entry1.Status)
	assert.Equal(t, "/api/second", entry2.Path)
	assert.Equal(t, http.StatusInternalServerError, entry2.Status)
}

func TestLogErrorResponse_MasksAuthorizationHeader(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	req := httptest.NewRequest("POST", "/api/test", nil)
	req.Header.Set("Authorization", "Bearer very-secret-token-12345")

	w := httptest.NewRecorder()
	rc := newResponseCapture(w)
	rc.WriteHeader(http.StatusUnauthorized)

	err := logErrorResponse(logFile, req, rc, []byte{})
	assert.NoError(t, err)

	content, _ := os.ReadFile(logFile)
	var entry ErrorLogEntry
	_ = json.Unmarshal(content, &entry)

	// Authorization should be masked
	assert.Equal(t, "Bearer ***", entry.Request.Headers["Authorization"])
	assert.NotContains(t, entry.Request.Headers["Authorization"], "secret-token")
}

func TestLogErrorResponse_PreservesOtherHeaders(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	req := httptest.NewRequest("POST", "/api/test", nil)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "test-client")

	w := httptest.NewRecorder()
	rc := newResponseCapture(w)
	rc.WriteHeader(http.StatusBadRequest)

	err := logErrorResponse(logFile, req, rc, []byte{})
	assert.NoError(t, err)

	content, _ := os.ReadFile(logFile)
	var entry ErrorLogEntry
	_ = json.Unmarshal(content, &entry)

	assert.Equal(t, "application/json", entry.Request.Headers["Content-Type"])
	assert.Equal(t, "application/json", entry.Request.Headers["Accept"])
	assert.Equal(t, "test-client", entry.Request.Headers["User-Agent"])
}

func TestLogErrorResponse_ResponseHeaders(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	req := httptest.NewRequest("GET", "/api/test", nil)

	w := httptest.NewRecorder()
	rc := newResponseCapture(w)
	rc.Header().Set("X-Custom-Header", "custom-value")
	rc.Header().Set("Content-Type", "text/plain")
	rc.WriteHeader(http.StatusForbidden)
	_, _ = rc.Write([]byte("forbidden"))

	err := logErrorResponse(logFile, req, rc, []byte{})
	assert.NoError(t, err)

	content, _ := os.ReadFile(logFile)
	var entry ErrorLogEntry
	_ = json.Unmarshal(content, &entry)

	assert.Equal(t, "custom-value", entry.Response.Headers["X-Custom-Header"])
	assert.Equal(t, "text/plain", entry.Response.Headers["Content-Type"])
}

func TestNewResponseCapture(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	assert.NotNil(t, rc)
	assert.Equal(t, http.StatusOK, rc.statusCode)
	assert.NotNil(t, rc.body)
	assert.Equal(t, "", rc.body.String())
}

func TestResponseCapture_WriteMultipleTimes(t *testing.T) {
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)

	data1 := []byte("chunk1")
	data2 := []byte("chunk2")
	data3 := []byte("chunk3")

	_, _ = rc.Write(data1)
	_, _ = rc.Write(data2)
	_, _ = rc.Write(data3)

	expected := "chunk1chunk2chunk3"
	assert.Equal(t, expected, rc.body.String())
	assert.Equal(t, expected, w.Body.String())
}

func TestCaptureRequestBody_StreamKeywordCaseSensitive(t *testing.T) {
	// "Stream" with capital S should also trigger the check since we're not doing case-sensitive check
	data := []byte(`{"Stream": true}`)
	req, err := http.NewRequest("POST", "/test", io.NopCloser(bytes.NewReader(data)))
	assert.NoError(t, err)

	body, err := captureRequestBody(req)

	// This should return error because the check uses strings.Contains which is case-sensitive
	// So "Stream" won't match "stream", body should be returned successfully
	assert.NoError(t, err)
	assert.Equal(t, data, body)
}

func TestLogErrorResponse_NoRequestBody(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "errors.log")

	req := httptest.NewRequest("GET", "/api/test", nil)
	w := httptest.NewRecorder()
	rc := newResponseCapture(w)
	rc.WriteHeader(http.StatusNotFound)

	err := logErrorResponse(logFile, req, rc, []byte{})
	assert.NoError(t, err)

	content, _ := os.ReadFile(logFile)
	var entry ErrorLogEntry
	_ = json.Unmarshal(content, &entry)

	assert.Equal(t, "", entry.Request.Body)
}
