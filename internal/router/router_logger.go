package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// responseCapture captures response status and body for logging
type responseCapture struct {
	http.ResponseWriter
	statusCode int
	body       *bytes.Buffer
}

func newResponseCapture(w http.ResponseWriter) *responseCapture {
	return &responseCapture{
		ResponseWriter: w,
		statusCode:     http.StatusOK, // default status
		body:           &bytes.Buffer{},
	}
}

func (rc *responseCapture) WriteHeader(statusCode int) {
	rc.statusCode = statusCode
	rc.ResponseWriter.WriteHeader(statusCode)
}

func (rc *responseCapture) Write(p []byte) (int, error) {
	rc.body.Write(p)
	return rc.ResponseWriter.Write(p)
}

func (rc *responseCapture) Flush() {
	if flusher, ok := rc.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// ErrorLogEntry represents a single error log entry
type ErrorLogEntry struct {
	Timestamp string       `json:"timestamp"`
	Path      string       `json:"path"`
	Method    string       `json:"method"`
	Status    int          `json:"status"`
	Request   RequestInfo  `json:"request"`
	Response  ResponseInfo `json:"response"`
}

type RequestInfo struct {
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
}

type ResponseInfo struct {
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
}

// logErrorResponse logs error responses to a file
func logErrorResponse(errorsLogPath string, req *http.Request, rc *responseCapture, requestBody []byte) error {
	// Check if path is empty
	if errorsLogPath == "" {
		return nil
	}

	// Read request body if it exists
	var reqBodyStr string
	if len(requestBody) > 0 {
		reqBodyStr = string(requestBody)
	}

	// Create request headers map
	reqHeaders := make(map[string]string)
	for key, values := range req.Header {
		if len(values) > 0 {
			// Mask Authorization header for security
			if key == "Authorization" {
				reqHeaders[key] = "Bearer ***"
			} else {
				reqHeaders[key] = values[0]
			}
		}
	}

	// Create response headers map
	respHeaders := make(map[string]string)
	for key, values := range rc.Header() {
		if len(values) > 0 {
			respHeaders[key] = values[0]
		}
	}

	// Create log entry
	entry := ErrorLogEntry{
		Timestamp: time.Now().Format(time.RFC3339),
		Path:      req.URL.Path,
		Method:    req.Method,
		Status:    rc.statusCode,
		Request: RequestInfo{
			Headers: reqHeaders,
			Body:    reqBodyStr,
		},
		Response: ResponseInfo{
			Headers: respHeaders,
			Body:    rc.body.String(),
		},
	}

	// Marshal to JSON
	entryJSON, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// Append to log file with newline
	file, err := os.OpenFile(errorsLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer func() {
		_ = file.Close()
	}()

	_, err = file.Write(append(entryJSON, '\n'))
	return err
}

// isErrorStatus checks if status code is an error (4xx or 5xx)
func isErrorStatus(statusCode int) bool {
	return statusCode >= 400
}

// captureRequestBody reads and captures the request body for logging
func captureRequestBody(req *http.Request) ([]byte, error) {
	if req.Body == nil {
		return []byte{}, nil
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}

	// Check if body contains "stream" substring
	if strings.Contains(string(body), "stream") {
		// Restore body before returning error so original request can be proxied
		req.Body = io.NopCloser(bytes.NewReader(body))
		return nil, http.ErrBodyNotAllowed
	}

	// Restore body for further processing
	req.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}
