package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/security"
)

// errorLogFileCache holds a cached file handle for error logging
type errorLogFileCache struct {
	mu      sync.Mutex
	handles map[string]*logFileHandle
}

var logFileCache = &errorLogFileCache{
	handles: make(map[string]*logFileHandle),
}

type logFileHandle struct {
	file *os.File
	mu   sync.Mutex
}

// getOrCreateLogFile returns a cached file handle or creates a new one
func (c *errorLogFileCache) getOrCreate(path string) (*logFileHandle, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if file, exists := c.handles[path]; exists {
		return file, nil
	}

	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	handle := &logFileHandle{file: file}
	c.handles[path] = handle
	return handle, nil
}

func (c *errorLogFileCache) closeAll() error {
	c.mu.Lock()
	handles := c.handles
	c.handles = make(map[string]*logFileHandle)
	c.mu.Unlock()

	var firstErr error
	for _, handle := range handles {
		handle.mu.Lock()
		err := handle.file.Close()
		handle.mu.Unlock()
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

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
				authValue := values[0]
				if strings.HasPrefix(authValue, "Bearer ") {
					token := strings.TrimPrefix(authValue, "Bearer ")
					reqHeaders[key] = "Bearer " + security.MaskToken(token)
				} else {
					reqHeaders[key] = security.MaskSecret(authValue, 4)
				}
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
		Timestamp: time.Now().UTC().Format(time.RFC3339),
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

	// Get or create cached file handle
	file, err := logFileCache.getOrCreate(errorsLogPath)
	if err != nil {
		return err
	}

	file.mu.Lock()
	_, err = file.file.Write(append(entryJSON, '\n'))
	file.mu.Unlock()
	return err
}

// CloseErrorLogFiles closes any cached error log file handles.
func CloseErrorLogFiles() error {
	return logFileCache.closeAll()
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

	// Check if this is a streaming request by parsing "stream": true in JSON
	if isStreamingRequestBody(body) {
		// Restore body before returning error so original request can be proxied
		req.Body = io.NopCloser(bytes.NewReader(body))
		// Return io.EOF instead of ErrBodyNotAllowed for streaming requests
		return body, nil
	}

	// Restore body for further processing
	req.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

func isStreamingRequestBody(body []byte) bool {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	streamRaw, ok := payload["stream"]
	if !ok {
		return false
	}

	var stream bool
	if err := json.Unmarshal(streamRaw, &stream); err != nil {
		return false
	}

	return stream
}
