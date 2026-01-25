package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

type Proxy struct {
	balancer       *balancer.RoundRobin
	client         *http.Client
	logger         *slog.Logger
	maxBodySizeMB  int
	requestTimeout time.Duration
	metrics        *monitoring.Metrics
	masterKey      string
}

func New(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics, masterKey string) *Proxy {
	return &Proxy{
		balancer:       bal,
		logger:         logger,
		maxBodySizeMB:  maxBodySizeMB,
		requestTimeout: requestTimeout,
		metrics:        metrics,
		masterKey:      masterKey,
		client: &http.Client{
			Timeout: requestTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

func (p *Proxy) ProxyRequest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Verify master_key from Authorization header
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		p.logger.Error("Missing Authorization header")
		http.Error(w, "Unauthorized: Missing Authorization header", http.StatusUnauthorized)
		return
	}

	// Extract token from "Bearer <token>"
	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == authHeader {
		// No "Bearer " prefix found
		p.logger.Error("Invalid Authorization header format")
		http.Error(w, "Unauthorized: Invalid Authorization header format", http.StatusUnauthorized)
		return
	}

	// Check if token matches master_key
	if token != p.masterKey {
		p.logger.Error("Invalid master key", "provided_key_prefix", maskKey(token))
		http.Error(w, "Unauthorized: Invalid master key", http.StatusUnauthorized)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, int64(p.maxBodySizeMB)*1024*1024))
	if err != nil {
		p.logger.Error("Failed to read request body", "error", err)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	defer func() {
		if closeErr := r.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close request body", "error", closeErr)
		}
	}()

	// Extract model from request body if present
	modelID := extractModelFromBody(body)

	// Select credential based on model availability
	var cred *balancer.Credential
	if modelID != "" {
		cred, err = p.balancer.NextForModel(modelID)
	} else {
		cred, err = p.balancer.Next()
	}

	if err != nil {
		p.logger.Error("Failed to get credential", "error", err, "model", modelID)
		if errors.Is(err, balancer.ErrRateLimitExceeded) {
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
		} else {
			http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
		}
		return
	}

	// Log request details at DEBUG level
	p.logger.Debug("Processing request",
		"credential", cred.Name,
		"method", r.Method,
		"path", r.URL.Path,
		"model", modelID,
	)

	targetURL := strings.TrimSuffix(cred.BaseURL, "/") + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		p.logger.Error("Failed to create proxy request", "error", err, "url", targetURL)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	for key, values := range r.Header {
		if key == "Authorization" {
			continue
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)

	// Detailed debug logging
	p.logger.Debug("Proxy request details",
		"target_url", targetURL,
		"credential", cred.Name,
		"request_body", string(body),
	)

	// Log headers (mask Authorization for security)
	debugHeaders := make(map[string]string)
	for key, values := range proxyReq.Header {
		if key == "Authorization" {
			debugHeaders[key] = "Bearer " + maskKey(cred.APIKey)
		} else {
			debugHeaders[key] = strings.Join(values, ", ")
		}
	}
	p.logger.Debug("Proxy request headers", "headers", debugHeaders)

	resp, err := p.client.Do(proxyReq)
	if err != nil {
		p.logger.Error("Upstream request failed",
			"credential", cred.Name,
			"error", err,
			"url", targetURL,
		)
		statusCode := http.StatusBadGateway
		p.balancer.RecordResponse(cred.Name, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
		http.Error(w, "Bad Gateway", statusCode)
		return
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close response body", "error", closeErr)
		}
	}()

	p.balancer.RecordResponse(cred.Name, resp.StatusCode)
	p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

	// Log response headers
	debugRespHeaders := make(map[string]string)
	for key, values := range resp.Header {
		debugRespHeaders[key] = strings.Join(values, ", ")
	}
	p.logger.Debug("Proxy response received",
		"status_code", resp.StatusCode,
		"credential", cred.Name,
		"headers", debugRespHeaders,
	)

	// Read and log response body for non-streaming responses
	var responseBody []byte
	if isStreamingResponse(resp) {
		p.logger.Debug("Response is streaming", "credential", cred.Name)
	} else {
		responseBody, err = io.ReadAll(resp.Body)
		if err != nil {
			p.logger.Error("Failed to read response body", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)

		p.logger.Debug("Proxy response body",
			"credential", cred.Name,
			"content_encoding", contentEncoding,
			"body", decodedBody,
		)
		// Replace resp.Body with a new reader for subsequent processing
		resp.Body = io.NopCloser(bytes.NewReader(responseBody))
	}

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.WriteHeader(resp.StatusCode)

	if isStreamingResponse(resp) {
		p.handleStreaming(w, resp, cred.Name)
	} else {
		if _, err := io.Copy(w, resp.Body); err != nil {
			p.logger.Error("Failed to copy response body", "error", err)
		}
	}
}

func (p *Proxy) handleStreaming(w http.ResponseWriter, resp *http.Response, credName string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return
	}

	p.logger.Debug("Starting streaming response", "credential", credName)

	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk",
					"error", writeErr,
					"credential", credName,
				)
				return
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error",
					"error", err,
					"credential", credName,
				)
			}
			break
		}
	}

	p.logger.Debug("Streaming response completed", "credential", credName)
}

func isStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json")
}

func (p *Proxy) HealthCheck() (bool, map[string]interface{}) {
	totalCreds := len(p.balancer.GetCredentials())
	availableCreds := p.balancer.GetAvailableCount()
	bannedCreds := p.balancer.GetBannedCount()

	healthy := availableCreds > 0

	status := map[string]interface{}{
		"status":                "healthy",
		"credentials_available": availableCreds,
		"credentials_banned":    bannedCreds,
		"total_credentials":     totalCreds,
	}

	if !healthy {
		status["status"] = "unhealthy"
	}

	return healthy, status
}

// extractModelFromBody extracts the model ID from the request body
func extractModelFromBody(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var reqBody struct {
		Model string `json:"model"`
	}

	if err := json.Unmarshal(body, &reqBody); err != nil {
		return ""
	}

	return reqBody.Model
}

// decodeResponseBody decodes the response body based on Content-Encoding
func decodeResponseBody(body []byte, encoding string) string {
	// Check if response is gzip-encoded
	if strings.Contains(strings.ToLower(encoding), "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return string(body) // Return as-is if can't decode
		}
		defer func() {
			_ = reader.Close()
		}()

		decoded, err := io.ReadAll(reader)
		if err != nil {
			return string(body) // Return as-is if can't read
		}
		return string(decoded)
	}

	// Return as plain text
	return string(body)
}

// maskKey masks the API key for logging (shows only first 7 chars)
func maskKey(key string) string {
	if len(key) <= 7 {
		return "***"
	}
	return key[:7] + "..."
}
