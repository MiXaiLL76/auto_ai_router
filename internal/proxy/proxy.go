package proxy

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/transform/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/transform/vertex"
)

// hopByHopHeaders are headers that should not be proxied
// See RFC 7230 Section 6.1
var hopByHopHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"TE":                  true,
	"Trailers":            true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

// ResponseBodyMultiplier scales maxBodySizeMB for proxy response bodies.
// Allows responses to be significantly larger than request bodies (e.g., large files, exports).
const ResponseBodyMultiplier = 20

//go:embed health.html
var healthHTML string

type Proxy struct {
	balancer       *balancer.RoundRobin
	client         *http.Client
	logger         *slog.Logger
	maxBodySizeMB  int
	requestTimeout time.Duration
	metrics        *monitoring.Metrics
	masterKey      string
	rateLimiter    *ratelimit.RPMLimiter
	tokenManager   *auth.VertexTokenManager
	healthTemplate *template.Template // Cached template
	modelManager   *models.Manager    // Model manager for getting configured models
}

var (
	Version = "dev"
	Commit  = "unknown"
)

func New(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics, masterKey string, rateLimiter *ratelimit.RPMLimiter, tokenManager *auth.VertexTokenManager, modelManager *models.Manager, version, commit string) *Proxy {
	// Parse template once at startup
	tmpl, err := template.New("health").Funcs(template.FuncMap{
		"div": func(a, b int) int {
			if b == 0 {
				return 0
			}
			return a / b
		},
		"mul": func(a, b int) int {
			return a * b
		},
		"version": func() string {
			return version
		},
		"commit": func() string {
			return commit
		},
	}).Parse(healthHTML)
	if err != nil {
		logger.Error("Failed to parse health template at startup", "error", err)
		// Continue without template - will cause error on /vhealth requests
	}

	return &Proxy{
		balancer:       bal,
		logger:         logger,
		maxBodySizeMB:  maxBodySizeMB,
		requestTimeout: requestTimeout,
		metrics:        metrics,
		masterKey:      masterKey,
		rateLimiter:    rateLimiter,
		tokenManager:   tokenManager,
		healthTemplate: tmpl,
		modelManager:   modelManager,
		client: &http.Client{
			Timeout: requestTimeout,
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment, // Support HTTP_PROXY, HTTPS_PROXY, NO_PROXY
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				DisableKeepAlives:   false,
			},
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

// isHopByHopHeader checks if a header should not be proxied
func isHopByHopHeader(key string) bool {
	return hopByHopHeaders[key]
}

// ProxyResponse holds response details from a proxy credential
type ProxyResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

// executeProxyRequest executes a request to a proxy credential and returns response details.
// This is a private helper method to avoid code duplication between forwardToProxy and related functions.
func (p *Proxy) executeProxyRequest(
	r *http.Request,
	cred *config.CredentialConfig,
	body []byte,
	start time.Time,
) (*ProxyResponse, error) {
	// Build target URL
	proxyBaseURL := strings.TrimSuffix(cred.BaseURL, "/")
	targetURL := proxyBaseURL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	// Create proxy request
	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		p.logger.Error("Failed to create proxy request", "error", err, "url", targetURL)
		return nil, err
	}

	// Copy headers (skip hop-by-hop headers)
	for key, values := range r.Header {
		if isHopByHopHeader(key) {
			continue
		}
		if key == "Authorization" {
			// Handle Authorization header: use credential API key if available, otherwise copy original
			if cred.APIKey != "" {
				proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
			} else {
				// Copy original Authorization header if no API key configured
				for _, value := range values {
					proxyReq.Header.Add(key, value)
				}
			}
		} else {
			for _, value := range values {
				proxyReq.Header.Add(key, value)
			}
		}
	}

	// Send request
	resp, err := p.client.Do(proxyReq)
	if err != nil {
		p.logger.Error("Failed to proxy request",
			"credential", cred.Name,
			"error", err,
			"url", targetURL,
		)
		p.balancer.RecordResponse(cred.Name, http.StatusBadGateway)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, http.StatusBadGateway, time.Since(start))
		return nil, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close proxy response body", "error", closeErr)
		}
	}()

	// Record response
	p.balancer.RecordResponse(cred.Name, resp.StatusCode)
	p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

	p.logger.Debug("Proxy request forwarded",
		"credential", cred.Name,
		"target_url", targetURL,
		"status_code", resp.StatusCode,
		"duration", time.Since(start),
	)

	// Read response body to set correct Content-Length
	// Limit response body size: allow ResponseBodyMultiplier larger than request limit
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, int64(p.maxBodySizeMB)*ResponseBodyMultiplier*1024*1024))
	if err != nil {
		p.logger.Error("Failed to read proxy response body", "error", err)
		return nil, err
	}

	// Return complete response information
	return &ProxyResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       respBody,
	}, nil
}

// forwardToProxy forwards a request to a proxy credential and returns response details.
// This enables fallback retry logic at the caller level.
//
// Protection against infinite fallback recursion:
// - The caller (main proxy handler or TryFallbackProxy) decides if fallback retry should be attempted
// - This function does NOT perform fallback retry
// - This ensures each credential (fallback or not) is tried only once per request chain
// - Streaming responses are not retried (architectural limitation)
func (p *Proxy) forwardToProxy(
	w http.ResponseWriter,
	r *http.Request,
	modelID string,
	cred *config.CredentialConfig,
	body []byte,
	start time.Time,
) (*ProxyResponse, error) {
	return p.executeProxyRequest(r, cred, body, start)
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
	modelID, streaming, body := extractMetadataFromBody(body)

	// Select credential based on model availability
	if modelID == "" {
		p.logger.Error("Model not specified in request body")
		http.Error(w, "Bad Request: model field is required", http.StatusBadRequest)
		return
	}

	// Two-stage credential selection
	cred, err := p.balancer.NextForModel(modelID)
	if err != nil {
		// First stage failed, attempt fallback proxy
		// This includes rate limit errors - if primary credentials are exhausted,
		// fallback credentials should be tried
		fallbackErr := error(nil)
		cred, fallbackErr = p.balancer.NextFallbackForModel(modelID)

		if fallbackErr != nil {
			// Both stages failed
			err_code := http.StatusServiceUnavailable
			err_line := "Service Unavailable"
			if errors.Is(err, balancer.ErrRateLimitExceeded) {
				err_code = http.StatusTooManyRequests
				err_line = "Too Many Requests"
			}

			p.logger.Error("No credentials available (regular and fallback)",
				"model", modelID,
				"primary_error", err,
				"fallback_error", fallbackErr,
			)
			http.Error(w, err_line, err_code)
			return
		}
	}

	// Log request details at DEBUG level
	p.logger.Debug("Processing request",
		"credential", cred.Name,
		"method", r.Method,
		"path", r.URL.Path,
		"model", modelID,
		"type", cred.Type,
	)

	// Handle proxy credential type with fallback retry support
	if cred.Type == config.ProviderTypeProxy {
		proxyResp, err := p.forwardToProxy(w, r, modelID, cred, body, start)
		if err != nil {
			// Network error
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
			return
		}

		// Check if response is streaming (streaming responses don't support fallback retry)
		isStreaming := IsStreamingResponse(&http.Response{
			StatusCode: proxyResp.StatusCode,
			Header:     proxyResp.Headers,
		})

		if isStreaming {
			p.logger.Debug("Response is streaming (no fallback retry for streaming)",
				"credential", cred.Name,
				"status", proxyResp.StatusCode,
			)
		}

		// Check if we should retry with fallback proxy
		// - Only if credential is NOT already a fallback
		// - Only if response is NOT streaming
		var shouldRetry bool
		var retryReason RetryReason
		if !cred.IsFallback && !isStreaming {
			shouldRetry, retryReason = ShouldRetryWithFallback(proxyResp.StatusCode, proxyResp.Body)
		}

		if shouldRetry {
			p.logger.Info("Attempting fallback retry for proxy credential",
				"credential", cred.Name,
				"status", proxyResp.StatusCode,
				"reason", retryReason,
				"model", modelID,
			)
			// Try fallback - it will write response directly if successful
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, proxyResp.StatusCode, retryReason, body, start)
			if success {
				return
			}
			// If fallback didn't work, continue with original response
			p.logger.Debug("Fallback retry failed, using original response",
				"credential", cred.Name,
				"status", proxyResp.StatusCode,
				"fallback_reason", fallbackReason,
			)
		}

		// Write response headers
		for key, values := range proxyResp.Headers {
			if isHopByHopHeader(key) {
				continue
			}
			// Skip Content-Length and Transfer-Encoding - we'll set them correctly
			if key == "Content-Length" || key == "Transfer-Encoding" {
				continue
			}
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}

		// Set correct Content-Length for the actual response body
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(proxyResp.Body)))

		w.WriteHeader(proxyResp.StatusCode)
		if _, err := w.Write(proxyResp.Body); err != nil {
			p.logger.Error("Failed to write proxy response body", "error", err)
		}
		return
	}

	// Build target URL based on credential type
	var targetURL string
	var vertexToken string
	var requestBody = body // Default to original body
	var isImageGeneration = strings.Contains(r.URL.Path, "/images/generations")

	switch cred.Type {
	case config.ProviderTypeVertexAI:
		vertexBody, err := vertex.OpenAIToVertex(body, isImageGeneration, modelID)
		if err != nil {
			p.logger.Error("Failed to vertex request to Vertex AI format",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to vertex request", http.StatusInternalServerError)
			return
		}
		requestBody = vertexBody

		if isImageGeneration && !strings.Contains(modelID, "gemini") {
			// For non-Gemini image generation (e.g., Imagen), use different endpoint
			targetURL = vertex.BuildVertexImageURL(cred, modelID)
		} else {
			// For text generation or Gemini image generation (which uses chat API)
			targetURL = vertex.BuildVertexURL(cred, modelID, streaming)
		}

		// Get OAuth2 token for Vertex AI
		vertexToken, err = p.tokenManager.GetToken(cred.Name, cred.CredentialsFile, cred.CredentialsJSON)
		if err != nil {
			p.logger.Error("Failed to get Vertex AI token",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to authenticate with Vertex AI", http.StatusInternalServerError)
			return
		}

	case config.ProviderTypeAnthropic:
		if isImageGeneration {
			p.logger.Error("Failed to Anthropic image request",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to Anthropic image request", http.StatusInternalServerError)
			return
		}

		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		targetURL = baseURL + "/v1/messages"

		anthropicBody, err := anthropic.OpenAIToAnthropic(body)
		if err != nil {
			p.logger.Error("Failed to Anthropic request transformation",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to transform request", http.StatusInternalServerError)
			return
		}
		requestBody = anthropicBody

	default:
		// For OpenAI and other providers, use baseURL + path
		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		path := r.URL.Path

		// If baseURL already ends with /v1 and path starts with /v1, remove /v1 from path to avoid duplication
		// Example: baseURL="https://api.openai.azure.com/openai/v1" + path="/v1/chat/completions"
		// Should become: "https://api.openai.azure.com/openai/v1/chat/completions" (not /v1/v1/...)
		if strings.HasSuffix(baseURL, "/v1") && strings.HasPrefix(path, "/v1") {
			path = strings.TrimPrefix(path, "/v1")
		}

		targetURL = baseURL + path
		if r.URL.RawQuery != "" {
			targetURL += "?" + r.URL.RawQuery
		}
	}

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(requestBody))
	if err != nil {
		p.logger.Error("Failed to create proxy request", "error", err, "url", targetURL)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	// Copy headers (skip hop-by-hop headers and Authorization)
	for key, values := range r.Header {
		if isHopByHopHeader(key) || key == "Authorization" {
			continue
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Set Authorization header based on credential type
	switch cred.Type {
	case config.ProviderTypeVertexAI:
		// For Vertex AI, use OAuth2 token
		proxyReq.Header.Set("Authorization", "Bearer "+vertexToken)
	case config.ProviderTypeAnthropic:
		// For Anthropic, use X-Api-Key and required version header
		proxyReq.Header.Set("X-Api-Key", cred.APIKey)
		proxyReq.Header.Set("anthropic-version", "2023-06-01")
	default:
		// For OpenAI and other providers, use API key
		proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
	}

	// Detailed debug logging (truncate long fields for readability)
	p.logger.Debug("Proxy request details",
		"target_url", targetURL,
		"credential", cred.Name,
		"request_body", logger.TruncateLongFields(string(requestBody), 500),
	)

	// Log headers (mask Authorization for security)
	debugHeaders := make(map[string]string)
	for key, values := range proxyReq.Header {
		switch key {
		case "Authorization":
			debugHeaders[key] = "Bearer " + maskKey(cred.APIKey)
		case "X-Api-Key":
			debugHeaders[key] = maskKey(cred.APIKey)
		default:
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
	var finalResponseBody []byte // Define here for broader scope
	if IsStreamingResponse(resp) {
		p.logger.Debug("Response is streaming", "credential", cred.Name)
	} else {
		responseBody, err = io.ReadAll(resp.Body)
		if err != nil {
			p.logger.Error("Failed to read response body", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Check if we should retry with fallback proxy
		shouldRetry, retryReason := ShouldRetryWithFallback(resp.StatusCode, responseBody)
		if shouldRetry {
			p.logger.Info("Attempting fallback retry",
				"credential", cred.Name,
				"status", resp.StatusCode,
				"reason", retryReason,
				"model", modelID,
			)
			// Try fallback - it will write response directly if successful
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, resp.StatusCode, retryReason, body, start)
			if success {
				return
			}
			// If fallback didn't work, continue with original response
			p.logger.Debug("Fallback retry failed, using original response",
				"credential", cred.Name,
				"status", resp.StatusCode,
				"fallback_reason", fallbackReason,
			)
		}

		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)

		// Transform response back to OpenAI format
		switch cred.Type {
		case config.ProviderTypeVertexAI:
			if isImageGeneration {
				// Transform image response
				var vertexedBody []byte
				var err error

				if !strings.Contains(modelID, "gemini") {
					// Non-Gemini image generation response (e.g., Imagen)
					vertexedBody, err = vertex.VertexImageToOpenAI([]byte(decodedBody))
				} else {
					// Gemini image generation response (from chat API)
					vertexedBody, err = vertex.VertexChatResponseToOpenAIImage([]byte(decodedBody))
				}

				if err != nil {
					p.logger.Error("Failed to transform Vertex AI image response to OpenAI format",
						"credential", cred.Name,
						"error", err,
					)
					finalResponseBody = responseBody
				} else {
					finalResponseBody = vertexedBody
					p.logger.Debug("Transformed Vertex AI image response to OpenAI format",
						"credential", cred.Name,
						"vertexed_body", logger.TruncateLongFields(string(vertexedBody), 200),
					)
				}
			} else {
				// Transform text response
				vertexedBody, err := vertex.VertexToOpenAI([]byte(decodedBody), modelID)
				if err != nil {
					p.logger.Error("Failed to vertex Vertex AI response",
						"credential", cred.Name,
						"error", err,
					)
					finalResponseBody = responseBody
				} else {
					finalResponseBody = vertexedBody
					p.logger.Debug("Transformed Vertex AI response to OpenAI format",
						"credential", cred.Name,
						"vertexed_body", logger.TruncateLongFields(string(vertexedBody), 500),
					)
				}
			}
		case config.ProviderTypeAnthropic:
			// Transform text response
			anthropicBody, err := anthropic.AnthropicToOpenAI([]byte(decodedBody), modelID)
			if err != nil {
				p.logger.Error("Failed to vertex Anthropic response",
					"credential", cred.Name,
					"error", err,
				)
				finalResponseBody = responseBody
			} else {
				finalResponseBody = anthropicBody
				p.logger.Debug("Transformed Anthropic response to OpenAI format",
					"credential", cred.Name,
					"vertexed_body", logger.TruncateLongFields(string(anthropicBody), 500),
				)
			}
		default:
			finalResponseBody = responseBody
		}

		// Extract and record token usage (after transformation to OpenAI format)
		bodyForTokenExtraction := finalResponseBody
		if len(bodyForTokenExtraction) == 0 {
			// For direct OpenAI responses (no transformation), use decoded body
			bodyForTokenExtraction = []byte(decodedBody)
		}
		tokens := extractTokensFromResponse(string(bodyForTokenExtraction), config.ProviderTypeOpenAI)
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.Debug("Token usage recorded",
				"credential", cred.Name,
				"model", modelID,
				"tokens", tokens,
			)
		}

		p.logger.Debug("Proxy response body",
			"credential", cred.Name,
			"content_encoding", contentEncoding,
			"body", logger.TruncateLongFields(decodedBody, 500),
		)
		// Replace resp.Body with a new reader for subsequent processing
		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	for key, values := range resp.Header {
		// Skip hop-by-hop headers
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length and Content-Encoding as we may have transformed the response
		switch cred.Type {
		case config.ProviderTypeVertexAI, config.ProviderTypeAnthropic:
			if key == "Content-Length" || key == "Content-Encoding" {
				continue
			}
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set correct Content-Length for vertexed responses
	switch cred.Type {
	case config.ProviderTypeVertexAI, config.ProviderTypeAnthropic:
		if len(finalResponseBody) > 0 {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(finalResponseBody)))
		}
	}

	w.WriteHeader(resp.StatusCode)

	if IsStreamingResponse(resp) {
		switch cred.Type {
		case config.ProviderTypeVertexAI:
			// Handle Vertex AI streaming with token tracking
			err := p.handleVertexStreaming(w, resp, cred.Name, modelID)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
			}
		case config.ProviderTypeAnthropic:
			// Handle Anthropic streaming with token tracking
			err := p.handleAnthropicStreaming(w, resp, cred.Name, modelID)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
			}
		default:
			p.handleStreamingWithTokens(w, resp, cred.Name, modelID)
		}

	} else {
		if _, err := io.Copy(w, resp.Body); err != nil {
			p.logger.Error("Failed to copy response body", "error", err)
		}
	}
}

type tokenCapturingWriter struct {
	writer io.Writer
	tokens *int
	logger *slog.Logger
}

func (tcw *tokenCapturingWriter) Write(p []byte) (n int, err error) {
	// Extract tokens from the data being written
	tokens := extractTokensFromStreamingChunk(string(p))
	if tokens > 0 {
		*tcw.tokens += tokens
	}
	return tcw.writer.Write(p)
}

func extractTokensFromStreamingChunk(chunk string) int {
	// Look for usage information in streaming chunks
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}

			var chunkData map[string]interface{}
			if err := json.Unmarshal([]byte(jsonData), &chunkData); err != nil {
				continue
			}

			// Check for usage field in OpenAI format
			if usage, ok := chunkData["usage"].(map[string]interface{}); ok {
				if totalTokens, ok := usage["total_tokens"].(float64); ok {
					return int(totalTokens)
				}
			}
		}
	}
	return 0
}

// extractMetadataFromBody extracts the model ID from the request body
func extractMetadataFromBody(body []byte) (string, bool, []byte) {
	if len(body) == 0 {
		return "", false, body
	}

	// var reqBody struct {
	// 	Model string `json:"model"`
	// 	Stream bool `json:"stream,omitempty"`
	// 	StreamOptions map[string]interface{} `json:"stream_options,omitempty"`
	// }
	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return "", false, body // Return original if parsing fails
	}

	model, ok := reqBody["model"].(string)
	if !ok {
		return "", false, body // Return original if model is missing or not streaming
	}

	// Check if this is a streaming request
	stream, ok := reqBody["stream"].(bool)
	if !ok || !stream {
		return model, false, body // Not a streaming request, return as-is
	}

	// Ensure stream_options exists and include_usage is true
	streamOptions, exists := reqBody["stream_options"]
	if !exists {
		// Create stream_options if it doesn't exist
		reqBody["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	} else if streamOptionsMap, ok := streamOptions.(map[string]interface{}); ok {
		// Update existing stream_options to ensure include_usage is true
		streamOptionsMap["include_usage"] = true
	} else {
		// stream_options exists but is not a map, replace it
		reqBody["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	}

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(reqBody)
	if err != nil {
		return model, stream, body // Return original if marshaling fails
	}

	return model, stream, modifiedBody
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

// extractTokensFromResponse extracts total_tokens from the response body
// Supports both OpenAI format (usage.total_tokens) and Vertex AI format (usageMetadata.totalTokenCount)
func extractTokensFromResponse(body string, credType config.ProviderType) int {
	if body == "" {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == config.ProviderTypeVertexAI {
		var vertexResp struct {
			UsageMetadata struct {
				TotalTokenCount int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}

		if err := json.Unmarshal([]byte(body), &vertexResp); err != nil {
			return 0
		}

		return vertexResp.UsageMetadata.TotalTokenCount
	}

	// For OpenAI and other providers, use standard format
	var openAIResp struct {
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal([]byte(body), &openAIResp); err != nil {
		return 0
	}

	return openAIResp.Usage.TotalTokens
}

// maskKey masks the API key for logging (shows only first 7 chars)
func maskKey(key string) string {
	if len(key) <= 7 {
		return "***"
	}
	return key[:7] + "..."
}
