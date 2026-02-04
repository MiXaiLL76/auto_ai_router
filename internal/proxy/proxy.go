package proxy

import (
	"bytes"
	"context"
	_ "embed"
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

// ResponseBodyMultiplier scales maxBodySizeMB for proxy response bodies.
// Allows responses to be significantly larger than request bodies (e.g., large files, exports).
const ResponseBodyMultiplier = 20

//go:embed health.html
var healthHTML string

// Config holds all configuration needed to create a Proxy
type Config struct {
	Balancer       *balancer.RoundRobin
	Logger         *slog.Logger
	MaxBodySizeMB  int
	RequestTimeout time.Duration
	Metrics        *monitoring.Metrics
	MasterKey      string
	RateLimiter    *ratelimit.RPMLimiter
	TokenManager   *auth.VertexTokenManager
	ModelManager   *models.Manager
	Version        string
	Commit         string
}

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

func New(cfg *Config) *Proxy {
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
			return cfg.Version
		},
		"commit": func() string {
			return cfg.Commit
		},
	}).Parse(healthHTML)
	if err != nil {
		cfg.Logger.Error("Failed to parse health template at startup", "error", err)
		// Continue without template - will cause error on /vhealth requests
	}

	return &Proxy{
		balancer:       cfg.Balancer,
		logger:         cfg.Logger,
		maxBodySizeMB:  cfg.MaxBodySizeMB,
		requestTimeout: cfg.RequestTimeout,
		metrics:        cfg.Metrics,
		masterKey:      cfg.MasterKey,
		rateLimiter:    cfg.RateLimiter,
		tokenManager:   cfg.TokenManager,
		healthTemplate: tmpl,
		modelManager:   cfg.ModelManager,
		client: &http.Client{
			Timeout: cfg.RequestTimeout,
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
	copyRequestHeaders(proxyReq, r, cred.APIKey)

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

		// Check if response is streaming by examining Content-Type header directly
		// (avoid creating temporary http.Response object)
		contentType := strings.ToLower(proxyResp.Headers.Get("Content-Type"))
		isStreaming := strings.Contains(contentType, "text/event-stream")

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
			p.logger.Error("Failed to convert request to Vertex AI format",
				"credential", cred.Name,
				"error", err,
			)
			http.Error(w, "Internal Server Error: Failed to convert request to Vertex AI format", http.StatusInternalServerError)
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
	copyHeadersSkipAuth(proxyReq, r)

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
	if p.logger.Enabled(context.Background(), slog.LevelDebug) {
		p.logger.Debug("Proxy request details",
			"target_url", targetURL,
			"credential", cred.Name,
			"request_body", logger.TruncateLongFields(string(requestBody), 500),
		)
	}

	// Log headers (omit auth headers for security)
	debugHeaders := make(map[string]string)
	for key, values := range proxyReq.Header {
		switch key {
		case "Authorization":
			continue
		case "X-Api-Key":
			continue
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

	// Check if response is streaming once and cache the result
	isStreamingResp := IsStreamingResponse(resp)

	// Read and log response body for non-streaming responses
	var responseBody []byte
	finalResponseBody := make([]byte, 0) // Initialize to empty slice to avoid nil panics
	if isStreamingResp {
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
					p.logTransformedResponse(cred.Name, "Vertex AI image", vertexedBody)
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
					p.logTransformedResponse(cred.Name, "Vertex AI", vertexedBody)
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
				p.logTransformedResponse(cred.Name, "Anthropic", anthropicBody)
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

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			p.logger.Debug("Proxy response body",
				"credential", cred.Name,
				"content_encoding", contentEncoding,
				"body", logger.TruncateLongFields(decodedBody, 500),
			)
		}
		// Replace resp.Body with a new reader for subsequent processing.
		// NopCloser is required because http.Response.Body must be an io.ReadCloser.
		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	copyResponseHeaders(w, resp.Header, cred.Type)

	// Set correct Content-Length for vertexed responses
	switch cred.Type {
	case config.ProviderTypeVertexAI, config.ProviderTypeAnthropic:
		if len(finalResponseBody) > 0 {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(finalResponseBody)))
		}
	}

	w.WriteHeader(resp.StatusCode)

	if isStreamingResp {
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
			err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID)
			if err != nil {
				p.logger.Error("Failed to handle streaming response", "error", err)
			}
		}

	} else {
		if _, err := p.streamResponseBody(w, resp.Body); err != nil {
			p.logger.Error("Failed to copy response body", "error", err)
		}
	}
}

// streamResponseBody streams a response body to the client using a pooled buffer
// to minimize memory allocations for large responses
func (p *Proxy) streamResponseBody(w io.Writer, reader io.Reader) (int64, error) {
	buf := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(buf)
	return io.CopyBuffer(w, reader, *buf)
}

// logTransformedResponse logs a transformed response at debug level
func (p *Proxy) logTransformedResponse(credName, providerName string, body []byte) {
	if p.logger.Enabled(context.Background(), slog.LevelDebug) {
		p.logger.Debug("Transformed response to OpenAI format",
			"credential", credName,
			"provider", providerName,
			"body", logger.TruncateLongFields(string(body), 500),
		)
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
