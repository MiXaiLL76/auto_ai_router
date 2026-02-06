package proxy

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/mixaill76/auto_ai_router/internal/transform/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/transform/vertex"
)

// ResponseBodyMultiplier scales maxBodySizeMB for proxy response bodies.
// Allows responses to be significantly larger than request bodies (e.g., large files, exports).
const ResponseBodyMultiplier = 20

//go:embed health.html
var healthHTML string

// HealthChecker provides cached database health status
type HealthChecker interface {
	IsDBHealthy() bool
}

// Config holds all configuration needed to create a Proxy
type Config struct {
	Balancer            *balancer.RoundRobin
	Logger              *slog.Logger
	MaxBodySizeMB       int
	RequestTimeout      time.Duration
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
	Metrics             *monitoring.Metrics
	MasterKey           string
	RateLimiter         *ratelimit.RPMLimiter
	TokenManager        *auth.VertexTokenManager
	ModelManager        *models.Manager
	Version             string
	Commit              string
	LiteLLMDB           litellmdb.Manager // LiteLLM database integration (optional)
	HealthChecker       HealthChecker     // Optional: cached DB health status (updated by health monitor)
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
	litellmDB      litellmdb.Manager  // LiteLLM database integration
	healthChecker  HealthChecker      // Cached DB health status (optional)
}

var (
	Version = "dev"
	Commit  = "unknown"
)

// isTimeoutError checks if an error is a timeout error
func isTimeoutError(err error) bool {
	if err == nil {
		return false
	}

	// Check for context deadline exceeded
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	// Check for net.Error timeout
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	return false
}

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
		litellmDB:      cfg.LiteLLMDB,
		healthChecker:  cfg.HealthChecker,
		client: &http.Client{
			Timeout: cfg.RequestTimeout,
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment, // Support HTTP_PROXY, HTTPS_PROXY, NO_PROXY
				MaxIdleConns:        cfg.MaxIdleConns,
				MaxIdleConnsPerHost: cfg.MaxIdleConnsPerHost,
				IdleConnTimeout:     cfg.IdleConnTimeout,
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
		statusCode := http.StatusBadGateway
		if isTimeoutError(err) {
			statusCode = http.StatusRequestTimeout
			p.logger.Error("Proxy request timeout",
				"credential", cred.Name,
				"error", err,
				"url", targetURL,
			)
		} else {
			p.logger.Error("Failed to proxy request",
				"credential", cred.Name,
				"error", err,
				"url", targetURL,
			)
		}
		p.balancer.RecordResponse(cred.Name, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
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
	start := time.Now().UTC()
	requestID := uuid.New().String()

	// Initialize request context with fallback retry tracking
	// This prevents infinite loops by tracking which credentials have been attempted
	ctx := r.Context()
	triedCreds := make(map[string]bool)
	ctx = SetTried(ctx, triedCreds)
	// Initialize attempt counter to 0 (incremented before first attempt)
	ctx = context.WithValue(ctx, AttemptCountKey{}, 0)
	r = r.WithContext(ctx)

	// Cache LiteLLM DB health check once per request to reduce concurrent calls
	// Use cached health status from health monitor if available, otherwise call IsHealthy()
	var isLiteLLMHealthy bool
	if p.litellmDB != nil && p.litellmDB.IsEnabled() {
		if p.healthChecker != nil {
			// Use cached health status from health monitor
			isLiteLLMHealthy = p.healthChecker.IsDBHealthy()
		} else {
			// Fallback to direct health check if no health checker provided
			isLiteLLMHealthy = p.litellmDB.IsHealthy()
		}
	}

	// Verify Authorization header
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

	// Auth variables for logging
	var tokenInfo *litellmdb.TokenInfo

	// Try LiteLLM DB authentication first if enabled and healthy
	if isLiteLLMHealthy {
		var err error
		tokenInfo, err = p.litellmDB.ValidateToken(r.Context(), token)
		if err != nil {
			// Handle specific auth errors
			if p.handleLiteLLMAuthError(w, err, token) {
				return
			}
			// Connection failed - fallback to master_key
			p.logger.Warn("LiteLLM DB unavailable, falling back to master_key")
			if !p.validateMasterKeyOrError(w, token) {
				return
			}
		} else if tokenInfo != nil {
			// Successfully authenticated via LiteLLM DB
			p.logger.Debug("Token validated via LiteLLM DB",
				"user_id", tokenInfo.UserID,
				"team_id", tokenInfo.TeamID,
			)
		}
	} else {
		// LiteLLM DB disabled - use master_key only
		if !p.validateMasterKeyOrError(w, token) {
			return
		}
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

	// Extract model and session ID from request body if present
	modelID, streaming, sessionID, body := extractMetadataFromBody(body)

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

	// Add primary credential to tried set (for fallback retry tracking)
	ctx = r.Context()
	currentTriedCreds := GetTried(ctx)
	currentTriedCreds[cred.Name] = true
	ctx = SetTried(ctx, currentTriedCreds)
	r = r.WithContext(ctx)

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
			statusCode := http.StatusBadGateway
			statusMessage := "Bad Gateway"
			if isTimeoutError(err) {
				statusCode = http.StatusRequestTimeout
				statusMessage = "Request Timeout"
			}
			http.Error(w, statusMessage, statusCode)
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
		statusCode := http.StatusBadGateway
		statusMessage := "Bad Gateway"

		if isTimeoutError(err) {
			statusCode = http.StatusRequestTimeout
			statusMessage = "Request Timeout"
			p.logger.Error("Upstream request timeout",
				"credential", cred.Name,
				"error", err,
				"url", targetURL,
			)
		} else {
			p.logger.Error("Upstream request failed",
				"credential", cred.Name,
				"error", err,
				"url", targetURL,
			)
		}

		p.balancer.RecordResponse(cred.Name, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
		http.Error(w, statusMessage, statusCode)
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

		// Log to LiteLLM DB (non-streaming)
		promptTokens, completionTokens := extractTokenCounts(bodyForTokenExtraction)
		status := "success"
		if resp.StatusCode >= 400 {
			status = "failure"
		}
		if err := p.logSpendToLiteLLMDB(requestID, start, r, token, modelID, promptTokens, completionTokens, status, cred, sessionID, targetURL, tokenInfo); err != nil {
			p.logger.Warn("Failed to queue spend log",
				"error", err,
				"request_id", requestID,
			)
		}
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

// ==================== LiteLLM DB Integration ====================

// validateMasterKeyOrError validates the master key token and writes HTTP error if invalid
// Returns true if token is valid, false otherwise
func (p *Proxy) validateMasterKeyOrError(w http.ResponseWriter, token string) bool {
	if token != p.masterKey {
		p.logger.Error("Invalid master key", "provided_key_prefix", security.MaskAPIKey(token))
		http.Error(w, "Unauthorized: Invalid master key", http.StatusUnauthorized)
		return false
	}
	return true
}

// handleLiteLLMAuthError handles LiteLLM authentication errors
// Returns true if error was handled and response was written
func (p *Proxy) handleLiteLLMAuthError(w http.ResponseWriter, err error, token string) bool {
	// Map error types to HTTP status and message
	errorMap := map[error]struct {
		status  int
		message string
		logMsg  string
	}{
		litellmdb.ErrTokenNotFound:  {http.StatusUnauthorized, "Invalid token", "Token not found"},
		litellmdb.ErrTokenBlocked:   {http.StatusForbidden, "Token blocked", "Token blocked"},
		litellmdb.ErrTokenExpired:   {http.StatusUnauthorized, "Token expired", "Token expired"},
		litellmdb.ErrBudgetExceeded: {http.StatusPaymentRequired, "Budget exceeded", "Budget exceeded"},
	}

	// Check for connection failure first (requires fallback, not an error response)
	if errors.Is(err, litellmdb.ErrConnectionFailed) {
		return false
	}

	// Check for known auth errors
	for errType, info := range errorMap {
		if errors.Is(err, errType) {
			p.logger.Error(info.logMsg, "token_prefix", security.MaskAPIKey(token))
			http.Error(w, "Unauthorized: "+info.message, info.status)
			return true
		}
	}

	// Unknown error
	p.logger.Error("Auth error", "error", err, "token_prefix", security.MaskAPIKey(token))
	http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	return true
}

// logSpendToLiteLLMDB logs request to LiteLLM_SpendLogs table
// Returns error if the log entry cannot be queued (e.g., queue full)
func (p *Proxy) logSpendToLiteLLMDB(
	requestID string,
	start time.Time,
	r *http.Request,
	token string,
	modelID string,
	promptTokens, completionTokens int,
	status string,
	cred *config.CredentialConfig,
	sessionID string,
	targetURL string,
	tokenInfo *litellmdb.TokenInfo,
) error {
	if p.litellmDB == nil || !p.litellmDB.IsEnabled() {
		return nil
	}

	// Fallback to request ID if session ID not provided
	if sessionID == "" {
		sessionID = requestID
	}

	// Build model_id as credential.name:model_name
	modelIDFormatted := cred.Name + ":" + modelID
	hashedToken := litellmdb.HashToken(token)

	// Extract user info from tokenInfo (or use empty strings as fallback)
	var userID, teamID, organizationID string
	if tokenInfo != nil {
		userID = tokenInfo.UserID
		teamID = tokenInfo.TeamID
		organizationID = tokenInfo.OrganizationID
	}

	// Build metadata with optional alias fields from tokenInfo
	metadata := buildMetadata(hashedToken, tokenInfo)

	// Determine end user - prefer user email from tokenInfo
	endUser := extractEndUser(r)
	if tokenInfo != nil && tokenInfo.UserEmail != "" {
		endUser = tokenInfo.UserEmail
	}

	// Extract domain from targetURL for APIBase (e.g., "https://api.openai.com/..." -> "api.openai.com")
	apiBase := "auto_ai_router"
	if targetURL != "" {
		if u, err := url.Parse(targetURL); err == nil && u.Host != "" {
			apiBase = u.Host
		}
	}

	return p.litellmDB.LogSpend(&litellmdb.SpendLogEntry{
		RequestID:         requestID,
		StartTime:         start,
		EndTime:           time.Now().UTC(),
		CallType:          r.URL.Path,
		APIBase:           apiBase,
		Model:             modelID,           // Model name
		ModelID:           modelIDFormatted,  // credential.name:model_name
		ModelGroup:        "",                // Empty for now
		CustomLLMProvider: string(cred.Type), // Provider type as string
		PromptTokens:      promptTokens,
		CompletionTokens:  completionTokens,
		TotalTokens:       promptTokens + completionTokens,
		Metadata:          metadata,
		Spend:             1, // TODO: implement cost calculation
		APIKey:            hashedToken,
		UserID:            userID,
		TeamID:            teamID,
		OrganizationID:    organizationID,
		EndUser:           endUser,
		RequesterIP:       getClientIP(r),
		Status:            status,
		SessionID:         sessionID,
	})
}

// buildMetadata builds metadata JSON with user and team alias information
func buildMetadata(hashedToken string, tokenInfo *litellmdb.TokenInfo) string {
	// Extract user info from tokenInfo (or use empty strings as fallback)
	var userID, teamID, organizationID string
	if tokenInfo != nil {
		userID = tokenInfo.UserID
		teamID = tokenInfo.TeamID
		organizationID = tokenInfo.OrganizationID
	}

	// Base metadata always includes additional_usage_values
	metadata := map[string]interface{}{
		"additional_usage_values": map[string]interface{}{
			"prompt_tokens_details":     nil,
			"completion_tokens_details": nil,
		},
		"user_api_key":         hashedToken,
		"user_api_key_org_id":  organizationID,
		"user_api_key_team_id": teamID,
		"user_api_key_user_id": userID,
	}

	// Add aliases from tokenInfo if available
	if tokenInfo != nil {
		if tokenInfo.KeyAlias != "" {
			metadata["user_api_key_alias"] = tokenInfo.KeyAlias
		}
		if tokenInfo.UserAlias != "" {
			metadata["user_api_key_user_alias"] = tokenInfo.UserAlias
		}
		if tokenInfo.TeamAlias != "" {
			metadata["user_api_key_team_alias"] = tokenInfo.TeamAlias
		}
	}

	// Convert to JSON
	jsonBytes, err := json.Marshal(metadata)
	if err != nil {
		// Fallback to simple format if marshaling fails
		return fmt.Sprintf(`{"user_api_key":"%s","user_api_key_org_id":"%s","user_api_key_team_id":"%s","user_api_key_user_id":"%s"}`, hashedToken, organizationID, teamID, userID)
	}
	return string(jsonBytes)
}

// extractEndUser extracts end_user from request headers or body
func extractEndUser(r *http.Request) string {
	// Check X-End-User header first
	if endUser := r.Header.Get("X-End-User"); endUser != "" {
		return endUser
	}
	return ""
}

// getClientIP gets the client IP address
func getClientIP(r *http.Request) string {
	// X-Forwarded-For header (first IP)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	// X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	// RemoteAddr
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// extractTokenCounts extracts token counts from response body
func extractTokenCounts(body []byte) (promptTokens, completionTokens int) {
	var resp struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return 0, 0
	}
	return resp.Usage.PromptTokens, resp.Usage.CompletionTokens
}
