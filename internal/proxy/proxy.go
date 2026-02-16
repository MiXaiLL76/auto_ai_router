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
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// DefaultResponseBodyMultiplier is the default multiplier for response body size limit
// relative to maxBodySizeMB. Responses can be larger than requests (e.g., base64 images).
// Can be overridden via Config.ResponseBodyMultiplier.
const DefaultResponseBodyMultiplier = 10

//go:embed health.html
var healthHTML string

// RequestLogContext holds all data needed for logging a request to LiteLLM DB
// Filled throughout request processing and logged at the end via defer
type RequestLogContext struct {
	RequestID            string                   // Request ID (UUID)
	StartTime            time.Time                // Request start time
	Request              *http.Request            // HTTP request
	Token                string                   // Auth token (raw, will be hashed)
	ModelID              string                   // Model name
	Status               string                   // "success" or "failure"
	HTTPStatus           int                      // HTTP response status code
	ErrorMsg             string                   // Error message (added to metadata on failure)
	TokenUsage           *converter.TokenUsage    // Token usage with detailed breakdown
	Credential           *config.CredentialConfig // Credential used
	SessionID            string                   // Session ID
	TargetURL            string                   // Target URL (for APIBase extraction)
	TokenInfo            *litellmdb.TokenInfo     // User/team/org info
	IsImageGeneration    bool                     // True if this is an image generation request
	ImageCount           int                      // Number of images to generate (from 'n' param)
	Logged               bool                     // True if already logged (prevents duplicate logging)
	PromptTokensEstimate int                      // Estimated prompt tokens for streaming responses (since streaming doesn't provide prompt tokens in headers)
}

// HealthChecker provides cached database health status
type HealthChecker interface {
	IsDBHealthy() bool
}

// Config holds all configuration needed to create a Proxy
type Config struct {
	Balancer               *balancer.RoundRobin
	Logger                 *slog.Logger
	MaxBodySizeMB          int
	ResponseBodyMultiplier int // Multiplier for response body size limit (default: DefaultResponseBodyMultiplier)
	RequestTimeout         time.Duration
	MaxIdleConns           int
	MaxIdleConnsPerHost    int
	IdleConnTimeout        time.Duration
	Metrics                *monitoring.Metrics
	MasterKey              string
	RateLimiter            *ratelimit.RPMLimiter
	TokenManager           *auth.VertexTokenManager
	ModelManager           *models.Manager
	Version                string
	Commit                 string
	LiteLLMDB              litellmdb.Manager          // LiteLLM database integration (optional)
	HealthChecker          HealthChecker              // Optional: cached DB health status (updated by health monitor)
	PriceRegistry          *models.ModelPriceRegistry // Model pricing information (optional)
}

type Proxy struct {
	balancer            *balancer.RoundRobin
	client              *http.Client
	logger              *slog.Logger
	maxBodySizeMB       int
	maxResponseBodySize int64 // Pre-computed max response body size in bytes
	requestTimeout      time.Duration
	metrics             *monitoring.Metrics
	masterKey           string
	rateLimiter         *ratelimit.RPMLimiter
	tokenManager        *auth.VertexTokenManager
	healthTemplate      *template.Template         // Cached template
	modelManager        *models.Manager            // Model manager for getting configured models
	litellmDB           litellmdb.Manager          // LiteLLM database integration
	healthChecker       HealthChecker              // Cached DB health status (optional)
	priceRegistry       *models.ModelPriceRegistry // Model pricing information (optional)
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

// isClientDisconnectError checks if an error indicates the client disconnected
// (broken pipe, connection reset, context canceled). These are expected during
// normal operation and should be logged at lower severity.
func isClientDisconnectError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return true
	}
	if errors.Is(err, syscall.EPIPE) {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "write: broken pipe") ||
		strings.Contains(msg, "connection reset by peer")
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

	// Create HTTP client using centralized factory with request-specific timeout
	httpClientCfg := httputil.DefaultHTTPClientConfig()
	httpClientCfg.Timeout = cfg.RequestTimeout
	httpClientCfg.MaxIdleConns = cfg.MaxIdleConns
	httpClientCfg.MaxIdleConnsPerHost = cfg.MaxIdleConnsPerHost
	httpClientCfg.IdleConnTimeout = cfg.IdleConnTimeout

	// Compute max response body size from multiplier
	multiplier := cfg.ResponseBodyMultiplier
	if multiplier <= 0 {
		multiplier = DefaultResponseBodyMultiplier
	}
	maxResponseBodySize := int64(cfg.MaxBodySizeMB) * int64(multiplier) * 1024 * 1024

	return &Proxy{
		balancer:            cfg.Balancer,
		logger:              cfg.Logger,
		maxBodySizeMB:       cfg.MaxBodySizeMB,
		maxResponseBodySize: maxResponseBodySize,
		requestTimeout:      cfg.RequestTimeout,
		metrics:             cfg.Metrics,
		masterKey:           cfg.MasterKey,
		rateLimiter:         cfg.RateLimiter,
		tokenManager:        cfg.TokenManager,
		healthTemplate:      tmpl,
		modelManager:        cfg.ModelManager,
		litellmDB:           cfg.LiteLLMDB,
		healthChecker:       cfg.HealthChecker,
		priceRegistry:       cfg.PriceRegistry,
		client:              httputil.NewHTTPClient(httpClientCfg),
	}
}

// ProxyResponse holds response details from a proxy credential
type ProxyResponse struct {
	StatusCode  int
	Headers     http.Header
	Body        []byte
	StreamBody  io.ReadCloser
	IsStreaming bool
}

// executeProxyRequest executes a request to a proxy credential and returns response details.
// This is a private helper method to avoid code duplication between forwardToProxy and related functions.
func (p *Proxy) executeProxyRequest(
	r *http.Request,
	cred *config.CredentialConfig,
	modelID string,
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
		p.balancer.RecordResponse(cred.Name, modelID, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
		return nil, err
	}
	// Record response
	p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
	p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

	p.logger.Debug("Proxy request forwarded",
		"credential", cred.Name,
		"target_url", targetURL,
		"status_code", resp.StatusCode,
		"duration", time.Since(start),
	)

	// For streaming responses, return body reader directly to avoid buffering entire stream.
	if IsStreamingResponse(resp) {
		return &ProxyResponse{
			StatusCode:  resp.StatusCode,
			Headers:     resp.Header,
			StreamBody:  resp.Body,
			IsStreaming: true,
		}, nil
	}

	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close proxy response body", "error", closeErr)
		}
	}()

	// Read response body with size limit protection
	respBody, err := p.readLimitedResponseBody(resp.Body)
	if err != nil {
		p.logger.Error("Failed to read proxy response body", "error", err)
		return nil, err
	}

	// Return complete response information
	return &ProxyResponse{
		StatusCode:  resp.StatusCode,
		Headers:     resp.Header,
		Body:        respBody,
		IsStreaming: false,
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
	return p.executeProxyRequest(r, cred, modelID, body, start)
}

func (p *Proxy) ProxyRequest(w http.ResponseWriter, r *http.Request) {
	start := utils.NowUTC()
	requestID := uuid.New().String()

	// Create logging context that will be filled throughout request processing
	// and logged at the end via defer to ensure all requests are logged
	logCtx := &RequestLogContext{
		RequestID: requestID,
		StartTime: start,
		Request:   r,
		Status:    "unknown",
	}

	// Ensure request is logged at the end regardless of which path is taken
	defer func() {
		if !logCtx.Logged && logCtx.Token != "" {
			// Log request only if we have a credential (successful auth path)
			// For auth/credential selection errors, log directly at the error point instead
			if logCtx.Credential != nil {
				if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
					p.logger.Warn("Failed to queue spend log",
						"error", err,
						"request_id", requestID,
					)
				}
			}
			logCtx.Logged = true
		}
	}()

	prepared, ok := p.orchestrateRequest(w, r, logCtx)
	if !ok {
		return
	}

	r = prepared.request
	logCtx.Request = r
	body := prepared.body
	modelID := prepared.modelID
	streaming := prepared.streaming
	cred := prepared.cred

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
			statusCode := http.StatusBadGateway
			statusMessage := "Bad Gateway"
			errorMsg := fmt.Sprintf("Proxy forward error: %v", err)
			if isTimeoutError(err) {
				statusCode = http.StatusRequestTimeout
				statusMessage = "Request Timeout"
				errorMsg = "Request timeout"
			} else if errors.Is(err, ErrResponseBodyTooLarge) {
				statusMessage = "Bad Gateway: upstream response too large"
				errorMsg = "Response body too large"
			}
			logCtx.Status = "failure"
			logCtx.HTTPStatus = statusCode
			logCtx.ErrorMsg = errorMsg
			logCtx.TargetURL = cred.BaseURL
			http.Error(w, statusMessage, statusCode)
			return
		}

		isStreaming := proxyResp.IsStreaming

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
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, proxyResp.StatusCode, retryReason, body, start, logCtx)
			if success {
				// Fallback succeeded - TryFallbackProxy handles logging
				return
			}
			// If fallback didn't work, continue with original response
			p.logger.Debug("Fallback retry failed, using original response",
				"credential", cred.Name,
				"status", proxyResp.StatusCode,
				"fallback_reason", fallbackReason,
			)
		}

		if isStreaming {
			totalTokens, err := p.writeProxyStreamingResponseWithTokens(w, proxyResp, cred.Name)
			if err != nil {
				p.logger.Error("Failed to write streaming proxy response",
					"credential", cred.Name,
					"error", err)
			}
			if totalTokens > 0 {
				p.rateLimiter.ConsumeTokens(cred.Name, totalTokens)
				if modelID != "" {
					p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, totalTokens)
				}
				p.logger.Debug("Proxy streaming token usage recorded",
					"credential", cred.Name,
					"model", modelID,
					"tokens", totalTokens,
				)
			}
		} else {
			p.writeProxyResponse(w, proxyResp)
			tokens := extractTokensFromResponse(string(proxyResp.Body), config.ProviderTypeOpenAI)
			if tokens > 0 {
				p.rateLimiter.ConsumeTokens(cred.Name, tokens)
				if modelID != "" {
					p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
				}
				p.logger.Debug("Proxy token usage recorded",
					"credential", cred.Name,
					"model", modelID,
					"tokens", tokens,
				)
			}
		}

		// Log proxy response
		logCtx.Status = "success"
		if proxyResp.StatusCode >= 400 {
			logCtx.Status = "failure"
		}
		logCtx.HTTPStatus = proxyResp.StatusCode
		logCtx.TargetURL = cred.BaseURL
		// Proxy responses typically don't have token counts, so leave at 0
		return
	}

	// Build target URL based on credential type
	var targetURL string
	var vertexToken string
	var requestBody []byte // Default to original body
	var err error

	// Track image generation request and extract image count
	logCtx.IsImageGeneration = strings.Contains(r.URL.Path, "/images/generations")
	if logCtx.IsImageGeneration {
		// Extract 'n' parameter (number of images) from request body
		var imgReq struct {
			N *int `json:"n"`
		}
		if err := json.Unmarshal(body, &imgReq); err == nil && imgReq.N != nil {
			logCtx.ImageCount = *imgReq.N
		} else {
			logCtx.ImageCount = 1 // Default to 1 image if not specified
		}
	}

	// Create provider converter for this request
	conv := converter.New(cred.Type, converter.RequestMode{
		IsImageGeneration: logCtx.IsImageGeneration,
		IsStreaming:       streaming,
		ModelID:           modelID,
	})

	// Convert request body to provider format
	requestBody, err = conv.RequestFrom(body)
	if err != nil {
		p.logger.Error("Failed to convert request to provider format",
			"credential", cred.Name,
			"type", cred.Type,
			"error", err,
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusInternalServerError
		logCtx.ErrorMsg = fmt.Sprintf("Request conversion failed: %v", err)
		logCtx.TargetURL = cred.BaseURL
		http.Error(w, "Internal Server Error: Failed to convert request", http.StatusInternalServerError)
		return
	}

	// Build target URL
	targetURL = conv.BuildURL(cred)
	if targetURL == "" {
		// For OpenAI and other passthrough providers, build URL from baseURL + path
		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		path := r.URL.Path
		// If baseURL already ends with /v1 and path starts with /v1, remove /v1 from path to avoid duplication
		if strings.HasSuffix(baseURL, "/v1") && strings.HasPrefix(path, "/v1") {
			path = strings.TrimPrefix(path, "/v1")
		}
		targetURL = baseURL + path
		if r.URL.RawQuery != "" {
			targetURL += "?" + r.URL.RawQuery
		}
	}

	// For Vertex AI, obtain OAuth2 token
	if cred.Type == config.ProviderTypeVertexAI {
		vertexToken, err = p.tokenManager.GetToken(cred.Name, cred.CredentialsFile, cred.CredentialsJSON)
		if err != nil {
			p.logger.Error("Failed to get Vertex AI token",
				"credential", cred.Name,
				"error", err,
			)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusInternalServerError
			logCtx.ErrorMsg = fmt.Sprintf("Vertex AI token error: %v", err)
			logCtx.TargetURL = targetURL
			http.Error(w, "Internal Server Error: Failed to authenticate with Vertex AI", http.StatusInternalServerError)
			return
		}
	}

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(requestBody))
	if err != nil {
		p.logger.Error("Failed to create proxy request", "error", err, "url", targetURL)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusInternalServerError
		logCtx.ErrorMsg = fmt.Sprintf("Failed to create request: %v", err)
		logCtx.TargetURL = targetURL
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

		p.balancer.RecordResponse(cred.Name, modelID, statusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
		http.Error(w, statusMessage, statusCode)
		return
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.Error("Failed to close response body", "error", closeErr)
		}
	}()

	p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
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
		logCtx.TargetURL = targetURL // Store target URL for logging
	} else {
		// Set a body read timeout for non-streaming responses to prevent
		// hanging if upstream stalls mid-body (Client.Timeout was removed
		// to support streaming, so we need an explicit guard here).
		bodyReadTimer := time.AfterFunc(p.requestTimeout, func() {
			_ = resp.Body.Close()
		})
		responseBody, err = p.readLimitedResponseBody(resp.Body)
		bodyReadTimer.Stop()
		if err != nil {
			statusCode := http.StatusInternalServerError
			statusMsg := "Internal Server Error"
			if errors.Is(err, ErrResponseBodyTooLarge) {
				statusCode = http.StatusBadGateway
				statusMsg = "Bad Gateway: upstream response too large"
			}
			p.logger.Error("Failed to read response body", "error", err)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = statusCode
			logCtx.ErrorMsg = fmt.Sprintf("Failed to read response body: %v", err)
			logCtx.TargetURL = targetURL
			http.Error(w, statusMsg, statusCode)
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
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, resp.StatusCode, retryReason, body, start, logCtx)
			if success {
				// Fallback succeeded - TryFallbackProxy handles logging
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

		// Transform response to OpenAI format (only for successful responses).
		// For error responses (4xx/5xx) pass the provider body through unchanged —
		// attempting to parse an error body as a success response produces garbled output.
		if resp.StatusCode >= 200 && resp.StatusCode < 300 && !conv.IsPassthrough() {
			convertedBody, convErr := conv.ResponseTo([]byte(decodedBody))
			if convErr != nil {
				p.logger.Error("Failed to transform provider response to OpenAI format",
					"credential", cred.Name,
					"type", cred.Type,
					"error", convErr,
				)
				finalResponseBody = []byte(decodedBody)
			} else {
				finalResponseBody = convertedBody
				p.logTransformedResponse(cred.Name, string(cred.Type), finalResponseBody)
			}
		} else {
			finalResponseBody = []byte(decodedBody)
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
		logCtx.TokenUsage = converter.ExtractTokenUsage(bodyForTokenExtraction)
		logCtx.Status = "success"
		logCtx.HTTPStatus = resp.StatusCode
		logCtx.TargetURL = targetURL

		// For image generation requests, use the image count from request
		if logCtx.IsImageGeneration && logCtx.TokenUsage != nil {
			logCtx.TokenUsage.ImageCount = logCtx.ImageCount
		}

		// Extract error message if status code indicates error
		if resp.StatusCode >= 400 {
			logCtx.Status = "failure"
			logCtx.ErrorMsg = extractErrorMessage(finalResponseBody)
		}

		// Log the request to LiteLLM DB before marking as logged
		if logCtx.Token != "" && logCtx.Credential != nil {
			if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
				p.logger.Warn("Failed to queue spend log",
					"error", err,
					"request_id", logCtx.RequestID,
				)
			}
		}
		logCtx.Logged = true // Mark as logged to prevent defer from logging again
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	copyResponseHeaders(w, resp.Header, cred.Type)

	// Set correct Content-Length for transformed responses
	if !conv.IsPassthrough() && len(finalResponseBody) > 0 {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(finalResponseBody)))
	}

	rc := http.NewResponseController(w)

	if isStreamingResp {
		// For streaming: no write deadline on WriteHeader — streamToClient
		// manages per-chunk deadlines. Setting one here risks killing the
		// connection if upstream is slow to produce the first chunk.
		w.WriteHeader(resp.StatusCode)

		// Estimate prompt tokens before streaming for logging
		// Streaming responses don't provide prompt token counts in headers,
		// so we estimate based on request body content before streaming begins
		if logCtx != nil {
			logCtx.PromptTokensEstimate = estimatePromptTokens(body)
			p.logger.Debug("Estimated prompt tokens for streaming response",
				"estimate", logCtx.PromptTokensEstimate,
				"request_id", logCtx.RequestID,
			)
		}

		switch cred.Type {
		case config.ProviderTypeVertexAI:
			// Handle Vertex AI streaming with token tracking
			err := p.handleVertexStreaming(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Vertex streaming error: %v", err)
				logCtx.Logged = false // Allow defer to log error
			}
		case config.ProviderTypeAnthropic:
			// Handle Anthropic streaming with token tracking
			err := p.handleAnthropicStreaming(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Anthropic streaming error: %v", err)
				logCtx.Logged = false // Allow defer to log error
			}
		default:
			err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to handle streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Streaming error: %v", err)
				logCtx.Logged = false // Allow defer to log error
			}
		}

	} else {
		// For non-streaming: set write deadline before header + body writes
		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		w.WriteHeader(resp.StatusCode)

		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		if _, err := p.streamResponseBody(w, resp.Body); err != nil {
			if isClientDisconnectError(err) {
				p.logger.Debug("Client disconnected during response body copy", "error", err)
			} else {
				p.logger.Error("Failed to copy response body", "error", err)
			}
		}
	}
}

// ErrResponseBodyTooLarge is returned when a response body exceeds the configured size limit.
var ErrResponseBodyTooLarge = errors.New("response body too large")

// readLimitedResponseBody reads a response body with size limit protection.
// Returns ErrResponseBodyTooLarge if the response exceeds maxResponseBodySize.
// Logs a warning when response size exceeds 50% of the limit for observability.
func (p *Proxy) readLimitedResponseBody(body io.Reader) ([]byte, error) {
	maxSize := p.maxResponseBodySize
	// Read one extra byte to detect overflow without allocating the full oversized buffer
	limitedReader := io.LimitReader(body, maxSize+1)
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxSize {
		p.logger.Error("Response body exceeds size limit",
			"limit_mb", maxSize/(1024*1024),
		)
		return nil, ErrResponseBodyTooLarge
	}
	// Warn when response is large (>50% of limit) for observability
	if int64(len(data)) > maxSize/2 {
		p.logger.Warn("Large response body detected",
			"size_bytes", len(data),
			"limit_bytes", maxSize,
			"usage_pct", int(float64(len(data))/float64(maxSize)*100),
		)
	}
	return data, nil
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
	writer  io.Writer
	tokens  *int
	logger  *slog.Logger
	onChunk func([]byte) // Callback invoked for each chunk (optional, for capturing last chunk)
}

func (tcw *tokenCapturingWriter) Write(p []byte) (n int, err error) {
	// Extract tokens from the data being written
	tokens := extractTokensFromStreamingChunk(string(p))
	if tokens > 0 {
		*tcw.tokens += tokens
	}

	// Invoke callback if provided (used to capture last chunk for usage extraction)
	if tcw.onChunk != nil {
		tcw.onChunk(p)
	}

	return tcw.writer.Write(p)
}

// ==================== LiteLLM DB Integration ====================
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
func (p *Proxy) logSpendToLiteLLMDB(logCtx *RequestLogContext) error {
	if p.litellmDB == nil || !p.litellmDB.IsEnabled() {
		return nil
	}

	if logCtx == nil || logCtx.Credential == nil || logCtx.Request == nil {
		return nil
	}

	// Fallback to request ID if session ID not provided
	if logCtx.SessionID == "" {
		logCtx.SessionID = logCtx.RequestID
	}

	// Build model_id as credential.name:model_name
	modelIDFormatted := logCtx.Credential.Name + ":" + logCtx.ModelID
	hashedToken := litellmdb.HashToken(logCtx.Token)

	// Extract user info from tokenInfo (or use empty strings as fallback)
	var userID, teamID, organizationID string
	if logCtx.TokenInfo != nil {
		userID = logCtx.TokenInfo.UserID
		teamID = logCtx.TokenInfo.TeamID
		organizationID = logCtx.TokenInfo.OrganizationID
	}

	// Build metadata with optional alias fields from tokenInfo
	// Add error field if request failed
	metadata := buildMetadata(hashedToken, logCtx.TokenInfo, logCtx.ErrorMsg, logCtx.HTTPStatus)

	// Determine end user - prefer user email from tokenInfo
	endUser := extractEndUser(logCtx.Request)
	if logCtx.TokenInfo != nil && logCtx.TokenInfo.UserEmail != "" {
		endUser = logCtx.TokenInfo.UserEmail
	}

	// Extract domain from targetURL for APIBase (e.g., "https://api.openai.com/..." -> "api.openai.com")
	apiBase := "auto_ai_router"
	if logCtx.TargetURL != "" {
		if u, err := url.Parse(logCtx.TargetURL); err == nil && u.Host != "" {
			apiBase = u.Host
		}
	}

	// Determine final status if not explicitly set
	status := logCtx.Status
	if status == "" {
		if logCtx.HTTPStatus >= 400 {
			status = "failure"
		} else {
			status = "success"
		}
	}

	// Ensure TokenUsage is not nil to prevent nil pointer dereference
	if logCtx.TokenUsage == nil {
		logCtx.TokenUsage = &converter.TokenUsage{}
	}

	// Calculate cost based on model pricing and token usage
	var cost float64
	if p.priceRegistry == nil {
		p.logger.Warn("Price registry not available, using 0 cost for spend log")
		cost = 0.0
	} else {
		// Try to get price for the model
		modelPrice := p.priceRegistry.GetPrice(logCtx.ModelID)
		if modelPrice == nil {
			p.logger.Warn("Model price not found in registry, using 0 cost",
				"model_name", logCtx.ModelID)
			cost = 0.0
		} else {
			cost = modelPrice.CalculateCost(logCtx.TokenUsage)
			p.logger.Debug("Calculated cost for model",
				"model_name", logCtx.ModelID,
				"cost", cost,
				"prompt_tokens", logCtx.TokenUsage.PromptTokens,
				"completion_tokens", logCtx.TokenUsage.CompletionTokens)
		}
	}

	return p.litellmDB.LogSpend(&litellmdb.SpendLogEntry{
		RequestID:         logCtx.RequestID,
		StartTime:         logCtx.StartTime,
		EndTime:           utils.NowUTC(),
		CallType:          logCtx.Request.URL.Path,
		APIBase:           apiBase,
		Model:             logCtx.ModelID,                                               // Model name
		ModelID:           modelIDFormatted,                                             // credential.name:model_name
		ModelGroup:        logCtx.ModelID,                                               // Model name
		CustomLLMProvider: strings.Replace(string(logCtx.Credential.Type), "-", "_", 1), // Provider type as string
		PromptTokens:      logCtx.TokenUsage.PromptTokens,
		CompletionTokens:  logCtx.TokenUsage.CompletionTokens,
		TotalTokens:       logCtx.TokenUsage.Total(),
		Metadata:          metadata,
		Spend:             cost, // Calculated cost based on model pricing and token usage
		APIKey:            hashedToken,
		UserID:            userID,
		TeamID:            teamID,
		OrganizationID:    organizationID,
		EndUser:           endUser,
		RequesterIP:       getClientIP(logCtx.Request),
		Status:            status,
		SessionID:         logCtx.SessionID,
	})
}

// extractErrorMessage returns the raw error response body as a string
// The HTTP status code is captured separately in error_code
func extractErrorMessage(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	// Return raw body (truncate if too large)
	const maxLen = 512
	if len(body) > maxLen {
		return string(body[:maxLen]) + "..."
	}
	return string(body)
}

// mapHTTPStatusToErrorClass maps HTTP status codes to LiteLLM exception class names
// Reference: https://docs.litellm.ai/docs/exception_mapping
func mapHTTPStatusToErrorClass(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "BadRequestError"
	case http.StatusUnauthorized:
		return "AuthenticationError"
	case http.StatusForbidden:
		return "PermissionDeniedError"
	case http.StatusNotFound:
		return "NotFoundError"
	case http.StatusRequestTimeout:
		return "Timeout"
	case http.StatusUnprocessableEntity:
		return "UnprocessableEntityError"
	case http.StatusTooManyRequests:
		return "RateLimitError"
	case http.StatusServiceUnavailable:
		return "ServiceUnavailableError"
	case http.StatusInternalServerError:
		return "InternalServerError"
	default:
		if statusCode >= 400 && statusCode < 500 {
			return "BadRequestError"
		} else if statusCode >= 500 {
			return "APIConnectionError"
		}
		return "APIError"
	}
}

// buildMetadata builds metadata JSON with user/team alias and optional error info
func buildMetadata(hashedToken string, tokenInfo *litellmdb.TokenInfo, errorMsg string, httpStatus int) string {
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
		"status":               "success",
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

	// Add error field if request failed
	if errorMsg != "" {
		// Determine error class based on HTTP status code (using LiteLLM exception types)
		errorClass := mapHTTPStatusToErrorClass(httpStatus)

		metadata["error_information"] = map[string]interface{}{
			"error_message": errorMsg,
			"error_code":    httpStatus,
			"error_class":   errorClass,
		}
		metadata["status"] = "failure"
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
