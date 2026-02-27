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
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"sync"
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
	MaxProviderRetries     int                        // Max same-type credential retries (default: 2)
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
	maxProviderRetries  int                        // Max same-type credential retries on provider errors
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
		maxProviderRetries:  cfg.MaxProviderRetries,
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

	// Handle proxy credential type with same-type retry + fallback
	if cred.Type == config.ProviderTypeProxy {
		triedCreds := GetTried(r.Context())
		var proxyResp *ProxyResponse
		var lastProxyErr error
		var shouldRetry bool
		var retryReason RetryReason

		for attempt := 0; attempt <= p.maxProviderRetries; attempt++ {
			if attempt > 0 {
				nextCred, err := p.balancer.NextForModelExcluding(modelID, triedCreds)
				if err != nil {
					p.logger.Debug("No more same-type proxy credentials for retry",
						"model", modelID, "attempt", attempt, "error", err)
					break
				}
				cred = nextCred
				triedCreds[cred.Name] = true
				logCtx.Credential = cred
				p.logger.Info("Retrying with next same-type proxy credential",
					"credential", cred.Name, "model", modelID,
					"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1,
					"retry_reason", retryReason)
				time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
			}

			shouldRetry = false

			proxyResp, lastProxyErr = p.forwardToProxy(w, r, modelID, cred, body, start)
			if lastProxyErr != nil {
				shouldRetry = true
				retryReason = RetryReasonNetErr
				continue
			}

			if proxyResp.IsStreaming {
				break // can't retry streaming
			}

			if !cred.IsFallback {
				shouldRetry, retryReason = ShouldRetryWithFallback(proxyResp.StatusCode, proxyResp.Body)
			}

			if !shouldRetry {
				break
			}

			p.logger.Info("Proxy credential returned retryable error",
				"credential", cred.Name, "status", proxyResp.StatusCode,
				"reason", retryReason, "model", modelID,
				"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1)
		}

		// After retry loop: try fallback proxy as last resort
		if shouldRetry {
			fallbackStatus := 0
			if lastProxyErr != nil {
				fallbackStatus = http.StatusBadGateway
				if isTimeoutError(lastProxyErr) {
					fallbackStatus = http.StatusRequestTimeout
				}
			} else if proxyResp != nil {
				fallbackStatus = proxyResp.StatusCode
			}

			p.logger.Info("All same-type proxy credentials exhausted, attempting fallback",
				"credential", cred.Name, "model", modelID,
				"last_status", fallbackStatus, "reason", retryReason)
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, fallbackStatus, retryReason, body, start, logCtx)
			if success {
				return
			}
			p.logger.Debug("Fallback retry failed, using original response",
				"credential", cred.Name, "fallback_reason", fallbackReason)
		}

		// Handle transport error (no successful response)
		if lastProxyErr != nil {
			statusCode := http.StatusBadGateway
			statusMessage := "Bad Gateway"
			errorMsg := fmt.Sprintf("Proxy forward error: %v", lastProxyErr)
			if isTimeoutError(lastProxyErr) {
				statusCode = http.StatusRequestTimeout
				statusMessage = "Request Timeout"
				errorMsg = "Request timeout"
			} else if errors.Is(lastProxyErr, ErrResponseBodyTooLarge) {
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

		// Write response (streaming or non-streaming)
		if proxyResp.IsStreaming {
			p.logger.Debug("Response is streaming (no retry for streaming)",
				"credential", cred.Name, "status", proxyResp.StatusCode)
			totalTokens, err := p.writeProxyStreamingResponseWithTokens(w, proxyResp, r, cred.Name)
			if err != nil {
				p.logger.Error("Failed to write streaming proxy response",
					"credential", cred.Name, "error", err)
			}
			if totalTokens > 0 {
				p.rateLimiter.ConsumeTokens(cred.Name, totalTokens)
				if modelID != "" {
					p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, totalTokens)
				}
				p.logger.Debug("Proxy streaming token usage recorded",
					"credential", cred.Name, "model", modelID, "tokens", totalTokens)
			}
		} else {
			p.writeProxyResponse(w, proxyResp, r)
			tokens := extractTokensFromResponse(string(proxyResp.Body), config.ProviderTypeOpenAI)
			if tokens > 0 {
				p.rateLimiter.ConsumeTokens(cred.Name, tokens)
				if modelID != "" {
					p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
				}
				p.logger.Debug("Proxy token usage recorded",
					"credential", cred.Name, "model", modelID, "tokens", tokens)
			}
		}

		// Log proxy response
		logCtx.Status = "success"
		if proxyResp.StatusCode >= 400 {
			logCtx.Status = "failure"
		}
		logCtx.HTTPStatus = proxyResp.StatusCode
		logCtx.TargetURL = cred.BaseURL
		return
	}

	// === Direct provider path with same-type credential retry ===

	// Track image generation request and extract image count (once, before retry loop)
	logCtx.IsImageGeneration = strings.Contains(r.URL.Path, "/images/generations")
	if logCtx.IsImageGeneration {
		var imgReq struct {
			N *int `json:"n"`
		}
		if err := json.Unmarshal(body, &imgReq); err == nil && imgReq.N != nil {
			logCtx.ImageCount = *imgReq.N
		} else {
			logCtx.ImageCount = 1
		}
	}

	// Retry loop: try same-type credentials on provider errors (429/5xx/auth)
	triedCreds := GetTried(r.Context())
	var (
		resp            *http.Response
		responseBody    []byte
		targetURL       string
		conv            *converter.ProviderConverter
		closeBody       func()
		isStreamingResp bool
		shouldRetry     bool
		retryReason     RetryReason
		transportErr    error
	)

	for attempt := 0; attempt <= p.maxProviderRetries; attempt++ {
		if attempt > 0 {
			// Close previous response body before retrying
			if closeBody != nil {
				closeBody()
				closeBody = nil
			}
			resp = nil
			responseBody = nil

			nextCred, err := p.balancer.NextForModelExcluding(modelID, triedCreds)
			if err != nil {
				p.logger.Debug("No more same-type credentials for retry",
					"model", modelID, "attempt", attempt, "error", err)
				break
			}
			cred = nextCred
			triedCreds[cred.Name] = true
			logCtx.Credential = cred

			p.logger.Info("Retrying with next same-type credential",
				"credential", cred.Name, "model", modelID,
				"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1,
				"retry_reason", retryReason)

			time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
		}

		// Reset retry state for this attempt
		shouldRetry = false
		retryReason = ""
		transportErr = nil

		// Create provider converter for this request
		conv = converter.New(cred.Type, converter.RequestMode{
			IsImageGeneration: logCtx.IsImageGeneration,
			IsStreaming:       streaming,
			ModelID:           modelID,
		})

		// Convert request body to provider format
		requestBody, convErr := conv.RequestFrom(body)
		if convErr != nil {
			// Fatal: conversion error won't be fixed by another credential
			p.logger.Error("Failed to convert request to provider format",
				"credential", cred.Name, "type", cred.Type, "error", convErr)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusInternalServerError
			logCtx.ErrorMsg = fmt.Sprintf("Request conversion failed: %v", convErr)
			logCtx.TargetURL = cred.BaseURL
			http.Error(w, "Internal Server Error: Failed to convert request", http.StatusInternalServerError)
			return
		}

		// Build target URL
		targetURL = conv.BuildURL(cred)
		if targetURL == "" {
			baseURL := strings.TrimSuffix(cred.BaseURL, "/")
			urlPath := r.URL.Path

			// Strip version prefix from urlPath if baseURL already ends with a version.
			// This prevents double-versioning like /v4/v1/... when baseURL contains /v4
			// and the incoming request path starts with /v1.
			if versionPrefix := extractVersionSuffix(baseURL); versionPrefix != "" {
				if pathVersion := extractVersionPrefix(urlPath); pathVersion != "" {
					urlPath = strings.TrimPrefix(urlPath, pathVersion)
				}
			}

			targetURL = baseURL + urlPath
			if r.URL.RawQuery != "" {
				targetURL += "?" + r.URL.RawQuery
			}
		}

		// For Vertex AI, obtain OAuth2 token
		var vertexToken string
		if cred.Type == config.ProviderTypeVertexAI {
			var tokenErr error
			vertexToken, tokenErr = p.tokenManager.GetToken(cred.Name, cred.CredentialsFile, cred.CredentialsJSON)
			if tokenErr != nil {
				p.logger.Error("Failed to get Vertex AI token",
					"credential", cred.Name, "error", tokenErr)
				// Token error is retryable (different credential may have valid token)
				shouldRetry = true
				retryReason = RetryReasonAuthErr
				p.balancer.RecordResponse(cred.Name, modelID, http.StatusInternalServerError)
				p.metrics.RecordRequest(cred.Name, r.URL.Path, http.StatusInternalServerError, time.Since(start))
				continue
			}
		}

		proxyReq, reqErr := http.NewRequest(r.Method, targetURL, bytes.NewReader(requestBody))
		if reqErr != nil {
			// Fatal: request creation error
			p.logger.Error("Failed to create proxy request", "error", reqErr, "url", targetURL)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusInternalServerError
			logCtx.ErrorMsg = fmt.Sprintf("Failed to create request: %v", reqErr)
			logCtx.TargetURL = targetURL
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Copy headers and set auth
		copyHeadersSkipAuth(proxyReq, r)
		switch cred.Type {
		case config.ProviderTypeVertexAI:
			proxyReq.Header.Set("Authorization", "Bearer "+vertexToken)
		case config.ProviderTypeGemini:
			proxyReq.Header.Set("x-goog-api-key", cred.APIKey)
		case config.ProviderTypeAnthropic:
			proxyReq.Header.Set("X-Api-Key", cred.APIKey)
			proxyReq.Header.Set("anthropic-version", "2023-06-01")
		default:
			proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
		}

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			p.logger.Debug("Proxy request details",
				"target_url", targetURL, "credential", cred.Name,
				"request_body", logger.TruncateLongFields(string(requestBody), 500))
		}

		debugHeaders := make(map[string]string)
		for key, values := range proxyReq.Header {
			if key == "Authorization" || key == "X-Api-Key" || key == "X-Goog-Api-Key" {
				continue
			}
			debugHeaders[key] = strings.Join(values, ", ")
		}
		p.logger.Debug("Proxy request headers", "headers", debugHeaders)

		// Execute HTTP request
		var doErr error
		resp, doErr = p.client.Do(proxyReq)
		if doErr != nil {
			statusCode := http.StatusBadGateway
			if isTimeoutError(doErr) {
				statusCode = http.StatusRequestTimeout
				p.logger.Error("Upstream request timeout",
					"credential", cred.Name, "error", doErr, "url", targetURL)
			} else {
				p.logger.Error("Upstream request failed",
					"credential", cred.Name, "error", doErr, "url", targetURL)
			}
			p.balancer.RecordResponse(cred.Name, modelID, statusCode)
			p.metrics.RecordRequest(cred.Name, r.URL.Path, statusCode, time.Since(start))
			shouldRetry = true
			retryReason = RetryReasonNetErr
			transportErr = doErr
			continue
		}

		// Setup close body with sync.Once to prevent double-close
		var closeOnce sync.Once
		closeBody = func() {
			closeOnce.Do(func() {
				if closeErr := resp.Body.Close(); closeErr != nil {
					p.logger.Error("Failed to close response body", "error", closeErr)
				}
			})
		}

		p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
		p.metrics.RecordRequest(cred.Name, r.URL.Path, resp.StatusCode, time.Since(start))

		// Debug: log response headers
		maskedRespHeaders := security.MaskSensitiveHeaders(resp.Header)
		debugRespHeaders := make(map[string]string)
		for key, values := range maskedRespHeaders {
			debugRespHeaders[key] = strings.Join(values, ", ")
		}
		p.logger.Debug("Proxy response received",
			"status_code", resp.StatusCode, "credential", cred.Name,
			"headers", debugRespHeaders)

		isStreamingResp = IsStreamingResponse(resp)
		if isStreamingResp {
			// Cannot retry streaming responses
			logCtx.TargetURL = targetURL
			break
		}

		// Read response body (non-streaming)
		currentCloseBody := closeBody // capture for timer closure
		bodyReadTimer := time.AfterFunc(p.requestTimeout, func() { currentCloseBody() })
		var readErr error
		responseBody, readErr = p.readLimitedResponseBody(resp.Body)
		bodyReadTimer.Stop()
		if readErr != nil {
			closeBody()
			if errors.Is(readErr, ErrResponseBodyTooLarge) {
				// Response too large — fatal, another credential won't help
				p.logger.Error("Failed to read response body", "error", readErr)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusBadGateway
				logCtx.ErrorMsg = fmt.Sprintf("Failed to read response body: %v", readErr)
				logCtx.TargetURL = targetURL
				http.Error(w, "Bad Gateway: upstream response too large", http.StatusBadGateway)
				return
			}
			// Transport error reading body — retryable with another credential
			p.logger.Warn("Failed to read response body, will retry", "error", readErr,
				"credential", cred.Name, "attempt", attempt+1)
			shouldRetry = true
			retryReason = RetryReasonNetErr
			transportErr = readErr
			continue
		}

		// Check if we should retry with another same-type credential
		shouldRetry, retryReason = ShouldRetryWithFallback(resp.StatusCode, responseBody)
		if !shouldRetry {
			break
		}

		p.logger.Info("Provider returned retryable error",
			"credential", cred.Name, "status", resp.StatusCode,
			"reason", retryReason, "model", modelID,
			"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1)
	}

	// After retry loop: try proxy fallback as last resort
	if shouldRetry && !isStreamingResp {
		fallbackStatus := 0
		if transportErr != nil {
			fallbackStatus = http.StatusBadGateway
			if isTimeoutError(transportErr) {
				fallbackStatus = http.StatusRequestTimeout
			}
		} else if resp != nil {
			fallbackStatus = resp.StatusCode
		}

		p.logger.Info("All same-type credentials exhausted, attempting fallback proxy",
			"credential", cred.Name, "model", modelID,
			"last_status", fallbackStatus, "reason", retryReason)
		success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, fallbackStatus, retryReason, body, start, logCtx)
		if success {
			if closeBody != nil {
				closeBody()
			}
			return
		}
		p.logger.Debug("Fallback retry failed, using original response",
			"credential", cred.Name, "fallback_reason", fallbackReason)
	}

	// Handle case where all attempts were transport errors (no response at all)
	if resp == nil {
		if closeBody != nil {
			closeBody()
		}
		statusCode := http.StatusBadGateway
		statusMessage := "Bad Gateway"
		if transportErr != nil && isTimeoutError(transportErr) {
			statusCode = http.StatusRequestTimeout
			statusMessage = "Request Timeout"
		}
		logCtx.Status = "failure"
		logCtx.HTTPStatus = statusCode
		logCtx.ErrorMsg = "All provider attempts failed"
		logCtx.TargetURL = targetURL
		http.Error(w, statusMessage, statusCode)
		return
	}

	// Ensure response body is closed at function exit
	if closeBody != nil {
		defer closeBody()
	}

	// === Process final response ===
	var finalResponseBody []byte

	if isStreamingResp {
		p.logger.Debug("Response is streaming", "credential", cred.Name)
	} else {
		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)

		// Transform response to OpenAI format (only for successful responses).
		// For error responses (4xx/5xx) pass the provider body through unchanged.
		if resp.StatusCode >= 200 && resp.StatusCode < 300 && !conv.IsPassthrough() {
			convertedBody, convErr := conv.ResponseTo([]byte(decodedBody))
			if convErr != nil {
				p.logger.Error("Failed to transform provider response to OpenAI format",
					"credential", cred.Name, "type", cred.Type, "error", convErr)
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
			bodyForTokenExtraction = []byte(decodedBody)
		}
		tokens := extractTokensFromResponse(string(bodyForTokenExtraction), config.ProviderTypeOpenAI)
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.Debug("Token usage recorded",
				"credential", cred.Name, "model", modelID, "tokens", tokens)
		}

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			p.logger.Debug("Proxy response body",
				"credential", cred.Name, "content_encoding", contentEncoding,
				"body", logger.TruncateLongFields(decodedBody, 500))
		}

		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))

		// Log to LiteLLM DB (non-streaming)
		logCtx.TokenUsage = converter.ExtractTokenUsage(bodyForTokenExtraction)
		logCtx.Status = "success"
		logCtx.HTTPStatus = resp.StatusCode
		logCtx.TargetURL = targetURL

		if logCtx.IsImageGeneration && logCtx.TokenUsage != nil {
			logCtx.TokenUsage.ImageCount = logCtx.ImageCount
		}

		if resp.StatusCode >= 400 {
			logCtx.Status = "failure"
			logCtx.ErrorMsg = extractErrorMessage(finalResponseBody)
		}

		if logCtx.Token != "" && logCtx.Credential != nil {
			if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
				p.logger.Warn("Failed to queue spend log",
					"error", err, "request_id", logCtx.RequestID)
			}
		}
		logCtx.Logged = true
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	copyResponseHeaders(w, resp.Header, cred.Type)

	rc := http.NewResponseController(w)

	if isStreamingResp {
		w.WriteHeader(resp.StatusCode)

		if logCtx != nil {
			logCtx.PromptTokensEstimate = estimatePromptTokens(body)
			p.logger.Debug("Estimated prompt tokens for streaming response",
				"estimate", logCtx.PromptTokensEstimate,
				"request_id", logCtx.RequestID)
		}

		switch cred.Type {
		case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
			err := p.handleVertexStreaming(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Vertex streaming error: %v", err)
				logCtx.Logged = false
			}
		case config.ProviderTypeAnthropic:
			err := p.handleAnthropicStreaming(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to vertex streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Anthropic streaming error: %v", err)
				logCtx.Logged = false
			}
		default:
			err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
			if err != nil {
				p.logger.Error("Failed to handle streaming response", "error", err)
				logCtx.Status = "failure"
				logCtx.ErrorMsg = fmt.Sprintf("Streaming error: %v", err)
				logCtx.Logged = false
			}
		}

	} else {
		acceptEncoding := r.Header.Get("Accept-Encoding")
		acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
		targetEncoding := SelectBestEncoding(acceptedEncodings)

		outputBody := finalResponseBody
		if targetEncoding != "identity" && len(finalResponseBody) > 0 {
			compressedBody, usedEncoding, compErr := CompressBody(finalResponseBody, targetEncoding)
			if compErr != nil {
				p.logger.Warn("Failed to compress response body",
					"credential", cred.Name, "encoding", targetEncoding, "error", compErr)
			} else {
				p.logger.Debug("Response body compressed for client",
					"credential", cred.Name, "encoding", usedEncoding,
					"original_size", len(finalResponseBody), "compressed_size", len(compressedBody))
				outputBody = compressedBody
				w.Header().Set("Content-Encoding", usedEncoding)
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(outputBody)))

		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		w.WriteHeader(resp.StatusCode)

		_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		if _, err := p.streamResponseBody(w, bytes.NewReader(outputBody)); err != nil {
			if isClientDisconnectError(err) {
				p.logger.Debug("Client disconnected during response body copy", "error", err)
			} else {
				p.logger.Error("Failed to copy response body", "error", err)
			}
		}
	}
}

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
