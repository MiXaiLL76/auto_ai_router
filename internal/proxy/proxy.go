package proxy

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/mixaill76/auto_ai_router/internal/utils"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// DefaultResponseBodyMultiplier is the default multiplier for response body size limit
// relative to maxBodySizeMB. Responses can be larger than requests (e.g., base64 images).
// Can be overridden via Config.ResponseBodyMultiplier.
const DefaultResponseBodyMultiplier = 10

// respCtx returns the context of the request that produced this upstream
// response (it carries the OTEL span for log/trace correlation), or
// context.Background() when unavailable (e.g. in unit tests).
func respCtx(resp *http.Response) context.Context {
	if resp == nil || resp.Request == nil {
		return context.Background()
	}
	return resp.Request.Context()
}

func requestWithPath(r *http.Request, path string) *http.Request {
	if r == nil || path == "" || r.URL == nil || r.URL.Path == path {
		return r
	}
	clone := r.Clone(r.Context())
	clone.URL.Path = path
	return clone
}

// Context returns the originating client request's context (carrying the OTEL
// span for log/trace correlation), or context.Background() when no request is
// set (e.g. in unit tests).
func (logCtx *RequestLogContext) Context() context.Context {
	if logCtx == nil || logCtx.Request == nil {
		return context.Background()
	}
	return logCtx.Request.Context()
}

// RequestLogContext holds all data needed for logging a request to LiteLLM DB
// Filled throughout request processing and logged at the end via defer
type RequestLogContext struct {
	RequestID            string                   // Request ID (UUID)
	StartTime            time.Time                // Request start time
	CompletionStartTime  time.Time                // Timestamp of the first real content/tool/reasoning delta (TTFT), not just the first byte/chunk; zero if not streamed or never reached
	Request              *http.Request            // HTTP request
	Token                string                   // Auth token (raw, will be hashed)
	ModelID              string                   // Model alias name (what client requested)
	RealModelID          string                   // Real model name sent to provider (for price lookup; equals ModelID if no alias)
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
	IsResponsesAPI       bool                     // True if this is a Responses API request (converted to Chat Completions)
	RequestCompleted     bool                     // True only after the response was fully and successfully delivered
	ActualCredentialName string                   // Real credential name from upstream when Credential.Type == ProviderTypeProxy
	IsProxyRequest       bool                     // True when this request came from another auto_ai_router (X-Aar-Proxy-Client header)
	Scope                scope.Context

	// ReservedEntities lists budget-hierarchy levels reserved for this request via
	// enforceBudgetAndRateLimits; reconcileBudgetAndRateLimits settles each one.
	ReservedEntities []reservedEntity
	// RateLimitedEntitiesForTPM lists entity keys that had a TPM limit, so the real
	// token count can be recorded post-call via ConsumeTokensCtx.
	RateLimitedEntitiesForTPM []string
	// budgetReconciled guards against double reconciliation: the first call (from
	// logSpendToLiteLLMDB with the real cost, or the ProxyRequest defer safety-net
	// with cost 0) wins; later calls are no-ops.
	budgetReconciled bool
}

// HealthChecker provides cached database health status
type HealthChecker interface {
	IsDBHealthy() bool
}

// Config holds all configuration needed to create a Proxy
type Config struct {
	Balancer                   *balancer.RoundRobin
	Logger                     *slog.Logger
	MaxBodySizeMB              int
	ResponseBodyMultiplier     int // Multiplier for response body size limit (default: DefaultResponseBodyMultiplier)
	RequestTimeout             time.Duration
	MaxIdleConns               int
	MaxIdleConnsPerHost        int
	IdleConnTimeout            time.Duration
	Metrics                    *monitoring.Metrics
	MasterKey                  string
	RateLimiter                *ratelimit.RPMLimiter
	TokenManager               *auth.VertexTokenManager
	ModelManager               *models.Manager
	Version                    string
	Commit                     string
	LiteLLMDB                  litellmdb.Manager          // LiteLLM database integration (optional)
	KafkaLog                   kafkalog.Manager           // Kafka spend-log publishing (optional, analytics write-path)
	HealthChecker              HealthChecker              // Optional: cached DB health status (updated by health monitor)
	PriceRegistry              *models.ModelPriceRegistry // Model pricing information (optional)
	MaxProviderRetries         int                        // Max same-type credential retries (default: 2)
	MaxFallbackAttempts        int                        // Max fallback proxy hops per request chain (default: 5)
	ResponseStore              responsestore.Store        // Optional: Responses API store (bbolt or Redis)
	SessionStickyEnabled       bool
	SessionStickyAutoCacheCtrl bool // Auto-inject Anthropic cache_control markers when session is active (default: true)
	SessionStoreTTL            time.Duration
	RouterID                   string // Human-readable name for this router (shown in /trace); defaults to hostname
	DrainUpstreamOnAbort       bool   // When true, keep reading upstream after client disconnect to get real usage (default: false)

	BudgetReserver                   *budget.Reserver      // Atomic Redis budget reservation (nil if Redis disabled — feature is a no-op)
	KeyRateLimiter                   *ratelimit.RPMLimiter // Key/user/team/org RPM/TPM enforcement (nil if Redis disabled)
	BudgetReservationEnabled         bool                  // Config toggle; independent of nil-check for clearer intent
	DefaultEstimatedCompletionTokens int                   // Completion-token estimate when max_tokens is absent (default: 1000)
}

type Proxy struct {
	balancer                         *balancer.RoundRobin
	client                           *http.Client
	logger                           *slog.Logger
	maxBodySizeMB                    int
	maxResponseBodySize              int64 // Pre-computed max response body size in bytes
	requestTimeout                   time.Duration
	metrics                          *monitoring.Metrics
	masterKey                        string
	rateLimiter                      *ratelimit.RPMLimiter
	tokenManager                     *auth.VertexTokenManager
	routerID                         string                     // Identifier for this router used in /trace responses
	modelManager                     *models.Manager            // Model manager for getting configured models
	LiteLLMDB                        litellmdb.Manager          // LiteLLM database integration
	kafkaLog                         kafkalog.Manager           // Kafka spend-log publishing (optional, analytics write-path)
	healthChecker                    HealthChecker              // Cached DB health status (optional)
	priceRegistry                    *models.ModelPriceRegistry // Model pricing information (optional)
	maxProviderRetries               int                        // Max same-type credential retries on provider errors
	maxFallbackAttempts              int                        // Max fallback proxy hops per request chain
	responseStore                    responsestore.Store        // Optional: Responses API store (bbolt or Redis)
	sessionStore                     *SessionStore              // Optional: session-sticky credential routing
	stickyAutoCacheCtrl              bool                       // Auto-inject Anthropic cache_control when session is active
	drainUpstreamOnAbort             bool                       // Keep reading upstream after client disconnect to get real usage chunk
	bedrockDailyQuota                *bedrockDailyQuotaTracker
	budgetReserver                   *budget.Reserver
	keyRateLimiter                   *ratelimit.RPMLimiter
	budgetReservationEnabled         bool
	defaultEstimatedCompletionTokens int
	version                          string
	commit                           string
}

func New(cfg *Config) *Proxy {
	if cfg.Balancer != nil && cfg.ModelManager != nil {
		cfg.Balancer.SetModelChecker(cfg.ModelManager)
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

	routerID := cfg.RouterID
	if routerID == "" {
		if h, err := os.Hostname(); err == nil {
			routerID = h
		} else {
			routerID = "unknown"
		}
	}

	var sessionStore *SessionStore
	if cfg.SessionStickyEnabled {
		ttl := cfg.SessionStoreTTL
		if ttl == 0 {
			ttl = 6 * time.Minute
		}
		sessionStore = NewSessionStore(ttl)
	}

	return &Proxy{
		routerID:                         routerID,
		balancer:                         cfg.Balancer,
		logger:                           cfg.Logger,
		maxBodySizeMB:                    cfg.MaxBodySizeMB,
		maxResponseBodySize:              maxResponseBodySize,
		requestTimeout:                   cfg.RequestTimeout,
		metrics:                          cfg.Metrics,
		masterKey:                        cfg.MasterKey,
		rateLimiter:                      cfg.RateLimiter,
		tokenManager:                     cfg.TokenManager,
		modelManager:                     cfg.ModelManager,
		LiteLLMDB:                        cfg.LiteLLMDB,
		kafkaLog:                         cfg.KafkaLog,
		healthChecker:                    cfg.HealthChecker,
		priceRegistry:                    cfg.PriceRegistry,
		maxProviderRetries:               cfg.MaxProviderRetries,
		maxFallbackAttempts:              cfg.MaxFallbackAttempts,
		responseStore:                    cfg.ResponseStore,
		sessionStore:                     sessionStore,
		stickyAutoCacheCtrl:              cfg.SessionStickyAutoCacheCtrl,
		drainUpstreamOnAbort:             cfg.DrainUpstreamOnAbort,
		bedrockDailyQuota:                newBedrockDailyQuotaTracker(),
		budgetReserver:                   cfg.BudgetReserver,
		keyRateLimiter:                   cfg.KeyRateLimiter,
		budgetReservationEnabled:         cfg.BudgetReservationEnabled,
		defaultEstimatedCompletionTokens: cfg.DefaultEstimatedCompletionTokens,
		client:                           httputil.NewHTTPClient(httpClientCfg),
		version:                          cfg.Version,
		commit:                           cfg.Commit,
	}
}

// Start launches background workers owned by Proxy.
func (p *Proxy) Start(ctx context.Context) {
	if p.sessionStore != nil {
		go p.sessionStore.StartCleanup(ctx, 2*time.Minute)
	}
}

func (p *Proxy) setSessionBinding(sessionID, modelID, credentialName string) {
	if p.sessionStore == nil || sessionID == "" || modelID == "" || credentialName == "" {
		return
	}
	p.sessionStore.Set(sessionID, modelID, credentialName)
}

func (p *Proxy) clearSessionBinding(sessionID, modelID string) {
	if p.sessionStore == nil || sessionID == "" || modelID == "" {
		return
	}
	p.sessionStore.Delete(sessionID, modelID)
}

// GetMasterKey returns the proxy master key.
func (p *Proxy) GetMasterKey() string {
	return p.masterKey
}

// isMasterKey reports whether token equals the master key using a constant-time
// comparison (timing-attack safe). An empty master key never matches, so an
// empty/absent token can never be mistaken for the master key.
func (p *Proxy) isMasterKey(token string) bool {
	if p.masterKey == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(token), []byte(p.masterKey)) == 1
}

// GetVersion returns the build version string.
func (p *Proxy) GetVersion() string {
	return p.version
}

// GetCommit returns the build commit hash.
func (p *Proxy) GetCommit() string {
	return p.commit
}

// ProxyResponse holds response details from a proxy credential
type ProxyResponse struct {
	StatusCode           int
	Headers              http.Header
	Body                 []byte
	StreamBody           io.ReadCloser
	IsStreaming          bool
	ActualCredentialName string // Credential name from upstream X-Credential-Name header
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

	// Create proxy request. The incoming request context carries the OTEL span,
	// so the client span parents correctly and traceparent is propagated, and the
	// upstream call is canceled if the client disconnects.
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		p.logger.ErrorContext(r.Context(), "Failed to create proxy request", "error", err, "url", targetURL)
		return nil, err
	}

	// Copy headers (skip hop-by-hop headers)
	copyRequestHeaders(proxyReq, r, cred.APIKey)
	// Mark request as coming from an internal proxy client so the upstream router
	// knows to include the X-Credential-Name response header.
	proxyReq.Header.Set("X-Aar-Proxy-Client", "1")

	// Send request
	resp, err := p.client.Do(proxyReq)
	if err != nil {
		// Transport failure on one credential — the caller retries with another
		// credential or fallback, so this is WARN; the final outcome (success or
		// exhausted attempts) is logged at the appropriate level by the caller.
		statusCode := http.StatusBadGateway
		if isTimeoutError(err) {
			statusCode = http.StatusRequestTimeout
			p.logger.WarnContext(r.Context(), "Proxy request timeout",
				"credential", cred.Name,
				"model", modelID,
				"error", err,
				"url", targetURL,
			)
		} else {
			p.logger.WarnContext(r.Context(), "Failed to proxy request",
				"credential", cred.Name,
				"model", modelID,
				"error", err,
				"url", targetURL,
			)
		}
		// Proxy credentials are dynamic relays — don't record them in fail2ban.
		// Their 429/5xx reflect downstream capacity, not a permanent credential failure.
		if cred.Type != config.ProviderTypeProxy {
			p.balancer.RecordResponse(cred.Name, modelID, statusCode)
		}
		p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, statusCode, time.Since(start))
		return nil, err
	}
	// Proxy credentials are dynamic relays — don't record them in fail2ban.
	if cred.Type != config.ProviderTypeProxy {
		p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
	}
	p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, resp.StatusCode, time.Since(start))

	p.logger.DebugContext(r.Context(), "Proxy request forwarded",
		"credential", cred.Name,
		"target_url", targetURL,
		"status_code", resp.StatusCode,
		"duration", time.Since(start),
	)

	actualCredName := resp.Header.Get("X-Credential-Name")

	// For streaming responses, return body reader directly to avoid buffering entire stream.
	if IsStreamingResponse(resp) {
		return &ProxyResponse{
			StatusCode:           resp.StatusCode,
			Headers:              resp.Header,
			StreamBody:           resp.Body,
			IsStreaming:          true,
			ActualCredentialName: actualCredName,
		}, nil
	}

	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.WarnContext(r.Context(), "Failed to close proxy response body", "error", closeErr)
		}
	}()

	// Read response body with size limit protection
	respBody, err := p.readLimitedResponseBody(resp.Body)
	if err != nil {
		p.logger.WarnContext(r.Context(), "Failed to read proxy response body, caller may retry",
			"credential", cred.Name, "model", modelID, "error", err)
		return nil, err
	}

	// Return complete response information
	return &ProxyResponse{
		StatusCode:           resp.StatusCode,
		Headers:              resp.Header,
		Body:                 respBody,
		IsStreaming:          false,
		ActualCredentialName: actualCredName,
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

	// Save and strip internal proxy marker. The marker is set by executeProxyRequest when
	// acting as a proxy client. We save it here and delete it to prevent external spoofing.
	isProxyRequest := r.Header.Get("X-Aar-Proxy-Client") == "1"
	r.Header.Del("X-Aar-Proxy-Client")

	// Create logging context that will be filled throughout request processing
	// and logged at the end via defer to ensure all requests are logged
	logCtx := &RequestLogContext{
		RequestID:      requestID,
		StartTime:      start,
		Request:        r,
		Status:         "unknown",
		IsProxyRequest: isProxyRequest,
	}

	// Ensure request is logged at the end regardless of which path is taken
	defer func() {
		if !logCtx.Logged && logCtx.Token != "" {
			// Log request only if we have a credential (successful auth path)
			// For auth/credential selection errors, log directly at the error point instead
			if logCtx.Credential != nil {
				if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
					p.logger.WarnContext(r.Context(), "Failed to queue spend log",
						"error", err,
						"request_id", requestID,
					)
				}
			}
			logCtx.Logged = true
		}
		// Safety net: release any budget reservation that never reached
		// logSpendToLiteLLMDB (e.g. an early failure before a credential was set).
		// The guard makes this a no-op when reconciliation already happened.
		p.reconcileBudgetAndRateLimits(logCtx, 0)
		if !logCtx.RequestCompleted {
			p.clearSessionBinding(logCtx.SessionID, logCtx.ModelID)
		}
	}()

	prepared, ok := p.orchestrateRequest(w, r, logCtx)
	if !ok {
		return
	}

	r = prepared.request
	logCtx.Request = r
	body := prepared.body
	proxyBody := prepared.proxyBody
	modelID := prepared.modelID
	realModelID := prepared.realModelID
	streaming := prepared.streaming
	cred := prepared.cred
	logCtx.IsResponsesAPI = prepared.isResponsesAPI
	logCtx.RealModelID = realModelID

	// Build a callback that saves the completed Responses API response if store=true.
	// Captured by streaming handlers and called when the stream completes.
	// The callback enriches the response with request-echoed fields (store, previous_response_id,
	// metadata) before persisting so GET /responses/{id} returns the canonical record.
	var saveResponseFn func(*responses.Response)
	if prepared.responsesMetadata != nil && prepared.responsesMetadata.Store && p.responseStore != nil {
		apiKeyHash := litellmdb.HashToken(logCtx.Token)
		meta := prepared.responsesMetadata
		saveResponseFn = func(resp *responses.Response) {
			if resp == nil {
				return
			}
			applyResponsesMetadata(resp, meta)
			if err := p.responseStore.SaveResponse(
				context.Background(), apiKeyHash, resp, meta.Metadata, meta.TTL, meta.AccumulatedInput, cred.Name,
			); err != nil {
				p.logger.WarnContext(r.Context(), "Failed to save response to store", "id", resp.ID, "error", err)
			} else {
				p.logger.DebugContext(r.Context(), "Saved response to store", "id", resp.ID)
			}
		}
	}

	// Annotate the server span (created by otelhttp in main) with routing details.
	// No-op when OTEL tracing is disabled.
	if span := trace.SpanFromContext(r.Context()); span.IsRecording() {
		span.SetAttributes(
			attribute.String("gen_ai.request.model", modelID),
			attribute.String("aar.real_model", realModelID),
			attribute.String("aar.credential", cred.Name),
			attribute.String("aar.provider", string(cred.Type)),
			attribute.Bool("aar.streaming", streaming),
			attribute.String("aar.request_id", requestID),
		)
	}

	// Log request details at DEBUG level
	p.logger.DebugContext(r.Context(), "Processing request",
		"credential", cred.Name,
		"method", r.Method,
		"path", r.URL.Path,
		"model", modelID,
		"type", cred.Type,
	)

	// Track image generation requests before both proxy and direct provider paths
	isImageGeneration := strings.Contains(r.URL.Path, "/images/generations")
	isImageEdit := strings.Contains(r.URL.Path, "/images/edits")
	logCtx.IsImageGeneration = isImageGeneration || isImageEdit
	if logCtx.IsImageGeneration {
		logCtx.ImageCount = extractImageCountFromBody(body, r.Header.Get("Content-Type"))
		if logCtx.ImageCount <= 0 {
			logCtx.ImageCount = 1
		}
	}

	// Handle proxy credential type with same-type retry + fallback
	if cred.Type == config.ProviderTypeProxy {
		logCtx.Credential = cred
		triedCreds := GetTried(r.Context())
		var proxyResp *ProxyResponse
		var lastProxyErr error
		var shouldRetry bool
		var retryReason RetryReason

		for attempt := 0; attempt <= p.maxProviderRetries; attempt++ {
			if attempt > 0 {
				nextCred, err := p.balancer.NextSameTypeForModelExcludingScoped(modelID, config.ProviderTypeProxy, triedCreds, logCtx.Scope)
				if err != nil {
					p.logger.DebugContext(r.Context(), "No more same-type proxy credentials for retry",
						"model", modelID, "attempt", attempt, "error", err)
					break
				}
				cred = nextCred
				triedCreds[cred.Name] = true
				logCtx.Credential = cred
				p.logger.WarnContext(r.Context(), "Retrying with next same-type proxy credential",
					"credential", cred.Name, "model", modelID,
					"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1,
					"retry_reason", retryReason)
				time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
			}

			shouldRetry = false

			resp, fwdErr := p.forwardToProxy(w, r, modelID, cred, proxyBody, start)
			lastProxyErr = fwdErr
			if fwdErr != nil {
				shouldRetry = true
				retryReason = RetryReasonNetErr
				continue
			}
			proxyResp = resp
			if proxyResp.ActualCredentialName != "" {
				logCtx.ActualCredentialName = proxyResp.ActualCredentialName
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

			// Mid-retry failure — the request will be retried with another credential.
			// The final failure (if all attempts fail) is logged at ERROR when the
			// response is written to the client.
			retryLogArgs := []any{
				"error_code", proxyResp.StatusCode, "credential", cred.Name,
				"reason", retryReason, "model", modelID,
				"attempt", attempt + 1, "max_attempts", p.maxProviderRetries + 1,
			}
			retryLogArgs = appendResponseBodyForLogs(retryLogArgs, cred, string(proxyResp.Body))
			p.logger.WarnContext(r.Context(), "Proxy credential returned retryable error, will retry", retryLogArgs...)
		}

		// After retry loop: try fallback proxy as last resort
		if shouldRetry {
			fallbackStatus := 0
			if proxyResp != nil {
				// Prefer the last HTTP status code we actually received.
				fallbackStatus = proxyResp.StatusCode
			} else if lastProxyErr != nil {
				fallbackStatus = http.StatusBadGateway
				if isTimeoutError(lastProxyErr) {
					fallbackStatus = http.StatusRequestTimeout
				}
			}

			p.logger.InfoContext(r.Context(), "All same-type proxy credentials exhausted, attempting fallback",
				"credential", cred.Name, "model", modelID,
				"last_status", fallbackStatus, "reason", retryReason)
			success, fallbackReason := p.TryFallbackProxy(w, r, modelID, cred.Name, fallbackStatus, retryReason, proxyBody, start, logCtx)
			if success {
				return
			}
			p.logger.DebugContext(r.Context(), "Fallback retry failed, using original response",
				"credential", cred.Name, "fallback_reason", fallbackReason)
		}

		// Handle transport error (no successful response at all).
		// If we have a saved proxyResp (e.g. a 429 from an earlier attempt that was
		// followed by a network error on the retry), fall through to writeProxyResponse
		// so the real HTTP status is returned to the client instead of 502.
		if lastProxyErr != nil && proxyResp == nil {
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
			p.logUpstreamError(r.Context(), "Proxy request failed: no upstream response", statusCode, cred, modelID, nil,
				"error", lastProxyErr,
				"url", cred.BaseURL,
				"request_id", logCtx.RequestID)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = statusCode
			logCtx.ErrorMsg = errorMsg
			logCtx.TargetURL = cred.BaseURL
			if statusCode == http.StatusRequestTimeout {
				WriteErrorTimeout(w, statusMessage)
			} else {
				WriteErrorBadGateway(w, statusMessage)
			}
			return
		}

		// Write response (streaming or non-streaming)
		if proxyResp.IsStreaming {
			p.logger.DebugContext(r.Context(), "Response is streaming (no retry for streaming)",
				"credential", cred.Name, "status", proxyResp.StatusCode)
			streamCompleted := false

			if prepared.convertedResp {
				// Proxy streaming + Responses API: need to convert Chat Completions SSE
				// to Responses API SSE. Wrap StreamBody in http.Response for handleResponsesAPIStreaming.
				defer func() {
					if closeErr := proxyResp.StreamBody.Close(); closeErr != nil {
						p.logger.WarnContext(r.Context(), "Failed to close proxy streaming response body", "error", closeErr)
					}
				}()
				copyResponseHeaders(w, proxyResp.Headers, cred.Type)
				if logCtx.IsProxyRequest && logCtx.ActualCredentialName != "" {
					w.Header().Set("X-Credential-Name", logCtx.ActualCredentialName)
				}
				w.WriteHeader(proxyResp.StatusCode)
				logCtx.PromptTokensEstimate = estimatePromptTokensForModel(body, realModelID)
				fakeResp := &http.Response{
					StatusCode: proxyResp.StatusCode,
					Header:     proxyResp.Headers,
					Body:       proxyResp.StreamBody,
				}
				err := p.handleResponsesAPIStreaming(w, fakeResp, cred, realModelID, logCtx, saveResponseFn, prepared.responsesMetadata)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle proxy Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
					streamCompleted = true
				}
			} else if prepared.passthroughResponses {
				// Codex passthrough: provider returns native Responses API SSE — stream as-is.
				defer func() {
					if closeErr := proxyResp.StreamBody.Close(); closeErr != nil {
						p.logger.WarnContext(r.Context(), "Failed to close proxy streaming response body", "error", closeErr)
					}
				}()
				copyResponseHeaders(w, proxyResp.Headers, cred.Type)
				if logCtx.IsProxyRequest && logCtx.ActualCredentialName != "" {
					w.Header().Set("X-Credential-Name", logCtx.ActualCredentialName)
				}
				w.WriteHeader(proxyResp.StatusCode)
				logCtx.PromptTokensEstimate = estimatePromptTokensForModel(body, realModelID)
				fakeResp := &http.Response{
					StatusCode: proxyResp.StatusCode,
					Header:     proxyResp.Headers,
					Body:       proxyResp.StreamBody,
				}
				if err := p.handlePassthroughResponsesStreaming(w, fakeResp, cred.Name, realModelID, logCtx, saveResponseFn); err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle proxy passthrough Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
					streamCompleted = true
				}
			} else {
				if logCtx.IsProxyRequest && logCtx.ActualCredentialName != "" {
					w.Header().Set("X-Credential-Name", logCtx.ActualCredentialName)
				}
				tokenizerModelID := realModelID
				if tokenizerModelID == "" {
					tokenizerModelID = modelID
				}
				logCtx.PromptTokensEstimate = estimatePromptTokensForModel(proxyBody, tokenizerModelID)
				streamUsage, err := p.writeProxyStreamingResponseWithTokens(w, proxyResp, r, cred.Name, modelID, tokenizerModelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to write streaming proxy response", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
					streamCompleted = true
				}
				if streamUsage != nil {
					// Backfill PromptTokens from estimate when provider didn't include it
					// (e.g. stream cut before usage chunk, or provider omits prompt tokens).
					if streamUsage.PromptTokens == 0 {
						streamUsage.PromptTokens = logCtx.PromptTokensEstimate
					}
					logCtx.TokenUsage = streamUsage
					if proxyResp.StatusCode < 400 {
						p.metrics.RecordTokenUsage(cred.Name, modelID,
							streamUsage.PromptTokens, streamUsage.CompletionTokens,
							streamUsage.ReasoningTokens, streamUsage.CachedInputTokens)
						totalTokens := streamUsage.Total()
						if totalTokens > 0 {
							p.rateLimiter.ConsumeTokens(cred.Name, totalTokens)
							if modelID != "" {
								p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, totalTokens)
							}
						}
						p.logger.DebugContext(r.Context(), "Proxy streaming token usage recorded",
							"credential", cred.Name, "model", modelID,
							"prompt_tokens", streamUsage.PromptTokens,
							"completion_tokens", streamUsage.CompletionTokens)
					}
				}
			}
			if streamCompleted {
				logCtx.RequestCompleted = true
				p.setSessionBinding(logCtx.SessionID, modelID, cred.Name)
			}
		} else {
			// Save passthrough Responses API response or convert Chat Completions response if needed
			if prepared.passthroughResponses && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				// Codex passthrough: body is already in Responses API format — just enrich and save.
				var respObj responses.Response
				if err := json.Unmarshal(proxyResp.Body, &respObj); err == nil {
					applyResponsesMetadata(&respObj, prepared.responsesMetadata)
					if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
						proxyResp.Body = enriched
					}
					if saveResponseFn != nil {
						saveResponseFn(&respObj)
					}
				}
			} else if prepared.convertedResp && proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				responsesBody, convErr := responses.ChatToResponse(proxyResp.Body)
				if convErr != nil {
					p.logger.ErrorContext(r.Context(), "Failed to convert proxy response to Responses API format",
						"credential", cred.Name, "model", modelID, "error", convErr,
						"request_id", logCtx.RequestID)
				} else {
					// Enrich the response with request-echoed fields (store, previous_response_id,
					// metadata) for both the client payload and the store record.
					var respObj responses.Response
					if err := json.Unmarshal(responsesBody, &respObj); err == nil {
						applyResponsesMetadata(&respObj, prepared.responsesMetadata)
						if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
							responsesBody = enriched
						}
						if saveResponseFn != nil {
							saveResponseFn(&respObj)
						}
					} else if saveResponseFn != nil {
						// Fallback: save unenriched on unmarshal failure (shouldn't happen)
						var r2 responses.Response
						if json.Unmarshal(responsesBody, &r2) == nil {
							saveResponseFn(&r2)
						}
					}
					proxyResp.Body = responsesBody
				}
			}

			if logCtx.IsProxyRequest && logCtx.ActualCredentialName != "" {
				w.Header().Set("X-Credential-Name", logCtx.ActualCredentialName)
			}
			p.writeProxyResponse(w, proxyResp, r, cred.Name, modelID)
			tokens := extractTokensFromResponse(string(proxyResp.Body), config.ProviderTypeOpenAI)
			if tokens > 0 {
				p.rateLimiter.ConsumeTokens(cred.Name, tokens)
				if modelID != "" {
					p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
				}
				p.logger.DebugContext(r.Context(), "Proxy token usage recorded",
					"credential", cred.Name, "model", modelID, "tokens", tokens)
			}
			if proxyResp.StatusCode >= 200 && proxyResp.StatusCode < 300 {
				logCtx.RequestCompleted = true
				p.setSessionBinding(logCtx.SessionID, modelID, cred.Name)
			}
		}

		// Log proxy response
		logCtx.Status = "success"
		if proxyResp.StatusCode >= 400 {
			logCtx.Status = "failure"
			// Final error returned to the client — single unified ERROR record.
			// For streaming responses the body was forwarded to the client and is
			// not available here (response_body is omitted).
			p.logUpstreamError(r.Context(), "Proxy request completed with error status", proxyResp.StatusCode, cred, modelID, proxyResp.Body,
				"url", cred.BaseURL,
				"streaming", proxyResp.IsStreaming,
				"actual_credential", logCtx.ActualCredentialName,
				"request_id", logCtx.RequestID)
		}
		logCtx.HTTPStatus = proxyResp.StatusCode
		logCtx.TargetURL = cred.BaseURL
		if !proxyResp.IsStreaming {
			logCtx.TokenUsage = converter.ExtractTokenUsage(proxyResp.Body)
			if logCtx.TokenUsage != nil && proxyResp.StatusCode < 400 {
				p.metrics.RecordTokenUsage(cred.Name, modelID,
					logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
					logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
			}
			// Image generation responses have no usage field, so ExtractTokenUsage returns nil.
			// Ensure ImageCount is always propagated for cost calculation.
			if logCtx.IsImageGeneration && proxyResp.StatusCode < 400 {
				if logCtx.TokenUsage == nil {
					logCtx.TokenUsage = &converter.TokenUsage{}
				}
				logCtx.TokenUsage.ImageCount = logCtx.ImageCount
			}
			if proxyResp.StatusCode >= 400 {
				logCtx.ErrorMsg = extractErrorMessage(proxyResp.Body)
			}
		}
		return
	}

	// === Direct provider path with credential retry ===

	// Track embeddings requests (once, before retry loop)
	isEmbeddings := strings.Contains(r.URL.Path, "/embeddings")

	// Retry loop: try same-type credentials by default, or fallback_priority tiers when configured.
	triedCreds := GetTried(r.Context())
	var (
		resp            *http.Response
		responseBody    []byte
		targetURL       string
		conv            *converter.ProviderConverter
		provResponses   responses.ProviderResponses // non-nil when prepared.nativeResponses
		closeBody       func()
		isStreamingResp bool
		shouldRetry     bool
		retryReason     RetryReason
		transportErr    error
	)

	for attempt := 0; attempt <= p.maxProviderRetries; attempt++ {
		if attempt > 0 {
			// Check for next credential BEFORE resetting resp/responseBody.
			// If no credential is available, break while preserving the last HTTP response
			// (e.g. a 429 from the provider) so the caller returns it instead of 502.
			var nextCred *config.CredentialConfig
			var nextReq credentialPreparedRequest
			retryReady := false
			for {
				candidate, err := p.balancer.NextRetryForModelExcludingScoped(modelID, cred, triedCreds, logCtx.Scope)
				if err != nil {
					p.logger.DebugContext(r.Context(), "No more retry credentials available",
						"model", modelID, "attempt", attempt, "error", err)
					break
				}
				preparedRetry, prepErr := p.prepareRequestForCredential(
					r,
					prepared.baseBody,
					prepared.baseProxyBody,
					modelID,
					prepared.baseRealModelID,
					prepared.basePath,
					streaming,
					candidate,
					prepared.isResponsesAPI,
					prepared.responsesPrevHandled,
					prepared.stickyCacheEligible,
				)
				if prepErr == nil {
					nextCred = candidate
					nextReq = preparedRetry
					retryReady = true
					break
				}
				triedCreds[candidate.Name] = true
				p.logger.WarnContext(r.Context(), "Failed to prepare retry request for credential",
					"credential", candidate.Name,
					"provider", string(candidate.Type),
					"model", modelID,
					"attempt", attempt,
					"error", prepErr)
			}
			if !retryReady {
				break
			}

			// Only reset after we know there is a credential to retry with.
			if closeBody != nil {
				closeBody()
				closeBody = nil
			}
			resp = nil
			responseBody = nil

			cred = nextCred
			triedCreds[cred.Name] = true
			logCtx.Credential = cred
			body = nextReq.body
			proxyBody = nextReq.proxyBody
			realModelID = nextReq.realModelID
			r.URL.Path = nextReq.path
			prepared.body = nextReq.body
			prepared.proxyBody = nextReq.proxyBody
			prepared.proxyPath = nextReq.proxyPath
			prepared.realModelID = nextReq.realModelID
			prepared.convertedResp = nextReq.convertedResp
			prepared.passthroughResponses = nextReq.passthroughResponses
			prepared.nativeResponses = nextReq.nativeResponses
			logCtx.RealModelID = realModelID

			p.logger.InfoContext(r.Context(), "Retrying with next credential",
				"credential", cred.Name, "model", modelID,
				"attempt", attempt+1, "max_attempts", p.maxProviderRetries+1,
				"retry_reason", retryReason)

			time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)
		}

		// Reset retry state for this attempt
		shouldRetry = false
		retryReason = ""
		transportErr = nil

		// Create converter and build request body / target URL.
		// nativeResponses path uses ProviderResponses (Vertex AI, Anthropic) directly.
		// All other paths use the Chat Completions ProviderConverter.
		var requestBody []byte
		if prepared.nativeResponses {
			mode := responses.ResponsesRequestMode{
				ModelID:        realModelID,
				DisplayModelID: modelID,
				IsStreaming:    streaming,
			}
			provResponses = responses.NewProviderResponses(cred.Type, mode)
			if provResponses == nil {
				p.logger.ErrorContext(r.Context(), "Native Responses converter unavailable",
					"error_code", http.StatusInternalServerError,
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID,
					"request_id", logCtx.RequestID)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusInternalServerError
				logCtx.ErrorMsg = "Native Responses converter unavailable"
				logCtx.TargetURL = cred.BaseURL
				WriteErrorInternal(w, "Failed to convert request")
				return
			}
			var convErr error
			requestBody, _, convErr = provResponses.RequestFrom(body)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert Responses API request to provider format",
					"error_code", http.StatusInternalServerError,
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusInternalServerError
				logCtx.ErrorMsg = fmt.Sprintf("Request conversion failed: %v", convErr)
				logCtx.TargetURL = cred.BaseURL
				WriteErrorInternal(w, "Failed to convert request")
				return
			}
			targetURL = provResponses.BuildURL(cred)
		} else {
			// Use realModelID for URL construction and body conversion (provider-facing name).
			// modelID (alias) is used for credential selection and rate limiting.
			conv = converter.New(cred.Type, converter.RequestMode{
				IsImageGeneration: logCtx.IsImageGeneration,
				IsImageEdit:       isImageEdit,
				IsEmbeddings:      isEmbeddings,
				IsStreaming:       streaming,
				ModelID:           realModelID,
				DisplayModelID:    modelID,
				ContentType:       r.Header.Get("Content-Type"),
			})
			var convErr error
			requestBody, convErr = conv.RequestFrom(body)
			if convErr != nil {
				// Fatal: conversion error won't be fixed by another credential
				p.logger.ErrorContext(r.Context(), "Failed to convert request to provider format",
					"error_code", http.StatusInternalServerError,
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusInternalServerError
				logCtx.ErrorMsg = fmt.Sprintf("Request conversion failed: %v", convErr)
				logCtx.TargetURL = cred.BaseURL
				WriteErrorInternal(w, "Failed to convert request")
				return
			}
			targetURL = conv.BuildURL(cred)
		}
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
				p.logger.ErrorContext(r.Context(), "Failed to get Vertex AI token",
					"error_code", http.StatusInternalServerError,
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", tokenErr)
				// Token error is retryable (different credential may have valid token)
				shouldRetry = true
				retryReason = RetryReasonAuthErr
				p.balancer.RecordResponse(cred.Name, modelID, http.StatusInternalServerError)
				p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, http.StatusInternalServerError, time.Since(start))
				continue
			}
		}

		proxyReq, reqErr := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(requestBody))
		if reqErr != nil {
			// Fatal: request creation error
			p.logger.ErrorContext(r.Context(), "Failed to create proxy request", "error", reqErr, "url", targetURL)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusInternalServerError
			logCtx.ErrorMsg = fmt.Sprintf("Failed to create request: %v", reqErr)
			logCtx.TargetURL = targetURL
			WriteErrorInternal(w, "Internal Server Error")
			return
		}

		// Copy headers and set auth
		copyHeadersSkipAuth(proxyReq, r)
		// For passthrough providers (OpenAI/Proxy) with multipart/form-data requests
		// (e.g. /v1/images/edits), preserve the original Content-Type so the boundary
		// parameter is forwarded intact. All other paths (Vertex, Anthropic, Bedrock,
		// and non-multipart OpenAI) always send JSON.
		originalContentType := r.Header.Get("Content-Type")
		isMultipartPassthrough := conv != nil && conv.IsPassthrough() &&
			strings.HasPrefix(strings.ToLower(originalContentType), "multipart/form-data")
		if !isMultipartPassthrough {
			proxyReq.Header.Set("Content-Type", "application/json")
		} else if rewrittenCT := conv.RewrittenContentType(); rewrittenCT != "" {
			// RequestFrom rewrote the multipart body (e.g. fixed image MIME types or
			// stripped response_format).  The boundary has changed so we must update the
			// Content-Type header to match the new boundary.
			proxyReq.Header.Set("Content-Type", rewrittenCT)
		}
		switch cred.Type {
		case config.ProviderTypeVertexAI:
			proxyReq.Header.Set("Authorization", "Bearer "+vertexToken)
		case config.ProviderTypeGemini:
			proxyReq.Header.Set("x-goog-api-key", cred.APIKey)
		case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
			if cred.AuthType == "bearer" {
				proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
			} else {
				proxyReq.Header.Set("X-Api-Key", cred.APIKey)
			}
			proxyReq.Header.Set("anthropic-version", "2023-06-01")
		case config.ProviderTypeBedrock:
			proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
		default:
			proxyReq.Header.Set("Authorization", "Bearer "+cred.APIKey)
		}

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			p.logger.DebugContext(r.Context(), "Proxy request details",
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
		p.logger.DebugContext(r.Context(), "Proxy request headers", "headers", debugHeaders)

		// Execute HTTP request
		var doErr error
		resp, doErr = p.client.Do(proxyReq)
		if doErr != nil {
			// Transport failure on one credential — retried with the next one;
			// the final failure is logged at ERROR after the retry loop.
			statusCode := http.StatusBadGateway
			if isTimeoutError(doErr) {
				statusCode = http.StatusRequestTimeout
				p.logger.WarnContext(r.Context(), "Upstream request timeout, will retry",
					"credential", cred.Name, "model", modelID, "error", doErr, "url", targetURL)
			} else {
				p.logger.WarnContext(r.Context(), "Upstream request failed, will retry",
					"credential", cred.Name, "model", modelID, "error", doErr, "url", targetURL)
			}
			p.balancer.RecordResponse(cred.Name, modelID, statusCode)
			p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, statusCode, time.Since(start))
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
					p.logger.WarnContext(r.Context(), "Failed to close response body", "error", closeErr)
				}
			})
		}

		p.metrics.RecordRequest(cred.Name, r.URL.Path, modelID, resp.StatusCode, time.Since(start))

		// Debug: log response headers
		maskedRespHeaders := security.MaskSensitiveHeaders(resp.Header)
		debugRespHeaders := make(map[string]string)
		for key, values := range maskedRespHeaders {
			debugRespHeaders[key] = strings.Join(values, ", ")
		}
		p.logger.DebugContext(r.Context(), "Proxy response received",
			"status_code", resp.StatusCode, "credential", cred.Name,
			"headers", debugRespHeaders)

		isStreamingResp = IsStreamingResponse(resp)
		// Bedrock HTTP errors must be buffered even when the content type is an
		// event stream so the provider-specific error body can be classified.
		if cred.Type == config.ProviderTypeBedrock && resp.StatusCode >= http.StatusBadRequest {
			isStreamingResp = false
		}
		if isStreamingResp {
			p.recordProviderResponse(r.Context(), cred, modelID, realModelID, resp.StatusCode, resp.Header, nil)
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
			p.balancer.RecordResponse(cred.Name, modelID, resp.StatusCode)
			if errors.Is(readErr, ErrResponseBodyTooLarge) {
				// Response too large — fatal, another credential won't help
				p.logUpstreamError(r.Context(), "Failed to read response body: too large", http.StatusBadGateway, cred, modelID, nil,
					"error", readErr,
					"url", targetURL,
					"request_id", logCtx.RequestID)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusBadGateway
				logCtx.ErrorMsg = fmt.Sprintf("Failed to read response body: %v", readErr)
				logCtx.TargetURL = targetURL
				WriteErrorBadGateway(w, "upstream response too large")
				return
			}
			// Transport error reading body — retryable with another credential
			p.logger.WarnContext(r.Context(), "Failed to read response body, will retry", "error", readErr,
				"credential", cred.Name, "attempt", attempt+1)
			shouldRetry = true
			retryReason = RetryReasonNetErr
			transportErr = readErr
			continue
		}

		p.recordProviderResponse(r.Context(), cred, modelID, realModelID, resp.StatusCode, resp.Header, responseBody)

		// Check if we should retry with another same-type credential
		shouldRetry, retryReason = ShouldRetryWithFallback(resp.StatusCode, responseBody)
		if !shouldRetry {
			break
		}

		// Mid-retry failure — the request will be retried with another credential.
		// The final failure (if all attempts fail) is logged at ERROR when the
		// response is written to the client.
		retryLogArgs := []any{
			"error_code", resp.StatusCode, "credential", cred.Name,
			"reason", retryReason, "model", modelID,
			"attempt", attempt + 1, "max_attempts", p.maxProviderRetries + 1,
		}
		retryLogArgs = appendResponseBodyForLogs(retryLogArgs, cred, string(responseBody))
		p.logger.WarnContext(r.Context(), "Provider returned retryable error, will retry", retryLogArgs...)
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

		p.logger.InfoContext(r.Context(), "All retry credentials exhausted, attempting fallback proxy",
			"credential", cred.Name, "model", modelID,
			"last_status", fallbackStatus, "reason", retryReason)
		fallbackReq := requestWithPath(r, prepared.proxyPath)
		success, fallbackReason := p.TryFallbackProxy(w, fallbackReq, modelID, cred.Name, fallbackStatus, retryReason, proxyBody, start, logCtx)
		if success {
			if closeBody != nil {
				closeBody()
			}
			return
		}
		p.logger.DebugContext(r.Context(), "Fallback retry failed, using original response",
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
		p.logUpstreamError(r.Context(), "All provider attempts failed: no upstream response", statusCode, cred, modelID, nil,
			"error", transportErr,
			"url", targetURL,
			"request_id", logCtx.RequestID)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = statusCode
		logCtx.ErrorMsg = "All provider attempts failed"
		logCtx.TargetURL = targetURL
		if statusCode == http.StatusRequestTimeout {
			WriteErrorTimeout(w, statusMessage)
		} else {
			WriteErrorBadGateway(w, statusMessage)
		}
		return
	}

	// Ensure response body is closed at function exit
	if closeBody != nil {
		defer closeBody()
	}

	// === Process final response ===
	var finalResponseBody []byte

	if isStreamingResp {
		p.logger.DebugContext(r.Context(), "Response is streaming", "credential", cred.Name)
	} else {
		// Decode the response body for logging (handles gzip, etc.)
		contentEncoding := resp.Header.Get("Content-Encoding")
		decodedBody := decodeResponseBody(responseBody, contentEncoding)

		// Transform response to OpenAI format (only for successful responses).
		// For error responses (4xx/5xx) pass the provider body through unchanged.
		// nativeResponses path skips this step — provider→Responses API conversion
		// happens further down via provResponses.ResponseTo().
		if resp.StatusCode >= 200 && resp.StatusCode < 300 && !prepared.nativeResponses && conv != nil && !conv.IsPassthrough() {
			convertedBody, convErr := conv.ResponseTo([]byte(decodedBody))
			if convErr != nil {
				args := []any{
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID,
				}
				args = appendResponseBodyForLogs(args, cred, decodedBody)
				p.logger.ErrorContext(r.Context(), "Failed to transform provider response to OpenAI format", args...)
				finalResponseBody = []byte(decodedBody)
			} else {
				finalResponseBody = convertedBody
				p.logTransformedResponse(r.Context(), cred.Name, string(cred.Type), finalResponseBody)
			}
		} else {
			finalResponseBody = []byte(decodedBody)
		}

		// bodyForTokenExtraction is set to finalResponseBody now and may be updated
		// after Responses API conversion (for nativeResponses the raw provider body
		// uses a provider-specific format that ExtractTokenUsage cannot parse).
		bodyForTokenExtraction := finalResponseBody
		if len(bodyForTokenExtraction) == 0 {
			bodyForTokenExtraction = []byte(decodedBody)
		}

		// Handle Responses API response body.
		if prepared.nativeResponses && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Native Responses converter: convert provider response → *responses.Response.
			nativeResp, convErr := provResponses.ResponseTo([]byte(decodedBody), modelID)
			if convErr != nil {
				args := []any{
					"credential", cred.Name, "provider", string(cred.Type),
					"model", modelID, "error", convErr,
					"request_id", logCtx.RequestID,
				}
				args = appendResponseBodyForLogs(args, cred, decodedBody)
				p.logger.ErrorContext(r.Context(), "Failed to convert native Responses API response", args...)
				// finalResponseBody already holds decodedBody — return as-is
			} else {
				applyResponsesMetadata(nativeResp, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(nativeResp); marshalErr == nil {
					finalResponseBody = enriched
					// Use the converted Responses API body for token extraction:
					// the raw provider body (e.g. Vertex usageMetadata) is not
					// parseable by ExtractTokenUsage which expects OpenAI-compatible fields.
					bodyForTokenExtraction = finalResponseBody
				}
				if saveResponseFn != nil {
					saveResponseFn(nativeResp)
				}
			}
		} else if prepared.passthroughResponses && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Codex passthrough: body is already in Responses API format — enrich and optionally save.
			var respObj responses.Response
			if err := json.Unmarshal(finalResponseBody, &respObj); err == nil {
				applyResponsesMetadata(&respObj, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
					finalResponseBody = enriched
				}
				if saveResponseFn != nil {
					saveResponseFn(&respObj)
				}
			}
		} else if prepared.convertedResp && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// Non-codex: convert Chat Completions response back to Responses API format.
			responsesBody, convErr := responses.ChatToResponse(finalResponseBody)
			if convErr != nil {
				p.logger.ErrorContext(r.Context(), "Failed to convert to Responses API format",
					"credential", cred.Name, "model", modelID, "error", convErr,
					"request_id", logCtx.RequestID)
				// fallback: use Chat Completions body
			} else {
				// Enrich the response with request-echoed fields (store, previous_response_id,
				// metadata) for both the client payload and the store record.
				var respObj responses.Response
				if err := json.Unmarshal(responsesBody, &respObj); err == nil {
					applyResponsesMetadata(&respObj, prepared.responsesMetadata)
					if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
						responsesBody = enriched
					}
					if saveResponseFn != nil {
						saveResponseFn(&respObj)
					}
				} else if saveResponseFn != nil {
					var r2 responses.Response
					if json.Unmarshal(responsesBody, &r2) == nil {
						saveResponseFn(&r2)
					}
				}
				finalResponseBody = responsesBody
			}
		}

		rawErrorBody := finalResponseBody
		if resp.StatusCode >= 400 && shouldMaskUpstreamErrors(cred) {
			finalResponseBody = maskedUpstreamErrorBody(resp.StatusCode)
			bodyForTokenExtraction = finalResponseBody
			resp.Header.Set("Content-Type", "application/json")
		}

		tokens := extractTokensFromResponse(string(bodyForTokenExtraction), config.ProviderTypeOpenAI)
		if tokens > 0 {
			p.rateLimiter.ConsumeTokens(cred.Name, tokens)
			if modelID != "" {
				p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, tokens)
			}
			p.logger.DebugContext(r.Context(), "Token usage recorded",
				"credential", cred.Name, "model", modelID, "tokens", tokens)
		}

		if p.logger.Enabled(context.Background(), slog.LevelDebug) {
			if resp.StatusCode >= 400 && shouldMaskUpstreamErrors(cred) {
				p.logger.DebugContext(r.Context(), "Proxy response body masked",
					"credential", cred.Name,
					"content_encoding", contentEncoding,
					"status_code", resp.StatusCode)
			} else {
				p.logger.DebugContext(r.Context(), "Proxy response body",
					"credential", cred.Name, "content_encoding", contentEncoding,
					"body", logger.TruncateLongFields(string(finalResponseBody), 500))
			}
		}

		resp.Body = io.NopCloser(bytes.NewReader(finalResponseBody))

		// Log to LiteLLM DB (non-streaming)
		logCtx.TokenUsage = converter.ExtractTokenUsage(bodyForTokenExtraction)
		logCtx.Status = "success"
		logCtx.HTTPStatus = resp.StatusCode
		logCtx.TargetURL = targetURL

		if resp.StatusCode >= 400 {
			logCtx.Status = "failure"
			if shouldMaskUpstreamErrors(cred) {
				logCtx.ErrorMsg = "Upstream provider error"
			} else {
				logCtx.ErrorMsg = extractErrorMessage(finalResponseBody)
			}
			// Final error returned to the client — single unified ERROR record
			// with everything needed for debugging.
			p.logUpstreamError(r.Context(), "Upstream request completed with error status", resp.StatusCode, cred, modelID, rawErrorBody,
				"url", targetURL,
				"request_id", logCtx.RequestID)
		} else if logCtx.TokenUsage != nil {
			p.metrics.RecordTokenUsage(cred.Name, modelID,
				logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
				logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
		}

		if logCtx.IsImageGeneration && logCtx.TokenUsage != nil {
			logCtx.TokenUsage.ImageCount = logCtx.ImageCount
		}
		if logCtx.Token != "" && logCtx.Credential != nil {
			if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
				p.logger.WarnContext(r.Context(), "Failed to queue spend log",
					"error", err, "request_id", logCtx.RequestID)
			}
		}
		logCtx.Logged = true
	}

	// Copy response headers (skip hop-by-hop headers and transformation-related headers)
	copyResponseHeaders(w, resp.Header, cred.Type)
	// Return credential name only to internal proxy clients, not to end users.
	if logCtx.IsProxyRequest {
		w.Header().Set("X-Credential-Name", cred.Name)
	}

	rc := http.NewResponseController(w)

	if isStreamingResp {
		if resp.StatusCode >= 400 {
			// Error status on a streaming response — the body is forwarded to the
			// client as a stream and is not available here for logging.
			p.logUpstreamError(r.Context(), "Upstream returned error status on streaming response", resp.StatusCode, cred, modelID, nil,
				"url", targetURL,
				"request_id", logCtx.RequestID)
			if shouldMaskUpstreamErrors(cred) {
				body := maskedUpstreamErrorBody(resp.StatusCode)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
				w.WriteHeader(resp.StatusCode)
				_, _ = w.Write(body)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = resp.StatusCode
				logCtx.ErrorMsg = "Upstream provider error"
				logCtx.TargetURL = targetURL
				return
			}
		}
		w.WriteHeader(resp.StatusCode)

		if logCtx != nil {
			logCtx.PromptTokensEstimate = estimatePromptTokensForModel(body, realModelID)

			p.logger.DebugContext(
				r.Context(),
				"Estimated prompt tokens for streaming response",
				"estimate", logCtx.PromptTokensEstimate,
				"request_id", logCtx.RequestID,
			)
		}

		p.logger.DebugContext(r.Context(), "Streaming handler selection",
			"is_responses_api", prepared.isResponsesAPI,
			"converted_resp", prepared.convertedResp,
			"provider", cred.Type,
			"model", modelID,
			"resp_content_type", resp.Header.Get("Content-Type"),
			"resp_status", resp.StatusCode)

		streamCompleted := false
		if prepared.nativeResponses {
			// Native Responses converter (Vertex AI, Anthropic): convert provider SSE
			// directly to Responses API SSE via ProviderResponses.StreamTo().
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				err := p.handleNativeResponsesStreaming(w, resp, provResponses, modelID, logCtx, saveResponseFn, prepared.responsesMetadata)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle native Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream as-is
				err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle streaming response", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				}
			}
		} else if prepared.convertedResp {
			// For Responses API: only transform successful responses (2xx status).
			// For error responses (4xx/5xx), pass through the provider error as-is.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				// Transform to Responses API SSE format
				err := p.handleResponsesAPIStreaming(w, resp, cred, modelID, logCtx, saveResponseFn, prepared.responsesMetadata)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
					// Note: finalizeStreamingLog inside handleTransformedStreaming already
					// logged the spend. We only update error metadata here for the defer
					// safety net, but don't reset Logged to avoid double logging.
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream using provider's native format instead
				err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
						"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		} else if prepared.passthroughResponses {
			// Codex passthrough: provider returns native Responses API SSE — forward as-is.
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				if err := p.handlePassthroughResponsesStreaming(w, resp, cred.Name, realModelID, logCtx, saveResponseFn); err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle passthrough Responses API streaming", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else {
					streamCompleted = true
				}
			} else {
				// Error response: stream as-is
				err := p.handleStreamingWithTokens(w, resp, cred.Name, modelID, logCtx)
				if err != nil {
					p.logStreamHandlerError(r.Context(), "Failed to handle streaming response", err,
						"credential", cred.Name, "model", modelID, "request_id", logCtx.RequestID)
				} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					streamCompleted = true
				}
			}
		} else {
			err := p.handleProviderStreaming(w, resp, cred, realModelID, modelID, logCtx)
			if err != nil {
				p.logStreamHandlerError(r.Context(), "Failed to handle provider streaming response", err,
					"credential", cred.Name, "provider", cred.Type, "model", modelID, "request_id", logCtx.RequestID)
			} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				streamCompleted = true
			}
		}
		if streamCompleted {
			logCtx.RequestCompleted = true
			p.setSessionBinding(logCtx.SessionID, modelID, cred.Name)
		}

	} else {
		acceptEncoding := r.Header.Get("Accept-Encoding")
		acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
		targetEncoding := SelectBestEncoding(acceptedEncodings)

		outputBody := finalResponseBody
		if targetEncoding != "identity" && len(finalResponseBody) > 0 {
			compressedBody, usedEncoding, compErr := CompressBody(finalResponseBody, targetEncoding)
			if compErr != nil {
				p.logger.WarnContext(r.Context(), "Failed to compress response body",
					"credential", cred.Name, "encoding", targetEncoding, "error", compErr)
			} else {
				p.logger.DebugContext(r.Context(), "Response body compressed for client",
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
				p.logger.DebugContext(r.Context(), "Client disconnected during response body copy", "error", err)
				p.recordAbortedRequest(cred.Name, r.URL.Path, modelID)
			} else {
				p.logger.ErrorContext(r.Context(), "Failed to copy response body", "error", err)
			}
		} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			logCtx.RequestCompleted = true
			p.setSessionBinding(logCtx.SessionID, modelID, cred.Name)
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

// applyResponsesMetadata echoes request-side fields (store, previous_response_id, metadata)
// into a Response object so the stored record and the client payload match the request.
// It is safe to call multiple times (idempotent field assignment).
func applyResponsesMetadata(resp *responses.Response, meta *responses.ResponsesMetadata) {
	if meta == nil || resp == nil {
		return
	}
	resp.Store = meta.Store
	if meta.PreviousResponseID != "" {
		resp.PreviousResponseID = meta.PreviousResponseID
	}
	if meta.Metadata != nil {
		resp.Metadata = meta.Metadata
	}
}

// HandleGetResponse handles GET /v1/responses/{response_id}.
// Returns the stored Responses API response if the caller owns it.
func (p *Proxy) HandleGetResponse(w http.ResponseWriter, r *http.Request) {
	if p.responseStore == nil {
		WriteErrorNotFound(w, "Response store not enabled")
		return
	}

	// Extract response_id from path (e.g. /v1/responses/resp_abc123)
	const prefix = "/v1/responses/"
	responseID := strings.TrimPrefix(r.URL.Path, prefix)
	if responseID == "" || strings.ContainsRune(responseID, '/') {
		WriteErrorNotFound(w, "Not Found")
		return
	}

	// Auth
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		WriteErrorUnauthorized(w, "Missing Authorization header")
		return
	}
	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == authHeader {
		WriteErrorUnauthorized(w, "Invalid Authorization header format")
		return
	}

	var resp *responses.Response
	var err error

	if p.isMasterKey(token) {
		// Master key: bypass ownership check
		resp, err = p.responseStore.GetResponseByID(r.Context(), responseID)
	} else {
		// Enforce blocked/expired/budget on the token before serving stored
		// responses, matching the checks other endpoints run via authenticateRequest.
		// On DB unavailability (handleLiteLLMAuthError returns false) fall back to the
		// legacy ownership-only path so reads stay available during a DB outage.
		if p.LiteLLMDB != nil && p.LiteLLMDB.IsEnabled() {
			if _, valErr := p.LiteLLMDB.ValidateToken(r.Context(), token); valErr != nil {
				if p.handleLiteLLMAuthError(r.Context(), w, valErr, token) {
					return
				}
			}
		}
		apiKeyHash := litellmdb.HashToken(token)
		resp, err = p.responseStore.GetResponse(r.Context(), responseID, apiKeyHash)
	}

	if err != nil {
		p.logger.DebugContext(r.Context(), "HandleGetResponse: not found or unauthorized", "id", responseID, "error", err)
		WriteErrorNotFound(w, "Not Found")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if encErr := json.NewEncoder(w).Encode(resp); encErr != nil {
		p.logger.ErrorContext(r.Context(), "HandleGetResponse: failed to encode response", "id", responseID, "error", encErr)
	}
}
