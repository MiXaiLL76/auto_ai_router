package proxy

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/users"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

type orchestratedRequest struct {
	request        *http.Request
	body           []byte
	modelID        string // alias name (for rate limiting, credential lookup, logging)
	realModelID    string // real model name sent to provider (equals modelID if no alias configured)
	streaming      bool
	cred           *config.CredentialConfig
	isResponsesAPI bool
	convertedResp  bool
}

// orchestrateRequest performs auth and credential selection for an incoming request.
func (p *Proxy) orchestrateRequest(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
) (*orchestratedRequest, bool) {
	r = initializeRetryTrackingContext(r)

	isLiteLLMHealthy := p.isLiteLLMHealthy()

	if !p.authenticateRequest(w, r, logCtx, isLiteLLMHealthy) {
		return nil, false
	}

	body, modelID, realModelID, streaming, ok := p.readRequestBodyAndSelectModel(w, r, logCtx)
	if !ok {
		return nil, false
	}

	// Detect Responses API requests and select credential before conversion.
	isResponsesAPI := responses.IsResponsesAPI(body) && strings.Contains(r.URL.Path, "/responses")

	cred, ok := p.selectCredentialForModel(w, modelID, logCtx)
	if !ok {
		return nil, false
	}

	p.logger.Debug("Responses API detection",
		"is_responses_api", isResponsesAPI,
		"provider", cred.Type,
		"model", modelID,
		"streaming", streaming,
		"url_path", r.URL.Path)

	convertedResp := false
	// Always convert Responses API requests for ALL providers.
	// Even "openai"-type credentials may point to Azure OpenAI or other
	// OpenAI-compatible endpoints that don't support the native /v1/responses
	// endpoint and return Chat Completions SSE instead of Responses API SSE.
	// Chat Completions API is universally supported, so converting is always safe.
	if isResponsesAPI {
		chatBody, convErr := responses.RequestToChat(body)
		if convErr != nil {
			p.logger.Error("Failed to convert Responses API request", "error", convErr)
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusBadRequest
			logCtx.ErrorMsg = "Failed to convert Responses API request: " + convErr.Error()
			WriteErrorBadRequest(w, "Failed to convert Responses API request")
			return nil, false
		}
		body = chatBody
		convertedResp = true
		// For streaming: inject stream_options.include_usage since extractMetadataFromBody
		// skipped it for Responses API (the original body had "input" not "messages").
		// Now that we've converted to Chat Completions format, providers need this.
		if streaming {
			body = injectStreamOptions(body)
		}
		// Rewrite URL path from /v1/responses to /v1/chat/completions
		// so passthrough providers (OpenAI, Proxy) send to the correct endpoint.
		r.URL.Path = strings.Replace(r.URL.Path, "/responses", "/chat/completions", 1)
		p.logger.Debug("Converted Responses API request to Chat Completions format",
			"model", modelID, "streaming", streaming)
	}

	logCtx.Credential = cred
	r = markCredentialAsTried(r, cred.Name)

	return &orchestratedRequest{
		request:        r,
		body:           body,
		modelID:        modelID,
		realModelID:    realModelID,
		streaming:      streaming,
		cred:           cred,
		isResponsesAPI: isResponsesAPI,
		convertedResp:  convertedResp,
	}, true
}

func initializeRetryTrackingContext(r *http.Request) *http.Request {
	ctx := r.Context()
	ctx = SetTried(ctx, make(map[string]bool))
	ctx = context.WithValue(ctx, AttemptCountKey{}, 0)
	return r.WithContext(ctx)
}

func markCredentialAsTried(r *http.Request, credentialName string) *http.Request {
	ctx := r.Context()
	triedCreds := GetTried(ctx)
	triedCreds[credentialName] = true
	ctx = SetTried(ctx, triedCreds)
	return r.WithContext(ctx)
}

func (p *Proxy) isLiteLLMHealthy() bool {
	if p.LiteLLMDB == nil || !p.LiteLLMDB.IsEnabled() {
		return false
	}
	if p.healthChecker != nil {
		return p.healthChecker.IsDBHealthy()
	}
	return p.LiteLLMDB.IsHealthy()
}

func (p *Proxy) authenticateRequest(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	isLiteLLMHealthy bool,
) bool {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		p.logger.Error("Missing Authorization header")
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized
		logCtx.ErrorMsg = "Missing Authorization header"
		WriteErrorUnauthorized(w, "Missing Authorization header")
		return false
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")
	logCtx.Token = token
	if token == authHeader {
		p.logger.Error("Invalid Authorization header format")
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized
		logCtx.ErrorMsg = "Invalid Authorization header format"
		WriteErrorUnauthorized(w, "Invalid Authorization header format")
		return false
	}

	if token == p.masterKey {
		return true
	}

	// JWT session token validation (tokens from /v2/login)
	if strings.HasPrefix(token, "eyJ") {
		claims, jwtErr := users.ValidateSessionJWT(token, p.masterKey)
		if jwtErr == nil && claims != nil {
			p.logger.Debug("Authenticated via session JWT",
				"user_id", claims.UserID,
				"user_role", claims.UserRole,
			)
			return true
		}
		// JWT validation failed — fall through to LiteLLM DB check
	}

	if isLiteLLMHealthy {
		tokenInfo, err := p.LiteLLMDB.ValidateToken(r.Context(), token)
		logCtx.TokenInfo = tokenInfo
		if err != nil {
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusUnauthorized

			if p.handleLiteLLMAuthError(w, err, token) {
				logCtx.ErrorMsg = "LiteLLM auth validation failed"
			} else {
				logCtx.ErrorMsg = "LiteLLM DB unavailable"
			}
			return false
		} else if tokenInfo != nil {
			p.logger.Debug("Token validated via LiteLLM DB",
				"user_id", tokenInfo.UserID,
				"team_id", tokenInfo.TeamID,
			)
		}
		return true
	} else {
		p.logger.Error("Invalid master key", "provided_key_prefix", security.MaskAPIKey(token))
		WriteErrorUnauthorized(w, "Invalid master key")
	}

	return false
}

func (p *Proxy) readRequestBodyAndSelectModel(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
) ([]byte, string, string, bool, bool) {
	maxBodyBytes := int64(p.maxBodySizeMB) * 1024 * 1024
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil {
		p.logger.Error("Failed to read request body", "error", err)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Failed to read request body: " + err.Error()
		WriteErrorBadRequest(w, "Failed to read request body")
		return nil, "", "", false, false
	}
	if closeErr := r.Body.Close(); closeErr != nil {
		p.logger.Error("Failed to close request body", "error", closeErr)
	}
	if int64(len(body)) > maxBodyBytes {
		p.logger.Error("Request body exceeds max size",
			"max_body_size_mb", p.maxBodySizeMB,
			"actual_size_bytes", len(body),
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusRequestEntityTooLarge
		logCtx.ErrorMsg = "Request body too large"
		WriteErrorTooLarge(w, "Request Entity Too Large")
		return nil, "", "", false, false
	}

	modelID, streaming, sessionID, body := extractMetadataFromBody(body)
	logCtx.ModelID = modelID
	logCtx.SessionID = sessionID

	if modelID == "" {
		p.logger.Error("Model not specified in request body")
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Model not specified in request body"
		WriteErrorBadRequest(w, "model field is required")
		return nil, "", "", false, false
	}

	// Resolve model_alias entries (changes modelID to real name; credential lookup uses real name)
	if resolved, isAlias := p.modelManager.ResolveAlias(modelID); isAlias {
		p.logger.Debug("Resolved model alias", "alias", modelID, "resolved", resolved)
		body = openai.ReplaceModelInBody(body, modelID, resolved)
		modelID = resolved
		logCtx.ModelID = modelID
	}

	// Resolve models[].model field: replace model in body for provider but keep alias as modelID
	// for rate limiting and credential lookup.
	realModelID := modelID
	if realName, hasReal := p.modelManager.GetRealModelName(modelID); hasReal {
		p.logger.Debug("Resolved model real name", "alias", modelID, "real", realName)
		body = openai.ReplaceModelInBody(body, modelID, realName)
		realModelID = realName
	}

	body = openai.ReplaceBodyParam(modelID, body)

	return body, modelID, realModelID, streaming, true
}

func (p *Proxy) selectCredentialForModel(
	w http.ResponseWriter,
	modelID string,
	logCtx *RequestLogContext,
) (*config.CredentialConfig, bool) {
	cred, err := p.balancer.NextForModel(modelID)
	if err == nil {
		return cred, true
	}

	fallbackErr := error(nil)
	cred, fallbackErr = p.balancer.NextFallbackForModel(modelID)
	if fallbackErr == nil {
		return cred, true
	}

	errCode := http.StatusTooManyRequests
	errorMsg := fmt.Sprintf("No credentials available: %v", err)
	if errors.Is(err, balancer.ErrRateLimitExceeded) || errors.Is(fallbackErr, balancer.ErrRateLimitExceeded) {
		errorMsg = "Rate limit exceeded"
	}

	p.logger.Error("No credentials available (regular and fallback)",
		"model", modelID,
		"primary_error", err,
		"fallback_error", fallbackErr,
	)

	logCtx.Status = "failure"
	logCtx.HTTPStatus = errCode
	logCtx.ErrorMsg = errorMsg
	logCtx.Credential = &config.CredentialConfig{
		Name: "system",
		Type: config.ProviderTypeProxy,
	}

	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.Warn("Failed to queue error log for no credentials",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
	logCtx.Logged = true

	WriteErrorRateLimit(w, errorMsg)
	return nil, false
}
