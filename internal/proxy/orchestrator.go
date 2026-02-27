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
	"github.com/mixaill76/auto_ai_router/internal/security"
)

type orchestratedRequest struct {
	request   *http.Request
	body      []byte
	modelID   string
	streaming bool
	cred      *config.CredentialConfig
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

	body, modelID, streaming, ok := p.readRequestBodyAndSelectModel(w, r, logCtx)
	if !ok {
		return nil, false
	}

	cred, ok := p.selectCredentialForModel(w, modelID, logCtx)
	if !ok {
		return nil, false
	}

	logCtx.Credential = cred
	r = markCredentialAsTried(r, cred.Name)

	return &orchestratedRequest{
		request:   r,
		body:      body,
		modelID:   modelID,
		streaming: streaming,
		cred:      cred,
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
	if p.litellmDB == nil || !p.litellmDB.IsEnabled() {
		return false
	}
	if p.healthChecker != nil {
		return p.healthChecker.IsDBHealthy()
	}
	return p.litellmDB.IsHealthy()
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
		http.Error(w, "Unauthorized: Missing Authorization header", http.StatusUnauthorized)
		return false
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")
	logCtx.Token = token
	if token == authHeader {
		p.logger.Error("Invalid Authorization header format")
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized
		logCtx.ErrorMsg = "Invalid Authorization header format"
		http.Error(w, "Unauthorized: Invalid Authorization header format", http.StatusUnauthorized)
		return false
	}

	if token == p.masterKey {
		return true
	} else if isLiteLLMHealthy {
		tokenInfo, err := p.litellmDB.ValidateToken(r.Context(), token)
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
		http.Error(w, "Unauthorized: Invalid master key", http.StatusUnauthorized)
	}

	return false
}

func (p *Proxy) readRequestBodyAndSelectModel(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
) ([]byte, string, bool, bool) {
	maxBodyBytes := int64(p.maxBodySizeMB) * 1024 * 1024
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil {
		p.logger.Error("Failed to read request body", "error", err)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Failed to read request body: " + err.Error()
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return nil, "", false, false
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
		http.Error(w, "Request Entity Too Large", http.StatusRequestEntityTooLarge)
		return nil, "", false, false
	}

	modelID, streaming, sessionID, body := extractMetadataFromBody(body)
	logCtx.ModelID = modelID
	logCtx.SessionID = sessionID

	if modelID == "" {
		p.logger.Error("Model not specified in request body")
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Model not specified in request body"
		http.Error(w, "Bad Request: model field is required", http.StatusBadRequest)
		return nil, "", false, false
	}

	// Resolve model alias
	if resolved, isAlias := p.modelManager.ResolveAlias(modelID); isAlias {
		p.logger.Debug("Resolved model alias", "alias", modelID, "resolved", resolved)
		body = replaceModelInBody(body, modelID, resolved)
		modelID = resolved
		logCtx.ModelID = modelID
	}

	return body, modelID, streaming, true
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

	http.Error(w, errorMsg, errCode)
	return nil, false
}
