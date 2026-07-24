package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/auth"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/responsestore"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/security"
)

type orchestratedRequest struct {
	request              *http.Request
	body                 []byte // body with realModelID substituted (for non-proxy providers)
	proxyBody            []byte // body with original modelID alias (for proxy forwarding)
	proxyPath            string
	baseBody             []byte
	baseProxyBody        []byte
	modelID              string // alias name (for rate limiting, credential lookup, logging)
	realModelID          string // real model name sent to provider (equals modelID if no alias configured)
	baseRealModelID      string
	basePath             string
	streaming            bool
	cred                 *config.CredentialConfig
	isResponsesAPI       bool
	convertedResp        bool
	passthroughResponses bool // true for codex/OpenAI models: Responses API forwarded as-is (no conversion)
	nativeResponses      bool // true when using Phase 4 ProviderResponses converter (Vertex/Anthropic)
	responsesPrevHandled bool
	responsesMetadata    *responses.ResponsesMetadata // non-nil for Responses API requests
	stickyCacheEligible  bool
}

type credentialPreparedRequest struct {
	body                 []byte
	proxyBody            []byte
	proxyPath            string
	realModelID          string
	path                 string
	convertedResp        bool
	passthroughResponses bool
	nativeResponses      bool
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
	if !p.enforceBudgetAndRateLimits(w, r, logCtx, modelID, realModelID, body) {
		return nil, false
	}

	// proxyBody: body with the original alias restored.
	// Proxy credentials handle their own model routing, so they must receive the
	// alias ("anthropic/claude-sonnet-4.6"), not the provider-specific real name
	// ("global.anthropic.claude-sonnet-4-6") that was substituted for direct providers.
	proxyBody := body
	if modelID != realModelID {
		proxyBody = openai.ReplaceModelInBody(body, realModelID, modelID)
	}
	baseBody := body
	baseProxyBody := proxyBody
	baseRealModelID := realModelID
	basePath := r.URL.Path

	// Detect Responses API requests and select credential before conversion.
	isResponsesAPI := responses.IsResponsesAPI(body) && strings.Contains(r.URL.Path, "/responses")

	var responsesMetadata *responses.ResponsesMetadata
	var prevEntry *responsestore.StoredEntry
	prevEntryHandled := false
	preferredCredentialName := ""

	if isResponsesAPI {
		// Extract Responses-API-only metadata before the fields are deleted.
		meta := responses.ExtractResponsesMetadata(body)
		responsesMetadata = &meta

		if meta.PreviousResponseID != "" && p.responseStore != nil {
			apiKeyHash := litellmdb.HashToken(logCtx.Token)
			entry, loadErr := p.responseStore.GetEntry(r.Context(), meta.PreviousResponseID, apiKeyHash)
			if loadErr != nil {
				p.logger.WarnContext(r.Context(), "Could not load previous_response_id, proceeding without history",
					"id", meta.PreviousResponseID, "error", loadErr)
			} else {
				prevEntry = entry
				preferredCredentialName = entry.CredentialName
			}
		}
	}

	cred, ok := p.selectCredentialForModel(w, modelID, logCtx.SessionID, preferredCredentialName, logCtx)
	if !ok {
		return nil, false
	}

	p.logger.DebugContext(r.Context(), "Responses API detection",
		"is_responses_api", isResponsesAPI,
		"provider", cred.Type,
		"model", modelID,
		"streaming", streaming,
		"url_path", r.URL.Path)

	if isResponsesAPI {
		// Handle previous_response_id: load the previous entry and prepend its
		// accumulated input + output so the model sees the full conversation history.
		if responsesMetadata.PreviousResponseID != "" && prevEntry != nil && prevEntry.ResponseJSON != nil {
			var accInput json.RawMessage
			if prevEntry.AccumulatedInput != nil {
				accInput = prevEntry.AccumulatedInput
			}
			newBody, prependErr := responses.PrependHistoryToInput(baseBody, accInput, prevEntry.ResponseJSON.Output)
			if prependErr != nil {
				p.logger.WarnContext(r.Context(), "Failed to prepend previous response history, ignoring",
					"id", responsesMetadata.PreviousResponseID, "error", prependErr)
			} else {
				baseBody = newBody
				prevEntryHandled = true
				baseProxyBody = baseBody
				if modelID != baseRealModelID {
					baseProxyBody = openai.ReplaceModelInBody(baseBody, baseRealModelID, modelID)
				}
				p.logger.DebugContext(r.Context(), "Prepended previous response history to input",
					"previous_response_id", responsesMetadata.PreviousResponseID,
					"output_items", len(prevEntry.ResponseJSON.Output),
					"credential", preferredCredentialName,
				)
			}
		}

		// Capture the full accumulated input (history + current) for storage.
		// This must happen after any history prepending but before RequestToChat removes "input".
		responsesMetadata.AccumulatedInput = responses.ExtractInputArray(baseBody)
	}

	stickyCacheEligible := logCtx.SessionID != "" || preferredCredentialName != ""
	credentialReq, prepErr := p.prepareRequestForCredential(
		r,
		baseBody,
		baseProxyBody,
		modelID,
		baseRealModelID,
		basePath,
		streaming,
		cred,
		isResponsesAPI,
		prevEntryHandled,
		stickyCacheEligible,
	)
	if prepErr != nil {
		p.logger.ErrorContext(r.Context(), "Failed to prepare request for credential",
			"error_code", http.StatusBadRequest,
			"credential", cred.Name, "provider", string(cred.Type),
			"model", modelID, "error", prepErr,
			"request_id", logCtx.RequestID)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Failed to convert Responses API request: " + prepErr.Error()
		WriteErrorBadRequest(w, "Failed to convert Responses API request")
		return nil, false
	}
	body = credentialReq.body
	proxyBody = credentialReq.proxyBody
	realModelID = credentialReq.realModelID
	r.URL.Path = credentialReq.path

	logCtx.Credential = cred
	r = markCredentialAsTried(r, cred.Name)

	return &orchestratedRequest{
		request:              r,
		body:                 body,
		proxyBody:            proxyBody,
		proxyPath:            credentialReq.proxyPath,
		baseBody:             baseBody,
		baseProxyBody:        baseProxyBody,
		modelID:              modelID,
		realModelID:          realModelID,
		baseRealModelID:      baseRealModelID,
		basePath:             basePath,
		streaming:            streaming,
		cred:                 cred,
		isResponsesAPI:       isResponsesAPI,
		convertedResp:        credentialReq.convertedResp,
		passthroughResponses: credentialReq.passthroughResponses,
		nativeResponses:      credentialReq.nativeResponses,
		responsesPrevHandled: prevEntryHandled,
		responsesMetadata:    responsesMetadata,
		stickyCacheEligible:  stickyCacheEligible,
	}, true
}

func (p *Proxy) prepareRequestForCredential(
	r *http.Request,
	baseBody []byte,
	baseProxyBody []byte,
	modelID string,
	baseRealModelID string,
	basePath string,
	streaming bool,
	cred *config.CredentialConfig,
	isResponsesAPI bool,
	prevEntryHandled bool,
	stickyCacheEligible bool,
) (credentialPreparedRequest, error) {
	body := baseBody
	proxyBody := baseProxyBody
	realModelID := baseRealModelID
	if !cred.IsProxyLike() && p.modelManager != nil {
		if credRealName, ok := p.modelManager.GetRealModelNameForCredential(modelID, cred.Name); ok && credRealName != realModelID {
			p.logger.DebugContext(r.Context(), "Re-resolved real model name for credential",
				"alias", modelID,
				"old_real", realModelID,
				"new_real", credRealName,
				"credential", cred.Name,
			)
			body = openai.ReplaceModelInBody(body, realModelID, credRealName)
			realModelID = credRealName
		}
	}

	if p.stickyAutoCacheCtrl &&
		stickyCacheEligible &&
		(cred.Type == config.ProviderTypeAnthropic || cred.Type == config.ProviderTypeCometAPI || cred.Type == config.ProviderTypeBedrock) {
		body = anthropicconv.InjectCacheControl(body)
	}

	req := credentialPreparedRequest{
		body:        body,
		proxyBody:   proxyBody,
		proxyPath:   basePath,
		realModelID: realModelID,
		path:        basePath,
	}
	if !isResponsesAPI {
		req.body = openai.ReplaceBodyParam(realModelID, body)
		return req, nil
	}

	switch {
	case p.modelManager != nil && p.modelManager.IsPassthroughResponsesForProvider(modelID, cred.Type):
		req.body = responses.PrepareCodexPassthrough(body, prevEntryHandled)
		req.proxyBody = responses.PrepareCodexPassthrough(proxyBody, prevEntryHandled)
		req.passthroughResponses = true
		p.logger.DebugContext(r.Context(), "Native Responses API passthrough",
			"model", modelID, "provider", cred.Type, "streaming", streaming)
	case responses.HasNativeResponsesForModel(cred.Type, realModelID) &&
		(p.modelManager == nil || !p.modelManager.HasPassthroughResponsesOverride(modelID)):
		req.nativeResponses = true
		p.logger.DebugContext(r.Context(), "Native Responses converter path",
			"model", modelID, "provider", cred.Type, "streaming", streaming)
	default:
		chatBody, err := responses.RequestToChat(body)
		if err != nil {
			return req, err
		}
		req.body = openai.ReplaceBodyParam(realModelID, chatBody)
		req.convertedResp = true
		if streaming {
			req.body = injectStreamOptions(req.body)
		}
		req.path = strings.Replace(basePath, "/responses", "/chat/completions", 1)
		p.logger.DebugContext(r.Context(), "Converted Responses API request to Chat Completions format",
			"model", modelID, "streaming", streaming)
	}

	return req, nil
}

func initializeRetryTrackingContext(r *http.Request) *http.Request {
	ctx := r.Context()
	ctx = SetTried(ctx, make(map[string]bool))
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

type clientCredentialState uint8

const (
	clientCredentialMissing clientCredentialState = iota
	clientCredentialMalformed
	clientCredentialPresent
)

// extractClientToken implements the public client-credential transport contract.
// Authorization is authoritative whenever the header is present: a malformed
// Bearer value must never fall through to a valid x-api-key value.
func extractClientToken(r *http.Request) (string, clientCredentialState) {
	if r == nil {
		return "", clientCredentialMissing
	}
	authorizationValues, authorizationPresent := headerValuesFold(r.Header, "Authorization")
	if authorizationPresent {
		if len(authorizationValues) != 1 {
			return "", clientCredentialMalformed
		}
		authHeader := strings.TrimSpace(authorizationValues[0])
		if authHeader == "" {
			return "", clientCredentialMissing
		}
		parts := strings.Fields(authHeader)
		if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") || strings.ContainsRune(parts[1], ',') {
			return "", clientCredentialMalformed
		}
		return parts[1], clientCredentialPresent
	}
	xAPIKeyValues, xAPIKeyPresent := headerValuesFold(r.Header, "X-Api-Key")
	if !xAPIKeyPresent {
		return "", clientCredentialMissing
	}
	if len(xAPIKeyValues) != 1 {
		return "", clientCredentialMalformed
	}
	token := strings.TrimSpace(xAPIKeyValues[0])
	if token == "" {
		return "", clientCredentialMissing
	}
	if strings.ContainsAny(token, ", \t\r\n") {
		return "", clientCredentialMalformed
	}
	return token, clientCredentialPresent
}

// headerValuesFold collects all values for a header name without relying on
// canonical map keys. This keeps precedence and duplicate detection intact for
// requests assembled by middleware that wrote directly to http.Header.
func headerValuesFold(header http.Header, name string) ([]string, bool) {
	var values []string
	present := false
	for key, currentValues := range header {
		if !strings.EqualFold(key, name) {
			continue
		}
		present = true
		values = append(values, currentValues...)
	}
	return values, present
}

// AuthenticateClientRequest authenticates a non-inference public endpoint by
// using the exact same master-key/LiteLLM validation path as ProxyRequest.
func (p *Proxy) AuthenticateClientRequest(w http.ResponseWriter, r *http.Request) (*models.TokenInfo, bool) {
	tokenInfo, _, ok := p.AuthenticateClientRequestScoped(w, r)
	return tokenInfo, ok
}

// AuthenticateClientRequestScoped authenticates once and returns the exact
// scope derived from that same credential. This keeps /v1/models visibility
// identical for Authorization and x-api-key transports and avoids a second DB
// validation with a different header parser.
func (p *Proxy) AuthenticateClientRequestScoped(w http.ResponseWriter, r *http.Request) (*models.TokenInfo, scope.Context, bool) {
	if p == nil {
		WriteErrorServiceUnavailable(w, "Service unavailable")
		return nil, scope.PublicContext(), false
	}
	logCtx := &RequestLogContext{Request: r}
	if !p.authenticateRequest(w, r, logCtx, p.isLiteLLMHealthy()) {
		return nil, scope.PublicContext(), false
	}
	return logCtx.TokenInfo, logCtx.Scope, true
}

func (p *Proxy) authenticateRequest(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	isLiteLLMHealthy bool,
) bool {
	if trusted, ok := trustedClientAuthFromRequest(r); ok {
		logCtx.Token = trusted.rawToken
		logCtx.TokenInfo = trusted.tokenInfo
		logCtx.Scope = scopeContextFromTokenInfo(trusted.tokenInfo)
		return true
	}

	token, credentialState := extractClientToken(r)
	if credentialState == clientCredentialMissing {
		// Client-side error (bad request from the caller), not a service failure
		p.logger.WarnContext(r.Context(), "Missing Authorization header",
			"error_code", http.StatusUnauthorized, "path", r.URL.Path)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized
		logCtx.ErrorMsg = "Missing Authorization header"
		WriteErrorUnauthorized(w, "Missing Authorization header")
		return false
	}
	if credentialState == clientCredentialMalformed {
		p.logger.WarnContext(r.Context(), "Invalid Authorization header format",
			"error_code", http.StatusUnauthorized, "path", r.URL.Path)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized
		logCtx.ErrorMsg = "Invalid Authorization header format"
		WriteErrorUnauthorized(w, "Invalid Authorization header format")
		return false
	}
	logCtx.Token = token

	if p.isMasterKey(token) {
		logCtx.TokenInfo = &models.TokenInfo{
			Token:       auth.HashToken(p.masterKey),
			KeyName:     liteLLMMasterKeyIdentity,
			UserID:      liteLLMMasterKeyIdentity,
			IsMasterKey: true,
		}
		logCtx.Scope = scope.AdminContext()
		return true
	}

	if !isLiteLLMHealthy {
		p.logger.WarnContext(r.Context(), "Invalid master key",
			"error_code", http.StatusUnauthorized,
			"provided_key_prefix", security.MaskAPIKey(token))
		WriteErrorUnauthorized(w, "Invalid master key")
		return false
	}

	tokenInfo, err := p.LiteLLMDB.ValidateToken(r.Context(), token)
	logCtx.TokenInfo = tokenInfo
	if err == nil && tokenInfo == nil {
		// A successful validation without identity is never an authenticated
		// result. Fail closed if a manager implementation violates its contract.
		err = litellmdb.ErrTokenNotFound
	}
	if err != nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusUnauthorized

		if p.handleLiteLLMAuthError(r.Context(), w, err, token) {
			logCtx.ErrorMsg = "LiteLLM auth validation failed"
		} else {
			logCtx.ErrorMsg = "LiteLLM DB unavailable"
		}
		return false
	}
	p.logger.DebugContext(r.Context(), "Token validated via LiteLLM DB",
		"user_id", tokenInfo.UserID,
		"team_id", tokenInfo.TeamID,
	)
	logCtx.Scope = scopeContextFromTokenInfo(tokenInfo)
	return true
}

func (p *Proxy) readRequestBodyAndSelectModel(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
) ([]byte, string, string, bool, bool) {
	maxBodyBytes := int64(p.maxBodySizeMB) * 1024 * 1024
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil {
		// Client-side transport problem while sending the body
		p.logger.WarnContext(r.Context(), "Failed to read request body",
			"error_code", http.StatusBadRequest, "error", err)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Failed to read request body: " + err.Error()
		WriteErrorBadRequest(w, "Failed to read request body")
		return nil, "", "", false, false
	}
	if closeErr := r.Body.Close(); closeErr != nil {
		p.logger.WarnContext(r.Context(), "Failed to close request body", "error", closeErr)
	}
	if int64(len(body)) > maxBodyBytes {
		p.logger.WarnContext(r.Context(), "Request body exceeds max size",
			"error_code", http.StatusRequestEntityTooLarge,
			"max_body_size_mb", p.maxBodySizeMB,
			"actual_size_bytes", len(body),
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusRequestEntityTooLarge
		logCtx.ErrorMsg = "Request body too large"
		WriteErrorTooLarge(w, "Request Entity Too Large")
		return nil, "", "", false, false
	}

	modelID, streaming, sessionID, body := extractMetadataFromBody(body, r.Header.Get("Content-Type"))
	logCtx.PublicModelID = modelID
	logCtx.ModelID = modelID
	logCtx.SessionID = sessionID

	if modelID == "" {
		p.logger.WarnContext(r.Context(), "Model not specified in request body",
			"error_code", http.StatusBadRequest, "path", r.URL.Path)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = "Model not specified in request body"
		WriteErrorBadRequest(w, "model field is required")
		return nil, "", "", false, false
	}
	// An unrestricted virtual key must still stay inside the configured product
	// model surface. Provider backend IDs remain available to the trusted
	// LiteLLM -> AIR hop authenticated with AIR's master key, but ordinary keys
	// cannot discover or invoke them even when their DB model ACL is empty.
	trustedInternalModelID := p.isMasterKey(logCtx.Token)
	// Token model scopes contain client-visible model IDs. Enforce the scope
	// before route-surface lookup or alias rewrites. LiteLLM reports a restricted
	// key's unknown/disallowed model as an authorization failure; unrestricted
	// keys continue to the product-surface check and receive a not-found error.
	modelAllowed := logCtx.TokenInfo == nil || logCtx.TokenInfo.IsModelAllowed(modelID)
	if logCtx.TokenInfo != nil && p.modelManager != nil {
		modelAllowed = logCtx.TokenInfo.IsModelAllowedBy(modelID, p.modelManager.IsModelIDAllowedByScope)
	}
	if !modelAllowed {
		p.logger.WarnContext(r.Context(), "Model is not allowed for token",
			"error_code", http.StatusForbidden,
			"model", modelID,
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusForbidden
		logCtx.ErrorMsg = "Model not allowed"
		WriteErrorForbidden(w, "Model not allowed")
		return nil, "", "", false, false
	}
	if p.modelManager != nil && !trustedInternalModelID && !p.modelManager.IsClientModelIDRoutable(modelID) {
		p.logger.WarnContext(r.Context(), "Client model identifier is not exposed",
			"error_code", http.StatusNotFound,
			"model", modelID,
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusNotFound
		logCtx.ErrorMsg = fmt.Sprintf("Model %s not found", modelID)
		// Product-surface rejections happen before a provider attempt. Suppress
		// the deferred zero-spend failure row just like an unknown model.
		logCtx.Logged = true
		WriteErrorNotFound(w, logCtx.ErrorMsg)
		return nil, "", "", false, false
	}

	// Resolve additional client-visible names to one exact LiteLLM deployment
	// identity first. The requested name remains in PublicModelID while routing
	// continues through the canonical public model and provider-backend alias.
	// A trusted LiteLLM hop may submit an exact configured backend whose string
	// also exists in the client accepted-alias map. Preserve that exact internal
	// route; non-master callers and non-routable aliases still use the fail-closed
	// public resolver.
	trustedExactModelID := trustedInternalModelID && len(p.modelManager.GetCredentialsForModel(modelID)) > 0
	if trustedExactModelID {
		p.logger.DebugContext(r.Context(), "Preserved trusted internal model identifier", "model", modelID)
	} else if canonical, isPublicAlias, aliasErr := p.modelManager.ResolvePublicModelAlias(modelID); aliasErr != nil {
		p.logger.WarnContext(r.Context(), "Public model alias is not uniquely routable",
			"error_code", http.StatusNotFound,
			"model", modelID,
			"error", aliasErr,
		)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusNotFound
		logCtx.ErrorMsg = fmt.Sprintf("Model %s not found", modelID)
		logCtx.Logged = true
		WriteErrorNotFound(w, logCtx.ErrorMsg)
		return nil, "", "", false, false
	} else if isPublicAlias {
		p.logger.DebugContext(r.Context(), "Resolved public model alias", "alias", modelID, "canonical", canonical)
		body = openai.ReplaceModelInBody(body, modelID, canonical)
		modelID = canonical
		logCtx.ModelID = modelID
	}

	// Resolve model_alias entries (changes modelID to real name; credential lookup uses real name)
	if resolved, isAlias := p.modelManager.ResolveAlias(modelID); isAlias {
		p.logger.DebugContext(r.Context(), "Resolved model alias", "alias", modelID, "resolved", resolved)
		body = openai.ReplaceModelInBody(body, modelID, resolved)
		modelID = resolved
		logCtx.ModelID = modelID
	}

	// Resolve models[].model field: replace model in body for provider but keep alias as modelID
	// for rate limiting and credential lookup.
	realModelID := modelID
	if realName, hasReal := p.modelManager.GetRealModelName(modelID); hasReal {
		p.logger.DebugContext(r.Context(), "Resolved model real name", "alias", modelID, "real", realName)
		body = openai.ReplaceModelInBody(body, modelID, realName)
		realModelID = realName
	}

	return body, modelID, realModelID, streaming, true
}

func (p *Proxy) selectCredentialForModel(
	w http.ResponseWriter,
	modelID string,
	sessionID string,
	preferredCredentialName string,
	logCtx *RequestLogContext,
) (*config.CredentialConfig, bool) {
	if p.modelManager != nil && p.modelManager.IsEnabled() && len(p.modelManager.GetCredentialsForModel(modelID)) == 0 {
		errorMsg := fmt.Sprintf("Model %s not found", modelID)
		p.logger.WarnContext(logCtx.Context(), "Model is not configured",
			"error_code", http.StatusNotFound,
			"model", modelID,
			"request_id", logCtx.RequestID,
		)

		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusNotFound
		logCtx.ErrorMsg = errorMsg
		// Unknown models are rejected before a credential exists. Mark the
		// request handled so the deferred logger cannot emit a zero-spend row.
		logCtx.Logged = true

		WriteErrorNotFound(w, errorMsg)
		return nil, false
	}

	if preferredCredentialName != "" {
		cred, err := p.balancer.NextSpecificScoped(preferredCredentialName, modelID, logCtx.Scope)
		if err == nil {
			p.logger.DebugContext(logCtx.Context(), "Responses API sticky routing: using credential from previous_response_id",
				"credential", cred.Name,
				"model", modelID,
			)
			return cred, true
		}

		p.logger.DebugContext(logCtx.Context(), "Responses API sticky routing: previous_response credential unavailable, falling back to standard selection",
			"credential", preferredCredentialName,
			"model", modelID,
			"error", err,
		)
	}

	if sessionID != "" && p.sessionStore != nil {
		if credName, ok := p.sessionStore.Get(sessionID, modelID); ok {
			cred, err := p.balancer.NextSpecificScoped(credName, modelID, logCtx.Scope)
			if err == nil {
				p.logger.DebugContext(logCtx.Context(), "Session-sticky routing: using cached credential",
					"session_id", sessionID,
					"credential", cred.Name,
					"model", modelID,
				)
				return cred, true
			}

			p.logger.DebugContext(logCtx.Context(), "Session-sticky routing: cached credential unavailable, falling back to standard selection",
				"session_id", sessionID,
				"credential", credName,
				"model", modelID,
				"error", err,
			)
		}
	}

	cred, err := p.balancer.NextForModelScoped(modelID, logCtx.Scope)
	if err == nil {
		return cred, true
	}

	fallbackErr := error(nil)
	cred, fallbackErr = p.balancer.NextFallbackForModelScoped(modelID, logCtx.Scope)
	if fallbackErr == nil {
		return cred, true
	}

	errCode := http.StatusTooManyRequests
	var errorMsg string
	if errors.Is(err, balancer.ErrRateLimitExceeded) || errors.Is(fallbackErr, balancer.ErrRateLimitExceeded) {
		errorMsg = "Rate limit exceeded"
	} else {
		errorMsg = fmt.Sprintf("No credentials available for model %s", modelID)
	}

	p.logger.ErrorContext(logCtx.Context(), "No credentials available (regular and fallback)",
		"error_code", errCode,
		"model", modelID,
		"primary_error", err,
		"fallback_error", fallbackErr,
		"request_id", logCtx.RequestID,
	)

	logCtx.Status = "failure"
	logCtx.HTTPStatus = errCode
	logCtx.ErrorMsg = errorMsg
	logCtx.Credential = &config.CredentialConfig{
		Name: "system",
		Type: config.ProviderTypeProxy,
	}

	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue error log for no credentials",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
	logCtx.Logged = true
	WriteErrorRateLimit(w, errorMsg)
	return nil, false
}
