package proxy

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/logger"
	"github.com/mixaill76/auto_ai_router/internal/security"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// ==================== Unified error logging ====================

// logUpstreamError emits the unified ERROR record for a failed upstream request.
// Every final provider failure must go through this helper so that a single ERROR
// line contains everything needed for debugging:
// error_code, credential (+provider), model, response_body.
// extra accepts additional context pairs (request_id, url, error, ...).
func (p *Proxy) logUpstreamError(ctx context.Context, msg string, errorCode int, cred *config.CredentialConfig, modelID string, responseBody []byte, extra ...any) {
	args := make([]any, 0, 10+len(extra))
	args = append(args, "error_code", errorCode)
	if cred != nil {
		args = append(args, "credential", cred.Name, "provider", string(cred.Type))
	}
	args = append(args, "model", modelID)
	if len(responseBody) > 0 {
		args = appendResponseBodyForLogs(args, cred, string(responseBody))
	}
	args = append(args, extra...)
	p.logger.ErrorContext(ctx, msg, args...)
}

func appendResponseBodyForLogs(args []any, cred *config.CredentialConfig, body string) []any {
	if shouldMaskUpstreamErrors(cred) {
		return append(args,
			"response_body_masked", true,
			"response_body", body,
		)
	}
	return append(args, "response_body", logger.TruncateLongFields(body, 500))
}

func shouldMaskUpstreamErrors(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	return cred.MaskUpstreamErrors || isCometAPICredential(cred) || isSosanaCredential(cred)
}

func isCometAPICredential(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	if cred.Type == config.ProviderTypeCometAPI {
		return true
	}
	name := strings.ToLower(cred.Name)
	return isCometAPIHost(cred.BaseURL) ||
		strings.Contains(name, "cometapi") ||
		strings.Contains(name, "comet-api")
}

func isCometAPIHost(rawBaseURL string) bool {
	host := normalizedHost(rawBaseURL)
	return host == "cometapi.com" || strings.HasSuffix(host, ".cometapi.com")
}

func normalizedHost(rawBaseURL string) string {
	baseURL := strings.TrimSpace(rawBaseURL)
	if baseURL == "" {
		return ""
	}
	u, err := url.Parse(baseURL)
	if err != nil || u.Hostname() == "" {
		u, err = url.Parse("https://" + baseURL)
		if err != nil {
			return ""
		}
	}
	return strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
}

func isSosanaCredential(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	if cred.Type == config.ProviderTypeSosana {
		return true
	}
	name := strings.ToLower(cred.Name)
	host := normalizedHost(cred.BaseURL)
	return isSosanaHost(cred.BaseURL) ||
		containsSosanaMarker(name) ||
		containsSosanaMarker(host)
}

func isSosanaHost(rawBaseURL string) bool {
	host := normalizedHost(rawBaseURL)
	return host == "sosana.art" || strings.HasSuffix(host, ".sosana.art")
}

// logStreamHandlerError logs a streaming handler failure. Client disconnects are
// expected during normal operation and go to DEBUG; real failures go to ERROR.
func (p *Proxy) logStreamHandlerError(ctx context.Context, msg string, err error, extra ...any) {
	args := append([]any{"error", err}, extra...)
	if isClientDisconnectError(err) {
		p.logger.DebugContext(ctx, msg+" (client disconnected)", args...)
		return
	}
	p.logger.ErrorContext(ctx, msg, args...)
}

// logTransformedResponse logs a transformed response at debug level
func (p *Proxy) logTransformedResponse(ctx context.Context, credName, providerName string, body []byte) {
	if p.logger.Enabled(ctx, slog.LevelDebug) {
		p.logger.DebugContext(ctx, "Transformed response to OpenAI format",
			"credential", credName,
			"provider", providerName,
			"body", logger.TruncateLongFields(string(body), 500),
		)
	}
}

// ==================== LiteLLM DB Integration ====================
// handleLiteLLMAuthError handles LiteLLM authentication errors
// Returns true if error was handled and response was written
func (p *Proxy) handleLiteLLMAuthError(ctx context.Context, w http.ResponseWriter, err error, token string) bool {
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

	// Check for known auth errors — client-side issues (bad/blocked/expired token,
	// budget), not service failures, so they are logged at WARN.
	for errType, info := range errorMap {
		if errors.Is(err, errType) {
			p.logger.WarnContext(ctx, info.logMsg,
				"error_code", info.status,
				"token_prefix", security.MaskAPIKey(token))
			switch info.status {
			case http.StatusForbidden:
				WriteErrorForbidden(w, info.message)
			case http.StatusPaymentRequired:
				WriteErrorPaymentRequired(w, info.message)
			default:
				WriteErrorUnauthorized(w, info.message)
			}
			return true
		}
	}

	// Unknown error — unexpected server-side failure, keep at ERROR
	p.logger.ErrorContext(ctx, "Auth error",
		"error_code", http.StatusInternalServerError,
		"error", err,
		"token_prefix", security.MaskAPIKey(token))
	WriteErrorInternal(w, "Internal Server Error")
	return true
}

// logSpendToLiteLLMDB logs request to LiteLLM_SpendLogs table
// Returns error if the log entry cannot be queued (e.g., queue full)
func (p *Proxy) logSpendToLiteLLMDB(logCtx *RequestLogContext) error {
	if p.LiteLLMDB == nil || !p.LiteLLMDB.IsEnabled() {
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
	// For proxy credentials, use the actual upstream credential name if available
	credName := logCtx.Credential.Name
	if logCtx.ActualCredentialName != "" {
		credName = logCtx.ActualCredentialName
	}
	modelIDFormatted := credName + ":" + logCtx.ModelID
	hashedToken := litellmdb.HashToken(logCtx.Token)

	// Extract user info from tokenInfo (or use empty strings as fallback)
	var userID, teamID, organizationID string
	if logCtx.TokenInfo != nil {
		userID = logCtx.TokenInfo.UserID
		teamID = logCtx.TokenInfo.TeamID
		organizationID = logCtx.TokenInfo.OrganizationID
	}

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

	// Calculate cost based on model pricing and token usage.
	// Try real model name first (from models[].model), then alias name.
	var cost float64
	var tokenCosts *converter.TokenCosts
	logSpendCtx := logCtx.Context()
	if p.priceRegistry == nil {
		p.logger.WarnContext(logSpendCtx, "Price registry not available, using 0 cost for spend log")
	} else {
		priceModelID := logCtx.ModelID
		if logCtx.RealModelID != "" && logCtx.RealModelID != logCtx.ModelID {
			priceModelID = logCtx.RealModelID
		}
		modelPrice := p.priceRegistry.GetPrice(priceModelID)
		if modelPrice == nil && priceModelID != logCtx.ModelID {
			modelPrice = p.priceRegistry.GetPrice(logCtx.ModelID)
		}
		if modelPrice == nil {
			p.logger.WarnContext(logSpendCtx, "Model price not found in registry, using 0 cost",
				"model_name", priceModelID)
		} else {
			tokenCosts = modelPrice.CalculateCosts(logCtx.TokenUsage)
			if tokenCosts != nil {
				cost = tokenCosts.TotalCost
			}
			p.logger.DebugContext(logSpendCtx, "Calculated cost for model",
				"model_name", priceModelID,
				"cost", cost,
				"prompt_tokens", logCtx.TokenUsage.PromptTokens,
				"completion_tokens", logCtx.TokenUsage.CompletionTokens)
		}
	}

	// Build metadata with usage, cost breakdown, requester IP, and optional error
	requesterIP := getClientIP(logCtx.Request)
	overheadMs := float64(time.Since(logCtx.StartTime).Microseconds()) / 1000.0
	metadata := buildMetadata(hashedToken, logCtx.TokenInfo, logCtx.ErrorMsg, logCtx.HTTPStatus, logCtx.TokenUsage, requesterIP, tokenCosts, logCtx.ModelID, overheadMs)

	customLLMProvider := strings.Replace(string(logCtx.Credential.Type), "-", "_", 1)
	if customLLMProvider == "proxy" {
		customLLMProvider = string(config.ProviderTypeOpenAI)
	}

	if teamID == "" {
		teamID = credName
	}

	return p.LiteLLMDB.LogSpend(&litellmdb.SpendLogEntry{
		RequestID:         logCtx.RequestID,
		StartTime:         logCtx.StartTime,
		EndTime:           utils.NowUTC(),
		CallType:          logCtx.Request.URL.Path,
		APIBase:           apiBase,
		Model:             logCtx.ModelID,    // Model name
		ModelID:           modelIDFormatted,  // credential.name:model_name
		ModelGroup:        logCtx.ModelID,    // Model name
		CustomLLMProvider: customLLMProvider, // Provider type as string
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
