package proxy

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/spendlog"
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
			"response_body", logger.TruncateLongFields(body, 500),
		)
	}
	return append(args, "response_body", logger.TruncateLongFields(body, 500))
}

func shouldMaskUpstreamErrors(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	return isCometAPICredential(cred) || isSosanaCredential(cred) || promanutils.IsCredential(cred)
}

func isCometAPICredential(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	if cred.Type == config.ProviderTypeCometAPI {
		return true
	}
	name := strings.ToLower(cred.Name)
	return isProviderHost(cred.BaseURL, "cometapi.com") ||
		strings.Contains(name, "cometapi") ||
		strings.Contains(name, "comet-api")
}

func isProviderHost(rawBaseURL, domain string) bool {
	baseURL := strings.TrimSpace(rawBaseURL)
	domain = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(domain)), ".")
	if baseURL == "" || domain == "" {
		return false
	}
	u, err := url.Parse(baseURL)
	if err != nil || u.Hostname() == "" {
		u, err = url.Parse("https://" + baseURL)
		if err != nil {
			return false
		}
	}
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	return host == domain || strings.HasSuffix(host, "."+domain)
}

func isSosanaCredential(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	if cred.Type == config.ProviderTypeSosana {
		return true
	}
	name := strings.ToLower(cred.Name)
	return isProviderHost(cred.BaseURL, "sosana.art") ||
		containsSosanaMarker(name) ||
		containsSosanaMarker(cred.BaseURL)
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

// dbUnavailableRetryAfterSeconds is the Retry-After hint sent to clients when
// LiteLLM DB is unreachable during auth. Kept short and fail-fast: AIR never
// blocks the request waiting on the DB, it just tells the client when to try
// again. Independent of litellm_db.health_check_interval — a fixed floor is
// simpler to reason about than deriving it from the pool's reconnect backoff.
const dbUnavailableRetryAfterSeconds = "5"

// ==================== LiteLLM DB Integration ====================
// handleLiteLLMAuthError handles a LiteLLM DB auth error: it always writes a
// client response and updates logCtx (Status/HTTPStatus/ErrorMsg) to match,
// so the access log and the actual response sent never disagree.
func (p *Proxy) handleLiteLLMAuthError(ctx context.Context, w http.ResponseWriter, logCtx *RequestLogContext, err error, token string) {
	logCtx.Status = "failure"

	// Connection failure means AIR cannot confirm or deny the key - fail closed
	// with 503, not 401. A 401 here would tell a client with a perfectly valid
	// key that its credentials are bad; 503+Retry-After correctly signals a
	// transient dependency outage that well-behaved clients retry on.
	if errors.Is(err, litellmdb.ErrConnectionFailed) {
		logCtx.HTTPStatus = http.StatusServiceUnavailable
		logCtx.ErrorMsg = "LiteLLM DB unavailable"
		p.logger.ErrorContext(ctx, "LiteLLM DB unavailable during auth",
			"error_code", http.StatusServiceUnavailable,
			"token_prefix", security.MaskAPIKey(token))
		w.Header().Set("Retry-After", dbUnavailableRetryAfterSeconds)
		WriteErrorServiceUnavailable(w, "Authentication service temporarily unavailable, please retry")
		return
	}

	// Map error types to HTTP status and message
	errorMap := map[error]struct {
		status  int
		message string
		logMsg  string
	}{
		litellmdb.ErrTokenNotFound:  {http.StatusUnauthorized, "Invalid token", "Token not found"},
		litellmdb.ErrTokenBlocked:   {http.StatusForbidden, "Token blocked", "Token blocked"},
		litellmdb.ErrTeamBlocked:    {http.StatusForbidden, "Team blocked", "Team blocked"},
		litellmdb.ErrTokenExpired:   {http.StatusUnauthorized, "Token expired", "Token expired"},
		litellmdb.ErrBudgetExceeded: {http.StatusPaymentRequired, "Budget exceeded", "Budget exceeded"},
	}

	// Check for known auth errors — client-side issues (bad/blocked/expired token,
	// budget), not service failures, so they are logged at WARN.
	for errType, info := range errorMap {
		if errors.Is(err, errType) {
			logCtx.HTTPStatus = info.status
			logCtx.ErrorMsg = "LiteLLM auth validation failed"
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
			return
		}
	}

	// Unknown error — unexpected server-side failure, keep at ERROR
	logCtx.HTTPStatus = http.StatusInternalServerError
	logCtx.ErrorMsg = "LiteLLM auth validation failed"
	p.logger.ErrorContext(ctx, "Auth error",
		"error_code", http.StatusInternalServerError,
		"error", err,
		"token_prefix", security.MaskAPIKey(token))
	WriteErrorInternal(w, "Internal Server Error")
}

// litellmCallType translates an AIR request path into the LiteLLM call_type.
// The single source of truth for the route table is the spendlog package.
func litellmCallType(path string) string {
	return spendlog.LiteLLMCallTypeForPath(path)
}

// logSpendToLiteLLMDB logs request to LiteLLM_SpendLogs table and, if enabled,
// publishes an expanded copy of the same event to Kafka for ClickHouse
// analytics (internal/kafkalog). The two write-paths are independent: either
// can be enabled/disabled on its own (see litellm_db.disable_spend_logs_write
// and kafka.enabled). Pricing and Postgres write errors are returned; Kafka
// publish failures are logged but never affect the caller.
func (p *Proxy) logSpendToLiteLLMDB(logCtx *RequestLogContext) error {
	litellmEnabled := p.postgresSpendTrackingEnabled()
	kafkaEnabled := p.kafkaLog != nil && p.kafkaLog.IsEnabled()
	if !litellmEnabled && !kafkaEnabled {
		return nil
	}
	if logCtx == nil || logCtx.Credential == nil || logCtx.Request == nil {
		return nil
	}

	// Marks entry into the logging path itself, used below to compute the
	// Kafka event's overhead_ms as the cost of this function (cost lookup +
	// metadata/event building), distinct from duration_ms (full request
	// lifetime) — see logSpendToKafka call below.
	spendLogFnStart := time.Now()

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
	// billingTeamID captures the token's real team assignment before the
	// credential-name substitution below. That substitution exists only to
	// attribute spend logs/Kafka/DailyTeamSpend by provider credential on
	// billing-aggregator instances; it must never make aggregateSpendUpdates
	// think a personal key belongs to a real, billable team.
	billingTeamID := teamID
	if teamID == "" && p.credentialNameAsTeamID {
		teamID = credName
	}

	// LiteLLM's end_user is the caller-supplied end-user identifier ("user" in
	// the request body, X-End-User for AIR). The key owner's email must NOT be
	// used as a fallback: LiteLLM leaves end_user empty for such traffic, and
	// substituting the email would fabricate EndUserTable/DailyEndUserSpend
	// rows that have no counterpart in the primary accounting.
	endUser := extractEndUser(logCtx.Request)

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
	if logCtx.IsImageGeneration && status == "success" && logCtx.TokenUsage.ImageCount <= 0 {
		logCtx.TokenUsage.ImageCount = logCtx.ImageCount
		if logCtx.TokenUsage.ImageCount <= 0 {
			logCtx.TokenUsage.ImageCount = 1
		}
	}
	logCtx.applyWebSearchUsageDefaults(status)
	logCtx.TokenUsage.Normalize()

	logSpendCtx := logCtx.Context()
	modelPrice := logCtx.ModelPrice
	priceModelID := logCtx.PriceModelID
	if modelPrice == nil {
		priceModelID, modelPrice = p.resolveBillingPrice(logCtx, logCtx.PublicModelID, logCtx.ModelID, logCtx.RealModelID)
	}
	var tokenCosts *converter.TokenCosts
	if modelPrice == nil {
		// No price and nothing was ever billable (e.g. rejected before any
		// provider was contacted, such as the "no credentials available" 429
		// path in selectCredentialForModel) — write a $0 audit row instead of
		// dropping it silently. Only fail closed when real usage exists that
		// we can't price, which is the actual billing-integrity risk.
		if !logCtx.TokenUsage.IsZero() {
			return fmt.Errorf("model price unavailable for %q", priceModelID)
		}
		tokenCosts = &converter.TokenCosts{}
	} else {
		tokenCosts = modelPrice.CalculateCosts(logCtx.TokenUsage)
		if tokenCosts == nil {
			return fmt.Errorf("cost calculation failed for %q", priceModelID)
		}
	}
	cost := tokenCosts.TotalCost
	p.logger.DebugContext(logSpendCtx, "Calculated cost for model",
		"model_name", priceModelID,
		"cost", cost,
		"prompt_tokens", logCtx.TokenUsage.PromptTokens,
		"completion_tokens", logCtx.TokenUsage.CompletionTokens)

	// Settle any Redis budget reservation / TPM usage against the real cost.
	// Guarded so it runs exactly once even though a defer safety-net may also call it.
	p.reconcileBudgetAndRateLimits(logCtx, cost)

	customLLMProvider := strings.Replace(string(logCtx.Credential.Type), "-", "_", 1)
	if customLLMProvider == "proxy" {
		customLLMProvider = string(config.ProviderTypeOpenAI)
	}

	endTime := utils.NowUTC()

	// Publish to Kafka *before* building/inserting the Postgres row (not
	// after) so a queue-full failure can be flagged in the same row's
	// metadata at insert time, instead of needing a separate update later —
	// see buildMetadata's kafkaFallbackReason parameter below.
	var kafkaFallbackReason string
	if kafkaEnabled && !logCtx.IsProxyRequest {
		// Deliberately distinct from the Postgres metadata's overheadMs (which
		// measures full elapsed time since the request started, near-identical
		// to duration_ms): kafkaOverheadMs measures only the cost of this
		// logging function itself, matching the ТЗ's intent for overhead_ms to
		// be a small, separate figure from duration_ms.
		kafkaOverheadMs := float64(time.Since(spendLogFnStart).Microseconds()) / 1000.0
		if err := p.logSpendToKafka(logCtx, credName, modelIDFormatted, hashedToken,
			userID, teamID, organizationID, endUser, apiBase, status,
			cost, tokenCosts, kafkaOverheadMs, endTime); err != nil {
			if errors.Is(err, kafkalog.ErrQueueFull) {
				kafkaFallbackReason = "queue_full"
			} else {
				kafkaFallbackReason = "publish_error"
			}
		}
	}

	// Build metadata with usage, cost breakdown, requester IP, and optional error
	requesterIP := getClientIP(logCtx.Request)
	overheadMs := float64(time.Since(logCtx.StartTime).Microseconds()) / 1000.0
	metadata := buildMetadata(hashedToken, logCtx.TokenInfo, logCtx.ErrorMsg, logCtx.HTTPStatus, logCtx.TokenUsage, requesterIP, tokenCosts, logCtx.ModelID, overheadMs, kafkaFallbackReason)
	metadata = addAIRSpendMetadata(metadata, logCtx.RequestID, logCtx.ClientResponseID, logCtx.IsProxyRequest, logCtx.ModelID, logCtx.PublicModelID)

	var completionStartTime *time.Time
	if !logCtx.CompletionStartTime.IsZero() {
		completionStartTime = &logCtx.CompletionStartTime
	}

	var pgErr error
	if litellmEnabled {
		pgErr = p.LiteLLMDB.LogSpend(&litellmdb.SpendLogEntry{
			RequestID:           logCtx.spendRequestID(),
			AirEventID:          logCtx.RequestID,
			StartTime:           logCtx.StartTime,
			EndTime:             endTime,
			CompletionStartTime: completionStartTime,
			CallType:            litellmCallType(logCtx.Request.URL.Path),
			APIBase:             apiBase,
			Model:               logCtx.ModelID,    // Model name
			ModelID:             modelIDFormatted,  // credential.name:model_name
			ModelGroup:          logCtx.ModelID,    // Model name
			CustomLLMProvider:   customLLMProvider, // Provider type as string
			PromptTokens:        logCtx.TokenUsage.PromptTokens,
			CompletionTokens:    logCtx.TokenUsage.CompletionTokens,
			TotalTokens:         logCtx.TokenUsage.Total(),
			Metadata:            metadata,
			Spend:               cost, // Calculated cost based on model pricing and token usage
			APIKey:              hashedToken,
			UserID:              userID,
			TeamID:              teamID,
			BillingTeamID:       billingTeamID,
			OrganizationID:      organizationID,
			EndUser:             endUser,
			RequesterIP:         requesterIP,
			Status:              status,
			SessionID:           logCtx.SessionID,
			SkipAccounting:      logCtx.IsProxyRequest,
		})
	}

	return pgErr
}
