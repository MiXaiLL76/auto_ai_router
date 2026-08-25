package proxy

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

type reservedEntity struct {
	key    string
	amount float64
}

type budgetLevel struct {
	entity    string
	maxBudget *float64
	dbSpend   float64
	rpm       *int64
	tpm       *int64
}

func entityKind(entity string) string {
	if kind, _, ok := strings.Cut(entity, ":"); ok {
		return kind
	}
	return "key"
}

func dereferenceFloat(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func limitOrUnlimited(value *int64) int {
	if value == nil {
		return -1
	}
	return int(*value)
}

// budgetLevels mirrors LiteLLM's independently enforced token, personal-user,
// team, organization, and membership hierarchy.
func budgetLevels(info *dbmodels.TokenInfo) []budgetLevel {
	if info == nil {
		return nil
	}
	levels := []budgetLevel{{
		entity:    "token:" + info.Token,
		maxBudget: info.MaxBudget,
		dbSpend:   info.Spend,
		rpm:       info.RPMLimit,
		tpm:       info.TPMLimit,
	}}
	if info.TeamID == "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "user:" + info.UserID,
			maxBudget: info.UserMaxBudget,
			dbSpend:   dereferenceFloat(info.UserSpend),
			rpm:       info.UserRPMLimit,
			tpm:       info.UserTPMLimit,
		})
	}
	if info.TeamID != "" {
		levels = append(levels, budgetLevel{
			entity:    "team:" + info.TeamID,
			maxBudget: info.TeamMaxBudget,
			dbSpend:   dereferenceFloat(info.TeamSpend),
			rpm:       info.TeamRPMLimit,
			tpm:       info.TeamTPMLimit,
		})
	}
	if info.OrganizationID != "" {
		levels = append(levels, budgetLevel{
			entity:    "org:" + info.OrganizationID,
			maxBudget: info.OrgMaxBudget,
			dbSpend:   dereferenceFloat(info.OrgSpend),
			rpm:       info.OrgRPMLimit,
			tpm:       info.OrgTPMLimit,
		})
	}
	if info.TeamID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "teammember:" + info.TeamID + ":" + info.UserID,
			maxBudget: info.TeamMemberMaxBudget,
			dbSpend:   dereferenceFloat(info.TeamMemberSpend),
			rpm:       info.TeamMemberRPMLimit,
			tpm:       info.TeamMemberTPMLimit,
		})
	}
	if info.OrganizationID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "orgmember:" + info.OrganizationID + ":" + info.UserID,
			maxBudget: info.OrgMemberMaxBudget,
			dbSpend:   dereferenceFloat(info.OrgMemberSpend),
			rpm:       info.OrgMemberRPMLimit,
			tpm:       info.OrgMemberTPMLimit,
		})
	}
	return levels
}

func (p *Proxy) estimateCompletionTokens(body []byte) int {
	fallback := p.defaultEstimatedCompletionTokens
	if fallback <= 0 {
		fallback = 1000
	}
	raw, ok := decodeRequestBody(body)
	if !ok {
		return fallback
	}
	for _, field := range []string{"max_completion_tokens", "max_tokens", "max_output_tokens"} {
		value, exists := raw[field]
		if !exists {
			continue
		}
		if number, valid := value.(float64); valid && number > 0 {
			return int(number)
		}
	}
	return fallback
}

func (p *Proxy) estimateRequestCost(logCtx *RequestLogContext, publicModelID, modelID, realModelID string, body []byte) (float64, bool) {
	if p.priceRegistry == nil {
		return 0, false
	}
	// Mirror final billing: reserve against the client-facing model price first,
	// then fall back to the provider-facing real model if the alias is unpriced.
	// Cached via logCtx so logSpendToLiteLLMDB reuses this exact resolution.
	_, modelPrice := p.resolveBillingPrice(logCtx, publicModelID, modelID, realModelID)
	if modelPrice == nil {
		return 0, false
	}
	// PromptTokens is skipped (not the whole estimate) when tiktoken_enabled
	// is off — this only undercounts the prompt-token portion of the budget
	// reservation, which is the documented trade-off of disabling local
	// tokenization; it must not also disable reservation entirely (costKnown
	// stays true — see estimateCompletionTokens below, which never touches
	// the tokenizer and so needs no such gate).
	var promptTokens int
	if p.tiktokenEnabled {
		promptTokens = estimatePromptTokensForModel(body, realModelID)
	}
	usage := &converter.TokenUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: p.estimateCompletionTokens(body),
	}
	return modelPrice.CalculateCost(usage), true
}

func (p *Proxy) spendTrackingEnabled() bool {
	return p.postgresSpendTrackingEnabled() ||
		p.kafkaLog != nil && p.kafkaLog.IsEnabled()
}

func (p *Proxy) postgresSpendTrackingEnabled() bool {
	if p.LiteLLMDB == nil || !p.LiteLLMDB.IsEnabled() {
		return false
	}
	manager, ok := p.LiteLLMDB.(interface{ SpendLoggingEnabled() bool })
	if !ok {
		return true
	}
	return manager.SpendLoggingEnabled()
}

// checkModelPriceAvailable resolves and caches the billing price for modelID
// on logCtx and, when spend tracking is enabled and no price exists, fails
// the request closed with a 503 instead of letting it reach a provider.
//
// This must be called from every path that can actually forward a request to
// a provider or a fallback proxy — not just from enforceBudgetAndRateLimits —
// because applyCredentialCompatibilityRouting's TryFallbackProxy branch can
// forward and return before enforceBudgetAndRateLimits ever runs. See PR
// #96/#115 follow-up review TODO 1.
func (p *Proxy) checkModelPriceAvailable(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	modelID string,
	realModelID string,
) bool {
	publicModelID := modelID
	if logCtx != nil && logCtx.PublicModelID != "" {
		publicModelID = logCtx.PublicModelID
	}
	priceModelID, modelPrice := lookupBillingModelPrice(
		p.priceRegistry, billingInstant(logCtx), publicModelID, modelID, realModelID,
	)
	if logCtx != nil {
		priceModelID, modelPrice = p.resolveBillingPrice(logCtx, publicModelID, modelID, realModelID)
	}
	if modelPrice == nil && p.spendTrackingEnabled() {
		requestID := ""
		if logCtx != nil {
			requestID = logCtx.RequestID
		}
		p.logger.ErrorContext(r.Context(), "Model price unavailable; request rejected",
			"error_code", http.StatusServiceUnavailable,
			"model", modelID,
			"real_model", realModelID,
			"request_id", requestID)
		if logCtx != nil {
			logCtx.Status = "failure"
			logCtx.HTTPStatus = http.StatusServiceUnavailable
			logCtx.ErrorMsg = "model pricing unavailable"
		}
		WriteErrorServiceUnavailable(w, "Model pricing unavailable")
		return false
	}
	if logCtx != nil {
		logCtx.ModelPrice = modelPrice
		logCtx.PriceModelID = priceModelID
	}
	return true
}

func (p *Proxy) enforceBudgetAndRateLimits(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	modelID string,
	realModelID string,
	body []byte,
) bool {
	if !p.checkModelPriceAvailable(w, r, logCtx, modelID, realModelID) {
		return false
	}
	if logCtx == nil || logCtx.Scope.Admin || logCtx.TokenInfo == nil {
		return true
	}
	budgetEnabled := p.budgetReservationEnabled && p.budgetReserver != nil
	rateEnabled := p.keyRateLimitsEnabled && p.keyRateLimiter != nil
	if !budgetEnabled && !rateEnabled {
		return true
	}

	levels := budgetLevels(logCtx.TokenInfo)
	estimatedCost, costKnown := p.estimateRequestCost(logCtx, logCtx.PublicModelID, modelID, realModelID, body)
	if budgetEnabled && !costKnown {
		p.logger.DebugContext(r.Context(), "Budget reservation skipped: model price unavailable", "model", modelID)
	}

	if budgetEnabled && costKnown {
		for _, level := range levels {
			if level.maxBudget == nil {
				continue
			}
			allowed, err := p.budgetReserver.TryReserve(
				r.Context(), level.entity, level.dbSpend, estimatedCost, *level.maxBudget,
			)
			if err != nil {
				// Preserve the existing PostgreSQL auth check as the fallback when
				// Redis is unavailable; availability wins over the race hardening.
				p.logger.WarnContext(r.Context(), "Budget reservation unavailable; request allowed",
					"entity_kind", entityKind(level.entity), "error", err)
				continue
			}
			if !allowed {
				p.releaseBudgetReservations(r.Context(), logCtx)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusPaymentRequired
				logCtx.ErrorMsg = "budget exceeded (" + level.entity + ")"
				WriteErrorPaymentRequired(w, "Budget exceeded")
				return false
			}
			logCtx.reservedEntities = append(logCtx.reservedEntities, reservedEntity{
				key: level.entity, amount: estimatedCost,
			})
		}
	}

	if rateEnabled {
		for _, level := range levels {
			if level.rpm == nil && level.tpm == nil {
				continue
			}
			rpm := limitOrUnlimited(level.rpm)
			tpm := limitOrUnlimited(level.tpm)
			p.keyRateLimiter.AddCredentialWithTPM(level.entity, rpm, tpm)
			if !p.keyRateLimiter.TryAllowAllCtx(r.Context(), level.entity, "") {
				p.releaseBudgetReservations(r.Context(), logCtx)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusTooManyRequests
				logCtx.ErrorMsg = "rate limit exceeded (" + level.entity + ")"
				WriteErrorRateLimit(w, "Rate limit exceeded for "+entityKind(level.entity))
				return false
			}
			if tpm >= 0 {
				logCtx.rateLimitedTPMEntities = append(logCtx.rateLimitedTPMEntities, level.entity)
			}
		}
	}
	return true
}

func (p *Proxy) releaseBudgetReservations(ctx context.Context, logCtx *RequestLogContext) {
	if p.budgetReserver == nil || logCtx == nil {
		return
	}
	for _, entity := range logCtx.reservedEntities {
		if err := p.budgetReserver.Reconcile(ctx, entity.key, -entity.amount); err != nil {
			p.logger.WarnContext(ctx, "Failed to release budget reservation",
				"entity_kind", entityKind(entity.key), "error", err)
		}
	}
	logCtx.reservedEntities = nil
}

func (p *Proxy) reconcileBudgetAndRateLimits(logCtx *RequestLogContext, actualCost float64) {
	if logCtx == nil || logCtx.budgetReconciled {
		return
	}
	logCtx.budgetReconciled = true
	if len(logCtx.reservedEntities) == 0 && len(logCtx.rateLimitedTPMEntities) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if p.budgetReserver != nil {
		for _, entity := range logCtx.reservedEntities {
			if err := p.budgetReserver.Reconcile(ctx, entity.key, actualCost-entity.amount); err != nil {
				p.logger.WarnContext(ctx, "Failed to reconcile budget reservation",
					"entity_kind", entityKind(entity.key), "error", err)
			}
		}
	}
	if p.keyRateLimiter != nil && logCtx.TokenUsage != nil {
		tokens := logCtx.TokenUsage.Total()
		for _, entity := range logCtx.rateLimitedTPMEntities {
			p.keyRateLimiter.ConsumeTokensCtx(ctx, entity, tokens)
		}
	}
}
