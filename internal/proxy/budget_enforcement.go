package proxy

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// entityKind extracts the hierarchy level ("token", "team", "org", ...) from an
// entity key ("team:<team_id>"), dropping the identifier so it is safe to
// include in a client-facing error message.
func entityKind(entity string) string {
	if kind, _, ok := strings.Cut(entity, ":"); ok {
		return kind
	}
	return "key"
}

// reservedEntity is one budget-hierarchy level for which a reservation was made
// within a single request.
type reservedEntity struct {
	key            string  // e.g. "token:<hash>"
	reservedAmount float64 // amount passed to TryReserve, needed for Reconcile's delta
}

// budgetLevel is one hierarchy level to check (token/user/team/org/members).
type budgetLevel struct {
	entity    string
	maxBudget *float64
	dbSpend   float64
	rpm       *int64
	tpm       *int64
}

func derefOr0(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func limitOrMinus1(v *int64) int {
	if v == nil {
		return -1
	}
	return int(*v)
}

// budgetLevels builds the list of hierarchy levels to enforce from TokenInfo,
// mirroring the semantics of TokenInfo.Validate's per-level checks.
func budgetLevels(info *dbmodels.TokenInfo) []budgetLevel {
	levels := make([]budgetLevel, 0, 6)

	// Token level — always present.
	levels = append(levels, budgetLevel{
		entity:    "token:" + info.Token,
		maxBudget: info.MaxBudget,
		dbSpend:   info.Spend,
		rpm:       info.RPMLimit,
		tpm:       info.TPMLimit,
	})

	// User level — personal keys only (no team), matching checkUserBudget.
	if info.TeamID == "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "user:" + info.UserID,
			maxBudget: info.UserMaxBudget,
			dbSpend:   derefOr0(info.UserSpend),
			rpm:       info.UserRPMLimit,
			tpm:       info.UserTPMLimit,
		})
	}

	if info.TeamID != "" {
		levels = append(levels, budgetLevel{
			entity:    "team:" + info.TeamID,
			maxBudget: info.TeamMaxBudget,
			dbSpend:   derefOr0(info.TeamSpend),
			rpm:       info.TeamRPMLimit,
			tpm:       info.TeamTPMLimit,
		})
	}

	if info.OrganizationID != "" {
		levels = append(levels, budgetLevel{
			entity:    "org:" + info.OrganizationID,
			maxBudget: info.OrgMaxBudget,
			dbSpend:   derefOr0(info.OrgSpend),
			rpm:       info.OrgRPMLimit,
			tpm:       info.OrgTPMLimit,
		})
	}

	if info.TeamID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "teammember:" + info.TeamID + ":" + info.UserID,
			maxBudget: info.TeamMemberMaxBudget,
			dbSpend:   derefOr0(info.TeamMemberSpend),
			rpm:       info.TeamMemberRPMLimit,
			tpm:       info.TeamMemberTPMLimit,
		})
	}

	if info.OrganizationID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "orgmember:" + info.OrganizationID + ":" + info.UserID,
			maxBudget: info.OrgMemberMaxBudget,
			dbSpend:   derefOr0(info.OrgMemberSpend),
			rpm:       info.OrgMemberRPMLimit,
			tpm:       info.OrgMemberTPMLimit,
		})
	}

	return levels
}

// estimateRequestCost estimates the max cost of a request for budget reservation.
// Returns ok=false when no price is known for the model (reservation is skipped).
func (p *Proxy) estimateRequestCost(modelID, realModelID string, body []byte) (float64, bool) {
	if p.priceRegistry == nil {
		return 0, false
	}
	// Prices are typically registered under the real provider model name, not the
	// alias — mirror the lookup order in logSpendToLiteLLMDB (proxy_log.go) so
	// aliased models (model_alias in config.yaml) don't silently skip reservation.
	modelPrice := p.priceRegistry.GetPrice(realModelID)
	if modelPrice == nil && realModelID != modelID {
		modelPrice = p.priceRegistry.GetPrice(modelID)
	}
	if modelPrice == nil {
		return 0, false
	}

	promptTokens := estimatePromptTokensForModel(body, modelID)
	completionTokens := p.estimateCompletionTokens(body)

	cost := modelPrice.CalculateCost(&converter.TokenUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	})
	return cost, true
}

// estimateCompletionTokens reads max_tokens/max_completion_tokens from the body,
// falling back to the configured default.
func (p *Proxy) estimateCompletionTokens(body []byte) int {
	def := 1000
	if p.defaultEstimatedCompletionTokens > 0 {
		def = p.defaultEstimatedCompletionTokens
	}
	raw, ok := decodeRequestBody(body)
	if !ok {
		return def
	}
	for _, key := range []string{"max_completion_tokens", "max_tokens", "max_output_tokens"} {
		if v, exists := raw[key]; exists {
			if n, ok := v.(float64); ok && n > 0 {
				return int(n)
			}
		}
	}
	return def
}

// enforceBudgetAndRateLimits runs AFTER modelID is known and TokenInfo is populated.
// It is a no-op (always allowed) when the reserver and rate limiter are both nil
// or the feature toggle is off, or when the request is authenticated via master
// key (admin scope has no budget/rate limits). Returns false — and has already
// written an HTTP error — when a limit is exceeded.
func (p *Proxy) enforceBudgetAndRateLimits(w http.ResponseWriter, r *http.Request, logCtx *RequestLogContext, modelID, realModelID string, body []byte) bool {
	if logCtx.Scope.Admin {
		return true
	}
	if !p.budgetReservationEnabled || (p.budgetReserver == nil && p.keyRateLimiter == nil) {
		return true
	}
	if logCtx.TokenInfo == nil {
		return true
	}

	ctx := r.Context()
	levels := budgetLevels(logCtx.TokenInfo)

	estimatedCost, costKnown := p.estimateRequestCost(modelID, realModelID, body)
	if !costKnown {
		p.logger.DebugContext(ctx, "Budget reservation skipped: no price for model",
			"model", modelID)
	}

	// Budget reservation pass.
	if p.budgetReserver != nil && costKnown {
		for _, lvl := range levels {
			if lvl.maxBudget == nil {
				continue // unlimited — don't create a Redis key
			}
			allowed, err := p.budgetReserver.TryReserve(ctx, lvl.entity, lvl.dbSpend, estimatedCost, *lvl.maxBudget)
			if err != nil {
				// Redis infra error — fail open (DB snapshot check remains the safety net).
				p.logger.WarnContext(ctx, "Budget reservation error, allowing request",
					"entity", lvl.entity, "error", err)
				continue
			}
			if !allowed {
				p.releaseReservations(ctx, logCtx)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusPaymentRequired
				logCtx.ErrorMsg = "budget exceeded (" + lvl.entity + ")"
				WriteErrorPaymentRequired(w, "Budget exceeded")
				return false
			}
			logCtx.ReservedEntities = append(logCtx.ReservedEntities, reservedEntity{
				key:            lvl.entity,
				reservedAmount: estimatedCost,
			})
		}
	}

	// RPM/TPM enforcement pass.
	if p.keyRateLimiter != nil {
		for _, lvl := range levels {
			if lvl.rpm == nil && lvl.tpm == nil {
				continue
			}
			rpm := limitOrMinus1(lvl.rpm)
			tpm := limitOrMinus1(lvl.tpm)
			p.keyRateLimiter.AddCredentialWithTPM(lvl.entity, rpm, tpm)
			if !p.keyRateLimiter.TryAllowAllCtx(ctx, lvl.entity, "") {
				p.releaseReservations(ctx, logCtx)
				logCtx.Status = "failure"
				logCtx.HTTPStatus = http.StatusTooManyRequests
				// Keep the exact entity (may contain internal team/org/user IDs or a
				// token hash) only in server-side logs/metadata — never in the
				// client-facing body, to avoid leaking internal identifiers.
				logCtx.ErrorMsg = "rate limit exceeded (" + lvl.entity + ")"
				WriteErrorRateLimit(w, "Rate limit exceeded for "+entityKind(lvl.entity))
				return false
			}
			if tpm != -1 {
				logCtx.RateLimitedEntitiesForTPM = append(logCtx.RateLimitedEntitiesForTPM, lvl.entity)
			}
		}
	}

	return true
}

// releaseReservations rolls back all budget reservations already made for this
// request (used when a later level rejects the request).
func (p *Proxy) releaseReservations(ctx context.Context, logCtx *RequestLogContext) {
	if p.budgetReserver == nil {
		return
	}
	for _, e := range logCtx.ReservedEntities {
		if err := p.budgetReserver.Reconcile(ctx, e.key, -e.reservedAmount); err != nil {
			p.logger.WarnContext(ctx, "Failed to release budget reservation", "entity", e.key, "error", err)
		}
	}
	logCtx.ReservedEntities = nil
}

// reconcileBudgetAndRateLimits must be called exactly once per request that went
// through enforceBudgetAndRateLimits (guarded by logCtx.budgetReconciled), settling
// each reservation to the real cost and recording actual token usage for TPM.
func (p *Proxy) reconcileBudgetAndRateLimits(logCtx *RequestLogContext, actualCost float64) {
	if logCtx == nil || logCtx.budgetReconciled {
		return
	}
	logCtx.budgetReconciled = true

	if len(logCtx.ReservedEntities) == 0 && len(logCtx.RateLimitedEntitiesForTPM) == 0 {
		return
	}

	// The request context may already be canceled (client disconnected) by the
	// time this runs, so use a fresh bounded context — reconciliation must complete.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if p.budgetReserver != nil {
		for _, e := range logCtx.ReservedEntities {
			delta := actualCost - e.reservedAmount
			if err := p.budgetReserver.Reconcile(ctx, e.key, delta); err != nil {
				p.logger.WarnContext(ctx, "Failed to reconcile budget reservation", "entity", e.key, "error", err)
			}
		}
	}
	if p.keyRateLimiter != nil && logCtx.TokenUsage != nil {
		totalTokens := logCtx.TokenUsage.Total()
		for _, entity := range logCtx.RateLimitedEntitiesForTPM {
			p.keyRateLimiter.ConsumeTokensCtx(ctx, entity, totalTokens)
		}
	}
}
