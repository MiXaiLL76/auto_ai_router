package yandexsearch

import (
	"context"
	"log/slog"
	"strconv"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// budgetLevel is one entry of the token/user/team/org budget hierarchy to
// reserve against before the upstream call.
type budgetLevel struct {
	entity    string
	level     dbmodels.BudgetLevel
	window    string
	maxBudget *float64
	dbSpend   float64
}

// budgetLevelsFor mirrors internal/proxy/budget_enforcement.go's
// budgetLevels() (entity naming and field provenance kept identical:
// "token:"+Token, "token:"+Token+":window:"+Duration+":"+<window start, unix>,
// "user:"+UserID, "team:"+TeamID, "org:"+OrganizationID,
// "teammember:"+TeamID+":"+UserID, "orgmember:"+OrganizationID+":"+UserID)
// — duplicated rather than imported because internal/yandexsearch is
// deliberately decoupled from internal/proxy (mirrors internal/video's own
// decoupling) and that function is unexported. If the LLM path's hierarchy
// logic changes, check here too.
func budgetLevelsFor(info *dbmodels.TokenInfo) []budgetLevel {
	if info == nil {
		return nil
	}
	levels := []budgetLevel{{
		entity:    "token:" + info.Token,
		level:     dbmodels.BudgetLevelKey,
		maxBudget: info.MaxBudget,
		dbSpend:   info.Spend,
	}}
	for _, window := range info.BudgetWindows {
		if window.Start.IsZero() {
			continue
		}
		levels = append(levels, budgetLevel{
			entity:    "token:" + info.Token + ":window:" + window.Duration + ":" + strconv.FormatInt(window.Start.Unix(), 10),
			level:     dbmodels.BudgetLevelKey,
			window:    window.Duration,
			maxBudget: &window.MaxBudget,
			dbSpend:   window.Spend,
		})
	}
	if info.TeamID == "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "user:" + info.UserID,
			level:     dbmodels.BudgetLevelUser,
			maxBudget: info.UserMaxBudget,
			dbSpend:   dereferenceFloat(info.UserSpend),
		})
	}
	if info.TeamID != "" {
		levels = append(levels, budgetLevel{
			entity:    "team:" + info.TeamID,
			level:     dbmodels.BudgetLevelTeam,
			maxBudget: info.TeamMaxBudget,
			dbSpend:   dereferenceFloat(info.TeamSpend),
		})
	}
	if info.OrganizationID != "" {
		levels = append(levels, budgetLevel{
			entity:    "org:" + info.OrganizationID,
			level:     dbmodels.BudgetLevelOrganization,
			maxBudget: info.OrgMaxBudget,
			dbSpend:   dereferenceFloat(info.OrgSpend),
		})
	}
	if info.TeamID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "teammember:" + info.TeamID + ":" + info.UserID,
			level:     dbmodels.BudgetLevelTeamMember,
			maxBudget: info.TeamMemberMaxBudget,
			dbSpend:   dereferenceFloat(info.TeamMemberSpend),
		})
	}
	if info.OrganizationID != "" && info.UserID != "" {
		levels = append(levels, budgetLevel{
			entity:    "orgmember:" + info.OrganizationID + ":" + info.UserID,
			level:     dbmodels.BudgetLevelOrganizationMember,
			maxBudget: info.OrgMemberMaxBudget,
			dbSpend:   dereferenceFloat(info.OrgMemberSpend),
		})
	}
	return levels
}

func dereferenceFloat(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

// reservation tracks the levels a budgetReserve call actually reserved
// against, so they can be settled (or rolled back) exactly once.
type reservation struct {
	entities    []budgetLevel
	estimateUSD float64
	reconciled  bool
}

// budgetReserve reserves estimateUSD against every applicable level of
// info's budget hierarchy (unlimited levels, maxBudget == nil, are
// skipped — same convention as the LLM path). On a Reserver error it logs
// and treats that level as allowed (fails OPEN — same "availability wins"
// rationale as internal/proxy/budget_enforcement.go; this is money, not
// the shared Yandex quota, which fails closed in quota.go for a different
// reason). If any level rejects, every level already reserved by this call
// is rolled back and a *dbmodels.BudgetExceededError naming that level is
// returned.
func budgetReserve(ctx context.Context, reserver *budget.Reserver, info *dbmodels.TokenInfo, estimateUSD float64, logger *slog.Logger) (*reservation, error) {
	levels := budgetLevelsFor(info)
	reserved := make([]budgetLevel, 0, len(levels))
	for _, level := range levels {
		if level.maxBudget == nil {
			continue
		}
		allowed, err := reserver.TryReserve(ctx, level.entity, level.dbSpend, estimateUSD, *level.maxBudget)
		if err != nil {
			if logger != nil {
				logger.WarnContext(ctx, "Yandex search: budget reservation unavailable; request allowed",
					"entity", level.entity, "error", err)
			}
			continue
		}
		if !allowed {
			for _, done := range reserved {
				if err := reserver.Reconcile(ctx, done.entity, -estimateUSD); err != nil && logger != nil {
					logger.WarnContext(ctx, "Yandex search: failed to roll back budget reservation",
						"entity", done.entity, "error", err)
				}
			}
			return nil, &dbmodels.BudgetExceededError{Level: level.level, Window: level.window}
		}
		reserved = append(reserved, level)
	}
	return &reservation{entities: reserved, estimateUSD: estimateUSD}, nil
}

// settle reconciles every reserved level to actualUSD exactly once; a
// repeated call is a no-op. Call with actualUSD == estimateUSD on a billed
// success, or 0 on any outcome that isn't billed (non-2xx, local error,
// timeout) — budget.Reserver.Reconcile itself no-ops when the resulting
// delta is 0.
func (r *reservation) settle(ctx context.Context, reserver *budget.Reserver, actualUSD float64, logger *slog.Logger) {
	if r == nil || r.reconciled {
		return
	}
	r.reconciled = true
	for _, level := range r.entities {
		if err := reserver.Reconcile(ctx, level.entity, actualUSD-r.estimateUSD); err != nil && logger != nil {
			logger.WarnContext(ctx, "Yandex search: failed to reconcile budget reservation",
				"entity", level.entity, "error", err)
		}
	}
}

// budgetExceededMessage is the 402 text of a refused reservation. The key's
// own limit gets its own wording, a key budget window names its duration;
// every level above the key keeps the long-standing "budget exceeded".
func budgetExceededMessage(err error) string {
	refusal, ok := dbmodels.AsBudgetExceeded(err)
	switch {
	case ok && refusal.Level == dbmodels.BudgetLevelKey && refusal.Window != "":
		return "key " + refusal.Window + " budget exceeded"
	case ok && refusal.Level == dbmodels.BudgetLevelKey:
		return "key budget exceeded"
	default:
		return "budget exceeded"
	}
}
