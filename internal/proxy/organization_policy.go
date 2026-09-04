package proxy

import (
	"net/http"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
)

func (p *Proxy) effectiveOrganizationPolicy(info *dbmodels.TokenInfo) (*routermodels.OrganizationPolicy, string, bool) {
	if p == nil || p.organizationPolicies == nil || p.organizationPolicies.Empty() || info == nil || info.IsMasterKey {
		return nil, "", false
	}

	// Only the token's own organization counts as a direct organization here.
	// info.OrganizationID is documented as "resolved from token or team", so it
	// must not be used as a fallback: a team-derived organization would then be
	// admitted through the direct branch and skip the TeamDangling /
	// TeamOrganizationDangling checks below.
	directOrg := info.DirectOrganizationID
	if directOrg != "" {
		policy, ok := p.organizationPolicies.Policy(directOrg)
		if !ok {
			return nil, "", false
		}
		if info.DirectOrganizationDangling {
			return nil, directOrg, true
		}
		return policy, directOrg, false
	}

	teamOrg := info.TeamOrganizationID
	if teamOrg == "" {
		return nil, "", false
	}
	policy, ok := p.organizationPolicies.Policy(teamOrg)
	if !ok {
		return nil, "", false
	}
	if info.TeamDangling || info.TeamOrganizationDangling {
		return nil, teamOrg, true
	}
	return policy, teamOrg, false
}

func (p *Proxy) selectOrganizationPolicy(w http.ResponseWriter, logCtx *RequestLogContext) (*routermodels.OrganizationPolicy, bool) {
	policy, organizationID, dangling := p.effectiveOrganizationPolicy(logCtx.TokenInfo)
	if dangling {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusForbidden
		logCtx.ErrorMsg = "Organization not available"
		logCtx.Logged = true
		WriteErrorForbidden(w, "Forbidden")
		return nil, false
	}
	if policy == nil {
		return nil, true
	}
	logCtx.OrganizationPolicy = policy
	logCtx.BillingProfileID = policy.PriceProfileID
	logCtx.BillingProfileSHA256 = policy.ProfileSHA256
	logCtx.BillingOrganizationID = organizationID
	if logCtx.TokenInfo != nil && logCtx.TokenInfo.OrganizationID != organizationID {
		logCtx.TokenInfo = logCtx.TokenInfo.Clone()
		logCtx.TokenInfo.OrganizationID = organizationID
	}
	return policy, true
}

func (p *Proxy) resolveRetryBillingPrice(
	logCtx *RequestLogContext,
	publicModelID string,
	modelID string,
	realModelID string,
) (string, *routermodels.ModelPrice) {
	if logCtx != nil && logCtx.OrganizationPolicy != nil {
		return logCtx.PriceModelID, logCtx.ModelPrice
	}
	return lookupBillingModelPrice(p.priceRegistry, publicModelID, modelID, realModelID)
}

func replaceModelInBodyPreserveContentType(body []byte, contentType, oldModel, newModel string) []byte {
	if oldModel == "" || newModel == "" || oldModel == newModel {
		return body
	}
	if replaced, err := replaceRequestModel(body, contentType, newModel); err == nil {
		return replaced
	}
	return body
}

func (p *Proxy) isAnyModelAllowedForToken(tokenInfo *dbmodels.TokenInfo, candidates []string) bool {
	if tokenInfo == nil || !p.strictAllTeamModelsACL {
		return true
	}
	var matcher dbmodels.ModelScopeMatcher
	if p.modelManager != nil {
		matcher = func(_ string, allowedModels []string) bool {
			return p.modelManager.IsAnyModelIDAllowedByScope(candidates, allowedModels)
		}
	}
	return tokenInfo.IsModelAllowedByPolicy(candidates[0], matcher, dbmodels.ModelAccessPolicy{
		StrictAllTeamModelsACL: true,
	})
}
