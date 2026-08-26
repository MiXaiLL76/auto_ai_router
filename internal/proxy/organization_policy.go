package proxy

import (
	"fmt"
	"net/http"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
)

func (p *Proxy) effectiveOrganizationPolicy(info *dbmodels.TokenInfo) (*routermodels.OrganizationPolicy, string, bool) {
	if p == nil || p.organizationPolicies == nil || p.organizationPolicies.Empty() || info == nil || info.IsMasterKey {
		return nil, "", false
	}

	directOrg := info.DirectOrganizationID
	if directOrg == "" {
		directOrg = info.OrganizationID
	}
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

func (p *Proxy) OrganizationPolicyForTokenInfo(info *dbmodels.TokenInfo) (*routermodels.OrganizationPolicy, bool) {
	policy, _, dangling := p.effectiveOrganizationPolicy(info)
	return policy, dangling
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

func (p *Proxy) admitOrganizationModel(
	w http.ResponseWriter,
	r *http.Request,
	body []byte,
	publicModelID string,
	logCtx *RequestLogContext,
) ([]byte, string, string, bool) {
	policy, ok := p.selectOrganizationPolicy(w, logCtx)
	if !ok {
		return nil, "", "", false
	}
	if policy == nil {
		return body, "", "", true
	}

	resolution, err := p.modelManager.ResolveOrganizationModelScoped(policy, publicModelID, logCtx.Scope)
	if err != nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusNotFound
		logCtx.ErrorMsg = fmt.Sprintf("Model %s not found", publicModelID)
		logCtx.Logged = true
		WriteErrorNotFound(w, logCtx.ErrorMsg)
		return nil, "", "", false
	}

	candidates := []string{resolution.PublicModelID, resolution.CanonicalModelID, resolution.ModelID}
	if !p.isAnyModelAllowedForToken(logCtx.TokenInfo, candidates) {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusForbidden
		logCtx.ErrorMsg = "Model not allowed"
		WriteErrorForbidden(w, "Model not allowed")
		return nil, "", "", false
	}
	if resolution.ModelPrice == nil {
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusServiceUnavailable
		logCtx.ErrorMsg = "model pricing unavailable"
		logCtx.Logged = true
		WriteErrorServiceUnavailable(w, "Model pricing unavailable")
		return nil, "", "", false
	}

	if resolution.CanonicalModelID != publicModelID {
		body = replaceModelInBodyPreserveContentType(body, r.Header.Get("Content-Type"), publicModelID, resolution.CanonicalModelID)
	}
	if resolution.ModelID != resolution.CanonicalModelID {
		body = replaceModelInBodyPreserveContentType(body, r.Header.Get("Content-Type"), resolution.CanonicalModelID, resolution.ModelID)
	}
	if resolution.RealModelID != resolution.ModelID {
		body = replaceModelInBodyPreserveContentType(body, r.Header.Get("Content-Type"), resolution.ModelID, resolution.RealModelID)
	}

	logCtx.PublicModelID = resolution.PublicModelID
	logCtx.CanonicalModelID = resolution.CanonicalModelID
	logCtx.ModelID = resolution.ModelID
	logCtx.RealModelID = resolution.RealModelID
	logCtx.PriceModelID = resolution.PriceModelID
	logCtx.ModelPrice = resolution.ModelPrice
	logCtx.billingPriceResolved = true
	logCtx.billingPriceModelID = resolution.PriceModelID
	logCtx.billingPrice = resolution.ModelPrice

	return body, resolution.ModelID, resolution.RealModelID, true
}

func (p *Proxy) IsOrganizationModelAllowedForToken(
	tokenInfo *dbmodels.TokenInfo,
	policy *routermodels.OrganizationPolicy,
	publicModelID string,
) bool {
	if p == nil || p.modelManager == nil || policy == nil {
		return false
	}
	resolution, err := p.modelManager.ResolveOrganizationModel(policy, publicModelID)
	if err != nil {
		return false
	}
	return p.isAnyModelAllowedForToken(tokenInfo, []string{
		resolution.PublicModelID,
		resolution.CanonicalModelID,
		resolution.ModelID,
	})
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
		matcher = func(model string, allowedModels []string) bool {
			return p.modelManager.IsAnyModelIDAllowedByScope(candidates, allowedModels)
		}
	}
	return tokenInfo.IsModelAllowedByPolicy(candidates[0], matcher, dbmodels.ModelAccessPolicy{
		StrictAllTeamModelsACL: true,
	})
}
