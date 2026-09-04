package proxy

import (
	"errors"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

var (
	errModelNotAvailable = errors.New("model not available")
	errModelNotAllowed   = errors.New("model not allowed")
	ErrOrganizationGone  = errors.New("organization not available")
)

type modelAdmission struct {
	publicModelID    string
	canonicalModelID string
	modelID          string
	realModelID      string
	priceModelID     string
	modelPrice       *routermodels.ModelPrice
	excluded         map[string]bool
}

func (p *Proxy) resolveModelAdmission(
	tokenInfo *dbmodels.TokenInfo,
	visibility scope.Context,
	policy *routermodels.OrganizationPolicy,
	requestedModelID string,
	trustedInternal bool,
) (modelAdmission, error) {
	if p == nil || p.modelManager == nil || p.balancer == nil {
		return modelAdmission{}, errModelNotAvailable
	}

	admission := modelAdmission{
		publicModelID: requestedModelID, canonicalModelID: requestedModelID,
		modelID: requestedModelID, realModelID: requestedModelID,
		excluded: make(map[string]bool),
	}
	if policy != nil {
		resolved, err := p.modelManager.ResolveOrganizationModelScoped(policy, requestedModelID, visibility)
		if err != nil {
			return modelAdmission{}, errModelNotAvailable
		}
		admission.publicModelID = resolved.PublicModelID
		admission.canonicalModelID = resolved.CanonicalModelID
		admission.modelID = resolved.ModelID
		admission.realModelID = resolved.RealModelID
		admission.priceModelID = resolved.PriceModelID
		admission.modelPrice = resolved.ModelPrice
		if !p.isAnyModelAllowedForToken(tokenInfo, []string{
			resolved.PublicModelID, resolved.CanonicalModelID, resolved.ModelID,
		}) {
			return modelAdmission{}, errModelNotAllowed
		}
		if resolved.ModelPrice == nil {
			return modelAdmission{}, errModelNotAvailable
		}
	} else {
		if !p.IsModelAllowedForToken(tokenInfo, requestedModelID) {
			return modelAdmission{}, errModelNotAllowed
		}
		resolved, ok := p.modelManager.ResolveModel(requestedModelID, nil, trustedInternal)
		if !ok {
			return modelAdmission{}, errModelNotAvailable
		}
		admission.publicModelID = resolved.PublicModelID
		admission.canonicalModelID = resolved.CanonicalModelID
		admission.modelID = resolved.ModelID
		admission.realModelID = resolved.RealModelID
	}

	if len(p.modelManager.GetCredentialsForModel(admission.modelID)) == 0 {
		return modelAdmission{}, errModelNotAvailable
	}

	denylist := map[string]bool{}
	if policy != nil {
		for _, name := range policy.CredentialDenylist() {
			denylist[name] = true
		}
	}
	eligible := false
	credentials := p.balancer.GetCredentialsSnapshot()
	for i := range credentials {
		cred := &credentials[i]
		if !cred.VisibleTo(visibility) || !p.modelManager.HasModelScoped(cred.Name, admission.modelID, visibility) {
			continue
		}
		if len(denylist) > 0 && !p.credentialAllowedByDenylist(cred, admission.modelID, visibility, denylist) {
			admission.excluded[cred.Name] = true
			continue
		}
		if policy == nil && p.spendTrackingEnabled() {
			realModelID := admission.realModelID
			if !cred.IsProxyLike() {
				realModelID, _ = p.modelManager.GetRealModelNameForCredential(admission.modelID, cred.Name)
			}
			_, price := lookupBillingModelPrice(p.priceRegistry, admission.publicModelID, admission.modelID, realModelID)
			if price == nil {
				admission.excluded[cred.Name] = true
				continue
			}
		}
		eligible = true
	}
	if !eligible {
		return modelAdmission{}, errModelNotAvailable
	}
	return admission, nil
}

func (p *Proxy) credentialAllowedByDenylist(cred *config.CredentialConfig, modelID string, visibility scope.Context, denylist map[string]bool) bool {
	if !cred.IsProxyLike() {
		return !denylist[cred.Name]
	}
	if !carriesCredentialDenylist(cred) {
		return true
	}
	for name, expression := range p.modelManager.ProviderRoutesForModel(cred, modelID) {
		if !denylist[name] && visibility.AllowsExpression(expression) {
			return true
		}
	}
	return false
}

func (p *Proxy) globalRealModelID(modelID string) string {
	if p != nil && p.modelManager != nil {
		if realModelID, mapped := p.modelManager.GetRealModelName(modelID); mapped {
			return realModelID
		}
	}
	return modelID
}

// ListModelsForToken returns exactly the stable model surface admitted by the
// same resolver used before inference. Temporary bans and live RPM/TPM usage do
// not make the catalog flicker.
// TODO: fix O(models x credentials) is too much
func (p *Proxy) ListModelsForToken(tokenInfo *dbmodels.TokenInfo, visibility scope.Context) (routermodels.ModelsResponse, error) {
	policy, _, dangling := p.effectiveOrganizationPolicy(tokenInfo)
	if dangling {
		return routermodels.ModelsResponse{}, ErrOrganizationGone
	}
	if p == nil || p.modelManager == nil {
		return routermodels.ModelsResponse{Object: "list", Data: []routermodels.Model{}}, nil
	}
	var response routermodels.ModelsResponse
	switch {
	case policy != nil:
		response = p.modelManager.GetAllModelsScopedForOrganization(visibility, policy)
	case tokenInfo != nil && tokenInfo.IsMasterKey:
		response = p.modelManager.GetAllModelsScopedIncludingInternal(visibility)
	default:
		response = p.modelManager.GetAllModelsScoped(visibility)
	}
	filtered := make([]routermodels.Model, 0, len(response.Data))
	for _, model := range response.Data {
		_, err := p.resolveModelAdmission(tokenInfo, visibility, policy, model.ID, tokenInfo != nil && tokenInfo.IsMasterKey)
		if err == nil {
			filtered = append(filtered, model)
		}
	}
	response.Data = filtered
	return response, nil
}
