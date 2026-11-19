package proxy

import (
	"errors"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

var errKeyRoutingPolicy = errors.New("credential is outside the permitted routing pool")

type keyRoutingRestrictedContextKey struct{}

func (p *Proxy) applyKeyRoutingPolicy(logCtx *RequestLogContext, modelID, realModelID string) {
	if p.keyRoutingPolicies == nil || logCtx.TokenInfo == nil || logCtx.TokenInfo.IsMasterKey {
		return
	}
	visibility, policyID := p.keyRoutingPolicies.Restrict(logCtx.Scope, logCtx.TokenInfo.Token, logCtx.PublicModelID, modelID, realModelID)
	logCtx.Scope = visibility
	if policyID != "" {
		p.logger.InfoContext(logCtx.Context(), "Key routing policy applied", "routing_policy_id", policyID, "model", logCtx.PublicModelID)
	}
}

// Defense in depth at the send boundary. Normal selections already apply this
// filter through CredentialConfig.VisibleTo, including exact/sticky picks.
func checkKeyRoutingCredential(logCtx *RequestLogContext, credential *config.CredentialConfig) error {
	if logCtx != nil && logCtx.Scope.HasCredentialRestriction() && (credential == nil || !credential.VisibleTo(logCtx.Scope)) {
		return errKeyRoutingPolicy
	}
	return nil
}

func rejectKeyRoutingSend(w http.ResponseWriter, logCtx *RequestLogContext) {
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusServiceUnavailable
	logCtx.ErrorMsg = "No permitted route available"
	WriteErrorServiceUnavailable(w, logCtx.ErrorMsg)
}
