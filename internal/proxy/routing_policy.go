package proxy

import (
	"errors"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
)

var errRoutingPool = errors.New("credential is outside the permitted routing pool")

type routingRestrictedContextKey struct{}

func (p *Proxy) applyRoutingPolicy(w http.ResponseWriter, logCtx *RequestLogContext, modelID string) bool {
	info := logCtx.TokenInfo
	if info == nil || info.IsMasterKey {
		return true
	}
	policy, err := models.ParseKeyRoutingPolicy(info.Metadata, info.TeamMetadata, info.OrganizationMetadata)
	if err == nil {
		logCtx.Scope, err = p.modelManager.RestrictToModelPools(logCtx.Scope, policy, logCtx.PublicModelID, modelID)
	}
	if err != nil {
		logCtx.Scope = logCtx.Scope.RestrictCredentials(nil)
		p.logger.WarnContext(logCtx.Context(), "Routing policy is invalid or unavailable")
		rejectRoutingSend(w, logCtx)
		return false
	}
	return true
}

func checkRoutingCredential(logCtx *RequestLogContext, credential *config.CredentialConfig) error {
	if logCtx != nil && logCtx.Scope.HasCredentialRestriction() && (credential == nil || !credential.VisibleTo(logCtx.Scope)) {
		return errRoutingPool
	}
	return nil
}

func rejectRoutingSend(w http.ResponseWriter, logCtx *RequestLogContext) {
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusServiceUnavailable
	logCtx.ErrorMsg = "No permitted route available"
	WriteErrorServiceUnavailable(w, logCtx.ErrorMsg)
}
