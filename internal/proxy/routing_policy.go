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
	if err == nil && policy == nil && info.Metadata[models.RoutingModelsField] != nil {
		// The selector is present but names no pool, so nothing is restricted.
		// Silence here would hide an operator mistake that drops a real isolation
		// to ordinary routing.
		p.logger.WarnContext(logCtx.Context(), "Model selector ignored: no routing profile list",
			"field", models.RoutingModelsField, "model", logCtx.PublicModelID,
			"request_id", logCtx.RequestID,
		)
	}
	if err == nil {
		logCtx.Scope, err = p.modelManager.RestrictToModelPools(logCtx.Scope, policy, logCtx.PublicModelID, modelID)
	}
	if err != nil {
		logCtx.Scope = logCtx.Scope.RestrictCredentials(nil)
		// The client gets one generic 503; the operator needs to know which
		// field and why, without echoing pool or model values.
		p.logger.WarnContext(logCtx.Context(), "Routing policy is invalid or unavailable",
			"reason", err.Error(),
			"model", logCtx.PublicModelID,
			"request_id", logCtx.RequestID,
		)
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
