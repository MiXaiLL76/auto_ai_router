package proxy

import (
	"errors"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

var errNoPermittedCredential = errors.New("no permitted route available")

func checkAllowedCredential(logCtx *RequestLogContext, credential *config.CredentialConfig) error {
	if logCtx != nil && logCtx.Scope.HasCredentialRestriction() &&
		(credential == nil || !credential.VisibleTo(logCtx.Scope)) {
		return errNoPermittedCredential
	}
	return nil
}

func rejectUnpermittedCredential(w http.ResponseWriter, logCtx *RequestLogContext) {
	logCtx.Status = "failure"
	logCtx.HTTPStatus = http.StatusServiceUnavailable
	logCtx.ErrorMsg = errNoPermittedCredential.Error()
	WriteErrorServiceUnavailable(w, logCtx.ErrorMsg)
}
