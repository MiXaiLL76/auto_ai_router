package router

import (
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/scope"
)

func (r *Router) visibilityScopeOrPublic(req *http.Request) scope.Context {
	if r.proxy == nil {
		return scope.PublicContext()
	}
	visibility, err := r.proxy.ScopeContextForRequest(req)
	if err != nil {
		return scope.PublicContext()
	}
	return visibility
}
