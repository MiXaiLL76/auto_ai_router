package proxy

import (
	"context"
	"net/http"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// trustedClientAuth is an in-process handoff used by handlers that transform
// one already-authorized public operation into another internal provider
// request. The private context key cannot be supplied over HTTP.
type trustedClientAuth struct {
	rawToken  string
	tokenInfo *dbmodels.TokenInfo
}

type trustedClientAuthContextKey struct{}

func withTrustedClientAuth(req *http.Request, rawToken string, tokenInfo *dbmodels.TokenInfo) *http.Request {
	if req == nil || tokenInfo == nil {
		return req
	}
	value := trustedClientAuth{rawToken: rawToken, tokenInfo: tokenInfo.Clone()}
	return req.WithContext(context.WithValue(req.Context(), trustedClientAuthContextKey{}, value))
}

func trustedClientAuthFromRequest(req *http.Request) (trustedClientAuth, bool) {
	if req == nil {
		return trustedClientAuth{}, false
	}
	value, ok := req.Context().Value(trustedClientAuthContextKey{}).(trustedClientAuth)
	if !ok || value.rawToken == "" || value.tokenInfo == nil {
		return trustedClientAuth{}, false
	}
	value.tokenInfo = value.tokenInfo.Clone()
	return value, true
}
