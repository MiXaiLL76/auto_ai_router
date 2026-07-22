package proxy

import (
	"errors"
	"net/http"
	"strings"

	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const liteLLMMasterKeyIdentity = "litellm-master-key"

var errInvalidScopeAuth = errors.New("invalid authorization")

func (p *Proxy) ScopeContextForRequest(r *http.Request) (scope.Context, error) {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return scope.PublicContext(), nil
	}

	token, ok := bearerToken(authHeader)
	if !ok {
		return scope.PublicContext(), errInvalidScopeAuth
	}
	if p.isMasterKey(token) {
		return scope.AdminContext(), nil
	}
	if !p.isLiteLLMHealthy() {
		return scope.PublicContext(), errInvalidScopeAuth
	}

	tokenInfo, err := p.LiteLLMDB.ValidateToken(r.Context(), token)
	if err != nil {
		return scope.PublicContext(), err
	}
	return scopeContextFromTokenInfo(tokenInfo), nil
}

func bearerToken(authHeader string) (string, bool) {
	token, ok := strings.CutPrefix(authHeader, "Bearer ")
	if !ok || strings.TrimSpace(token) == "" {
		return "", false
	}
	return strings.TrimSpace(token), true
}

func scopeContextFromTokenInfo(info *dbmodels.TokenInfo) scope.Context {
	if info == nil {
		return scope.PublicContext()
	}
	if isLiteLLMMasterTokenInfo(info) {
		return scope.AdminContext()
	}

	allowed := metadataScopes(info.Metadata, "air_scopes", "air_scope")
	if len(allowed) == 0 {
		allowed = append(allowed, info.KeyName)
		if info.KeyName == "" {
			allowed = append(allowed, info.KeyAlias)
		}
	}

	denied := metadataScopes(info.Metadata,
		"air_denied_scopes",
		"air_denied_scope",
		"air_forbidden_scopes",
		"air_forbidden_scope",
	)

	return scope.NewContext(allowed, denied)
}

func isLiteLLMMasterTokenInfo(info *dbmodels.TokenInfo) bool {
	return info != nil && info.UserID == liteLLMMasterKeyIdentity
}

func metadataScopes(metadata map[string]interface{}, keys ...string) []string {
	for _, key := range keys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		scopes := scopeValues(value)
		if len(scopes) > 0 {
			return scopes
		}
	}
	return nil
}

func scopeValues(value interface{}) []string {
	switch v := value.(type) {
	case string:
		return splitScopeString(v)
	case []string:
		return v
	case []interface{}:
		values := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				values = append(values, s)
			}
		}
		return values
	default:
		return nil
	}
}

func splitScopeString(value string) []string {
	if !strings.Contains(value, ",") {
		return []string{value}
	}
	parts := strings.Split(value, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		values = append(values, strings.TrimSpace(part))
	}
	return values
}
