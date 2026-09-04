package models

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// ProviderRoutesForModel returns the stable leaf alternatives behind a credential
// from the same cached health snapshot used for discovery. No network calls or
// live rate-limit checks are made. The caller owns the returned map.
func (m *Manager) ProviderRoutesForModel(cred *config.CredentialConfig, modelID string) map[string]*scope.Expression {
	if !cred.IsProxyLike() {
		return map[string]*scope.Expression{cred.Name: cred.ScopeExpression()}
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	cached := m.remoteModelsCache[cred.Name]
	if cached.health == nil || !cred.SameProviderIdentity(cached.credential) {
		// Missing or legacy health cannot prove which leaves serve this route.
		return map[string]*scope.Expression{"": cred.ScopeExpression()}
	}
	ownScope := scope.FromScopes(cred.Scopes, cred.DeniedScopes)
	routes := make(map[string]*scope.Expression)
	for _, model := range cached.health.Models {
		upstream, ok := cached.health.Credentials[model.Credential]
		if !ok || model.Model != modelID || (!cred.IsFallback && upstream.IsFallback) {
			continue
		}
		leaves := model.ProviderRoutes
		if leaves == nil {
			name := model.Credential
			if upstream.IsProxyLike || upstream.Type == string(config.ProviderTypeProxy) || upstream.Type == string(config.ProviderTypeAIR) {
				name = "" // RealCredential is display-only, not a complete provider list.
			}
			leaves = map[string]*scope.Expression{name: modelScopeExpression(model, upstream)}
		}
		for name, leafScope := range leaves {
			// Leaf metadata already includes the complete upstream path. Adding its
			// aggregate model scope again multiplies unrelated alternatives at each hop.
			expression := scope.And(ownScope, leafScope)
			if previous, exists := routes[name]; exists {
				expression = scope.Or(previous, expression)
			}
			routes[name] = expression
		}
	}
	return routes
}
