package models

import (
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// AggregateProviderScopesFromHealth ORs the scope expressions of every upstream
// credential in the /health response. All credentials are included regardless of their
// priority tier: a chained downstream discovers the upstream's full catalogue and tiers
// last-resort (priority: 999 / is_fallback) models locally rather than dropping them.
func AggregateProviderScopesFromHealth(health *httputil.ProxyHealthResponse) ScopeMetadata {
	expressions := make([]*scope.Expression, 0, len(health.Credentials))
	for _, stats := range health.Credentials {
		expressions = append(expressions, credentialScopeExpression(stats))
	}
	return scopeMetadataFromExpression(scope.Or(expressions...))
}

// AggregateModelScopesFromHealth builds the per-model scope expression from every
// upstream credential serving it. See AggregateProviderScopesFromHealth on tier handling.
func AggregateModelScopesFromHealth(health *httputil.ProxyHealthResponse) map[string]ScopeMetadata {
	expressions := make(map[string][]*scope.Expression)
	for _, modelStats := range health.Models {
		if modelStats.Model == "" {
			continue
		}
		credStats, ok := health.Credentials[modelStats.Credential]
		if !ok {
			continue
		}
		expressions[modelStats.Model] = append(expressions[modelStats.Model], modelScopeExpression(modelStats, credStats))
	}

	result := make(map[string]ScopeMetadata, len(expressions))
	for modelID, modelExpressions := range expressions {
		result[modelID] = scopeMetadataFromExpression(scope.Or(modelExpressions...))
	}
	return result
}

func credentialScopeExpression(stats httputil.CredentialHealthStats) *scope.Expression {
	if stats.ScopeExpression != nil {
		return scope.NormalizeExpression(stats.ScopeExpression)
	}
	return scope.FromScopes(stats.Scopes, stats.DeniedScopes)
}

func modelScopeExpression(modelStats httputil.ModelHealthStats, credStats httputil.CredentialHealthStats) *scope.Expression {
	if modelStats.ScopeExpression != nil {
		return scope.NormalizeExpression(modelStats.ScopeExpression)
	}
	if len(modelStats.Scopes) > 0 || len(modelStats.DeniedScopes) > 0 {
		return scope.FromScopes(modelStats.Scopes, modelStats.DeniedScopes)
	}
	return credentialScopeExpression(credStats)
}

func scopeMetadataFromExpression(expression *scope.Expression) ScopeMetadata {
	scopes, deniedScopes := expression.LegacyProjection()
	return ScopeMetadata{
		Scopes:          scopes,
		DeniedScopes:    deniedScopes,
		ScopeExpression: scope.NormalizeExpression(expression),
	}
}
