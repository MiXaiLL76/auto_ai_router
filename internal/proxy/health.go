package proxy

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/proxy/webui"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

func (p *Proxy) HealthCheck() (bool, *httputil.ProxyHealthResponse) {
	return p.HealthCheckScoped(scope.AdminContext())
}

func (p *Proxy) HasAvailableCredentials() bool {
	return p.balancer.GetAvailableCount() > 0
}

func (p *Proxy) HealthCheckScoped(visibility scope.Context) (bool, *httputil.ProxyHealthResponse) {
	ctx := context.Background()

	creds := visibleCredentials(p.balancer.GetCredentialsSnapshot(), visibility)
	totalCreds := len(creds)
	visibleCreds := credentialNameSet(creds)
	availableCreds := 0
	for _, cred := range creds {
		if !p.balancer.HasAnyBan(cred.Name) {
			availableCreds++
		}
	}
	bannedCreds := 0
	for _, bp := range p.balancer.GetBannedPairs() {
		if visibleCreds[bp.Credential] {
			bannedCreds++
		}
	}

	healthy := availableCreds > 0

	if creds == nil {
		creds = []config.CredentialConfig{}
	}

	// Collect all (credential, model) pairs we'll need stats for.
	allTrackedModels := p.rateLimiter.GetAllModelPairs()
	allModelPairs := make([]ratelimit.ModelPair, 0, len(allTrackedModels))
	seenModelKeys := make(map[string]struct{}, len(allTrackedModels))
	modelScopeExpressions := make(map[string]*scope.Expression, len(allTrackedModels))
	for _, pair := range allTrackedModels {
		if !visibleCreds[pair.Credential] {
			continue
		}
		k := pair.Credential + ":" + pair.Model
		expression := p.modelScopeExpression(creds, pair.Credential, pair.Model)
		if !visibility.AllowsExpression(expression) {
			continue
		}
		seenModelKeys[k] = struct{}{}
		modelScopeExpressions[k] = expression
		allModelPairs = append(allModelPairs, pair)
	}
	if p.modelManager != nil {
		for _, cred := range creds {
			for _, model := range p.modelManager.GetModelsForCredential(cred.Name) {
				k := cred.Name + ":" + model.ID
				if _, ok := seenModelKeys[k]; ok {
					continue
				}
				expression := p.modelScopeExpression(creds, cred.Name, model.ID)
				if !visibility.AllowsExpression(expression) {
					continue
				}
				seenModelKeys[k] = struct{}{}
				modelScopeExpressions[k] = expression
				allModelPairs = append(allModelPairs, ratelimit.ModelPair{Credential: cred.Name, Model: model.ID})
			}
		}
	}

	// Fetch all RPM/TPM counters in a single backend round-trip.
	credNames := make([]string, len(creds))
	for i, c := range creds {
		credNames[i] = c.Name
	}
	credStats, modelStats := p.rateLimiter.BatchCurrentStats(ctx, credNames, allModelPairs)

	// Collect credentials info
	credentialsInfo := make(map[string]httputil.CredentialHealthStats, len(creds))
	for _, cred := range creds {
		limitRPM := cred.RPM
		limitTPM := cred.TPM
		if cred.Type == config.ProviderTypeProxy {
			rateLimiterRPM := p.rateLimiter.GetLimitRPM(cred.Name)
			rateLimiterTPM := p.rateLimiter.GetLimitTPM(cred.Name)
			if rateLimiterRPM != -1 {
				limitRPM = rateLimiterRPM
			}
			if rateLimiterTPM != -1 {
				limitTPM = rateLimiterTPM
			}
		}

		cs := credStats[cred.Name]
		expression := cred.ScopeExpression()
		scopes, deniedScopes := expression.LegacyProjection()
		credentialsInfo[cred.Name] = httputil.CredentialHealthStats{
			Type:             string(cred.Type),
			BaseURL:          cleanBaseURL(cred.BaseURL),
			IsFallback:       cred.IsFallback,
			IsBanned:         p.balancer.HasAnyBan(cred.Name),
			Weight:           balancer.EffectiveWeight(0, cred.Weight),
			FallbackPriority: cred.FallbackPriority,
			Scopes:           scopes,
			DeniedScopes:     deniedScopes,
			ScopeExpression:  expression,
			CurrentRPM:       cs.RPM,
			CurrentTPM:       cs.TPM,
			LimitRPM:         limitRPM,
			LimitTPM:         limitTPM,
		}
	}

	// Collect models info using the pre-fetched stats.
	modelsInfo := make(map[string]httputil.ModelHealthStats, len(allModelPairs))
	for _, pair := range allModelPairs {
		modelKey := pair.Credential + ":" + pair.Model
		p.addModelHealthStats(modelsInfo, creds, pair.Credential, pair.Model, modelStats, modelScopeExpressions[modelKey])
	}

	// Enrich models and credentials with error code counts from banned pairs
	bannedPairs := p.balancer.GetBannedPairs()
	// credentialErrorCounts accumulates error counts per credential across all its banned models
	credentialErrorCounts := make(map[string]map[int]int)
	for _, bp := range bannedPairs {
		if !visibleCreds[bp.Credential] {
			continue
		}
		modelKey := bp.Credential + ":" + bp.Model
		if ms, ok := modelsInfo[modelKey]; ok {
			if len(bp.ErrorCodeCounts) > 0 {
				counts := make(map[int]int, len(bp.ErrorCodeCounts))
				for code, cnt := range bp.ErrorCodeCounts {
					counts[code] = cnt
				}
				ms.ErrorCodeCounts = counts
			}
			ms.ProviderError = bp.Reason
			if !bp.BanUntil.IsZero() {
				banUntil := bp.BanUntil
				ms.BanUntil = &banUntil
			}
			modelsInfo[modelKey] = ms
		}
		// Aggregate into per-credential counts
		if len(bp.ErrorCodeCounts) > 0 {
			if credentialErrorCounts[bp.Credential] == nil {
				credentialErrorCounts[bp.Credential] = make(map[int]int)
			}
			for code, cnt := range bp.ErrorCodeCounts {
				credentialErrorCounts[bp.Credential][code] += cnt
			}
		}
	}
	// Apply aggregated error counts to credential info
	for credName, counts := range credentialErrorCounts {
		if cs, ok := credentialsInfo[credName]; ok {
			cs.BannedErrorCounts = counts
			credentialsInfo[credName] = cs
		}
	}

	status := &httputil.ProxyHealthResponse{
		Status:               "healthy",
		CredentialsAvailable: availableCreds,
		CredentialsBanned:    bannedCreds,
		TotalCredentials:     totalCreds,
		Credentials:          credentialsInfo,
		Models:               modelsInfo,
	}

	if !healthy {
		status.Status = "unhealthy"
	}

	return healthy, status
}

func visibleCredentials(creds []config.CredentialConfig, visibility scope.Context) []config.CredentialConfig {
	if len(creds) == 0 {
		return nil
	}
	result := make([]config.CredentialConfig, 0, len(creds))
	for _, cred := range creds {
		if cred.VisibleTo(visibility) {
			result = append(result, cred)
		}
	}
	return result
}

func credentialNameSet(creds []config.CredentialConfig) map[string]bool {
	result := make(map[string]bool, len(creds))
	for _, cred := range creds {
		result[cred.Name] = true
	}
	return result
}

func cleanBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	u.User = nil
	u.RawQuery = ""
	u.Fragment = ""
	return strings.TrimRight(u.String(), "/")
}

func (p *Proxy) addModelHealthStats(
	modelsInfo map[string]httputil.ModelHealthStats,
	creds []config.CredentialConfig,
	credentialName string,
	modelID string,
	stats map[string]ratelimit.KeyStats,
	expression *scope.Expression,
) {
	modelKey := credentialName + ":" + modelID
	credWeight := credentialWeight(creds, credentialName)
	modelWeight := 0
	if p.modelManager != nil {
		modelWeight = p.modelManager.GetModelWeightForCredential(modelID, credentialName)
	}
	ms := stats[modelKey]
	scopes, deniedScopes := expression.LegacyProjection()
	modelsInfo[modelKey] = httputil.ModelHealthStats{
		Credential:      credentialName,
		Model:           modelID,
		IsBanned:        p.balancer.IsBanned(credentialName, modelID),
		Weight:          balancer.EffectiveWeight(modelWeight, credWeight),
		CurrentRPM:      ms.RPM,
		CurrentTPM:      ms.TPM,
		LimitRPM:        p.rateLimiter.GetModelLimitRPM(credentialName, modelID),
		LimitTPM:        p.rateLimiter.GetModelLimitTPM(credentialName, modelID),
		Scopes:          scopes,
		DeniedScopes:    deniedScopes,
		ScopeExpression: expression,
	}
}

func (p *Proxy) modelScopeExpression(creds []config.CredentialConfig, credentialName, modelID string) *scope.Expression {
	var credential *config.CredentialConfig
	for _, cred := range creds {
		if cred.Name == credentialName {
			credentialCopy := cred
			credential = &credentialCopy
			break
		}
	}
	if credential == nil {
		return scope.FalseExpression()
	}
	if p.modelManager == nil {
		return credential.ScopeExpression()
	}
	modelExpression := p.modelManager.GetModelScopeExpressionForCredential(modelID, credentialName)
	if modelExpression == nil {
		return credential.ScopeExpression()
	}
	return scope.And(scope.FromScopes(credential.Scopes, credential.DeniedScopes), modelExpression)
}

func credentialWeight(creds []config.CredentialConfig, credentialName string) int {
	for _, cred := range creds {
		if cred.Name == credentialName {
			return cred.Weight
		}
	}
	return 0
}

// VisualHealthCheck serves the static health dashboard HTML.
func (p *Proxy) VisualHealthCheck(w http.ResponseWriter, r *http.Request) {
	webui.ServeHealth(w, r)
}
