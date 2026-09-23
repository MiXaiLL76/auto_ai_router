package proxy

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
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
		if cred.IsProxyLike() {
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
			IsProxyLike:      cred.IsProxyLike(),
			IsBanned:         p.balancer.HasAnyBan(cred.Name),
			Weight:           balancer.EffectiveWeight(0, cred.Weight),
			FallbackPriority: cred.FallbackPriority,
			// Priority carries only the explicit priority: field, not EffectivePriority():
			// a downstream proxy folds this into its own primary-pool grouping via
			// EffectiveHealthPriority, and fallback_priority (reported separately above)
			// must not leak in as a hard primary tier there.
			Priority:        cred.Priority,
			Scopes:          scopes,
			DeniedScopes:    deniedScopes,
			ScopeExpression: expression,
			CurrentRPM:      cs.RPM,
			CurrentTPM:      cs.TPM,
			LimitRPM:        limitRPM,
			LimitTPM:        limitTPM,
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
	now := time.Now().UTC()
	// credentialErrorCounts accumulates error counts per credential across all its banned models
	credentialErrorCounts := make(map[string]map[int]int)
	// primaryBans holds, per credential, the active ban that best explains why
	// it is banned (see betterBan).
	primaryBans := make(map[string]fail2ban.BanPair)
	for _, bp := range bannedPairs {
		if !visibleCreds[bp.Credential] {
			continue
		}
		active := bp.BanUntil.IsZero() || bp.BanUntil.After(now)
		if active {
			if cur, ok := primaryBans[bp.Credential]; !ok || betterBan(bp, cur) {
				primaryBans[bp.Credential] = bp
			}
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
	applyBansToModels(modelsInfo, bannedPairs, visibleCreds, now)
	for credName, bp := range primaryBans {
		if cs, ok := credentialsInfo[credName]; ok {
			cs.BanOrigin = bp.Origin
			cs.BanReason = bp.Reason
			if !bp.BanUntil.IsZero() {
				until := bp.BanUntil
				cs.BanUntil = &until
			}
			credentialsInfo[credName] = cs
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

// applyBansToModels overlays ban details onto the per-model health stats. The
// result does not depend on the order of bannedPairs (GetBannedPairs iterates
// a map): an active exact ban wins over a credential-wide one, and an expired
// exact ban is ignored while the credential is banned credential-wide, so its
// stale reason, expiry and error counts never leak next to the active ban.
func applyBansToModels(
	modelsInfo map[string]httputil.ModelHealthStats,
	bannedPairs []fail2ban.BanPair,
	visibleCreds map[string]bool,
	now time.Time,
) {
	isActive := func(bp fail2ban.BanPair) bool {
		return bp.BanUntil.IsZero() || bp.BanUntil.After(now)
	}

	credentialWide := make(map[string]fail2ban.BanPair)
	for _, bp := range bannedPairs {
		if visibleCreds[bp.Credential] && bp.Model == fail2ban.WildcardModel && isActive(bp) {
			credentialWide[bp.Credential] = bp
		}
	}

	for _, bp := range bannedPairs {
		if !visibleCreds[bp.Credential] || bp.Model == fail2ban.WildcardModel {
			continue
		}
		active := isActive(bp)
		if _, wide := credentialWide[bp.Credential]; wide && !active {
			continue
		}
		key := bp.Credential + ":" + bp.Model
		if ms, ok := modelsInfo[key]; ok {
			applyBanToModelStats(&ms, bp, active)
			modelsInfo[key] = ms
		}
	}

	// A credential-wide ban covers every model of the credential, including
	// ones with no ban entry of their own; models already under an active
	// exact ban keep their more specific details.
	for key, ms := range modelsInfo {
		if bp, wide := credentialWide[ms.Credential]; wide && ms.BanOrigin == "" {
			applyBanToModelStats(&ms, bp, true)
			modelsInfo[key] = ms
		}
	}
}

// applyBanToModelStats copies one ban's details onto a model's health entry.
// An expired ban that has not been swept yet keeps its historical error
// counts (as before) but is not reported as the model's current ban.
func applyBanToModelStats(ms *httputil.ModelHealthStats, bp fail2ban.BanPair, active bool) {
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
	if active {
		ms.BanOrigin = bp.Origin
	}
}

// betterBan reports whether a explains a credential's ban better than b: an
// admin ban beats an automatic one, then the ban lasting longest wins (a
// permanent ban, with a zero BanUntil, lasts longest).
func betterBan(a, b fail2ban.BanPair) bool {
	if (a.Origin == fail2ban.OriginAdmin) != (b.Origin == fail2ban.OriginAdmin) {
		return a.Origin == fail2ban.OriginAdmin
	}
	if a.BanUntil.IsZero() != b.BanUntil.IsZero() {
		return a.BanUntil.IsZero()
	}
	return a.BanUntil.After(b.BanUntil)
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
	cred, credFound := findCredential(creds, credentialName)
	credWeight := 0
	credPriority := 0
	isProxyLike := false
	if credFound {
		credWeight = cred.Weight
		// cred.Priority (not EffectivePriority()) — the balancer's primary-pool grouping
		// keys off the explicit priority: field only (balancer.primaryPriority);
		// fallback_priority is a retry-only knob and must not show here as a primary tier.
		credPriority = cred.Priority
		isProxyLike = cred.IsProxyLike()
	}
	modelWeight := 0
	modelPriority := 0
	hasLearnedPriority := false
	var priorityTiers []httputil.ModelPriorityTier
	realCredential := ""
	locallyBanned := p.balancer.IsBanned(credentialName, modelID)
	if p.modelManager != nil {
		modelWeight = p.modelManager.GetModelWeightForCredential(modelID, credentialName)
		// Dynamic per-model priority is learned from an upstream proxy/AIR credential's
		// own /health poll. Gate the lookup on exactly what balancer.learnedProxyPriority
		// (internal/balancer/weighted.go) checks — proxy-like AND the model checker being
		// enabled — so the dashboard never shows a learned priority the balancer would not
		// actually route on (it falls back to the static field when the checker is off).
		if isProxyLike && p.modelManager.IsEnabled() {
			modelPriority, hasLearnedPriority = p.modelManager.LearnedModelPriorityForCredential(modelID, credentialName)
			// Re-emit the learned per-tier breakdown (Design B) so a router fronting this
			// one keeps the tier structure across the chain hop. Per-tier current usage is
			// the last upstream-poll value; the downstream adds its own local contribution
			// via its aggregate counter, same as the scalar current_rpm above.
			priorityTiers = p.modelManager.GetModelPriorityTiersForCredential(modelID, credentialName)
			if len(priorityTiers) > 0 && locallyBanned {
				// Second hop: this router has locally fail2ban-ed the (credential, model)
				// pair. Fold that into every re-emitted tier — the scalar IsBanned below
				// is not enough, a Design B downstream rebuilds live tier-candidates from
				// PriorityTiers and would keep routing primary traffic at a path we have
				// closed. Copy first: GetModelPriorityTiersForCredential may share backing.
				tiersCopy := make([]httputil.ModelPriorityTier, len(priorityTiers))
				copy(tiersCopy, priorityTiers)
				for i := range tiersCopy {
					tiersCopy[i].Banned = true
				}
				priorityTiers = tiersCopy
			}
		}
		realCredential = p.modelManager.GetModelSourceCredentialForCredential(modelID, credentialName)
	}
	// Falls back to the owning credential's static priority: field when nothing has been
	// learned yet (see LearnedModelPriorityForCredential — a learned 0 is authoritative
	// and does not fall through). Keeps the dashboard's Priority in sync with the number
	// the balancer actually routes on.
	priority := credPriority
	if hasLearnedPriority {
		priority = modelPriority
	}
	ms := stats[modelKey]
	scopes, deniedScopes := expression.LegacyProjection()
	modelsInfo[modelKey] = httputil.ModelHealthStats{
		Credential:      credentialName,
		RealCredential:  realCredential,
		Model:           modelID,
		IsBanned:        locallyBanned,
		Weight:          balancer.EffectiveWeight(modelWeight, credWeight),
		Priority:        priority,
		PriorityTiers:   priorityTiers,
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
	credential, ok := findCredential(creds, credentialName)
	if !ok {
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

// findCredential returns a copy of the credential named credentialName within creds
// (ok=false when not found). Returned by value rather than as a pointer into creds so a
// caller cannot accidentally mutate shared balancer state through it — creds is only a
// shallow snapshot, so its CredentialConfig scalar fields are safe to hand out by value
// but writes through an aliasing pointer would not be.
func findCredential(creds []config.CredentialConfig, credentialName string) (config.CredentialConfig, bool) {
	for i := range creds {
		if creds[i].Name == credentialName {
			return creds[i], true
		}
	}
	return config.CredentialConfig{}, false
}

// VisualHealthCheck serves the static health dashboard HTML.
func (p *Proxy) VisualHealthCheck(w http.ResponseWriter, r *http.Request) {
	webui.ServeHealth(w, r)
}
