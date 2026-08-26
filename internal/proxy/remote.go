package proxy

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// ModelManagerInterface for adding dynamically loaded models
type ModelManagerInterface interface {
	AddModel(credentialName, modelID string)
	SetModelWeightForCredential(modelID, credentialName string, weight int)
	ReplaceModelsForCredential(credentialName string, modelIDs []string)
	ReplaceModelWeightsForCredential(credentialName string, weights map[string]int)
	SetModelPriorityForCredential(modelID, credentialName string, priority int)
	ReplaceModelPrioritiesForCredential(credentialName string, priorities map[string]int)
	ReplaceModelSourceCredentialsForCredential(credentialName string, sourceCredentials map[string]string)
	HasModel(credentialName, modelID string) bool
}

type modelScopeUpdater interface {
	UpdateProviderScopesForCredential(credentialName string, metadata models.ScopeMetadata)
	ReplaceModelScopesForCredential(credentialName string, scopes map[string]models.ScopeMetadata)
}

// UpdateAllFromRemoteHealth polls every proxy/AIR credential's own /health and syncs
// weight, priority, limits, current usage, and (via ModelHealthStats.RealCredential)
// which real leaf credential is actually serving each model — everything
// UpdateStatsFromRemoteProxy/UpdateStatsFromHealth compute below.
//
// This is the counterpart to modelupdate.UpdateAllProxyCredentials, wired
// alongside it in cmd/server/main.go's startProxyStatsUpdater, not a replacement for
// it: that one discovers model *names* through a generic /v1/models-shaped call,
// which works for any OpenAI-compatible upstream (type: proxy) whether or not it is
// itself Auto AI Router. This one only has anything to sync when the upstream
// exposes AIR's own /health JSON shape — for a plain type: proxy upstream that
// doesn't, FetchHealthFromRemoteProxy's fetch just fails and is logged at Debug, so
// calling this unconditionally for every proxy-like credential is safe.
//
// Until 2026-08-27 this whole sync path (UpdateStatsFromRemoteProxy and everything
// under it) was fully implemented and tested but never called from production
// wiring — see the priority-propagation entry in todo_round_robin.md for the story.
func UpdateAllFromRemoteHealth(
	ctx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
	modelManager ModelManagerInterface,
) {
	credentials := bal.GetCredentialsSnapshot()

	var wg sync.WaitGroup
	for i := range credentials {
		cred := &credentials[i]
		if !cred.IsProxyLike() {
			continue
		}
		wg.Add(1)
		go func(c *config.CredentialConfig) {
			defer wg.Done()
			UpdateStatsFromRemoteProxy(ctx, c, rateLimiter, logger, modelManager)
		}(cred)
	}
	wg.Wait()
}

// UpdateStatsFromRemoteProxy fetches and updates RPM/TPM limits from remote /health endpoint
func UpdateStatsFromRemoteProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
	modelManager ModelManagerInterface,
) {
	health, err := FetchHealthFromRemoteProxy(ctx, cred, logger)
	if err != nil {
		return
	}

	UpdateStatsFromHealth(health, cred, rateLimiter, logger, modelManager)
}

// FetchHealthFromRemoteProxy retrieves proxy health data from the /health endpoint.
func FetchHealthFromRemoteProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	logger *slog.Logger,
) (*httputil.ProxyHealthResponse, error) {
	health, _, err := FetchHealthResponseFromRemoteProxy(ctx, cred, logger)
	return health, err
}

// FetchHealthResponseFromRemoteProxy retrieves proxy health data and response
// headers from the /health endpoint.
func FetchHealthResponseFromRemoteProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	logger *slog.Logger,
) (*httputil.ProxyHealthResponse, http.Header, error) {
	var health httputil.ProxyHealthResponse
	body, headers, err := httputil.FetchResponseFromProxy(ctx, cred, "/health", logger)
	if err != nil {
		logger.Debug("Failed to fetch remote proxy stats",
			"credential", cred.Name,
			"error", err,
		)
		return nil, headers, err
	}
	if err := json.Unmarshal(body, &health); err != nil {
		logger.Error("Failed to parse remote proxy health response",
			"credential", cred.Name,
			"error", err,
		)
		return nil, headers, err
	}

	return &health, headers, nil
}

// UpdateStatsFromHealth updates RPM/TPM limits from already-fetched health data.
func UpdateStatsFromHealth(
	health *httputil.ProxyHealthResponse,
	cred *config.CredentialConfig,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
	modelManager ModelManagerInterface,
) {
	// Update credential limits from remote credentials
	updateCredentialLimits(health, cred, rateLimiter, logger)

	// Update model limits from remote models
	updateModelLimits(health, cred, rateLimiter, logger, modelManager)

	updateModelScopes(health, cred, modelManager)
}

func updateModelScopes(
	health *httputil.ProxyHealthResponse,
	cred *config.CredentialConfig,
	modelManager ModelManagerInterface,
) {
	providerScopes := models.AggregateProviderScopesFromHealth(health, cred.IsFallback)
	cred.ProviderScopes = providerScopes.Scopes
	cred.ProviderDeniedScopes = providerScopes.DeniedScopes
	cred.ProviderScopeExpression = scope.NormalizeExpression(providerScopes.ScopeExpression)
	cred.ProviderScopeKnown = true

	if updater, ok := modelManager.(modelScopeUpdater); ok {
		updater.UpdateProviderScopesForCredential(cred.Name, providerScopes)
		updater.ReplaceModelScopesForCredential(cred.Name, models.AggregateModelScopesFromHealth(health, cred.IsFallback))
	}
}

type limitAggregation struct {
	rpm             int
	tpm             int
	weight          int
	priority        int
	hasPriority     bool
	currentRPM      int
	currentTPM      int
	hasUnlimitedRPM bool
	hasUnlimitedTPM bool
	hasLimitOrUsage bool
}

func newSumLimitAggregation() *limitAggregation {
	return &limitAggregation{}
}

func (agg *limitAggregation) applySum(rpm, tpm, currentRPM, currentTPM int) {
	agg.hasLimitOrUsage = true

	if rpm <= 0 {
		agg.hasUnlimitedRPM = true
	} else {
		agg.rpm += rpm
	}

	if tpm <= 0 {
		agg.hasUnlimitedTPM = true
	} else {
		agg.tpm += tpm
	}

	agg.currentRPM += currentRPM
	agg.currentTPM += currentTPM
}

func (agg *limitAggregation) applyWeight(weight int) {
	if weight > 0 {
		agg.weight += weight
	}
}

// applyPriorityMin folds another upstream credential's priority for this model into
// the aggregate. Unlike weight/limits (summed), priority uses MIN: when several
// upstream credentials offer the same model at different priorities (e.g. two grant
// credentials on the same node with different priority groups), the resulting
// proxy-local model priority is the lowest one — the group that would be tried first
// — not a sum or average.
func (agg *limitAggregation) applyPriorityMin(priority int) {
	if !agg.hasPriority || priority < agg.priority {
		agg.priority = priority
		agg.hasPriority = true
	}
}

func (agg *limitAggregation) finalizeLimits() (int, int) {
	rpm := agg.rpm
	tpm := agg.tpm

	if agg.hasUnlimitedRPM || rpm == 0 {
		rpm = -1
	}
	if agg.hasUnlimitedTPM || tpm == 0 {
		tpm = -1
	}

	return rpm, tpm
}

func hasRemoteModelLimitOrUsage(stats httputil.ModelHealthStats) bool {
	return stats.LimitRPM != 0 ||
		stats.LimitTPM != 0 ||
		stats.CurrentRPM > 0 ||
		stats.CurrentTPM > 0
}

// updateCredentialLimits updates credential limits from remote credentials data
func updateCredentialLimits(
	health *httputil.ProxyHealthResponse,
	cred *config.CredentialConfig,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
) {
	if len(health.Credentials) == 0 {
		logger.Debug("No credentials in remote health response",
			"proxy", cred.Name,
		)
		return
	}

	// Aggregate limits and current usage from remote credentials.
	// Use SUM aggregation: proxy's total capacity is the sum of all upstream credentials'
	// RPM/TPM limits (requests are distributed across them via round-robin).
	// Previously used MAX which underestimated capacity, causing false rate limiting
	// when total usage exceeded the highest single credential's limit.
	aggregation := newSumLimitAggregation()

	for _, credStats := range health.Credentials {
		// Previously skipped upstream credentials marked is_fallback here unless our own
		// proxy credential was also is_fallback — that dropped an upstream credential's
		// entire RPM/TPM contribution instead of just deprioritizing it, which made
		// models served only by a fallback-flagged upstream credential vanish from this
		// proxy's aggregated limits. The upstream credential's is_fallback status has no
		// bearing on our local proxy credential's capacity: it always counts toward the
		// aggregate now. Ordering/deprioritization is handled separately by priority
		// (see updateModelLimits below and EffectivePriority()).
		aggregation.applySum(
			credStats.LimitRPM,
			credStats.LimitTPM,
			credStats.CurrentRPM,
			credStats.CurrentTPM,
		)
	}

	totalRPM, totalTPM := aggregation.finalizeLimits()
	totalCurrentRPM := aggregation.currentRPM
	totalCurrentTPM := aggregation.currentTPM

	logger.Debug("Aggregated credential limits from remote",
		"proxy", cred.Name,
		"credentials_count", len(health.Credentials),
		"total_rpm", totalRPM,
		"total_tpm", totalTPM,
		"total_current_rpm", totalCurrentRPM,
		"total_current_tpm", totalCurrentTPM,
	)

	// Update our proxy credential with aggregated limits (even if both are -1, we still need to sync usage)
	rateLimiter.AddCredentialWithTPM(cred.Name, totalRPM, totalTPM)
	// Sync current usage from remote
	rateLimiter.SetCredentialCurrentUsage(cred.Name, totalCurrentRPM, totalCurrentTPM)
	logger.Debug("Updated proxy credential limits from remote",
		"proxy", cred.Name,
		"rpm_limit", totalRPM,
		"tpm_limit", totalTPM,
		"current_rpm", totalCurrentRPM,
		"current_tpm", totalCurrentTPM,
	)
}

// updateModelLimits updates model limits from remote models data
func updateModelLimits(
	health *httputil.ProxyHealthResponse,
	cred *config.CredentialConfig,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
	modelManager ModelManagerInterface,
) {
	if len(health.Models) == 0 {
		if modelManager != nil {
			modelManager.ReplaceModelsForCredential(cred.Name, nil)
			modelManager.ReplaceModelWeightsForCredential(cred.Name, nil)
			modelManager.ReplaceModelPrioritiesForCredential(cred.Name, nil)
			modelManager.ReplaceModelSourceCredentialsForCredential(cred.Name, nil)
		}
		removedModels := removeStaleModelLimits(cred.Name, map[string]bool{}, rateLimiter)
		if removedModels > 0 {
			logger.Debug("Removed stale model limits from remote proxy",
				"proxy", cred.Name,
				"models_removed", removedModels,
			)
		}
		return
	}

	// Aggregate limits per model from multiple credentials in remote proxy
	modelStats := make(map[string]*limitAggregation)
	modelIDs := make([]string, 0, len(health.Models))
	modelIDSet := make(map[string]bool, len(health.Models))
	modelWeights := make(map[string]int)
	modelPriorities := make(map[string]int)
	// Display-only: which real upstream credential is serving each model. When
	// several upstream credentials serve the same model behind this one proxy
	// link, the last one seen wins — same simplification weight/priority
	// aggregation already makes for "which one do we show," not a routing input.
	modelSourceCreds := make(map[string]string)

	for _, modelStats_data := range health.Models {
		credStats, ok := health.Credentials[modelStats_data.Credential]
		if !ok {
			continue
		}
		// Previously skipped upstream credentials marked is_fallback here unless our own
		// proxy credential was also is_fallback — see the matching comment in
		// updateCredentialLimits above. The upstream credential (and its models) always
		// participates in aggregation now; its priority (below) is what pushes it later
		// in the selection order, not exclusion from the model/limit set.
		modelID := modelStats_data.Model
		if modelID == "" {
			continue
		}
		if !modelIDSet[modelID] {
			modelIDSet[modelID] = true
			modelIDs = append(modelIDs, modelID)
		}

		// Aggregate (sum) limits and current usage for this model
		rpm := modelStats_data.LimitRPM
		tpm := modelStats_data.LimitTPM
		curRPM := modelStats_data.CurrentRPM
		curTPM := modelStats_data.CurrentTPM
		weight := httputil.EffectiveHealthWeight(modelStats_data, credStats)
		priority := httputil.EffectiveHealthPriority(modelStats_data, credStats)

		aggregation, ok := modelStats[modelID]
		if !ok {
			aggregation = newSumLimitAggregation()
			modelStats[modelID] = aggregation
		}
		if hasRemoteModelLimitOrUsage(modelStats_data) {
			aggregation.applySum(rpm, tpm, curRPM, curTPM)
		}
		aggregation.applyWeight(weight)
		modelWeights[modelID] = aggregation.weight
		aggregation.applyPriorityMin(priority)
		modelPriorities[modelID] = aggregation.priority
		if modelStats_data.Credential != "" {
			modelSourceCreds[modelID] = modelStats_data.Credential
		}
	}

	if modelManager != nil {
		modelManager.ReplaceModelsForCredential(cred.Name, modelIDs)
		modelManager.ReplaceModelWeightsForCredential(cred.Name, modelWeights)
		modelManager.ReplaceModelPrioritiesForCredential(cred.Name, modelPriorities)
		modelManager.ReplaceModelSourceCredentialsForCredential(cred.Name, modelSourceCreds)
	}

	// Update rate limiter with aggregated model limits
	modelsUpdated := 0
	limitedModelIDs := make(map[string]bool, len(modelStats))
	for modelID, stats := range modelStats {
		if !stats.hasLimitOrUsage {
			continue
		}
		rpm, tpm := stats.finalizeLimits()

		rateLimiter.AddModelWithTPM(cred.Name, modelID, rpm, tpm)
		limitedModelIDs[modelID] = true
		// Sync current usage for this model
		if stats.currentRPM > 0 || stats.currentTPM > 0 {
			rateLimiter.SetModelCurrentUsage(cred.Name, modelID, stats.currentRPM, stats.currentTPM)
		}

		modelsUpdated++
	}
	removedModels := removeStaleModelLimits(cred.Name, limitedModelIDs, rateLimiter)

	if modelsUpdated > 0 || removedModels > 0 {
		logger.Debug("Updated model limits from remote proxy",
			"proxy", cred.Name,
			"models_updated", modelsUpdated,
			"models_removed", removedModels,
		)
	}
}

func removeStaleModelLimits(credentialName string, currentModelLimits map[string]bool, rateLimiter *ratelimit.RPMLimiter) int {
	removed := 0
	for _, pair := range rateLimiter.GetAllModelPairs() {
		if pair.Credential != credentialName || currentModelLimits[pair.Model] {
			continue
		}
		rateLimiter.RemoveModel(pair.Credential, pair.Model)
		removed++
	}
	return removed
}
