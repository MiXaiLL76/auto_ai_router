package proxy

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"slices"

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
	// ReplaceModelPriorityTiersForCredential replaces the per-priority-tier breakdown a
	// proxy/AIR credential learned from its upstream /health (Design B). A model with
	// fewer than two distinct tiers is omitted — callers treat it as one implicit tier
	// from the scalar priority set via ReplaceModelPrioritiesForCredential.
	ReplaceModelPriorityTiersForCredential(credentialName string, tiers map[string][]httputil.ModelPriorityTier)
	ReplaceModelSourceCredentialsForCredential(credentialName string, sourceCredentials map[string]string)
	HasModel(credentialName, modelID string) bool
}

type modelScopeUpdater interface {
	UpdateProviderScopesForCredential(credentialName string, metadata models.ScopeMetadata)
	ReplaceModelScopesForCredential(credentialName string, scopes map[string]models.ScopeMetadata)
}

// UpdateStatsFromRemoteProxy fetches and updates RPM/TPM limits from remote /health
// endpoint in one call — kept as a single-credential convenience for tests and any
// future caller that doesn't need the fetch/write split the periodic poller uses;
// production's periodic sync no longer calls this directly (see UpdateStatsFromHealth).
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

// UpdateStatsFromHealth is the single sync point for everything a proxy/AIR credential
// learns from its upstream's own /health JSON: aggregated RPM/TPM limits, current usage,
// per-model weight/priority, provider/model scopes, and which real leaf credential is
// serving each model. It runs once per cycle from the sole periodic poller,
// modelupdate.UpdateAllProxyCredentials, fed the same /health response that populated the
// model-name snapshot (models.Manager.CachedRemoteHealth) so each upstream is polled once.
// A plain type: proxy upstream that doesn't expose AIR's /health shape has no health
// cached and this is simply not called for it.
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
	// is_fallback rules mirror the old model-snapshot path (models.fetchRemoteModelsFromHealth):
	// for a non-fallback connection, upstream credentials marked is_fallback are excluded
	// from aggregation (updateModelLimits below drops their models from the set entirely,
	// so there is nothing to scope), and for a fallback connection every upstream
	// credential is included as a last resort. Passing `true` unconditionally here (a) let
	// an unrestricted is_fallback upstream credential OR its lack-of-scope into the
	// aggregate and erase a restricted primary credential's scope, and (b) diverged from
	// the old path, which re-applies its own snapshot every 30s and would flip the state
	// back. Both paths now use the same rule.
	includeFallback := cred.IsFallback
	providerScopes := models.AggregateProviderScopesFromHealth(health, includeFallback)
	cred.ProviderScopes = providerScopes.Scopes
	cred.ProviderDeniedScopes = providerScopes.DeniedScopes
	cred.ProviderScopeExpression = scope.NormalizeExpression(providerScopes.ScopeExpression)
	cred.ProviderScopeKnown = true

	if updater, ok := modelManager.(modelScopeUpdater); ok {
		updater.UpdateProviderScopesForCredential(cred.Name, providerScopes)
		updater.ReplaceModelScopesForCredential(cred.Name, models.AggregateModelScopesFromHealth(health, includeFallback))
	}
}

type limitAggregation struct {
	rpm          int
	tpm          int
	weight       int
	priority     int
	priorityHigh int
	hasPriority  bool
	// priorityWorst/hasPriorityWorst track the highest priority seen across ALL entries
	// for this model, non-live entries included — the fallback used when no entry is live
	// (hasPriority ends up false). See trackWorstPriority's doc comment.
	priorityWorst    int
	hasPriorityWorst bool
	currentRPM       int
	currentTPM       int
	hasUnlimitedRPM  bool
	hasUnlimitedTPM  bool
	hasLimitOrUsage  bool
}

func newSumLimitAggregation() *limitAggregation {
	return &limitAggregation{}
}

// tierAccumulator sums one primary-priority tier's worth of upstream leaf credentials
// (Design B). agg reuses the SUM/unlimited handling of limitAggregation. anyNotBanned
// records whether at least one contributor is not *actually* banned. The emitted tier's
// Banned flag means only a real ban (every contributor banned) — RPM/TPM saturation is
// left to the tier's Current/Limit fields so the balancer can tell a 429 (rate-limited)
// from a 503 (banned).
type tierAccumulator struct {
	agg          *limitAggregation
	anyNotBanned bool
}

// accumulateTier folds one upstream leaf (or sub-tier) into the per-priority bucket.
// banned is the contributor's *real* ban state (ModelHealthStats.IsBanned /
// ModelPriorityTier.Banned), not its live status — a contributor that is merely
// RPM/TPM-saturated is not banned.
func accumulateTier(byPriority map[int]*tierAccumulator, priority, weight, rpm, tpm, curRPM, curTPM int, banned bool) {
	ta := byPriority[priority]
	if ta == nil {
		ta = &tierAccumulator{agg: newSumLimitAggregation()}
		byPriority[priority] = ta
	}
	// Capacity/usage always fold in — a saturated or banned tier still has real capacity
	// that will free up, and the balancer needs the cumulative cap to be right.
	ta.agg.applySum(rpm, tpm, curRPM, curTPM)
	ta.agg.applyWeight(weight)
	if !banned {
		ta.anyNotBanned = true
	}
}

// buildModelPriorityTiers turns the per-priority accumulators for one model into a
// sorted []ModelPriorityTier. It returns nil for the common single-tier case (the caller
// keeps the model on the scalar priority path) — EXCEPT when that sole tier is banned:
// scalar priority carries no ban state, so a single banned upstream tier would leave this
// credential a live candidate here and on the next router in the chain until local
// fail2ban trips. A lone banned tier is surfaced so the balancer drops it.
func buildModelPriorityTiers(byPriority map[int]*tierAccumulator) []httputil.ModelPriorityTier {
	if len(byPriority) == 0 {
		return nil
	}
	tiers := make([]httputil.ModelPriorityTier, 0, len(byPriority))
	for p, ta := range byPriority {
		lrpm, ltpm := ta.agg.finalizeLimits()
		tiers = append(tiers, httputil.ModelPriorityTier{
			Priority:   p,
			Weight:     ta.agg.weight,
			LimitRPM:   lrpm,
			LimitTPM:   ltpm,
			CurrentRPM: ta.agg.currentRPM,
			CurrentTPM: ta.agg.currentTPM,
			Banned:     !ta.anyNotBanned,
		})
	}
	if len(tiers) < 2 && !tiers[0].Banned {
		return nil
	}
	slices.SortFunc(tiers, func(a, b httputil.ModelPriorityTier) int { return a.Priority - b.Priority })
	return tiers
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
//
// Callers must only invoke this for upstream credentials that can actually serve the
// model right now (see updateModelLimits, which folds in only httputil.ModelHealthEntryLive
// entries — not banned and not RPM/TPM-exhausted): a down or saturated upstream's static
// priority has no business dragging the aggregate down via MIN when the upstream has in
// fact already cascaded past it. This function itself has no notion of live status — it
// folds in whatever priority it's given — so the filtering lives entirely in the caller.
//
// Also tracks the highest live priority seen (priorityHigh) alongside the lowest
// (priority): the gap between them is exactly the risk updateModelLimits warns about
// below — if the credential currently holding the low end ever stops being live
// (banned, saturated, removed), the aggregate jumps straight to priorityHigh with no
// intermediate warning at that moment, since a ticking sync loses no history.
func (agg *limitAggregation) applyPriorityMin(priority int) {
	if !agg.hasPriority || priority < agg.priority {
		agg.priority = priority
	}
	if !agg.hasPriority || priority > agg.priorityHigh {
		agg.priorityHigh = priority
	}
	agg.hasPriority = true
}

// trackWorstPriority records the highest priority seen across ALL entries for this
// model, non-live (banned / saturated) included — unlike applyPriorityMin, callers must
// call this unconditionally, before any live-status filtering. It is the fallback the
// caller exposes when no entry is live at all, so the proxy credential still lands in a
// real (worst-case) tier rather than defaulting to the tried-first group.
func (agg *limitAggregation) trackWorstPriority(priority int) {
	if !agg.hasPriorityWorst || priority > agg.priorityWorst {
		agg.priorityWorst = priority
		agg.hasPriorityWorst = true
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

	for credName, credStats := range health.Credentials {
		// For a non-fallback connection, skip upstream credentials marked is_fallback:
		// their RPM/TPM is reserved capacity for fallback traffic, and folding it into
		// this proxy credential's primary ceiling lets the client rate limiter admit
		// primary traffic past what the upstream will actually serve on its primary
		// credentials (upstream then 429s / burns reserved fallback capacity). For a
		// fallback connection, include every upstream credential. Mirrors the old
		// model-snapshot path's rule so the two 30s pollers agree.
		if !cred.IsFallback && credStats.IsFallback {
			logger.Debug("Skipping is_fallback upstream credential in primary limit aggregation",
				"proxy", cred.Name,
				"upstream_credential", credName,
			)
			continue
		}
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
			modelManager.ReplaceModelPriorityTiersForCredential(cred.Name, nil)
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
	// Design B: per-priority-tier accumulation. modelID -> priority -> summed capacity of
	// the upstream leaf credentials serving that model at that priority. Emitted as
	// []ModelPriorityTier only when a model spans >= 2 distinct priorities behind this one
	// proxy credential; the single-tier case stays on the scalar path above.
	modelTierAcc := make(map[string]map[int]*tierAccumulator)
	// Display-only: which real upstream credential is serving each model. When
	// several upstream credentials serve the same model behind this one proxy
	// link, the last one seen wins — same simplification weight/priority
	// aggregation already makes for "which one do we show," not a routing input.
	modelSourceCreds := make(map[string]string)

	for _, modelStatsData := range health.Models {
		credStats, ok := health.Credentials[modelStatsData.Credential]
		if !ok {
			continue
		}
		// For a non-fallback connection: skip upstream credentials marked is_fallback
		// (reserved for fallback traffic, must not serve primary requests). For a
		// fallback connection: include ALL upstream credentials as a last resort.
		// Mirrors the old model-snapshot path (models.fetchRemoteModelsFromHealth) so
		// the two 30s pollers write the same model set / weights / priorities.
		if !cred.IsFallback && credStats.IsFallback {
			continue
		}
		modelID := modelStatsData.Model
		if modelID == "" {
			continue
		}
		if !modelIDSet[modelID] {
			modelIDSet[modelID] = true
			modelIDs = append(modelIDs, modelID)
		}

		// Aggregate (sum) limits and current usage for this model
		rpm := modelStatsData.LimitRPM
		tpm := modelStatsData.LimitTPM
		curRPM := modelStatsData.CurrentRPM
		curTPM := modelStatsData.CurrentTPM
		weight := httputil.EffectiveHealthWeight(modelStatsData, credStats)
		priority := httputil.EffectiveHealthPriority(modelStatsData, credStats)

		aggregation, ok := modelStats[modelID]
		if !ok {
			aggregation = newSumLimitAggregation()
			modelStats[modelID] = aggregation
		}
		if hasRemoteModelLimitOrUsage(modelStatsData) {
			aggregation.applySum(rpm, tpm, curRPM, curTPM)
		}
		aggregation.applyWeight(weight)
		modelWeights[modelID] = aggregation.weight

		// Per-tier accumulation. If this upstream entry itself carries a tier breakdown
		// (a proxy-of-proxy hop — the upstream is also fronting a multi-tier node), fold
		// each of its tiers in at its own priority; otherwise the entry is one implicit
		// tier at EffectiveHealthPriority.
		byP := modelTierAcc[modelID]
		if byP == nil {
			byP = make(map[int]*tierAccumulator)
			modelTierAcc[modelID] = byP
		}
		if len(modelStatsData.PriorityTiers) > 0 {
			for _, st := range modelStatsData.PriorityTiers {
				accumulateTier(byP, st.Priority, st.Weight, st.LimitRPM, st.LimitTPM, st.CurrentRPM, st.CurrentTPM, st.Banned)
			}
		} else {
			accumulateTier(byP, priority, weight, rpm, tpm, curRPM, curTPM, modelStatsData.IsBanned)
		}

		// Unconditional, non-live entries included — see trackWorstPriority's doc comment.
		aggregation.trackWorstPriority(priority)
		// Fold this entry's priority into the MIN aggregation only when the upstream
		// credential can actually serve this model right now — not banned AND not
		// RPM/TPM-exhausted (httputil.ModelHealthEntryLive, mirroring the webui's
		// isRowLive). A saturated cheap tier must not keep pinning this proxy credential's
		// exposed priority to its lowest value while the upstream has in fact already
		// cascaded to a pricier tier; excluding it lets MIN rise to whichever tier is
		// live, so the local balancer can prefer a mid-priced alternative instead of
		// over-sending to a proxy that only looks cheap. Scoped to priority only:
		// rpm/tpm/current-usage (applySum) and weight (applyWeight) still fold in every
		// entry — total capacity and share don't hinge on "which tier is tried first".
		if httputil.ModelHealthEntryLive(modelStatsData) {
			aggregation.applyPriorityMin(priority)
		}
		if modelStatsData.Credential != "" {
			modelSourceCreds[modelID] = modelStatsData.Credential
		}
	}

	// Resolve each model's exposed priority now that every entry has been folded in,
	// and surface a heterogeneous priority group before it turns into a silent cost
	// surprise.
	for modelID, stats := range modelStats {
		switch {
		case stats.hasPriority:
			// At least one live (not banned, not saturated) entry — MIN over the live
			// entries, i.e. the tier the upstream is actually serving from right now.
			modelPriorities[modelID] = stats.priority

			if stats.priority != stats.priorityHigh {
				// Debug, not Info: a proxy credential whose upstream spans several priority
				// groups for one model is a fully-supported config (it expands into local
				// priority tiers, capacity enforced per tier), and this fires every poll —
				// once per such model, plus unconditionally for every is_fallback proxy
				// credential (pinned to group 999 vs the primary 0). lowest/highest are the
				// MIN and MAX over the currently-live tiers; with 3+ live tiers the priority
				// after the cheapest dies is the second-lowest, not highest_live_priority.
				logger.Debug("Model has upstream credentials in different priority groups behind one proxy credential — expanded into local priority tiers, capacity enforced per tier",
					"proxy", cred.Name,
					"model", modelID,
					"lowest_live_priority", stats.priority,
					"highest_live_priority", stats.priorityHigh,
				)
			}
		case stats.hasPriorityWorst:
			// No entry for this model is live right now (all banned or saturated) — expose
			// the worst (highest) priority ever seen rather than leaving the model out of
			// modelPriorities, which ReplaceModelPrioritiesForCredential reads as
			// "nothing learned" and falls back to this proxy credential's own static
			// priority: field — commonly 0, the *best* group. See trackWorstPriority.
			modelPriorities[modelID] = stats.priorityWorst
		}
	}

	modelTiers := make(map[string][]httputil.ModelPriorityTier)
	for modelID, byPriority := range modelTierAcc {
		if tiers := buildModelPriorityTiers(byPriority); tiers != nil {
			modelTiers[modelID] = tiers
		}
	}

	if modelManager != nil {
		modelManager.ReplaceModelsForCredential(cred.Name, modelIDs)
		modelManager.ReplaceModelWeightsForCredential(cred.Name, modelWeights)
		modelManager.ReplaceModelPrioritiesForCredential(cred.Name, modelPriorities)
		modelManager.ReplaceModelPriorityTiersForCredential(cred.Name, modelTiers)
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
