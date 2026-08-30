// Package modelupdate periodically refreshes model data from external sources.
package modelupdate

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
)

// ProxyHealthSync syncs a proxy/AIR credential's aggregated limits, current usage,
// per-model weight/priority, scopes and model set from the upstream /health response that
// RefreshRemoteModelsWithError just fetched and cached (models.Manager.CachedRemoteHealth).
// It returns true when it handled the credential (an AIR-shaped /health was cached), in
// which case UpdateAllProxyCredentials skips its legacy per-model default-limit pass.
//
// Injected as a callback (wired to proxy.UpdateStatsFromHealth in cmd/server) rather
// than called directly so this package does not import internal/proxy — internal/proxy's
// own tests import this package, which would be a cycle.
type ProxyHealthSync func(cred *config.CredentialConfig) (handled bool)

// UpdateAllProxyCredentials fetches the latest models from all proxy credentials
// and updates the balancer, rate limiter, and model manager with the results.
// This function is designed to be called periodically in a background goroutine.
//
// Parameters:
//   - ctx: Context for cancellation
//   - bal: Balancer for credential management
//   - rateLimiter: Rate limiter for model tracking
//   - log: Logger for operation details
//   - modelManager: Model manager for storing fetched models
//   - updateMutex: Synchronizes updates (prevents race conditions with metrics)
//   - syncFromHealth: optional; when set, owns limit/weight/priority/scope sync for
//     AIR-shaped upstreams (see ProxyHealthSync). nil disables health-stats sync.
func UpdateAllProxyCredentials(
	ctx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	log *slog.Logger,
	modelManager *models.Manager,
	updateMutex *sync.Mutex,
	syncFromHealth ProxyHealthSync,
) {
	// Get all proxy credentials
	credentials := bal.GetCredentialsSnapshot()
	proxyCredentials := make([]*config.CredentialConfig, 0)

	for i, cred := range credentials {
		if cred.IsProxyLike() {
			proxyCredentials = append(proxyCredentials, &credentials[i])
		}
	}

	if len(proxyCredentials) == 0 {
		return
	}

	// Fetch models from each proxy concurrently
	type proxyResult struct {
		credential *config.CredentialConfig
		models     []models.Model
		err        error
	}

	resultsChan := make(chan proxyResult, len(proxyCredentials))

	var wg sync.WaitGroup
	for _, cred := range proxyCredentials {
		wg.Add(1)
		go func(c *config.CredentialConfig) {
			defer wg.Done()

			remoteModels, err := modelManager.RefreshRemoteModelsWithError(ctx, c)
			modelManager.CopyProviderScopeMetadata(c)

			resultsChan <- proxyResult{
				credential: c,
				models:     remoteModels,
				err:        err,
			}
		}(cred)
	}

	// Close results channel when all goroutines complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Process results
	updatedCount := 0
	failedCount := 0

	for result := range resultsChan {
		if result.err != nil {
			bal.UpdateProviderScopes(
				*result.credential,
				result.credential.ProviderScopes,
				result.credential.ProviderDeniedScopes,
				result.credential.ProviderScopeExpression,
				result.credential.ProviderScopeKnown,
			)
			log.Warn("Failed to fetch models from proxy",
				"credential", result.credential.Name,
				"error", result.err,
			)
			failedCount++
			continue
		}

		addedCount := 0

		updateMutex.Lock()
		if syncFromHealth != nil && syncFromHealth(result.credential) {
			// AIR-shaped upstream: syncFromHealth (proxy.UpdateStatsFromHealth) is the
			// single sync point for this credential's model set, weights, priorities,
			// scopes, aggregated RPM/TPM limits and current usage — all derived from the
			// same /health response that RefreshRemoteModelsWithError just fetched and cached.
			// This is the only periodic poller now; proxy.UpdateAllFromRemoteHealth was
			// removed.
			addedCount = len(result.models)
		} else {
			if syncFromHealth != nil {
				// This credential's upstream is no longer serving an AIR-shaped /health
				// (legacy 404/405 fallback to /v1/models — a genuine non-AIR downgrade, not
				// a transient error: those set result.err and are handled above with
				// continue, leaving the last snapshot intact). Clear the frozen per-model
				// tier / priority / weight snapshot so the balancer stops expanding this
				// credential into stale (possibly Banned) priority tiers and stale
				// cumulative caps that would otherwise live until process restart.
				modelManager.ReplaceModelPriorityTiersForCredential(result.credential.Name, nil)
				modelManager.ReplaceModelPrioritiesForCredential(result.credential.Name, nil)
				modelManager.ReplaceModelWeightsForCredential(result.credential.Name, nil)
			}
			// Legacy /v1/models fallback (non-AIR upstream): only model names were
			// discovered, so apply model-manager default RPM/TPM and prune names that
			// disappeared upstream.
			modelIDSet := make(map[string]bool, len(result.models))
			for _, model := range result.models {
				if model.ID == "" || modelIDSet[model.ID] {
					continue
				}
				modelIDSet[model.ID] = true
			}
			for _, pair := range rateLimiter.GetAllModelPairs() {
				if pair.Credential != result.credential.Name || modelIDSet[pair.Model] {
					continue
				}
				if modelManager.HasModel(pair.Credential, pair.Model) {
					continue
				}
				rateLimiter.RemoveModel(pair.Credential, pair.Model)
			}
			for _, model := range result.models {
				modelRPM := modelManager.GetModelRPMForCredential(model.ID, result.credential.Name)
				modelTPM := modelManager.GetModelTPMForCredential(model.ID, result.credential.Name)
				rateLimiter.AddModelWithTPM(result.credential.Name, model.ID, modelRPM, modelTPM)
				addedCount++
			}
		}
		updateMutex.Unlock()

		// Push the provider scope this credential learned from its upstream /health into
		// the live balancer state after UpdateStatsFromHealth has run, so scope-based
		// routing / VisibleTo() see the current value.
		bal.UpdateProviderScopes(
			*result.credential,
			result.credential.ProviderScopes,
			result.credential.ProviderDeniedScopes,
			result.credential.ProviderScopeExpression,
			result.credential.ProviderScopeKnown,
		)

		if addedCount > 0 {
			log.Info("Updated proxy models",
				"credential", result.credential.Name,
				"added_models", addedCount,
				"total_models", len(result.models),
			)
			updatedCount++
		}
	}

	if failedCount > 0 {
		log.Warn("Proxy model update completed with failures",
			"total_proxies", len(proxyCredentials),
			"updated", updatedCount,
			"failed", failedCount,
		)
	} else {
		log.Debug("Proxy model update completed",
			"total_proxies", len(proxyCredentials),
			"updated", updatedCount,
		)
	}
}

// SplitCredentialModel parses a "credential:model" format string.
// Returns a slice of two strings: [credential, model].
// If the model name contains colons (e.g., "gpt-4o:turbo"), it splits on the first colon only.
// If the format is invalid, returns the entire string as a single element.
func SplitCredentialModel(key string) []string {
	// Use SplitN to split only on the first colon
	// This allows model names to contain colons
	parts := strings.SplitN(key, ":", 2)
	if len(parts) == 2 {
		return parts
	}
	// Fallback for unexpected format (no colon found)
	return []string{key}
}
