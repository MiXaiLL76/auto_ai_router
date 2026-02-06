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
func UpdateAllProxyCredentials(
	ctx context.Context,
	bal *balancer.RoundRobin,
	rateLimiter *ratelimit.RPMLimiter,
	log *slog.Logger,
	modelManager *models.Manager,
	updateMutex *sync.Mutex,
) {
	// Get all proxy credentials
	credentials := bal.GetCredentialsSnapshot()
	proxyCredentials := make([]*config.CredentialConfig, 0)

	for i, cred := range credentials {
		if cred.Type == config.ProviderTypeProxy {
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

			// Fetch models from proxy
			remoteModels := modelManager.GetRemoteModels(c)

			resultsChan <- proxyResult{
				credential: c,
				models:     remoteModels,
				err:        nil,
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

	updateMutex.Lock()
	defer updateMutex.Unlock()

	for result := range resultsChan {
		if result.err != nil {
			log.Warn("Failed to fetch models from proxy",
				"credential", result.credential.Name,
				"error", result.err,
			)
			failedCount++
			continue
		}

		// Update rate limiter with fetched models
		addedCount := 0
		for _, model := range result.models {
			// Get default RPM/TPM from model manager
			modelRPM := modelManager.GetModelRPMForCredential(model.ID, result.credential.Name)
			modelTPM := modelManager.GetModelTPMForCredential(model.ID, result.credential.Name)

			// AddModelWithTPM handles duplicates internally (overwrites existing)
			rateLimiter.AddModelWithTPM(result.credential.Name, model.ID, modelRPM, modelTPM)
			addedCount++
		}

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
