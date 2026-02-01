package proxy

import (
	"context"
	"log/slog"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
)

// UpdateStatsFromRemoteProxy fetches and updates RPM/TPM limits from remote /health endpoint
func UpdateStatsFromRemoteProxy(
	ctx context.Context,
	cred *config.CredentialConfig,
	rateLimiter *ratelimit.RPMLimiter,
	logger *slog.Logger,
) {
	// Fetch health data from remote proxy
	var health httputil.ProxyHealthResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/health", logger, &health); err != nil {
		logger.Debug("Failed to fetch remote proxy stats",
			"credential", cred.Name,
			"error", err,
		)
		return
	}

	// Update credential limits from remote credentials
	updateCredentialLimits(&health, cred, rateLimiter, logger)

	// Update model limits from remote models
	updateModelLimits(&health, cred, rateLimiter, logger)
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

	// Aggregate limits and current usage from remote credentials
	maxRPM := -1
	maxTPM := -1
	totalCurrentRPM := 0
	totalCurrentTPM := 0

	for _, credStats := range health.Credentials {
		if credStats.LimitRPM > 0 {
			if maxRPM == -1 || credStats.LimitRPM > maxRPM {
				maxRPM = credStats.LimitRPM
			}
		}
		if credStats.LimitTPM > 0 {
			if maxTPM == -1 || credStats.LimitTPM > maxTPM {
				maxTPM = credStats.LimitTPM
			}
		}
		totalCurrentRPM += credStats.CurrentRPM
		totalCurrentTPM += credStats.CurrentTPM
	}

	logger.Debug("Aggregated credential limits from remote",
		"proxy", cred.Name,
		"credentials_count", len(health.Credentials),
		"max_rpm", maxRPM,
		"max_tpm", maxTPM,
		"total_current_rpm", totalCurrentRPM,
		"total_current_tpm", totalCurrentTPM,
	)

	// Convert 0 (unlimited) to -1 for consistency
	if maxRPM == 0 {
		maxRPM = -1
	}
	if maxTPM == 0 {
		maxTPM = -1
	}

	// Update our proxy credential with aggregated limits (even if both are -1, we still need to sync usage)
	rateLimiter.AddCredentialWithTPM(cred.Name, maxRPM, maxTPM)
	// Sync current usage from remote
	rateLimiter.SetCredentialCurrentUsage(cred.Name, totalCurrentRPM, totalCurrentTPM)
	logger.Debug("Updated proxy credential limits from remote",
		"proxy", cred.Name,
		"rpm_limit", maxRPM,
		"tpm_limit", maxTPM,
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
) {
	if len(health.Models) == 0 {
		return
	}

	// Aggregate limits per model from multiple credentials in remote proxy
	type ModelStats struct {
		limitRPM   int
		limitTPM   int
		currentRPM int
		currentTPM int
	}
	modelStats := make(map[string]ModelStats)

	for _, modelStats_data := range health.Models {
		modelID := modelStats_data.Model

		// Aggregate (sum) limits and current usage for this model
		rpm := modelStats_data.LimitRPM
		tpm := modelStats_data.LimitTPM
		curRPM := modelStats_data.CurrentRPM
		curTPM := modelStats_data.CurrentTPM

		if rpm > 0 || tpm > 0 || curRPM > 0 || curTPM > 0 {
			if existing, ok := modelStats[modelID]; ok {
				// Sum the limits and usage
				if rpm > 0 {
					existing.limitRPM += rpm
				}
				if tpm > 0 {
					existing.limitTPM += tpm
				}
				existing.currentRPM += curRPM
				existing.currentTPM += curTPM
				modelStats[modelID] = existing
			} else {
				modelStats[modelID] = ModelStats{
					limitRPM:   rpm,
					limitTPM:   tpm,
					currentRPM: curRPM,
					currentTPM: curTPM,
				}
			}
		}
	}

	// Update rate limiter with aggregated model limits
	modelsUpdated := 0
	for modelID, stats := range modelStats {
		rpm := stats.limitRPM
		tpm := stats.limitTPM

		// Set -1 if not limited (0 in remote means unlimited)
		if rpm == 0 {
			rpm = -1
		}
		if tpm == 0 {
			tpm = -1
		}

		rateLimiter.AddModelWithTPM(cred.Name, modelID, rpm, tpm)
		// Sync current usage for this model
		if stats.currentRPM > 0 || stats.currentTPM > 0 {
			rateLimiter.SetModelCurrentUsage(cred.Name, modelID, stats.currentRPM, stats.currentTPM)
		}
		modelsUpdated++
	}

	if modelsUpdated > 0 {
		logger.Debug("Updated model limits from remote proxy",
			"proxy", cred.Name,
			"models_updated", modelsUpdated,
		)
	}
}
