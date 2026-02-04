package proxy

import (
	"context"
	"log/slog"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
)

// ModelManagerInterface for adding dynamically loaded models
type ModelManagerInterface interface {
	AddModel(credentialName, modelID string)
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
	var health httputil.ProxyHealthResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/health", logger, &health); err != nil {
		logger.Debug("Failed to fetch remote proxy stats",
			"credential", cred.Name,
			"error", err,
		)
		return nil, err
	}

	return &health, nil
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
}

type limitAggregation struct {
	rpm             int
	tpm             int
	currentRPM      int
	currentTPM      int
	hasUnlimitedRPM bool
	hasUnlimitedTPM bool
}

func newMaxLimitAggregation() *limitAggregation {
	return &limitAggregation{
		rpm: -1,
		tpm: -1,
	}
}

func newSumLimitAggregation() *limitAggregation {
	return &limitAggregation{}
}

func (agg *limitAggregation) applyMax(rpm, tpm, currentRPM, currentTPM int) {
	if rpm == 0 {
		agg.hasUnlimitedRPM = true
	} else if rpm > 0 {
		if agg.rpm == -1 || rpm > agg.rpm {
			agg.rpm = rpm
		}
	}

	if tpm == 0 {
		agg.hasUnlimitedTPM = true
	} else if tpm > 0 {
		if agg.tpm == -1 || tpm > agg.tpm {
			agg.tpm = tpm
		}
	}

	agg.currentRPM += currentRPM
	agg.currentTPM += currentTPM
}

func (agg *limitAggregation) applySum(rpm, tpm, currentRPM, currentTPM int) {
	if rpm == 0 {
		agg.hasUnlimitedRPM = true
	} else if rpm > 0 {
		agg.rpm += rpm
	}

	if tpm == 0 {
		agg.hasUnlimitedTPM = true
	} else if tpm > 0 {
		agg.tpm += tpm
	}

	agg.currentRPM += currentRPM
	agg.currentTPM += currentTPM
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
	aggregation := newMaxLimitAggregation()

	for _, credStats := range health.Credentials {
		aggregation.applyMax(
			credStats.LimitRPM,
			credStats.LimitTPM,
			credStats.CurrentRPM,
			credStats.CurrentTPM,
		)
	}

	maxRPM, maxTPM := aggregation.finalizeLimits()
	totalCurrentRPM := aggregation.currentRPM
	totalCurrentTPM := aggregation.currentTPM

	logger.Debug("Aggregated credential limits from remote",
		"proxy", cred.Name,
		"credentials_count", len(health.Credentials),
		"max_rpm", maxRPM,
		"max_tpm", maxTPM,
		"total_current_rpm", totalCurrentRPM,
		"total_current_tpm", totalCurrentTPM,
	)

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
	modelManager ModelManagerInterface,
) {
	if len(health.Models) == 0 {
		return
	}

	// Aggregate limits per model from multiple credentials in remote proxy
	modelStats := make(map[string]*limitAggregation)

	for _, modelStats_data := range health.Models {
		modelID := modelStats_data.Model

		// Aggregate (sum) limits and current usage for this model
		rpm := modelStats_data.LimitRPM
		tpm := modelStats_data.LimitTPM
		curRPM := modelStats_data.CurrentRPM
		curTPM := modelStats_data.CurrentTPM

		if rpm > 0 || tpm > 0 || curRPM > 0 || curTPM > 0 {
			aggregation, ok := modelStats[modelID]
			if !ok {
				aggregation = newSumLimitAggregation()
				modelStats[modelID] = aggregation
			}
			aggregation.applySum(rpm, tpm, curRPM, curTPM)
		}
	}

	// Update rate limiter with aggregated model limits
	modelsUpdated := 0
	for modelID, stats := range modelStats {
		rpm, tpm := stats.finalizeLimits()

		rateLimiter.AddModelWithTPM(cred.Name, modelID, rpm, tpm)
		// Sync current usage for this model
		if stats.currentRPM > 0 || stats.currentTPM > 0 {
			rateLimiter.SetModelCurrentUsage(cred.Name, modelID, stats.currentRPM, stats.currentTPM)
		}

		// Add model to modelManager for dynamic model filtering
		if modelManager != nil {
			modelManager.AddModel(cred.Name, modelID)
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
