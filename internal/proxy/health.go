package proxy

import (
	_ "embed"
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
)

func (p *Proxy) HealthCheck() (bool, *httputil.ProxyHealthResponse) {
	creds := p.balancer.GetCredentialsSnapshot()
	totalCreds := len(creds)
	availableCreds := p.balancer.GetAvailableCount()
	bannedCreds := p.balancer.GetBannedCount()

	healthy := availableCreds > 0

	// Collect credentials info
	credentialsInfo := make(map[string]httputil.CredentialHealthStats)
	if creds == nil {
		creds = []config.CredentialConfig{}
	}
	for _, cred := range creds {
		// For proxy credentials, get limits from rateLimiter (updated by UpdateStatsFromRemoteProxy)
		// For other credentials, use config values
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

		// Check if credential is banned from balancer
		isBanned := p.balancer.IsBanned(cred.Name)

		credentialsInfo[cred.Name] = httputil.CredentialHealthStats{
			Type:       string(cred.Type),
			IsFallback: cred.IsFallback,
			IsBanned:   isBanned,
			CurrentRPM: p.rateLimiter.GetCurrentRPM(cred.Name),
			CurrentTPM: p.rateLimiter.GetCurrentTPM(cred.Name),
			LimitRPM:   limitRPM,
			LimitTPM:   limitTPM,
		}
	}

	// Collect models info from rateLimiter (which tracks all credential:model pairs)
	modelsInfo := make(map[string]httputil.ModelHealthStats)

	// Get all tracked credential:model pairs from rateLimiter (pre-parsed)
	// This includes duplicates when same model is available from different credentials
	allTrackedModels := p.rateLimiter.GetAllModelPairs()
	for _, pair := range allTrackedModels {
		modelKey := pair.Credential + ":" + pair.Model
		modelsInfo[modelKey] = httputil.ModelHealthStats{
			Credential: pair.Credential,
			Model:      pair.Model,
			CurrentRPM: p.rateLimiter.GetCurrentModelRPM(pair.Credential, pair.Model),
			CurrentTPM: p.rateLimiter.GetCurrentModelTPM(pair.Credential, pair.Model),
			LimitRPM:   p.rateLimiter.GetModelLimitRPM(pair.Credential, pair.Model),
			LimitTPM:   p.rateLimiter.GetModelLimitTPM(pair.Credential, pair.Model),
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

// VisualHealthCheck renders an HTML dashboard with health check information
func (p *Proxy) VisualHealthCheck(w http.ResponseWriter, r *http.Request) {
	_, status := p.HealthCheck()

	if p.healthTemplate == nil {
		p.logger.Error("Health template not available")
		http.Error(w, "Internal Server Error: Template not available", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)

	// Convert to map[string]interface{} for template compatibility
	statusMap := map[string]interface{}{
		"status":                status.Status,
		"credentials_available": status.CredentialsAvailable,
		"credentials_banned":    status.CredentialsBanned,
		"total_credentials":     status.TotalCredentials,
		"credentials":           status.Credentials,
		"models":                status.Models,
	}

	if err := p.healthTemplate.Execute(w, statusMap); err != nil {
		p.logger.Error("Failed to execute health template", "error", err)
	}
}
