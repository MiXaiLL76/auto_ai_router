package proxy

import (
	_ "embed"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/httputil"
)

func (p *Proxy) HealthCheck() (bool, *httputil.ProxyHealthResponse) {
	totalCreds := len(p.balancer.GetCredentials())
	availableCreds := p.balancer.GetAvailableCount()
	bannedCreds := p.balancer.GetBannedCount()

	healthy := availableCreds > 0

	// Collect credentials info
	credentialsInfo := make(map[string]httputil.CredentialHealthStats)
	for _, cred := range p.balancer.GetCredentials() {
		// For proxy credentials, get limits from rateLimiter (updated by UpdateStatsFromRemoteProxy)
		// For other credentials, use config values
		limitRPM := cred.RPM
		limitTPM := cred.TPM
		if cred.Type == "proxy" {
			rateLimiterRPM := p.rateLimiter.GetLimitRPM(cred.Name)
			rateLimiterTPM := p.rateLimiter.GetLimitTPM(cred.Name)
			if rateLimiterRPM != -1 {
				limitRPM = rateLimiterRPM
			}
			if rateLimiterTPM != -1 {
				limitTPM = rateLimiterTPM
			}
		}

		credentialsInfo[cred.Name] = httputil.CredentialHealthStats{
			Type:       string(cred.Type),
			IsFallback: cred.IsFallback,
			CurrentRPM: p.rateLimiter.GetCurrentRPM(cred.Name),
			CurrentTPM: p.rateLimiter.GetCurrentTPM(cred.Name),
			LimitRPM:   limitRPM,
			LimitTPM:   limitTPM,
		}
	}

	// Collect models info from rateLimiter (which tracks all credential:model pairs)
	modelsInfo := make(map[string]httputil.ModelHealthStats)

	// Get all tracked credential:model pairs from rateLimiter
	// This includes duplicates when same model is available from different credentials
	allTrackedModels := p.rateLimiter.GetAllModels()
	for _, modelKey := range allTrackedModels {
		// Parse credential:model format
		parts := strings.Split(modelKey, ":")
		if len(parts) != 2 {
			continue
		}
		credName := parts[0]
		modelID := parts[1]

		modelsInfo[modelKey] = httputil.ModelHealthStats{
			Credential: credName,
			Model:      modelID,
			CurrentRPM: p.rateLimiter.GetCurrentModelRPM(credName, modelID),
			CurrentTPM: p.rateLimiter.GetCurrentModelTPM(credName, modelID),
			LimitRPM:   p.rateLimiter.GetModelLimitRPM(credName, modelID),
			LimitTPM:   p.rateLimiter.GetModelLimitTPM(credName, modelID),
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
