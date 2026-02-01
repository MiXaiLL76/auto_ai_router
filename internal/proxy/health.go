package proxy

import (
	_ "embed"
	"fmt"
	"net/http"

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
		credentialsInfo[cred.Name] = httputil.CredentialHealthStats{
			Type:       string(cred.Type),
			IsFallback: cred.IsFallback,
			CurrentRPM: p.rateLimiter.GetCurrentRPM(cred.Name),
			CurrentTPM: p.rateLimiter.GetCurrentTPM(cred.Name),
			LimitRPM:   cred.RPM,
			LimitTPM:   cred.TPM,
		}
	}

	// Collect models info from all configured models
	modelsInfo := make(map[string]httputil.ModelHealthStats)

	// Get all models from config (both credential-specific and global)
	allConfigModels := p.modelManager.GetAllModels()
	for _, model := range allConfigModels.Data {
		// For each model, check which credentials support it
		credentials := p.modelManager.GetCredentialsForModel(model.ID)
		if len(credentials) == 0 {
			// If no specific credentials, add for all credentials
			for _, cred := range p.balancer.GetCredentials() {
				modelKey := fmt.Sprintf("%s:%s", cred.Name, model.ID)
				modelsInfo[modelKey] = httputil.ModelHealthStats{
					Credential: cred.Name,
					Model:      model.ID,
					CurrentRPM: p.rateLimiter.GetCurrentModelRPM(cred.Name, model.ID),
					CurrentTPM: p.rateLimiter.GetCurrentModelTPM(cred.Name, model.ID),
					LimitRPM:   p.rateLimiter.GetModelLimitRPM(cred.Name, model.ID),
					LimitTPM:   p.rateLimiter.GetModelLimitTPM(cred.Name, model.ID),
				}
			}
		} else {
			// Add for specific credentials only
			for _, credName := range credentials {
				modelKey := fmt.Sprintf("%s:%s", credName, model.ID)
				modelsInfo[modelKey] = httputil.ModelHealthStats{
					Credential: credName,
					Model:      model.ID,
					CurrentRPM: p.rateLimiter.GetCurrentModelRPM(credName, model.ID),
					CurrentTPM: p.rateLimiter.GetCurrentModelTPM(credName, model.ID),
					LimitRPM:   p.rateLimiter.GetModelLimitRPM(credName, model.ID),
					LimitTPM:   p.rateLimiter.GetModelLimitTPM(credName, model.ID),
				}
			}
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
