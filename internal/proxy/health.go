package proxy

import (
	_ "embed"
	"fmt"
	"net/http"
)

func (p *Proxy) HealthCheck() (bool, map[string]interface{}) {
	totalCreds := len(p.balancer.GetCredentials())
	availableCreds := p.balancer.GetAvailableCount()
	bannedCreds := p.balancer.GetBannedCount()

	healthy := availableCreds > 0

	// Collect credentials info
	credentialsInfo := make(map[string]interface{})
	for _, cred := range p.balancer.GetCredentials() {
		credentialsInfo[cred.Name] = map[string]interface{}{
			"current_rpm": p.rateLimiter.GetCurrentRPM(cred.Name),
			"current_tpm": p.rateLimiter.GetCurrentTPM(cred.Name),
			"limit_rpm":   cred.RPM,
			"limit_tpm":   cred.TPM,
		}
	}

	// Collect models info from all configured models
	modelsInfo := make(map[string]interface{})

	// Get all models from config (both credential-specific and global)
	allConfigModels := p.modelManager.GetAllModels()
	for _, model := range allConfigModels.Data {
		// For each model, check which credentials support it
		credentials := p.modelManager.GetCredentialsForModel(model.ID)
		if len(credentials) == 0 {
			// If no specific credentials, add for all credentials
			for _, cred := range p.balancer.GetCredentials() {
				modelKey := fmt.Sprintf("%s:%s", cred.Name, model.ID)
				modelsInfo[modelKey] = map[string]interface{}{
					"credential":  cred.Name,
					"model":       model.ID,
					"current_rpm": p.rateLimiter.GetCurrentModelRPM(cred.Name, model.ID),
					"current_tpm": p.rateLimiter.GetCurrentModelTPM(cred.Name, model.ID),
					"limit_rpm":   p.rateLimiter.GetModelLimitRPM(cred.Name, model.ID),
					"limit_tpm":   p.rateLimiter.GetModelLimitTPM(cred.Name, model.ID),
				}
			}
		} else {
			// Add for specific credentials only
			for _, credName := range credentials {
				modelKey := fmt.Sprintf("%s:%s", credName, model.ID)
				modelsInfo[modelKey] = map[string]interface{}{
					"credential":  credName,
					"model":       model.ID,
					"current_rpm": p.rateLimiter.GetCurrentModelRPM(credName, model.ID),
					"current_tpm": p.rateLimiter.GetCurrentModelTPM(credName, model.ID),
					"limit_rpm":   p.rateLimiter.GetModelLimitRPM(credName, model.ID),
					"limit_tpm":   p.rateLimiter.GetModelLimitTPM(credName, model.ID),
				}
			}
		}
	}

	status := map[string]interface{}{
		"status":                "healthy",
		"credentials_available": availableCreds,
		"credentials_banned":    bannedCreds,
		"total_credentials":     totalCreds,
		"credentials":           credentialsInfo,
		"models":                modelsInfo,
	}

	if !healthy {
		status["status"] = "unhealthy"
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

	if err := p.healthTemplate.Execute(w, status); err != nil {
		p.logger.Error("Failed to execute health template", "error", err)
	}
}
