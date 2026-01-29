package models

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// Model represents a single model from OpenAI API
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

// ModelsResponse represents the response from /v1/models endpoint
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// Manager handles model discovery and mapping
type Manager struct {
	mu                 sync.RWMutex
	credentialModels   map[string][]string  // credential name -> list of model IDs
	allModels          []Model              // deduplicated list of all models
	modelToCredentials map[string][]string  // model ID -> list of credential names
	modelsConfig       *config.ModelsConfig // models.yaml config (merged with static models from config.yaml)
	modelsConfigPath   string               // path to models.yaml
	defaultModelsRPM   int                  // default RPM for models
	logger             *slog.Logger
	enabled            bool
}

// New creates a new model manager
func New(logger *slog.Logger, enabled bool, defaultModelsRPM int, modelsConfigPath string, staticModels []config.ModelRPMConfig) *Manager {
	// Load models.yaml if it exists
	modelsConfig, err := config.LoadModelsConfig(modelsConfigPath)
	if err != nil {
		logger.Error("Failed to load models config", "error", err, "path", modelsConfigPath)
		modelsConfig = &config.ModelsConfig{Models: []config.ModelRPMConfig{}}
	} else if len(modelsConfig.Models) > 0 {
		logger.Info("Loaded models config", "path", modelsConfigPath, "models_count", len(modelsConfig.Models))
	}

	// Merge static models from config.yaml into modelsConfig (they have priority)
	if len(staticModels) > 0 {
		logger.Info("Merging static models from config.yaml", "models_count", len(staticModels))
		for _, staticModel := range staticModels {
			// Check if model already exists in models.yaml
			exists := false
			for i, existingModel := range modelsConfig.Models {
				if existingModel.Name == staticModel.Name && existingModel.Credential == staticModel.Credential {
					// Update with values from config.yaml (static has priority)
					modelsConfig.Models[i] = staticModel
					exists = true
					logger.Debug("Updated model from config.yaml", "model", staticModel.Name, "credential", staticModel.Credential, "rpm", staticModel.RPM, "tpm", staticModel.TPM)
					break
				}
			}
			// Add new model if it doesn't exist
			if !exists {
				modelsConfig.Models = append(modelsConfig.Models, staticModel)
				logger.Debug("Added static model from config.yaml", "model", staticModel.Name, "rpm", staticModel.RPM, "tpm", staticModel.TPM)
			}
		}
	}

	return &Manager{
		credentialModels:   make(map[string][]string),
		allModels:          make([]Model, 0),
		modelToCredentials: make(map[string][]string),
		modelsConfig:       modelsConfig,
		modelsConfigPath:   modelsConfigPath,
		defaultModelsRPM:   defaultModelsRPM,
		logger:             logger,
		enabled:            enabled,
	}
}

// FetchModels fetches models from all credentials at startup
// Note: LoadModelsFromConfig should be called before this method
func (m *Manager) FetchModels(credentials []config.CredentialConfig, timeout time.Duration) {
	if !m.enabled {
		m.logger.Info("Model API fetching disabled", "replace_v1_models", false)
		return
	}

	m.logger.Info("Fetching models from credentials via API", "count", len(credentials))

	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment, // Support HTTP_PROXY, HTTPS_PROXY, NO_PROXY
			MaxIdleConns:        50,
			MaxIdleConnsPerHost: 5,
			IdleConnTimeout:     30 * time.Second,
			DisableKeepAlives:   false,
		},
	}
	defer client.CloseIdleConnections()

	var wg sync.WaitGroup
	modelsChan := make(chan struct {
		credName string
		models   []Model
		err      error
	}, len(credentials))

	// Fetch models from all credentials in parallel
	for _, cred := range credentials {
		wg.Add(1)
		go func(c config.CredentialConfig) {
			defer wg.Done()

			models, err := m.fetchModelsFromCredential(client, c)
			modelsChan <- struct {
				credName string
				models   []Model
				err      error
			}{
				credName: c.Name,
				models:   models,
				err:      err,
			}
		}(cred)
	}

	// Wait for all fetches to complete
	go func() {
		wg.Wait()
		close(modelsChan)
	}()

	// Collect results
	uniqueModels := make(map[string]Model)
	for result := range modelsChan {
		if result.err != nil {
			m.logger.Error("Failed to fetch models from credential",
				"credential", result.credName,
				"error", result.err,
			)
			continue
		}

		// Store models for this credential
		modelIDs := make([]string, len(result.models))
		for i, model := range result.models {
			modelIDs[i] = model.ID
			// Add to unique models set
			uniqueModels[model.ID] = model
			// Map model to credential (avoid duplicates)
			if !contains(m.modelToCredentials[model.ID], result.credName) {
				m.modelToCredentials[model.ID] = append(m.modelToCredentials[model.ID], result.credName)
			}
		}

		m.credentialModels[result.credName] = modelIDs
		m.logger.Debug("Fetched models from credential",
			"credential", result.credName,
			"models_count", len(modelIDs),
		)
	}

	// Convert unique models to slice
	m.allModels = make([]Model, 0, len(uniqueModels))
	for _, model := range uniqueModels {
		m.allModels = append(m.allModels, model)
	}

	m.logger.Info("Model fetching complete",
		"total_unique_models", len(m.allModels),
		"credentials_with_models", len(m.credentialModels),
	)

	// Update models.yaml with newly discovered models
	m.updateModelsConfig()
}

// LoadModelsFromConfig loads credential-specific models from config
// This should be called once during initialization before FetchModels
func (m *Manager) LoadModelsFromConfig(credentials []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.modelsConfig == nil || len(m.modelsConfig.Models) == 0 {
		m.logger.Debug("No models in config to load")
		return
	}

	// Create map of credential names for validation
	credNames := make(map[string]bool)
	for _, cred := range credentials {
		credNames[cred.Name] = true
	}

	credentialSpecificCount := 0
	globalModelsCount := 0

	// Process each model in config
	for _, modelConfig := range m.modelsConfig.Models {
		if modelConfig.Credential != "" {
			// Model is specific to a credential
			if !credNames[modelConfig.Credential] {
				m.logger.Warn("Model references non-existent credential",
					"model", modelConfig.Name,
					"credential", modelConfig.Credential,
				)
				continue
			}

			// Add to credentialModels map (avoid duplicates)
			if !contains(m.credentialModels[modelConfig.Credential], modelConfig.Name) {
				m.credentialModels[modelConfig.Credential] = append(
					m.credentialModels[modelConfig.Credential],
					modelConfig.Name,
				)
			}

			// Add to modelToCredentials map (avoid duplicates)
			if !contains(m.modelToCredentials[modelConfig.Name], modelConfig.Credential) {
				m.modelToCredentials[modelConfig.Name] = append(
					m.modelToCredentials[modelConfig.Name],
					modelConfig.Credential,
				)
			}

			credentialSpecificCount++

			m.logger.Debug("Registered credential-specific model",
				"model", modelConfig.Name,
				"credential", modelConfig.Credential,
			)
		} else {
			// Model is global (no credential specified)
			// Map to all credentials
			for credName := range credNames {
				// Add to credentialModels map (avoid duplicates)
				if !contains(m.credentialModels[credName], modelConfig.Name) {
					m.credentialModels[credName] = append(
						m.credentialModels[credName],
						modelConfig.Name,
					)
				}

				// Add to modelToCredentials map (avoid duplicates)
				if !contains(m.modelToCredentials[modelConfig.Name], credName) {
					m.modelToCredentials[modelConfig.Name] = append(
						m.modelToCredentials[modelConfig.Name],
						credName,
					)
				}
			}

			globalModelsCount++
			m.logger.Debug("Registered global model mapping",
				"model", modelConfig.Name,
			)
		}
	}

	m.logger.Info("Loaded models from config",
		"credential_specific", credentialSpecificCount,
		"global_models", globalModelsCount,
	)
}

// contains checks if a string slice contains a specific value
func contains(slice []string, val string) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}

// fetchModelsFromCredential fetches models from a single credential
func (m *Manager) fetchModelsFromCredential(client *http.Client, cred config.CredentialConfig) ([]Model, error) {
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")

	// Check if baseURL already ends with /v1 to avoid /v1/v1/models
	var url string
	if strings.HasSuffix(baseURL, "/v1") {
		url = baseURL + "/models"
	} else {
		url = baseURL + "/v1/models"
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+cred.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch models: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			m.logger.Error("Failed to close response body", "error", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var modelsResp ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return modelsResp.Data, nil
}

// GetAllModels returns all unique models across all credentials
func (m *Manager) GetAllModels() ModelsResponse {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// If model fetching is disabled but models.yaml exists, return models from config
	if !m.enabled && m.modelsConfig != nil && len(m.modelsConfig.Models) > 0 {
		models := make([]Model, 0, len(m.modelsConfig.Models))
		for _, modelConfig := range m.modelsConfig.Models {
			models = append(models, Model{
				ID:      modelConfig.Name,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "system",
			})
		}
		return ModelsResponse{
			Object: "list",
			Data:   models,
		}
	}

	return ModelsResponse{
		Object: "list",
		Data:   m.allModels,
	}
}

// GetCredentialsForModel returns list of credential names that support the given model
// Works with both fetched models (when enabled=true) and config-loaded models (when enabled=false)
func (m *Manager) GetCredentialsForModel(modelID string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check modelToCredentials map (populated by both loadModelsFromConfig and FetchModels)
	creds, ok := m.modelToCredentials[modelID]
	if !ok {
		return nil
	}

	// Return a copy to avoid race conditions
	result := make([]string, len(creds))
	copy(result, creds)
	return result
}

// HasModel checks if a credential supports a specific model
func (m *Manager) HasModel(credentialName, modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// If API fetching is disabled
	if !m.enabled {
		// If we have models in config, check credential-specific mappings first
		if m.modelsConfig != nil && len(m.modelsConfig.Models) > 0 {
			// Check modelToCredentials map (populated by LoadModelsFromConfig)
			if creds, ok := m.modelToCredentials[modelID]; ok {
				for _, cred := range creds {
					if cred == credentialName {
						return true
					}
				}
				// Model exists but not for this credential
				return false
			}
			// Model not found in modelToCredentials
			// Check if this credential exists in our system
			if _, credExists := m.credentialModels[credentialName]; credExists {
				// Credential exists but model not found - deny
				return false
			}
			// If modelToCredentials is empty (LoadModelsFromConfig not called),
			// check if model exists in config directly
			if len(m.modelToCredentials) == 0 {
				for _, modelConfig := range m.modelsConfig.Models {
					if modelConfig.Name == modelID {
						return true // Model exists in config
					}
				}
				return false
			}
			// Credential doesn't exist - allow (fallback behavior)
			return true
		}
		// No config available - allow all (fallback behavior)
		return true
	}

	// API fetching is enabled - check modelToCredentials map first
	if creds, ok := m.modelToCredentials[modelID]; ok {
		for _, cred := range creds {
			if cred == credentialName {
				return true
			}
		}
		// Model exists but not for this credential
		return false
	}

	// Check credentialModels map (for fetched models)
	if models, ok := m.credentialModels[credentialName]; ok {
		for _, model := range models {
			if model == modelID {
				return true
			}
		}
		// Credential exists but doesn't have this model
		return false
	}

	// Credential not found - allow (fallback behavior for unknown credentials)
	return true
}

// IsEnabled returns whether model filtering should be used
// Returns true if:
// 1. API fetching is enabled (replace_v1_models=true), OR
// 2. There are models defined in config (models.yaml or config.yaml)
func (m *Manager) IsEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// If API fetching is enabled, filtering is enabled
	if m.enabled {
		return true
	}

	// If API fetching is disabled, check if we have models in config
	if m.modelsConfig != nil && len(m.modelsConfig.Models) > 0 {
		return true
	}

	// No API fetching and no config - filtering is disabled
	return false
}

// GetModelRPM returns RPM limit for a specific model
func (m *Manager) GetModelRPM(modelID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelsConfig == nil {
		return m.defaultModelsRPM
	}

	return m.modelsConfig.GetModelRPM(modelID, m.defaultModelsRPM)
}

// GetModelRPMForCredential returns RPM limit for a specific model and credential
func (m *Manager) GetModelRPMForCredential(modelID, credentialName string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelsConfig == nil {
		return m.defaultModelsRPM
	}

	return m.modelsConfig.GetModelRPMForCredential(modelID, credentialName, m.defaultModelsRPM)
}

// GetModelTPM returns TPM limit for a specific model
func (m *Manager) GetModelTPM(modelID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelsConfig == nil {
		return -1 // Unlimited by default
	}

	return m.modelsConfig.GetModelTPM(modelID, -1)
}

// GetModelTPMForCredential returns TPM limit for a specific model and credential
func (m *Manager) GetModelTPMForCredential(modelID, credentialName string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelsConfig == nil {
		return -1 // Unlimited by default
	}

	return m.modelsConfig.GetModelTPMForCredential(modelID, credentialName, -1)
}

// GetModelsForCredential returns all models available for a specific credential
func (m *Manager) GetModelsForCredential(credentialName string) []Model {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []Model

	// If we have models in config, use them
	if m.modelsConfig != nil && len(m.modelsConfig.Models) > 0 {
		for _, modelConfig := range m.modelsConfig.Models {
			// Check if model is available for this credential
			if modelConfig.Credential == "" || modelConfig.Credential == credentialName {
				result = append(result, Model{
					ID:      modelConfig.Name,
					Object:  "model",
					Created: time.Now().Unix(),
					OwnedBy: "system",
				})
			}
		}
		return result
	}

	// If no config, check credentialModels map (from API fetching)
	if modelIDs, ok := m.credentialModels[credentialName]; ok {
		for _, modelID := range modelIDs {
			// Find the model in allModels
			for _, model := range m.allModels {
				if model.ID == modelID {
					result = append(result, model)
					break
				}
			}
		}
	}

	return result
}

// updateModelsConfig updates models.yaml with newly discovered models
func (m *Manager) updateModelsConfig() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.modelsConfig == nil {
		m.modelsConfig = &config.ModelsConfig{Models: []config.ModelRPMConfig{}}
	}

	// Add all discovered models that don't exist in config yet
	updated := false
	for _, model := range m.allModels {
		// Check if model already exists in config
		exists := false
		for _, existingModel := range m.modelsConfig.Models {
			if existingModel.Name == model.ID {
				exists = true
				break
			}
		}

		// Add new model with default RPM
		if !exists {
			m.modelsConfig.UpdateOrAddModel(model.ID, m.defaultModelsRPM)
			updated = true
			m.logger.Debug("Added model to config", "model", model.ID, "rpm", m.defaultModelsRPM)
		}
	}

	// Save updated config if changes were made
	if updated {
		if err := config.SaveModelsConfig(m.modelsConfigPath, m.modelsConfig); err != nil {
			m.logger.Error("Failed to save models config", "error", err, "path", m.modelsConfigPath)
		} else {
			m.logger.Info("Updated models config", "path", m.modelsConfigPath, "total_models", len(m.modelsConfig.Models))
		}
	}
}
