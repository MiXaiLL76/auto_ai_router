package models

import (
	"log/slog"
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

// ModelLimits stores RPM and TPM limits for a model
type ModelLimits struct {
	RPM        int
	TPM        int
	Credential string // If set, limits apply only to this credential
}

// Manager handles model discovery and mapping
type Manager struct {
	mu                 sync.RWMutex
	credentialModels   map[string][]string      // credential name -> list of model IDs
	allModels          []Model                  // deduplicated list of all models
	modelToCredentials map[string][]string      // model ID -> list of credential names
	modelLimits        map[string][]ModelLimits // model ID -> limits (may have multiple entries for different credentials)
	defaultModelsRPM   int                      // default RPM for models
	logger             *slog.Logger
}

// New creates a new model manager
func New(logger *slog.Logger, defaultModelsRPM int, staticModels []config.ModelRPMConfig) *Manager {
	m := &Manager{
		credentialModels:   make(map[string][]string),
		allModels:          make([]Model, 0),
		modelToCredentials: make(map[string][]string),
		modelLimits:        make(map[string][]ModelLimits),
		defaultModelsRPM:   defaultModelsRPM,
		logger:             logger,
	}

	// Load static models from config.yaml
	if len(staticModels) > 0 {
		logger.Info("Loading static models from config.yaml", "models_count", len(staticModels))
		for _, staticModel := range staticModels {
			m.modelLimits[staticModel.Name] = append(m.modelLimits[staticModel.Name], ModelLimits{
				RPM:        staticModel.RPM,
				TPM:        staticModel.TPM,
				Credential: staticModel.Credential,
			})
			logger.Debug("Added static model from config.yaml",
				"model", staticModel.Name,
				"credential", staticModel.Credential,
				"rpm", staticModel.RPM,
				"tpm", staticModel.TPM)
		}
	}

	return m
}

// addModelToMaps adds model to credential mapping, avoiding duplicates using sets
func addModelToMaps(
	credentialModels map[string][]string,
	modelToCredentials map[string][]string,
	credentialModelsSet map[string]map[string]bool,
	modelToCredentialsSet map[string]map[string]bool,
	credName, modelName string,
) {
	// Initialize sets if needed
	if credentialModelsSet[credName] == nil {
		credentialModelsSet[credName] = make(map[string]bool)
	}
	if modelToCredentialsSet[modelName] == nil {
		modelToCredentialsSet[modelName] = make(map[string]bool)
	}

	// Add to credentialModels if not present
	if !credentialModelsSet[credName][modelName] {
		credentialModels[credName] = append(credentialModels[credName], modelName)
		credentialModelsSet[credName][modelName] = true
	}

	// Add to modelToCredentials if not present
	if !modelToCredentialsSet[modelName][credName] {
		modelToCredentials[modelName] = append(modelToCredentials[modelName], credName)
		modelToCredentialsSet[modelName][credName] = true
	}
}

// LoadModelsFromConfig loads credential-specific models from static config
// This should be called once during initialization
func (m *Manager) LoadModelsFromConfig(credentials []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.modelLimits) == 0 {
		m.logger.Debug("No models in config to load")
		return
	}

	// Create map of credential names for validation
	credNames := make(map[string]bool)
	for _, cred := range credentials {
		credNames[cred.Name] = true
	}

	// Create sets for efficient duplicate checking
	credentialModelsSet := make(map[string]map[string]bool)
	modelToCredentialsSet := make(map[string]map[string]bool)

	credentialSpecificCount := 0
	globalModelsCount := 0

	// Process each model from static config
	for modelName, limits := range m.modelLimits {
		for _, limit := range limits {
			if limit.Credential != "" {
				// Model is specific to a credential
				if !credNames[limit.Credential] {
					m.logger.Warn("Model references non-existent credential",
						"model", modelName,
						"credential", limit.Credential,
					)
					continue
				}

				addModelToMaps(
					m.credentialModels,
					m.modelToCredentials,
					credentialModelsSet,
					modelToCredentialsSet,
					limit.Credential,
					modelName,
				)

				credentialSpecificCount++

				m.logger.Debug("Registered credential-specific model",
					"model", modelName,
					"credential", limit.Credential,
				)
			} else {
				// Model is global (no credential specified)
				// Map to all credentials
				for credName := range credNames {
					addModelToMaps(
						m.credentialModels,
						m.modelToCredentials,
						credentialModelsSet,
						modelToCredentialsSet,
						credName,
						modelName,
					)
				}

				globalModelsCount++
				m.logger.Debug("Registered global model mapping",
					"model", modelName,
				)
			}
		}
	}

	m.logger.Info("Loaded models from config",
		"credential_specific", credentialSpecificCount,
		"global_models", globalModelsCount,
	)
}

// GetAllModels returns all unique models across all credentials
func (m *Manager) GetAllModels() ModelsResponse {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// If static models are configured, return them
	if len(m.modelLimits) > 0 {
		models := make([]Model, 0, len(m.modelLimits))
		for modelName := range m.modelLimits {
			models = append(models, Model{
				ID:      modelName,
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

// hasModelInCredentials checks if modelID is assigned to credentialName in modelToCredentials map
func hasModelInCredentials(modelToCredentials map[string][]string, modelID, credentialName string) (bool, bool) {
	creds, modelExists := modelToCredentials[modelID]
	if !modelExists {
		return false, false // Model doesn't exist in map
	}

	for _, cred := range creds {
		if cred == credentialName {
			return true, true // Model exists and credential matches
		}
	}

	return false, true // Model exists but credential doesn't match
}

// HasModel checks if a credential supports a specific model
func (m *Manager) HasModel(credentialName, modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check modelToCredentials map
	hasModel, modelExists := hasModelInCredentials(m.modelToCredentials, modelID, credentialName)
	if hasModel {
		return true
	}
	if modelExists {
		// Model exists but not for this credential
		return false
	}

	// Model not found in modelToCredentials
	// Check if we have any models configured
	hasStaticModels := len(m.modelLimits) > 0
	credentialExists := false

	// Check credentialModels map
	if models, ok := m.credentialModels[credentialName]; ok {
		credentialExists = true
		// If credential exists, check if it has the model
		for _, model := range models {
			if model == modelID {
				return true
			}
		}
	}

	// If we have static models configured and credential exists but model not found - deny
	if hasStaticModels && credentialExists {
		return false
	}

	// If credential doesn't exist, allow (fallback behavior)
	// If no models configured, allow all (fallback behavior)
	return true
}

// IsEnabled returns whether model filtering should be used
// Returns true if there are models defined in static config
func (m *Manager) IsEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Filtering is enabled if we have static models configured
	return len(m.modelLimits) > 0
}

// findRPMLimit searches for RPM limit with optional credential filtering
func findRPMLimit(limits []ModelLimits, credentialName string) (int, bool) {
	if credentialName != "" {
		// Look for credential-specific limit first
		for _, limit := range limits {
			if limit.Credential == credentialName {
				return limit.RPM, true
			}
		}
	}

	// Fall back to global limit (no credential specified)
	for _, limit := range limits {
		if limit.Credential == "" {
			return limit.RPM, true
		}
	}

	// If only credential-specific limits exist and no credential specified, return first one
	if credentialName == "" && len(limits) > 0 {
		return limits[0].RPM, true
	}

	return 0, false
}

// GetModelRPM returns RPM limit for a specific model
func (m *Manager) GetModelRPM(modelID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	limits, ok := m.modelLimits[modelID]
	if !ok {
		return m.defaultModelsRPM
	}

	if rpm, found := findRPMLimit(limits, ""); found {
		return rpm
	}

	return m.defaultModelsRPM
}

// GetModelRPMForCredential returns RPM limit for a specific model and credential
func (m *Manager) GetModelRPMForCredential(modelID, credentialName string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	limits, ok := m.modelLimits[modelID]
	if !ok {
		return m.defaultModelsRPM
	}

	if rpm, found := findRPMLimit(limits, credentialName); found {
		return rpm
	}

	return m.defaultModelsRPM
}

// findTPMLimit searches for TPM limit with optional credential filtering
// Returns -1 for unlimited (when TPM is 0 or not set)
func findTPMLimit(limits []ModelLimits, credentialName string) (int, bool) {
	if credentialName != "" {
		// Look for credential-specific limit first
		for _, limit := range limits {
			if limit.Credential == credentialName {
				if limit.TPM == 0 {
					return -1, true // 0 means unlimited
				}
				return limit.TPM, true
			}
		}
	}

	// Fall back to global limit (no credential specified)
	for _, limit := range limits {
		if limit.Credential == "" {
			if limit.TPM == 0 {
				return -1, true // 0 means unlimited
			}
			return limit.TPM, true
		}
	}

	// If only credential-specific limits exist and no credential specified, return first one
	if credentialName == "" && len(limits) > 0 {
		if limits[0].TPM == 0 {
			return -1, true
		}
		return limits[0].TPM, true
	}

	return 0, false
}

// GetModelTPM returns TPM limit for a specific model
func (m *Manager) GetModelTPM(modelID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	limits, ok := m.modelLimits[modelID]
	if !ok {
		return -1 // Unlimited by default
	}

	if tpm, found := findTPMLimit(limits, ""); found {
		return tpm
	}

	return -1
}

// GetModelTPMForCredential returns TPM limit for a specific model and credential
func (m *Manager) GetModelTPMForCredential(modelID, credentialName string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	limits, ok := m.modelLimits[modelID]
	if !ok {
		return -1 // Unlimited by default
	}

	if tpm, found := findTPMLimit(limits, credentialName); found {
		return tpm
	}

	return -1
}

// GetModelsForCredential returns all models available for a specific credential
func (m *Manager) GetModelsForCredential(credentialName string) []Model {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []Model

	// If we have static models configured, use them
	if len(m.modelLimits) > 0 {
		for modelName, limits := range m.modelLimits {
			// Check if model is available for this credential
			available := false
			for _, limit := range limits {
				if limit.Credential == "" || limit.Credential == credentialName {
					available = true
					break
				}
			}
			if available {
				result = append(result, Model{
					ID:      modelName,
					Object:  "model",
					Created: time.Now().Unix(),
					OwnedBy: "system",
				})
			}
		}
		return result
	}

	// If no static config, check credentialModels map
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
