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
	credentialModels   map[string][]string      // credential name -> list of model IDs
	allModels          []Model                  // deduplicated list of all models
	modelToCredentials map[string][]string      // model ID -> list of credential names
	modelsConfig       *config.ModelsConfig     // models.yaml config
	modelsConfigPath   string                   // path to models.yaml
	defaultModelsRPM   int                      // default RPM for models
	logger             *slog.Logger
	enabled            bool
}

// New creates a new model manager
func New(logger *slog.Logger, enabled bool, defaultModelsRPM int, modelsConfigPath string) *Manager {
	// Load models.yaml if it exists
	modelsConfig, err := config.LoadModelsConfig(modelsConfigPath)
	if err != nil {
		logger.Error("Failed to load models config", "error", err, "path", modelsConfigPath)
		modelsConfig = &config.ModelsConfig{Models: []config.ModelRPMConfig{}}
	} else if len(modelsConfig.Models) > 0 {
		logger.Info("Loaded models config", "path", modelsConfigPath, "models_count", len(modelsConfig.Models))
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
func (m *Manager) FetchModels(credentials []config.CredentialConfig, timeout time.Duration) {
	if !m.enabled {
		m.logger.Info("Model fetching disabled", "replace_v1_models", false)
		return
	}

	m.logger.Info("Fetching models from credentials", "count", len(credentials))

	client := &http.Client{
		Timeout: timeout,
	}

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
			// Map model to credential
			m.modelToCredentials[model.ID] = append(m.modelToCredentials[model.ID], result.credName)
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

// fetchModelsFromCredential fetches models from a single credential
func (m *Manager) fetchModelsFromCredential(client *http.Client, cred config.CredentialConfig) ([]Model, error) {
	url := strings.TrimSuffix(cred.BaseURL, "/") + "/v1/models"

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
	defer resp.Body.Close()

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

	return ModelsResponse{
		Object: "list",
		Data:   m.allModels,
	}
}

// GetCredentialsForModel returns list of credential names that support the given model
func (m *Manager) GetCredentialsForModel(modelID string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.enabled {
		return nil
	}

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

	if !m.enabled {
		return true // If disabled, allow all models
	}

	models, ok := m.credentialModels[credentialName]
	if !ok {
		return true // If we don't have data, allow it
	}

	for _, m := range models {
		if m == modelID {
			return true
		}
	}
	return false
}

// IsEnabled returns whether model filtering is enabled
func (m *Manager) IsEnabled() bool {
	return m.enabled
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
