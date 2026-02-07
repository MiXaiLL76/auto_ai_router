package models

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/transform/openai"
)

// ModelPrice contains pricing information for a single model
type ModelPrice struct {
	// Regular tokens (input/output)
	InputCostPerToken  float64 `json:"input_cost_per_token"`
	OutputCostPerToken float64 `json:"output_cost_per_token"`

	// Audio tokens (can be more specific than regular tokens)
	InputCostPerAudioToken  float64 `json:"input_cost_per_audio_token,omitempty"`
	OutputCostPerAudioToken float64 `json:"output_cost_per_audio_token,omitempty"`

	// Image tokens (can be more specific than regular tokens)
	InputCostPerImageToken  float64 `json:"input_cost_per_image_token,omitempty"`
	OutputCostPerImageToken float64 `json:"output_cost_per_image_token,omitempty"`

	// Reasoning tokens (deep thinking models)
	OutputCostPerReasoningToken float64 `json:"output_cost_per_reasoning_token,omitempty"`

	// Cached/Prediction tokens
	OutputCostPerCachedToken     float64 `json:"output_cost_per_cached_token,omitempty"`
	InputCostPerCachedToken      float64 `json:"input_cost_per_cached_token,omitempty"`
	OutputCostPerPredictionToken float64 `json:"output_cost_per_prediction_token,omitempty"`

	// Vision/Images cost per image (not per token)
	OutputCostPerImage float64 `json:"output_cost_per_image,omitempty"`
}

// ModelPriceRegistry stores and manages cached model prices
type ModelPriceRegistry struct {
	mu         sync.RWMutex
	prices     map[string]*ModelPrice // key: normalized model name
	lastUpdate time.Time
}

// NewModelPriceRegistry creates a new price registry
func NewModelPriceRegistry() *ModelPriceRegistry {
	return &ModelPriceRegistry{
		prices: make(map[string]*ModelPrice),
	}
}

// GetPrice returns the price for a model, or nil if not found
func (r *ModelPriceRegistry) GetPrice(modelName string) *ModelPrice {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.prices[modelName]
}

// Update safely updates the registry with new prices
func (r *ModelPriceRegistry) Update(prices map[string]*ModelPrice) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prices = make(map[string]*ModelPrice)
	for k, v := range prices {
		r.prices[k] = v
	}
	r.lastUpdate = time.Now().UTC()
}

// LastUpdate returns the time of last successful update
func (r *ModelPriceRegistry) LastUpdate() time.Time {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.lastUpdate
}

// Count returns the number of models in the registry
func (r *ModelPriceRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.prices)
}

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

// remoteModelCache stores cached remote models with expiration time
type remoteModelCache struct {
	models    []Model
	expiresAt time.Time
}

// allModelsCache stores cached result of GetAllModels
type allModelsCache struct {
	response  ModelsResponse
	expiresAt time.Time
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
	credentials        []config.CredentialConfig   // credentials for fetching remote models
	remoteModelsCache  map[string]remoteModelCache // cache for remote models per credential (credentialName -> cache)
	cacheExpiration    time.Duration               // how long to cache remote models (default 5 minutes)
	allModelsCache     allModelsCache              // cached result of GetAllModels (3 second TTL)
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
		credentials:        make([]config.CredentialConfig, 0),
		remoteModelsCache:  make(map[string]remoteModelCache),
		cacheExpiration:    5 * time.Minute, // Default cache TTL: 5 minutes
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

// SetCredentials sets the credentials for fetching remote models from proxies
func (m *Manager) SetCredentials(credentials []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.credentials = credentials
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

// GetAllModels returns all unique models across all credentials with caching
func (m *Manager) GetAllModels() ModelsResponse {
	// Check cache first (fast path without holding full lock)
	m.mu.RLock()
	if !m.allModelsCache.expiresAt.IsZero() && time.Now().UTC().Before(m.allModelsCache.expiresAt) {
		// Copy response while holding lock to prevent TOCTOU
		cachedResponse := m.allModelsCache.response
		cachedCount := len(cachedResponse.Data)
		m.mu.RUnlock()
		m.logger.Debug("Returning cached all models",
			"models_count", cachedCount,
		)
		return cachedResponse
	}

	var models []Model
	modelMap := make(map[string]bool)
	allModelsSnapshot := append([]Model(nil), m.allModels...)

	// Add static models first (configured in model_limits)
	if len(m.modelLimits) > 0 {
		models = make([]Model, 0, len(m.modelLimits)+len(allModelsSnapshot))
		for modelName := range m.modelLimits {
			models = append(models, Model{
				ID:      modelName,
				Object:  "model",
				Created: openai.GetCurrentTimestamp(),
				OwnedBy: "system",
			})
			modelMap[modelName] = true
		}
	} else {
		models = make([]Model, 0, len(allModelsSnapshot))
	}

	// Also add models from credential config (allModels)
	for _, model := range allModelsSnapshot {
		if !modelMap[model.ID] {
			models = append(models, model)
			modelMap[model.ID] = true
		}
	}

	// Make a copy of credentials for fetching remote models
	credentials := make([]config.CredentialConfig, len(m.credentials))
	copy(credentials, m.credentials)

	m.mu.RUnlock()

	// Add models from proxy credentials only (not from other provider types)
	modelUpdates := make(map[string][]string) // model -> credentials to add
	for _, cred := range credentials {
		// Skip non-proxy credentials - we only fetch models from proxy credentials
		if cred.Type != config.ProviderTypeProxy {
			m.logger.Debug("Skipping model fetch for non-proxy credential",
				"credential", cred.Name,
				"type", cred.Type,
			)
			continue
		}

		m.logger.Debug("Fetching models from proxy credential",
			"credential", cred.Name,
		)
		remoteModels := m.GetRemoteModels(&cred)
		m.logger.Debug("Got models from proxy",
			"credential", cred.Name,
			"remote_models_count", len(remoteModels),
			"current_total", len(models),
		)
		added := 0
		for _, model := range remoteModels {
			if !modelMap[model.ID] {
				models = append(models, model)
				modelMap[model.ID] = true
				added++
				modelUpdates[model.ID] = append(modelUpdates[model.ID], cred.Name)
			}
		}
		m.logger.Debug("Processed proxy models",
			"credential", cred.Name,
			"added", added,
			"duplicates", len(remoteModels)-added,
			"total_now", len(models),
		)
	}

	response := ModelsResponse{
		Object: "list",
		Data:   models,
	}

	// Update cache and modelToCredentials atomically
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update modelToCredentials with new models
	for modelID, creds := range modelUpdates {
		if m.modelToCredentials[modelID] == nil {
			m.modelToCredentials[modelID] = []string{}
		}
		// Add credentials that aren't already in the list
		for _, cred := range creds {
			found := false
			for _, existing := range m.modelToCredentials[modelID] {
				if existing == cred {
					found = true
					break
				}
			}
			if !found {
				m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], cred)
			}
		}
	}

	// Cache the result for 3 seconds
	m.allModelsCache = allModelsCache{
		response:  response,
		expiresAt: time.Now().UTC().Add(3 * time.Second),
	}
	m.allModels = append([]Model(nil), models...)

	return response
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

// AddModel adds a model to the credential mapping (used for dynamically loaded models from proxy)
func (m *Manager) AddModel(credentialName, modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Add to credentialModels
	if !m.contains(m.credentialModels[credentialName], modelID) {
		m.credentialModels[credentialName] = append(m.credentialModels[credentialName], modelID)
	}

	// Add to modelToCredentials
	if !m.contains(m.modelToCredentials[modelID], credentialName) {
		m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], credentialName)
	}
}

// contains checks if a string slice contains a value
func (m *Manager) contains(slice []string, value string) bool {
	for _, item := range slice {
		if item == value {
			return true
		}
	}
	return false
}

// IsEnabled returns whether model filtering should be used
// Returns true if there are models defined in static config
func (m *Manager) IsEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Filtering is enabled if we have static models configured
	return len(m.modelLimits) > 0
}

// findLimit searches for a limit value with optional credential filtering
// The fieldFunc extracts the value from ModelLimits (e.g., func(ml) ml.RPM)
// The convertFunc optionally transforms the value (e.g., convert 0 to -1 for TPM)
func findLimit(limits []ModelLimits, credentialName string, fieldFunc func(*ModelLimits) int, convertFunc func(int) int) (int, bool) {
	if credentialName != "" {
		// Look for credential-specific limit first
		for i := range limits {
			if limits[i].Credential == credentialName {
				value := fieldFunc(&limits[i])
				return convertFunc(value), true
			}
		}
	}

	// Fall back to global limit (no credential specified)
	for i := range limits {
		if limits[i].Credential == "" {
			value := fieldFunc(&limits[i])
			return convertFunc(value), true
		}
	}

	// If only credential-specific limits exist and no credential specified, return first one
	if credentialName == "" && len(limits) > 0 {
		value := fieldFunc(&limits[0])
		return convertFunc(value), true
	}

	return 0, false
}

// findRPMLimit searches for RPM limit with optional credential filtering
func findRPMLimit(limits []ModelLimits, credentialName string) (int, bool) {
	return findLimit(limits, credentialName, func(ml *ModelLimits) int { return ml.RPM }, func(v int) int { return v })
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
	convertTPM := func(v int) int {
		if v == 0 {
			return -1 // 0 means unlimited
		}
		return v
	}
	return findLimit(limits, credentialName, func(ml *ModelLimits) int { return ml.TPM }, convertTPM)
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

// GetModelsForCredential returns all models available for a specific credential.
//
// Behavior:
//   - If the credential has explicitly configured models, returns those models
//   - If the credential is unknown but global models exist (with empty Credential field),
//     returns global models as a fallback for backward compatibility
//   - Returns empty slice if no models are found for the credential
//
// Note: This fallback behavior (returning global models for unknown credentials)
// differs from HasModel() which does not have this fallback behavior.
// For new code, prefer using HasModel() for stricter credential validation.
func (m *Manager) GetModelsForCredential(credentialName string) []Model {
	m.mu.RLock()
	modelIDs := make([]string, 0)
	seen := make(map[string]bool)
	for modelID, creds := range m.modelToCredentials {
		for _, cred := range creds {
			if cred == credentialName {
				if !seen[modelID] {
					modelIDs = append(modelIDs, modelID)
					seen[modelID] = true
				}
				break
			}
		}
	}

	// Preserve legacy behavior: unknown credentials still get global models
	if len(modelIDs) == 0 && len(m.modelLimits) > 0 {
		for modelName, limits := range m.modelLimits {
			for _, limit := range limits {
				if limit.Credential == "" {
					if !seen[modelName] {
						modelIDs = append(modelIDs, modelName)
						seen[modelName] = true
					}
					break
				}
			}
		}
	}

	if len(modelIDs) == 0 {
		m.mu.RUnlock()
		return nil
	}

	allModelsSnapshot := append([]Model(nil), m.allModels...)
	m.mu.RUnlock()

	modelLookup := make(map[string]Model, len(allModelsSnapshot))
	for _, model := range allModelsSnapshot {
		modelLookup[model.ID] = model
	}

	result := make([]Model, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		if model, ok := modelLookup[modelID]; ok {
			result = append(result, model)
			continue
		}
		result = append(result, Model{
			ID:      modelID,
			Object:  "model",
			Created: openai.GetCurrentTimestamp(),
			OwnedBy: "system",
		})
	}

	return result
}

// GetRemoteModels fetches models from a remote proxy credential with caching
func (m *Manager) GetRemoteModels(cred *config.CredentialConfig) []Model {
	if cred.Type != config.ProviderTypeProxy {
		return nil
	}

	// Check cache first
	m.mu.RLock()
	if cached, ok := m.remoteModelsCache[cred.Name]; ok && !cached.expiresAt.IsZero() && time.Now().UTC().Before(cached.expiresAt) {
		// Copy models slice reference while holding lock to prevent TOCTOU
		cachedModels := cached.models
		cachedCount := len(cachedModels)
		expiresIn := time.Until(cached.expiresAt).Seconds()
		m.mu.RUnlock()
		m.logger.Debug("Using cached remote models",
			"credential", cred.Name,
			"models_count", cachedCount,
			"expires_in", expiresIn,
		)
		return cachedModels
	}
	m.mu.RUnlock()

	m.logger.Debug("Fetching remote models from proxy",
		"credential", cred.Name,
		"base_url", cred.BaseURL,
	)

	// Fetch models using httputil helper
	ctx := context.Background()
	var modelsResp ModelsResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/v1/models", m.logger, &modelsResp); err != nil {
		m.logger.Error("Failed to fetch remote models",
			"credential", cred.Name,
			"error", err,
		)
		return nil
	}

	// Cache the result
	m.mu.Lock()
	m.remoteModelsCache[cred.Name] = remoteModelCache{
		models:    modelsResp.Data,
		expiresAt: time.Now().UTC().Add(m.cacheExpiration),
	}
	m.mu.Unlock()

	m.logger.Debug("Cached remote models",
		"credential", cred.Name,
		"models_count", len(modelsResp.Data),
		"expires_in", m.cacheExpiration.Seconds(),
	)

	return modelsResp.Data
}
