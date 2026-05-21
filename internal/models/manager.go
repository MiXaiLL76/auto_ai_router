package models

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

var errProxyHealthModelMetadataUnavailable = fmt.Errorf("proxy health model metadata unavailable")

// ModelPrice contains pricing information for a single model
type ModelPrice struct {
	// Regular tokens (input/output)
	InputCostPerToken  float64 `json:"input_cost_per_token"`
	OutputCostPerToken float64 `json:"output_cost_per_token"`

	// Tiered pricing: tokens above 200k threshold billed at a different rate
	InputCostPerTokenAbove200k  float64 `json:"input_cost_per_token_above_200k_tokens,omitempty"`
	OutputCostPerTokenAbove200k float64 `json:"output_cost_per_token_above_200k_tokens,omitempty"`

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
	CacheReadInputTokenCost      float64 `json:"cache_read_input_token_cost,omitempty"`     // LiteLLM alias for InputCostPerCachedToken
	CacheCreationInputTokenCost  float64 `json:"cache_creation_input_token_cost,omitempty"` // Anthropic cache write cost
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
	return r.prices[NormalizeModelName(modelName)]
}

// Update safely updates the registry with new prices
func (r *ModelPriceRegistry) Update(prices map[string]*ModelPrice) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prices = make(map[string]*ModelPrice)
	for k, v := range prices {
		r.prices[k] = v
	}
	r.lastUpdate = utils.NowUTC()
}

// MergeDB applies DB-sourced prices on top of the existing registry without
// removing prices that came from the file-based price list.
// DB prices take precedence for models that appear in both sources.
func (r *ModelPriceRegistry) MergeDB(dbPrices map[string]*ModelPrice) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for k, v := range dbPrices {
		r.prices[k] = v
	}
	r.lastUpdate = utils.NowUTC()
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
	mu                        sync.RWMutex
	credentialModels          map[string][]string      // credential name -> list of model IDs
	allModels                 []Model                  // deduplicated list of all models
	modelToCredentials        map[string][]string      // model ID -> list of credential names
	modelLimits               map[string][]ModelLimits // model ID -> limits (may have multiple entries for different credentials)
	staticModelLimits         map[string][]ModelLimits // immutable snapshot of limits from config.yaml (never modified after New())
	staticModelRealNames      map[string]string        // immutable snapshot of real names from config.yaml
	modelPassthroughResponses map[string]*bool         // model name -> explicit passthrough_responses override (nil = auto)
	dbModelNames              map[string]bool          // model names that were loaded from LiteLLM DB (for hot-reload diffing)
	modelAliases              map[string]string        // alias -> real model name (from model_alias config)
	modelRealNames            map[string]string        // alias name -> real model name (from models[].model field)
	defaultModelsRPM          int                      // default RPM for models
	logger                    *slog.Logger
	credentials               []config.CredentialConfig   // credentials for fetching remote models
	remoteModelsCache         map[string]remoteModelCache // cache for remote models per credential (credentialName -> cache)
	cacheExpiration           time.Duration               // how long to cache remote models (default 5 minutes)
	allModelsCache            allModelsCache              // cached result of GetAllModels (3 second TTL)
}

// New creates a new model manager
func New(logger *slog.Logger, defaultModelsRPM int, staticModels []config.ModelRPMConfig) *Manager {
	m := &Manager{
		credentialModels:          make(map[string][]string),
		allModels:                 make([]Model, 0),
		modelToCredentials:        make(map[string][]string),
		modelLimits:               make(map[string][]ModelLimits),
		staticModelLimits:         make(map[string][]ModelLimits),
		staticModelRealNames:      make(map[string]string),
		dbModelNames:              make(map[string]bool),
		modelAliases:              make(map[string]string),
		modelRealNames:            make(map[string]string),
		modelPassthroughResponses: make(map[string]*bool),
		defaultModelsRPM:          defaultModelsRPM,
		logger:                    logger,
		credentials:               make([]config.CredentialConfig, 0),
		remoteModelsCache:         make(map[string]remoteModelCache),
		cacheExpiration:           5 * time.Minute, // Default cache TTL: 5 minutes
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
			// Register real model name mapping if Model field differs from Name
			if staticModel.Model != "" && staticModel.Model != staticModel.Name {
				m.modelRealNames[staticModel.Name] = staticModel.Model
				logger.Debug("Registered model real name", "alias", staticModel.Name, "real", staticModel.Model)
			}
			// Register explicit passthrough_responses override if set
			if staticModel.PassthroughResponses != nil {
				m.modelPassthroughResponses[staticModel.Name] = staticModel.PassthroughResponses
				logger.Debug("Registered passthrough_responses override",
					"model", staticModel.Name, "value", *staticModel.PassthroughResponses)
			}
			logger.Debug("Added static model from config.yaml",
				"model", staticModel.Name,
				"real_model", staticModel.Model,
				"credential", staticModel.Credential,
				"rpm", staticModel.RPM,
				"tpm", staticModel.TPM)
		}
	}

	// Snapshot the static-only model limits so UpdateDBModels can always
	// restore them when rebuilding after a DB sync cycle.
	for k, v := range m.modelLimits {
		m.staticModelLimits[k] = append([]ModelLimits(nil), v...)
	}
	for k, v := range m.modelRealNames {
		m.staticModelRealNames[k] = v
	}

	return m
}

// GetRealModelName returns the real model name for a given alias (from models[].model config).
// Returns (alias, false) if no separate real name is configured.
func (m *Manager) GetRealModelName(alias string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if real, ok := m.modelRealNames[alias]; ok {
		return real, true
	}
	return alias, false
}

// responsesAPIModelPrefixes lists model name substrings that natively support
// the /v1/responses endpoint.  Checked case-insensitively via strings.Contains.
// Source: https://platform.openai.com/docs/api-reference/responses
var responsesAPIModelPrefixes = []string{
	"codex",
	"gpt-4o",
	"gpt-4.1",
	"gpt-5",
	"o1",
	"o3",
	"o4",
}

// isNativeResponsesModel returns true when modelID matches any known prefix
// that supports the native /v1/responses endpoint.
func isNativeResponsesModel(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, prefix := range responsesAPIModelPrefixes {
		if strings.Contains(lower, prefix) {
			return true
		}
	}
	return false
}

// providerPassthroughDefaults maps provider types to their default passthrough behaviour.
// OpenAI and Proxy natively support /v1/responses so they default to true.
// Vertex AI and Anthropic use the native ProviderResponses converter (Phase 4) instead.
var providerPassthroughDefaults = map[config.ProviderType]bool{
	config.ProviderTypeOpenAI:    true,
	config.ProviderTypeProxy:     true,
	config.ProviderTypeVertexAI:  false,
	config.ProviderTypeGemini:    false,
	config.ProviderTypeAnthropic: false,
	config.ProviderTypeBedrock:   false,
}

// IsPassthroughResponses reports whether Responses API requests for modelID
// should be forwarded to the provider's native /v1/responses endpoint as-is,
// without converting to Chat Completions format.
//
// Priority:
//  1. Explicit config override (passthrough_responses: true/false in models[])
//  2. Auto-detect: true for models in responsesAPIModelPrefixes
func (m *Manager) IsPassthroughResponses(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if v, ok := m.modelPassthroughResponses[modelID]; ok && v != nil {
		return *v
	}
	return isNativeResponsesModel(modelID)
}

// IsPassthroughResponsesForProvider reports whether Responses API requests for modelID
// on the given provider should use the native /v1/responses passthrough.
//
// Priority:
//  1. Explicit config override (passthrough_responses: true/false in models[])
//  2. Provider default (openai=true, others=false)
//  3. Auto-detect by model name prefix
func (m *Manager) IsPassthroughResponsesForProvider(modelID string, providerType config.ProviderType) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if v, ok := m.modelPassthroughResponses[modelID]; ok && v != nil {
		return *v
	}
	if def, ok := providerPassthroughDefaults[providerType]; ok {
		return def
	}
	return isNativeResponsesModel(modelID)
}

// SetCredentials sets the credentials for fetching remote models from proxies
func (m *Manager) SetCredentials(credentials []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.credentials = credentials
}

// SetModelAliases sets the model alias map (alias -> real model name)
func (m *Manager) SetModelAliases(aliases map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.modelAliases = make(map[string]string, len(aliases))
	for alias, target := range aliases {
		if alias == target {
			m.logger.Warn("Model alias points to itself, skipping", "alias", alias)
			continue
		}
		m.modelAliases[alias] = target
		m.logger.Info("Registered model alias", "alias", alias, "target", target)
	}
}

// ResolveAlias resolves a model alias to the real model name.
// Returns the resolved model name and true if it was an alias, or the original name and false otherwise.
func (m *Manager) ResolveAlias(modelID string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if resolved, ok := m.modelAliases[modelID]; ok {
		return resolved, true
	}
	return modelID, false
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

	// Register non-proxy credentials that have no models with an explicit empty list.
	// HasModel has a fallback "return true" for credentials not present in credentialModels;
	// that fallback is intentional for proxy credentials whose model list is fetched dynamically,
	// but it incorrectly allows non-proxy credentials (e.g. openai_backup with no models:)
	// to match any model when static models are configured for other credentials.
	for _, cred := range credentials {
		if cred.Type == config.ProviderTypeProxy {
			continue // proxy models are fetched dynamically via GetAllModels
		}
		if _, exists := m.credentialModels[cred.Name]; !exists {
			m.credentialModels[cred.Name] = []string{}
			m.logger.Debug("Registered non-proxy credential with no models",
				"credential", cred.Name,
				"type", cred.Type,
			)
		}
	}

	m.logger.Info("Loaded models from config",
		"credential_specific", credentialSpecificCount,
		"global_models", globalModelsCount,
	)
}

// UpdateDBModels atomically replaces DB-sourced model limits and credential mappings.
// Static models (from config.yaml snapshot) are always preserved.
//
// staticCreds is the YAML-only credential list. It is used as the "global" target when a
// model has no specific credential — ensuring that synthetic DB credentials (db-model-*)
// are never accidentally assigned to unrelated global models.
//
// allCreds is the complete list (static + DB) used solely for validating explicit credential
// references (checking that a named credential actually exists).
func (m *Manager) UpdateDBModels(dbModels []config.ModelRPMConfig, staticCreds []config.CredentialConfig, allCreds []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Rebuild modelLimits = static snapshot + new DB entries.
	//    Starting from the static snapshot guarantees that removing a DB model never
	//    destroys static limits for a model with the same name.
	newLimits := make(map[string][]ModelLimits, len(m.staticModelLimits)+len(dbModels))
	for k, v := range m.staticModelLimits {
		newLimits[k] = append([]ModelLimits(nil), v...)
	}

	// 2. Rebuild modelRealNames = static snapshot + new DB real names.
	newRealNames := make(map[string]string, len(m.staticModelRealNames)+len(dbModels))
	for k, v := range m.staticModelRealNames {
		newRealNames[k] = v
	}

	// 3. Apply DB model data.
	newDBNames := make(map[string]bool, len(dbModels))
	for _, dm := range dbModels {
		newLimits[dm.Name] = append(newLimits[dm.Name], ModelLimits{
			RPM:        dm.RPM,
			TPM:        dm.TPM,
			Credential: dm.Credential,
		})
		// Only apply DB real name if not already defined in static config.
		// Static YAML takes priority: model_table sync must not override
		// explicitly configured models[].model mappings.
		if dm.Model != "" && dm.Model != dm.Name {
			if _, isStatic := m.staticModelRealNames[dm.Name]; !isStatic {
				newRealNames[dm.Name] = dm.Model
			}
		}
		newDBNames[dm.Name] = true
	}

	m.modelLimits = newLimits
	m.modelRealNames = newRealNames
	m.dbModelNames = newDBNames

	// 4. Rebuild ALL credential↔model mappings from the merged modelLimits.
	//    Proxy-fetched entries (from GetAllModels) are discarded but auto-refresh
	//    on the next GetAllModels call (3-second cache).
	//
	//    allCredNames: full set for validating explicit credential references.
	//    staticCredNames: YAML-only set used when a model has no specific credential so
	//    that synthetic DB credentials (db-model-*) are not mapped to unrelated models.
	allCredNames := make(map[string]bool, len(allCreds))
	for _, c := range allCreds {
		allCredNames[c.Name] = true
	}
	staticCredNames := make(map[string]bool, len(staticCreds))
	for _, c := range staticCreds {
		staticCredNames[c.Name] = true
	}
	nonSyntheticCredNames := make(map[string]bool, len(allCreds))
	for _, c := range allCreds {
		if !strings.HasPrefix(c.Name, "db-model-") {
			nonSyntheticCredNames[c.Name] = true
		}
	}

	newCredentialModels := make(map[string][]string)
	newModelToCredentials := make(map[string][]string)
	credentialModelsSet := make(map[string]map[string]bool)
	modelToCredentialsSet := make(map[string]map[string]bool)

	for modelName, limits := range m.modelLimits {
		for _, limit := range limits {
			if limit.Credential != "" {
				if !allCredNames[limit.Credential] {
					m.logger.Warn("Model references unknown credential",
						"model", modelName, "credential", limit.Credential)
					continue
				}
				addModelToMaps(newCredentialModels, newModelToCredentials,
					credentialModelsSet, modelToCredentialsSet,
					limit.Credential, modelName)
			} else {
				// No specific credential: map to YAML-only (static) credentials.
				// If there are no static creds (DB-only setup), map to non-synthetic
				// DB credentials and still avoid db-model-* synthetic ones.
				credTargets := staticCredNames
				if len(credTargets) == 0 {
					credTargets = nonSyntheticCredNames
				}
				for credName := range credTargets {
					addModelToMaps(newCredentialModels, newModelToCredentials,
						credentialModelsSet, modelToCredentialsSet,
						credName, modelName)
				}
			}
		}
	}

	// Register non-proxy credentials with no models — same logic as in LoadModelsFromConfig.
	for _, c := range allCreds {
		if c.Type == config.ProviderTypeProxy {
			continue
		}
		if _, exists := newCredentialModels[c.Name]; !exists {
			newCredentialModels[c.Name] = []string{}
		}
	}

	// Preserve proxy credential model entries populated by AddModel/UpdateAllProxyCredentials.
	// Proxy models are not in modelLimits so the rebuild above omits them. Without this,
	// every DB sync cycle wipes dynamically-fetched proxy model data and causes routing gaps
	// until the next UpdateAllProxyCredentials tick.
	for _, c := range allCreds {
		if c.Type != config.ProviderTypeProxy {
			continue
		}
		if oldModels, ok := m.credentialModels[c.Name]; ok && len(oldModels) > 0 {
			newCredentialModels[c.Name] = append([]string(nil), oldModels...)
			// Restore modelToCredentials entries for this proxy credential.
			for _, modelID := range oldModels {
				if modelToCredentialsSet[modelID] == nil {
					modelToCredentialsSet[modelID] = make(map[string]bool)
				}
				if !modelToCredentialsSet[modelID][c.Name] {
					newModelToCredentials[modelID] = append(newModelToCredentials[modelID], c.Name)
					modelToCredentialsSet[modelID][c.Name] = true
				}
			}
		}
	}

	m.credentialModels = newCredentialModels
	m.modelToCredentials = newModelToCredentials

	// 5. Invalidate caches so next GetAllModels rebuilds from the updated modelLimits.
	m.allModels = nil
	m.allModelsCache = allModelsCache{}

	m.logger.Info("DB model data updated",
		"db_models", len(m.dbModelNames),
		"total_model_limits", len(m.modelLimits),
	)
}

// GetAllModels returns all unique models across all credentials with caching
func (m *Manager) GetAllModels() ModelsResponse {
	// Check cache first (fast path without holding full lock)
	m.mu.RLock()
	if !m.allModelsCache.expiresAt.IsZero() && utils.NowUTC().Before(m.allModelsCache.expiresAt) {
		// Copy response and its Data slice while holding lock to prevent TOCTOU
		// and to avoid sharing the backing array with callers.
		cachedData := append([]Model(nil), m.allModelsCache.response.Data...)
		cachedObject := m.allModelsCache.response.Object
		m.mu.RUnlock()
		m.logger.Debug("Returning cached all models",
			"models_count", len(cachedData),
		)
		return ModelsResponse{Object: cachedObject, Data: cachedData}
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
				Created: converterutil.GetCurrentTimestamp(),
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
	successfullyFetched := make(map[string]bool)
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
		remoteModels, err := m.GetRemoteModelsWithError(context.Background(), &cred)
		if err != nil {
			m.logger.Warn("Failed to fetch models from proxy during full model list refresh",
				"credential", cred.Name,
				"error", err,
			)
			continue
		}
		successfullyFetched[cred.Name] = true
		m.logger.Debug("Got models from proxy",
			"credential", cred.Name,
			"remote_models_count", len(remoteModels),
			"current_total", len(models),
		)
		added := 0
		for _, model := range remoteModels {
			// Always record credential→model mapping for ALL credentials that support this model
			// (not just the first one that returned it)
			modelUpdates[model.ID] = append(modelUpdates[model.ID], cred.Name)
			if !modelMap[model.ID] {
				models = append(models, model)
				modelMap[model.ID] = true
				added++
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

	// Remove stale mappings for successfully-fetched proxy credentials before adding fresh ones.
	// Only touch credentials we actually got a response from — failed fetches are left
	// unchanged to avoid false negatives on transient errors.
	if len(successfullyFetched) > 0 {
		for modelID, creds := range m.modelToCredentials {
			var kept []string
			for _, c := range creds {
				if !successfullyFetched[c] {
					kept = append(kept, c)
				}
			}
			if len(kept) == 0 {
				delete(m.modelToCredentials, modelID)
			} else {
				m.modelToCredentials[modelID] = kept
			}
		}
	}

	// Add fresh mappings from this refresh cycle.
	for modelID, creds := range modelUpdates {
		m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], creds...)
	}

	// Cache a copy so the cached backing array is independent from the returned response.
	m.allModelsCache = allModelsCache{
		response: ModelsResponse{
			Object: response.Object,
			Data:   append([]Model(nil), models...),
		},
		expiresAt: utils.NowUTC().Add(3 * time.Second),
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
		for _, model := range models {
			if model == modelID {
				return true
			}
		}
		// Credential has a non-empty registered model list (static or dynamic via AddModel)
		// but the requested model isn't in it — deny authoritatively.
		if len(models) > 0 {
			return false
		}
		// len==0: credential was registered with an explicit empty list (non-proxy cred with
		// no model config); fall through to the hasStaticModels check below.
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

// IsEnabled returns whether model filtering should be used.
// Returns true when static model limits are configured OR when dynamic proxy
// model data has been fetched (modelToCredentials is non-empty). This ensures
// that chain setups without a static models: section still benefit from
// per-credential model filtering once proxy model lists are discovered.
func (m *Manager) IsEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.modelLimits) > 0 || len(m.modelToCredentials) > 0
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
// Returns -1 for unlimited (when RPM is 0 or not set), same semantics as findTPMLimit.
func findRPMLimit(limits []ModelLimits, credentialName string) (int, bool) {
	convertRPM := func(v int) int {
		if v == 0 {
			return -1 // 0 means unlimited
		}
		return v
	}
	return findLimit(limits, credentialName, func(ml *ModelLimits) int { return ml.RPM }, convertRPM)
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

// providerTypeLiteLLMPrefix maps our provider type to the LiteLLM-compatible model prefix.
// vertex-ai uses underscore to match LiteLLM's "vertex_ai/model" convention.
var providerTypeLiteLLMPrefix = map[config.ProviderType]string{
	config.ProviderTypeOpenAI:    "openai",
	config.ProviderTypeVertexAI:  "vertex_ai",
	config.ProviderTypeGemini:    "gemini",
	config.ProviderTypeAnthropic: "anthropic",
	config.ProviderTypeBedrock:   "bedrock",
	config.ProviderTypeProxy:     "openai",
}

// GetAllModelsWithAccessGroups returns all models in "provider/model-id" format,
// used when the caller requests ?include_model_access_groups=True.
// Each (provider, model) pair appears at most once in the response.
func (m *Manager) GetAllModelsWithAccessGroups() ModelsResponse {
	m.mu.RLock()
	defer m.mu.RUnlock()

	credProvider := make(map[string]string, len(m.credentials))
	for _, cred := range m.credentials {
		prefix, ok := providerTypeLiteLLMPrefix[cred.Type]
		if !ok {
			prefix = string(cred.Type)
		}
		credProvider[cred.Name] = prefix
	}

	seen := make(map[string]bool)
	result := make([]Model, 0, len(m.modelToCredentials))

	for modelID, creds := range m.modelToCredentials {
		for _, credName := range creds {
			prefix, ok := credProvider[credName]
			if !ok {
				continue
			}
			prefixedID := prefix + "/" + modelID
			if seen[prefixedID] {
				continue
			}
			seen[prefixedID] = true
			result = append(result, Model{
				ID:      prefixedID,
				Object:  "model",
				Created: converterutil.GetCurrentTimestamp(),
				OwnedBy: prefix,
			})
		}
	}

	return ModelsResponse{Object: "list", Data: result}
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
			Created: converterutil.GetCurrentTimestamp(),
			OwnedBy: "system",
		})
	}

	return result
}

// GetRemoteModels fetches models from a remote proxy credential with caching.
// Deprecated: use GetRemoteModelsWithError to handle upstream fetch errors explicitly.
func (m *Manager) GetRemoteModels(cred *config.CredentialConfig) []Model {
	models, err := m.GetRemoteModelsWithError(context.Background(), cred)
	if err != nil {
		return nil
	}
	return models
}

// GetRemoteModelsWithError fetches models from a remote proxy credential with caching.
// Returns explicit error when remote fetch fails.
func (m *Manager) GetRemoteModelsWithError(ctx context.Context, cred *config.CredentialConfig) ([]Model, error) {
	if cred.Type != config.ProviderTypeProxy {
		return nil, nil
	}

	// Check cache first
	m.mu.RLock()
	if cached, ok := m.remoteModelsCache[cred.Name]; ok && !cached.expiresAt.IsZero() && utils.NowUTC().Before(cached.expiresAt) {
		// Defensive copy to prevent callers from corrupting cached data
		cachedModels := append([]Model(nil), cached.models...)
		cachedCount := len(cachedModels)
		expiresIn := time.Until(cached.expiresAt).Seconds()
		m.mu.RUnlock()
		m.logger.Debug("Using cached remote models",
			"credential", cred.Name,
			"models_count", cachedCount,
			"expires_in", expiresIn,
		)
		return cachedModels, nil
	}
	m.mu.RUnlock()

	m.logger.Debug("Fetching remote models from proxy",
		"credential", cred.Name,
		"base_url", cred.BaseURL,
	)

	models, err := m.fetchRemoteModelsFromHealth(ctx, cred)
	if err != nil {
		m.logger.Debug("Failed to fetch remote models from proxy health; falling back to /v1/models",
			"credential", cred.Name,
			"error", err,
		)

		// Fallback for non-AAR proxies that expose /v1/models but not /health.
		var modelsResp ModelsResponse
		if err := httputil.FetchJSONFromProxy(ctx, cred, "/v1/models", m.logger, &modelsResp); err != nil {
			m.logger.Error("Failed to fetch remote models",
				"credential", cred.Name,
				"error", err,
			)
			return nil, err
		}
		models = modelsResp.Data
	}

	// Cache the result
	m.mu.Lock()
	m.remoteModelsCache[cred.Name] = remoteModelCache{
		models:    models,
		expiresAt: utils.NowUTC().Add(m.cacheExpiration),
	}
	m.mu.Unlock()

	m.logger.Debug("Cached remote models",
		"credential", cred.Name,
		"models_count", len(models),
		"expires_in", m.cacheExpiration.Seconds(),
	)

	return models, nil
}

func (m *Manager) fetchRemoteModelsFromHealth(ctx context.Context, cred *config.CredentialConfig) ([]Model, error) {
	var health httputil.ProxyHealthResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/health", m.logger, &health); err != nil {
		return nil, err
	}

	if len(health.Credentials) == 0 || len(health.Models) == 0 {
		return nil, errProxyHealthModelMetadataUnavailable
	}

	modelsByID := make(map[string]Model)
	for _, modelStats := range health.Models {
		credStats, ok := health.Credentials[modelStats.Credential]
		if !ok {
			continue
		}
		if credStats.IsFallback != cred.IsFallback {
			continue
		}
		if modelStats.Model == "" {
			continue
		}
		if _, exists := modelsByID[modelStats.Model]; exists {
			continue
		}
		modelsByID[modelStats.Model] = Model{
			ID:      modelStats.Model,
			Object:  "model",
			OwnedBy: credStats.Type,
		}
	}

	models := make([]Model, 0, len(modelsByID))
	for _, model := range modelsByID {
		models = append(models, model)
	}
	slices.SortFunc(models, func(a, b Model) int {
		return strings.Compare(a.ID, b.ID)
	})

	return models, nil
}
