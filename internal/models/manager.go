// Package models manages the router's model catalog, aliases, pricing and routing metadata.
package models

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/utils"
)

var (
	errProxyHealthModelMetadataUnavailable = errors.New("proxy health model metadata unavailable")
	errProxyCredentialChanged              = errors.New("proxy credential changed during refresh")
)

// ModelPrice contains pricing information for a single model
type ModelPrice struct {
	// Regular tokens (input/output)
	InputCostPerToken  float64 `json:"input_cost_per_token"`
	OutputCostPerToken float64 `json:"output_cost_per_token"`

	// Tiered pricing: tokens above 200k threshold billed at a different rate
	InputCostPerTokenAbove200k  float64 `json:"input_cost_per_token_above_200k_tokens,omitempty"`
	OutputCostPerTokenAbove200k float64 `json:"output_cost_per_token_above_200k_tokens,omitempty"`
	InputCostPerTokenAbove272k  float64 `json:"input_cost_per_token_above_272k_tokens,omitempty"`
	OutputCostPerTokenAbove272k float64 `json:"output_cost_per_token_above_272k_tokens,omitempty"`

	// Full-session tiered pricing (Alibaba Cloud style): once the prompt
	// exceeds the threshold, the entire request is billed at that tier's
	// rate rather than only the tokens beyond it. Same semantics as the
	// 272k tier above, generalised to more brackets.
	InputCostPerTokenAbove32k           float64 `json:"input_cost_per_token_above_32k_tokens,omitempty"`
	OutputCostPerTokenAbove32k          float64 `json:"output_cost_per_token_above_32k_tokens,omitempty"`
	CacheReadInputTokenCostAbove32k     float64 `json:"cache_read_input_token_cost_above_32k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove32k float64 `json:"cache_creation_input_token_cost_above_32k_tokens,omitempty"`

	InputCostPerTokenAbove128k           float64 `json:"input_cost_per_token_above_128k_tokens,omitempty"`
	OutputCostPerTokenAbove128k          float64 `json:"output_cost_per_token_above_128k_tokens,omitempty"`
	CacheReadInputTokenCostAbove128k     float64 `json:"cache_read_input_token_cost_above_128k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove128k float64 `json:"cache_creation_input_token_cost_above_128k_tokens,omitempty"`

	InputCostPerTokenAbove256k           float64 `json:"input_cost_per_token_above_256k_tokens,omitempty"`
	OutputCostPerTokenAbove256k          float64 `json:"output_cost_per_token_above_256k_tokens,omitempty"`
	CacheReadInputTokenCostAbove256k     float64 `json:"cache_read_input_token_cost_above_256k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove256k float64 `json:"cache_creation_input_token_cost_above_256k_tokens,omitempty"`

	InputCostPerTokenAbove512k           float64 `json:"input_cost_per_token_above_512k_tokens,omitempty"`
	OutputCostPerTokenAbove512k          float64 `json:"output_cost_per_token_above_512k_tokens,omitempty"`
	CacheReadInputTokenCostAbove512k     float64 `json:"cache_read_input_token_cost_above_512k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove512k float64 `json:"cache_creation_input_token_cost_above_512k_tokens,omitempty"`

	// Audio tokens (can be more specific than regular tokens)
	InputCostPerAudioToken  float64 `json:"input_cost_per_audio_token,omitempty"`
	OutputCostPerAudioToken float64 `json:"output_cost_per_audio_token,omitempty"`

	// Image tokens (can be more specific than regular tokens)
	InputCostPerImageToken  float64 `json:"input_cost_per_image_token,omitempty"`
	OutputCostPerImageToken float64 `json:"output_cost_per_image_token,omitempty"`

	// Reasoning tokens (deep thinking models)
	OutputCostPerReasoningToken float64 `json:"output_cost_per_reasoning_token,omitempty"`

	// ReasoningTokensAdditive selects which reasoning-token counting semantics
	// this model's upstream uses. Default (false) is "included": ReasoningTokens
	// is already part of CompletionTokens, as OpenAI o-series and GLM report it
	// (CalculateTokenCosts subtracts it back out before pricing the regular
	// output tokens, then prices it again at the reasoning rate). true is
	// "additive": the upstream reports ReasoningTokens on top of CompletionTokens
	// (observed on Grok), so it must not be subtracted from CompletionTokens when
	// deriving regular output tokens, or those tokens would be priced as neither
	// output nor reasoning. Defaults to false so existing models are unaffected.
	ReasoningTokensAdditive bool `json:"reasoning_tokens_additive,omitempty"`

	// Cached/Prediction tokens
	OutputCostPerCachedToken                     float64 `json:"output_cost_per_cached_token,omitempty"`
	InputCostPerCachedToken                      float64 `json:"input_cost_per_cached_token,omitempty"`
	CacheReadInputTokenCost                      float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheCreationInputTokenCost                  float64 `json:"cache_creation_input_token_cost,omitempty"`
	CacheReadInputTokenCostAbove200k             float64 `json:"cache_read_input_token_cost_above_200k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove200k         float64 `json:"cache_creation_input_token_cost_above_200k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove1hr          float64 `json:"cache_creation_input_token_cost_above_1hr,omitempty"`
	CacheCreationInputTokenCostAbove1hrAbove200k float64 `json:"cache_creation_input_token_cost_above_1hr_above_200k_tokens,omitempty"`
	CacheReadInputTokenCostAbove272k             float64 `json:"cache_read_input_token_cost_above_272k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove272k         float64 `json:"cache_creation_input_token_cost_above_272k_tokens,omitempty"`
	CacheReadInputAudioTokenCost                 float64 `json:"cache_read_input_audio_token_cost,omitempty"`
	OutputCostPerPredictionToken                 float64 `json:"output_cost_per_prediction_token,omitempty"`

	// Vision/Images cost per image (not per token)
	OutputCostPerImage float64 `json:"output_cost_per_image,omitempty"`

	// Built-in web search tool pricing. Values are per query/call, keyed by
	// search_context_size_low|medium|high in LiteLLM's price format.
	SearchContextCostPerQuery map[string]float64 `json:"search_context_cost_per_query,omitempty"`
	WebSearchBillingUnit      string             `json:"web_search_billing_unit,omitempty"`
	LiteLLMProvider           string             `json:"litellm_provider,omitempty"`
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

// GetPrice returns the price for a model, or nil if not found. Tries the
// raw (case/whitespace-folded, unsplit) key before the normalized
// (provider-prefix-stripped) one, matching GetPriceAny's two-pass priority
// so the two functions never disagree about which entry a given name
// resolves to.
func (r *ModelPriceRegistry) GetPrice(modelName string) *ModelPrice {
	r.mu.RLock()
	defer r.mu.RUnlock()
	raw := strings.ToLower(strings.TrimSpace(modelName))
	if price := r.prices[raw]; price != nil {
		return price
	}
	return r.prices[NormalizeModelName(modelName)]
}

// GetPriceAny looks up prices for multiple candidate model names under a
// single read lock, returning the first match in the given priority order.
// This avoids a concurrent Update() straddling two map generations, which
// can happen when callers issue separate GetPrice calls per candidate.
//
// Each candidate is tried in full — raw lowercase key first (no prefix
// stripping, so an explicit entry like "google/gemini-3-flash-preview-
// highlimits" is found before any normalised fallback), then the
// NormalizeModelName (provider-prefix-stripped) key — before moving on to
// the next candidate. This preserves the caller's priority order (e.g.
// publicModelID > modelID > realModelID) exactly: a higher-priority
// candidate's normalised match always wins over a lower-priority candidate's
// raw match. Trying every candidate's raw key before any candidate's
// normalised key would let a coincidental raw hit on a low-priority
// candidate (realModelID, the literal provider model name, is especially
// likely to double as a raw price-file key) shadow the correct match on a
// higher-priority one.
func (r *ModelPriceRegistry) GetPriceAny(modelNames ...string) (string, *ModelPrice) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, modelName := range modelNames {
		if modelName == "" {
			continue
		}
		raw := strings.ToLower(strings.TrimSpace(modelName))
		if price := r.prices[raw]; price != nil {
			return modelName, price
		}
		if price := r.prices[NormalizeModelName(modelName)]; price != nil {
			return modelName, price
		}
	}

	return "", nil
}

// Update safely updates the registry with new prices
func (r *ModelPriceRegistry) Update(prices map[string]*ModelPrice) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prices = make(map[string]*ModelPrice)
	for k, v := range prices {
		r.prices[k] = v
		// Store raw lowercase key so that provider-prefixed model names
		// (e.g. "openrouter/gpt-5-mini") can be matched in GetPriceAny's
		// first pass before normalisation strips the prefix.
		raw := strings.ToLower(strings.TrimSpace(k))
		r.prices[raw] = v
	}
	r.lastUpdate = utils.NowUTC()
}

// MergeDB applies DB-sourced prices on top of the existing registry without
// removing prices that came from the file-based price list.
// DB prices take precedence for models that appear in both sources.
//
// A DB override only ever replaces the exact key it names (plus that key's
// own raw/lowercased form) — it deliberately does NOT sweep the registry for
// other keys that merely share a normalized form. A price entry deliberately
// keyed "provider/model" (e.g. "openrouter/gpt-5-mini", or any alias that
// needs a price distinct from its canonical model — see
// "gemini-3-flash-preview-highlimits" in client_a.libsonnet) is independent
// of the plain model it happens to normalize to; an override for the plain
// model must not silently leak into it. If a DB-sourced override is meant
// to apply to such an alias too, the DB record must name that alias's exact
// key itself.
func (r *ModelPriceRegistry) MergeDB(dbPrices map[string]*ModelPrice) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for k, v := range dbPrices {
		r.prices[k] = v
		// Store raw lowercase key (same logic as Update) so that DB-sourced
		// provider-prefixed entries are also findable in GetPriceAny pass 1.
		raw := strings.ToLower(strings.TrimSpace(k))
		r.prices[raw] = v
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
	Weight     int    // Weighted round-robin weight (0 = unset, falls back to credential default / 1)
	Credential string // If set, limits apply only to this credential
}

type ScopeMetadata struct {
	Scopes          []string
	DeniedScopes    []string
	ScopeExpression *scope.Expression
}

// remoteModelCache stores cached remote models with expiration time
type remoteModelCache struct {
	credential    config.CredentialConfig
	models        []Model
	scopeSnapshot *remoteScopeSnapshot
	expiresAt     time.Time
}

type remoteScopeSnapshot struct {
	providerScopes ScopeMetadata
	modelScopes    map[string]ScopeMetadata
	modelWeights   map[string]int
	scopeKnown     bool
}

// allModelsCache stores cached result of GetAllModels
type allModelsCache struct {
	response  ModelsResponse
	expiresAt time.Time
}

const (
	allModelsCacheTTL        = 3 * time.Second
	scopedAllModelsCacheSize = 256
)

// Manager handles model discovery and mapping
type Manager struct {
	mu                           sync.RWMutex
	credentialModels             map[string][]string          // credential name -> list of model IDs
	allModels                    []Model                      // deduplicated list of all models
	modelToCredentials           map[string][]string          // model ID -> list of credential names
	modelLimits                  map[string][]ModelLimits     // model ID -> limits (may have multiple entries for different credentials)
	staticModelLimits            map[string][]ModelLimits     // immutable snapshot of limits from config.yaml (never modified after New())
	staticModelRealNames         map[string]string            // immutable snapshot of global real names from config.yaml
	staticModelRealNamesPerCred  map[string]map[string]string // immutable snapshot of per-credential real names: credential -> alias -> real name
	modelPassthroughResponses    map[string]*bool             // model name -> explicit passthrough_responses override (nil = auto)
	modelPassthroughMessages     map[string]*bool             // model name -> explicit passthrough_messages override (nil = provider default)
	dynamicModelWeights          map[string]map[string]int    // model ID -> credential -> weight learned from upstream /health
	dynamicModelScopes           map[string]map[string]ScopeMetadata
	dbModelNames                 map[string]bool              // model names that were loaded from LiteLLM DB (for hot-reload diffing)
	modelAliases                 map[string]string            // alias -> real model name (from model_alias config)
	clientModelIDs               map[string]struct{}          // exact advertised canonical client IDs
	clientModelSurfaceConfigured bool                         // distinguishes an omitted boundary from an explicit empty boundary
	publicModelAliases           map[string]string            // client alias -> canonical LiteLLM public deployment identity
	acceptedModelAliases         map[string]string            // accepted client alias -> canonical model, hidden from discovery
	modelRealNames               map[string]string            // alias name -> real model name (global, no specific credential)
	modelRealNamesPerCred        map[string]map[string]string // credential -> alias -> real model name (for credential-specific entries)
	credentialMappingsReady      bool                         // true after static/DB credential mappings have been initialized
	defaultModelsRPM             int                          // default RPM for models
	logger                       *slog.Logger
	credentials                  []config.CredentialConfig // credentials for fetching remote models
	credentialsConfigured        bool
	remoteModelsCache            map[string]remoteModelCache        // cache for remote models per credential (credentialName -> cache)
	cacheExpiration              time.Duration                      // how long to cache remote models (default 5 minutes)
	allModelsCache               allModelsCache                     // cached result of GetAllModels (3 second TTL)
	scopedAllModelsCache         *lru.Cache[string, allModelsCache] // cached scoped /v1/models responses
}

// New creates a new model manager
func New(logger *slog.Logger, defaultModelsRPM int, staticModels []config.ModelRPMConfig) *Manager {
	m := &Manager{
		credentialModels:            make(map[string][]string),
		allModels:                   make([]Model, 0),
		modelToCredentials:          make(map[string][]string),
		modelLimits:                 make(map[string][]ModelLimits),
		staticModelLimits:           make(map[string][]ModelLimits),
		staticModelRealNames:        make(map[string]string),
		staticModelRealNamesPerCred: make(map[string]map[string]string),
		dbModelNames:                make(map[string]bool),
		modelAliases:                make(map[string]string),
		clientModelIDs:              make(map[string]struct{}),
		publicModelAliases:          make(map[string]string),
		acceptedModelAliases:        make(map[string]string),
		modelRealNames:              make(map[string]string),
		modelRealNamesPerCred:       make(map[string]map[string]string),
		modelPassthroughResponses:   make(map[string]*bool),
		modelPassthroughMessages:    make(map[string]*bool),
		dynamicModelWeights:         make(map[string]map[string]int),
		dynamicModelScopes:          make(map[string]map[string]ScopeMetadata),
		defaultModelsRPM:            defaultModelsRPM,
		logger:                      logger,
		credentials:                 make([]config.CredentialConfig, 0),
		remoteModelsCache:           make(map[string]remoteModelCache),
		cacheExpiration:             5 * time.Minute, // Default cache TTL: 5 minutes
		scopedAllModelsCache:        newScopedAllModelsCache(),
	}

	// Load static models from config.yaml
	if len(staticModels) > 0 {
		logger.Info("Loading static models from config.yaml", "models_count", len(staticModels))
		for _, staticModel := range staticModels {
			m.modelLimits[staticModel.Name] = append(m.modelLimits[staticModel.Name], ModelLimits{
				RPM:        staticModel.RPM,
				TPM:        staticModel.TPM,
				Weight:     staticModel.Weight,
				Credential: staticModel.Credential,
			})
			// Register real model name mapping if Model field differs from Name.
			// Credential-specific entries go into the per-credential map so that the
			// same alias (e.g. "claude-haiku-4.5") can map to a different real name on
			// each provider (e.g. Bedrock vs OpenRouter) without overwriting each other.
			if staticModel.Model != "" && staticModel.Model != staticModel.Name {
				if staticModel.Credential != "" {
					if m.modelRealNamesPerCred[staticModel.Credential] == nil {
						m.modelRealNamesPerCred[staticModel.Credential] = make(map[string]string)
					}
					m.modelRealNamesPerCred[staticModel.Credential][staticModel.Name] = staticModel.Model
				} else {
					m.modelRealNames[staticModel.Name] = staticModel.Model
				}
				logger.Debug("Registered model real name",
					"alias", staticModel.Name,
					"real", staticModel.Model,
					"credential", staticModel.Credential)
			}
			// Register explicit passthrough_responses override if set
			if staticModel.PassthroughResponses != nil {
				m.modelPassthroughResponses[staticModel.Name] = staticModel.PassthroughResponses
				logger.Debug("Registered passthrough_responses override",
					"model", staticModel.Name, "value", *staticModel.PassthroughResponses)
			}
			// Register explicit passthrough_messages override if set
			if staticModel.PassthroughMessages != nil {
				m.modelPassthroughMessages[staticModel.Name] = staticModel.PassthroughMessages
				logger.Debug("Registered passthrough_messages override",
					"model", staticModel.Name, "value", *staticModel.PassthroughMessages)
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
	for cred, names := range m.modelRealNamesPerCred {
		snapshot := make(map[string]string, len(names))
		for alias, realName := range names {
			snapshot[alias] = realName
		}
		m.staticModelRealNamesPerCred[cred] = snapshot
	}

	return m
}

// GetRealModelName returns the global real model name for a given alias (from models[].model
// config entries that have no specific credential). Returns (alias, false) if not found.
func (m *Manager) GetRealModelName(alias string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if realName, ok := m.modelRealNames[alias]; ok {
		return realName, true
	}
	return alias, false
}

// GetRealModelNameForCredential returns the real model name for a given alias and credential.
// It checks the per-credential map first (for entries with a specific credential in config),
// then falls back to the global map (entries without a credential).
// Returns (alias, false) if no real name mapping is configured for this combination.
func (m *Manager) GetRealModelNameForCredential(alias, credential string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if names, ok := m.modelRealNamesPerCred[credential]; ok {
		if realName, ok := names[alias]; ok {
			return realName, true
		}
	}
	if realName, ok := m.modelRealNames[alias]; ok {
		return realName, true
	}
	return alias, false
}

// GetAliasesForCredentialRealModel returns route-visible model IDs on a
// credential that resolve to the same provider-facing model name.
func (m *Manager) GetAliasesForCredentialRealModel(credential, realModel string) []string {
	if realModel == "" {
		return nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	aliases := make([]string, 0)
	seen := make(map[string]struct{})
	for _, alias := range m.credentialModels[credential] {
		resolved := alias
		resolvedPerCredential := false
		if names, ok := m.modelRealNamesPerCred[credential]; ok {
			if realName, ok := names[alias]; ok {
				resolved = realName
				resolvedPerCredential = true
			}
		}
		if !resolvedPerCredential {
			if realName, ok := m.modelRealNames[alias]; ok {
				resolved = realName
			}
		}
		if resolved != realModel {
			continue
		}
		if _, ok := seen[alias]; ok {
			continue
		}
		seen[alias] = struct{}{}
		aliases = append(aliases, alias)
	}
	slices.Sort(aliases)
	return aliases
}

// GetAliasesForModel returns every alias — across all credentials that serve
// modelID — that resolves to the same provider-facing realModelID. Unlike
// GetAliasesForCredentialRealModel this doesn't require a credential to already
// be picked, which matters for callers (e.g. key model-allow-list enforcement)
// that run before credential selection.
func (m *Manager) GetAliasesForModel(modelID, realModelID string) []string {
	aliases := []string{modelID}
	if realModelID == "" {
		return aliases
	}
	seen := map[string]struct{}{modelID: {}}
	for _, credential := range m.GetCredentialsForModel(modelID) {
		for _, alias := range m.GetAliasesForCredentialRealModel(credential, realModelID) {
			if _, ok := seen[alias]; ok {
				continue
			}
			seen[alias] = struct{}{}
			aliases = append(aliases, alias)
		}
	}
	return aliases
}

func (m *Manager) publicModelAliasTargetActiveLocked(target string) bool {
	_, active := m.clientCanonicalRouteTargetLocked(target)
	return active
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
	config.ProviderTypeAIR:       true,
	config.ProviderTypeVertexAI:  false,
	config.ProviderTypeGemini:    false,
	config.ProviderTypeAnthropic: false,
	config.ProviderTypeCometAPI:  false,
	config.ProviderTypeProMan:    false,
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

func (m *Manager) HasPassthroughResponsesOverride(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	value, ok := m.modelPassthroughResponses[modelID]
	return ok && value != nil
}

// IsPassthroughMessagesForProvider reports whether /v1/messages requests for modelID
// on the given provider should be forwarded to the provider's native /v1/messages
// endpoint as-is, instead of the Messages->Chat->Messages round trip.
//
// Priority:
//  1. Explicit config override (passthrough_messages: true/false in models[])
//  2. Provider default: true for Anthropic; true for CometAPI unless it's running in
//     its OpenAI-compatible protocol mode (openai_proto), where /v1/messages already
//     converts to the Chat Completions wire format it actually speaks; false otherwise.
func (m *Manager) IsPassthroughMessagesForProvider(modelID string, providerType config.ProviderType, openAIProtocol bool) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if v, ok := m.modelPassthroughMessages[modelID]; ok && v != nil {
		return *v
	}
	switch providerType {
	case config.ProviderTypeAnthropic:
		return true
	case config.ProviderTypeCometAPI:
		return !openAIProtocol
	default:
		return false
	}
}

// SetCredentials sets the credentials for fetching remote models from proxies
func (m *Manager) SetCredentials(credentials []config.CredentialConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()

	previous := make(map[string]config.CredentialConfig, len(m.credentials))
	for _, credential := range m.credentials {
		previous[credential.Name] = credential
	}
	next := append(make([]config.CredentialConfig, 0, len(credentials)), credentials...)
	active := make(map[string]bool, len(next))
	for i := range next {
		active[next[i].Name] = true
		old, ok := previous[next[i].Name]
		if ok && next[i].SameProviderIdentity(old) {
			next[i].ProviderScopes = append([]string(nil), old.ProviderScopes...)
			next[i].ProviderDeniedScopes = append([]string(nil), old.ProviderDeniedScopes...)
			next[i].ProviderScopeExpression = scope.NormalizeExpression(old.ProviderScopeExpression)
			next[i].ProviderScopeKnown = old.ProviderScopeKnown
			continue
		}
		delete(m.remoteModelsCache, next[i].Name)
		if ok && next[i].IsProxyLike() && next[i].ProviderScopeExpression == nil &&
			len(next[i].ProviderScopes) == 0 && len(next[i].ProviderDeniedScopes) == 0 {
			next[i].ProviderScopeExpression = scope.FalseExpression()
		}
	}
	for credentialName := range m.remoteModelsCache {
		if !active[credentialName] {
			delete(m.remoteModelsCache, credentialName)
		}
	}
	m.credentials = next
	m.credentialsConfigured = true
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
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
	// Public model discovery includes aliases, so a changed alias set invalidates
	// both the rendered response and the accumulated discovery snapshot.
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
}

// SetClientModelIDs installs an explicit product boundary between identifiers
// accepted from an ordinary client and internal/provider routing names. Calling
// this method with an empty slice intentionally creates a deny-all boundary;
// not calling it preserves the legacy inferred model surface.
func (m *Manager) SetClientModelIDs(modelIDs []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.clientModelSurfaceConfigured = true
	m.clientModelIDs = make(map[string]struct{}, len(modelIDs))
	for _, modelID := range modelIDs {
		if modelID == "" {
			m.logger.Warn("Invalid client model ID, skipping")
			continue
		}
		m.clientModelIDs[modelID] = struct{}{}
	}
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
}

// SetPublicModelAliases configures client-visible aliases that share the
// deployment identity of an exact LiteLLM public model. This mapping is kept
// separate from model_alias: model_alias translates a routable public model to
// its provider backend, while public_model_alias translates an additional
// client name to that canonical public model. Combining the two maps would
// create cycles for common short backend names (for example
// gpt-4.1 -> openai/gpt-4.1 -> gpt-4.1).
func (m *Manager) SetPublicModelAliases(aliases map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.publicModelAliases = make(map[string]string, len(aliases))
	for alias, target := range aliases {
		if alias == "" || target == "" || alias == target {
			m.logger.Warn("Invalid public model alias, skipping", "alias", alias, "target", target)
			continue
		}
		m.publicModelAliases[alias] = target
		m.logger.Info("Registered public model alias", "alias", alias, "target", target)
	}
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
}

func (m *Manager) SetAcceptedModelAliases(aliases map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acceptedModelAliases = make(map[string]string, len(aliases))
	for alias, target := range aliases {
		if alias == "" || target == "" || alias == target {
			m.logger.Warn("Invalid accepted model alias, skipping", "alias", alias, "target", target)
			continue
		}
		m.acceptedModelAliases[alias] = target
		m.logger.Info("Registered accepted model alias", "alias", alias, "target", target)
	}
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
}

func (m *Manager) clientAliasTargetLocked(modelID string) (string, bool, bool) {
	publicTarget, publicConfigured := m.publicModelAliases[modelID]
	acceptedTarget, acceptedConfigured := m.acceptedModelAliases[modelID]
	if publicConfigured && acceptedConfigured && publicTarget != acceptedTarget {
		return modelID, true, false
	}
	if publicConfigured {
		return publicTarget, true, true
	}
	if acceptedConfigured {
		return acceptedTarget, true, true
	}
	return modelID, false, true
}

func (m *Manager) IsClientModelIDRoutable(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if target, aliased, unambiguous := m.clientAliasTargetLocked(modelID); aliased {
		if !unambiguous || !m.publicModelAliasTargetActiveLocked(target) {
			return false
		}
		if m.clientModelSurfaceConfigured {
			_, allowed := m.clientModelIDs[target]
			return allowed
		}
		return true
	}
	if m.clientModelSurfaceConfigured {
		if _, allowed := m.clientModelIDs[modelID]; !allowed {
			return false
		}
	}
	_, routable := m.clientCanonicalRouteTargetLocked(modelID)
	return routable
}

func (m *Manager) ResolvePublicModelAlias(modelID string) (string, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	target, aliased, unambiguous := m.clientAliasTargetLocked(modelID)
	if !aliased {
		return modelID, false, nil
	}
	if !unambiguous || !m.publicModelAliasTargetActiveLocked(target) {
		return modelID, false, fmt.Errorf("public model alias %q is not routable", modelID)
	}
	return target, true, nil
}

func (m *Manager) clientCanonicalRouteTargetLocked(modelID string) (string, bool) {
	if len(m.modelToCredentials[modelID]) > 0 {
		return modelID, true
	}
	target, aliased := m.modelAliases[modelID]
	if aliased && isActiveModelAlias(
		modelID,
		target,
		len(m.modelToCredentials[target]) > 0,
		m.modelAliases,
	) {
		return target, true
	}
	return modelID, false
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

// isActiveModelAlias is the shared activation rule for catalog projection and
// model-scope equivalence. Routability belongs to the immediate target; alias
// keys are not recursively made routable by another active alias.
func isActiveModelAlias(alias, target string, targetRoutable bool, aliases map[string]string) bool {
	if alias == "" || target == "" || !targetRoutable {
		return false
	}
	configuredTarget, configured := aliases[alias]
	if !configured || configuredTarget != target {
		return false
	}
	return !modelAliasPathHasCycle(alias, aliases)
}

func modelAliasPathHasCycle(start string, aliases map[string]string) bool {
	seen := make(map[string]struct{}, len(aliases))
	current := start
	for {
		if _, repeated := seen[current]; repeated {
			return true
		}
		seen[current] = struct{}{}

		next, isAlias := aliases[current]
		if !isAlias {
			return false
		}
		current = next
	}
}

// modelScopePatternMatch implements LiteLLM's '*' model-scope wildcard while
// treating every other byte literally. This deliberately avoids turning DB
// allowlists into regular expressions.
func modelScopePatternMatch(model, pattern string) bool {
	modelIndex, patternIndex := 0, 0
	starIndex, starModelIndex := -1, 0
	for modelIndex < len(model) {
		if patternIndex < len(pattern) && pattern[patternIndex] != '*' && pattern[patternIndex] == model[modelIndex] {
			modelIndex++
			patternIndex++
			continue
		}
		if patternIndex < len(pattern) && pattern[patternIndex] == '*' {
			starIndex = patternIndex
			starModelIndex = modelIndex
			patternIndex++
			continue
		}
		if starIndex < 0 {
			return false
		}
		patternIndex = starIndex + 1
		starModelIndex++
		modelIndex = starModelIndex
	}
	for patternIndex < len(pattern) && pattern[patternIndex] == '*' {
		patternIndex++
	}
	return patternIndex == len(pattern)
}

func modelScopeEntryAllows(modelID, allowedModelID string) bool {
	if allowedModelID == "*" || allowedModelID == "all-proxy-models" || modelID == allowedModelID {
		return true
	}
	return strings.ContainsRune(allowedModelID, '*') && modelScopePatternMatch(modelID, allowedModelID)
}

func modelScopeProviderPrefix(modelID string) (string, bool) {
	separator := strings.IndexByte(modelID, '/')
	if separator <= 0 {
		return "", false
	}
	prefix := modelID[:separator]
	if prefix == "google" {
		return "vertex_ai", true
	}
	return prefix, true
}

func (m *Manager) liteLLMProviderPrefixForShortModelLocked(modelID string) (string, bool) {
	if len(m.modelToCredentials[modelID]) == 0 {
		return "", false
	}

	prefix := ""
	for publicModel, backendModel := range m.modelAliases {
		if backendModel != modelID {
			continue
		}
		candidate, ok := modelScopeProviderPrefix(publicModel)
		if !ok {
			continue
		}
		if prefix == "" {
			prefix = candidate
			continue
		}
		if prefix != candidate {
			return "", false
		}
	}
	return prefix, prefix != ""
}

func scopeAllowsAnyModelID(modelIDs []string, allowedModelIDs []string) bool {
	for _, modelID := range modelIDs {
		for _, allowedModelID := range allowedModelIDs {
			if modelScopeEntryAllows(modelID, allowedModelID) {
				return true
			}
		}
	}
	return false
}

func scopeAllowsProviderQualifiedWildcard(modelID string, allowedModelIDs []string) bool {
	for _, allowedModelID := range allowedModelIDs {
		// LiteLLM's provider-qualified fallback exists only for wildcard
		// patterns. An exact entry such as openai/gpt-4o-mini must not widen
		// access to the distinct short request ID gpt-4o-mini.
		if strings.ContainsRune(allowedModelID, '*') && modelScopeEntryAllows(modelID, allowedModelID) {
			return true
		}
	}
	return false
}

// IsModelIDAllowedByScope is the shared model-scope predicate for discovery
// and inference. An empty scope means all models, matching LiteLLM semantics.
// A request alias may inherit permission from its configured target, but the
// reverse is forbidden: AIR's internal routing target must not gain permission
// merely because a DB scope contains the client-visible alias.
func (m *Manager) IsModelIDAllowedByScope(modelID string, allowedModelIDs []string) bool {
	if len(allowedModelIDs) == 0 {
		return true
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	requestIDs := []string{modelID}
	if target, isClientAlias, unambiguous := m.clientAliasTargetLocked(modelID); isClientAlias {
		if !unambiguous || !m.publicModelAliasTargetActiveLocked(target) {
			return false
		}
		requestIDs = append(requestIDs, target)
	}
	if target, isAlias := m.modelAliases[modelID]; isAlias && isActiveModelAlias(
		modelID,
		target,
		len(m.modelToCredentials[target]) > 0,
		m.modelAliases,
	) {
		requestIDs = append(requestIDs, target)
	}
	if scopeAllowsAnyModelID(requestIDs, allowedModelIDs) {
		return true
	}

	// LiteLLM also evaluates provider-qualified wildcard patterns for short
	// model IDs. Infer provider identity from the pinned LiteLLM model registry,
	// while requiring unambiguous routing metadata. Unknown IDs, missing
	// credentials and mixed transports all fail closed.
	for _, requestID := range requestIDs {
		if strings.ContainsRune(requestID, '/') {
			continue
		}
		prefix, ok := m.liteLLMProviderPrefixForShortModelLocked(requestID)
		if !ok {
			continue
		}
		if scopeAllowsProviderQualifiedWildcard(prefix+"/"+requestID, allowedModelIDs) {
			return true
		}
	}
	return false
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
	m.credentialMappingsReady = true

	if len(m.modelLimits) == 0 {
		m.allModels = nil
		m.invalidateAllModelsCachesLocked()
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
		if cred.IsProxyLike() {
			continue // proxy/AIR models are fetched dynamically via GetAllModels
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
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
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

	// 2b. Rebuild modelRealNamesPerCred = static snapshot + new DB per-credential real names.
	newRealNamesPerCred := make(map[string]map[string]string, len(m.staticModelRealNamesPerCred))
	for cred, names := range m.staticModelRealNamesPerCred {
		snapshot := make(map[string]string, len(names))
		for alias, realName := range names {
			snapshot[alias] = realName
		}
		newRealNamesPerCred[cred] = snapshot
	}
	// 3. Apply DB model data.
	newDBNames := make(map[string]bool, len(dbModels))
	for _, dm := range dbModels {
		newLimits[dm.Name] = append(newLimits[dm.Name], ModelLimits{
			RPM:        dm.RPM,
			TPM:        dm.TPM,
			Weight:     dm.Weight,
			Credential: dm.Credential,
		})
		// Only apply DB real name if not already defined in static config.
		// Static YAML takes priority: model_table sync must not override
		// explicitly configured models[].model mappings.
		if dm.Model != "" && dm.Model != dm.Name {
			if dm.Credential != "" {
				staticCred := m.staticModelRealNamesPerCred[dm.Credential]
				if _, isStatic := staticCred[dm.Name]; !isStatic {
					if newRealNamesPerCred[dm.Credential] == nil {
						newRealNamesPerCred[dm.Credential] = make(map[string]string)
					}
					newRealNamesPerCred[dm.Credential][dm.Name] = dm.Model
				}
			} else {
				if _, isStatic := m.staticModelRealNames[dm.Name]; !isStatic {
					newRealNames[dm.Name] = dm.Model
				}
			}
		}
		newDBNames[dm.Name] = true
	}

	m.modelLimits = newLimits
	m.modelRealNames = newRealNames
	m.modelRealNamesPerCred = newRealNamesPerCred
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
		if c.IsProxyLike() {
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
		if !c.IsProxyLike() {
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
	m.credentialMappingsReady = true

	// 5. Invalidate caches so next GetAllModels rebuilds from the updated modelLimits.
	m.allModels = nil
	m.invalidateAllModelsCachesLocked()

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
	routableModels := make(map[string]struct{}, len(m.modelToCredentials))
	for modelID, credentialNames := range m.modelToCredentials {
		if len(credentialNames) > 0 {
			routableModels[modelID] = struct{}{}
		}
	}
	credentialMappingsReady := m.credentialMappingsReady
	// Add static models first (configured in model_limits)
	if len(m.modelLimits) > 0 {
		models = make([]Model, 0, len(m.modelLimits)+len(allModelsSnapshot))
		for modelName := range m.modelLimits {
			if credentialMappingsReady {
				if _, routable := routableModels[modelName]; !routable {
					continue
				}
			}
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
		if credentialMappingsReady {
			if _, routable := routableModels[model.ID]; !routable {
				continue
			}
		}
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
		// Skip non-remote-router credentials - we only fetch models from proxy/AIR credentials.
		if !cred.IsProxyLike() {
			m.logger.Debug("Skipping model fetch for non-proxy-like credential",
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
			routableModels[model.ID] = struct{}{}
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

	// Keep the routable/discovered IDs internally. Public projection is applied
	// only after the refreshed credential mappings are installed under the lock;
	// doing it earlier lets the reconciliation step re-introduce backend IDs.
	internalModels := append([]Model(nil), models...)
	response := ModelsResponse{Object: "list", Data: internalModels}

	// Update cache and modelToCredentials atomically
	m.mu.Lock()
	defer m.mu.Unlock()

	// Replace mappings only for proxy credentials whose refresh succeeded.
	// Failed fetches retain their previous mappings to avoid transient false denies.
	if len(successfullyFetched) > 0 {
		for modelID, creds := range m.modelToCredentials {
			kept := make([]string, 0, len(creds))
			for _, credential := range creds {
				if !successfullyFetched[credential] {
					kept = append(kept, credential)
				}
			}
			if len(kept) == 0 {
				delete(m.modelToCredentials, modelID)
			} else {
				m.modelToCredentials[modelID] = kept
			}
		}
	}
	for modelID, creds := range modelUpdates {
		m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], creds...)
	}

	currentCredentials := make(map[string]bool, len(m.credentials))
	for _, credential := range m.credentials {
		currentCredentials[credential.Name] = true
	}
	if !m.credentialsConfigured {
		// LoadModelsFromConfig is also a supported standalone initialization
		// path in embedders/tests. In that mode the mapping itself is the only
		// authoritative credential inventory.
		for credentialName := range m.credentialModels {
			currentCredentials[credentialName] = true
		}
	}
	internalResponse := m.currentModelsLocked(response, currentCredentials)
	internalModels = internalResponse.Data
	response = internalResponse

	// Cache a copy so the cached backing array is independent from the returned response.
	m.allModelsCache = allModelsCache{
		response: ModelsResponse{
			Object: response.Object,
			Data:   append([]Model(nil), response.Data...),
		},
		expiresAt: utils.NowUTC().Add(allModelsCacheTTL),
	}
	m.allModels = append([]Model(nil), internalModels...)
	m.invalidateScopedAllModelsCacheLocked()

	return response
}

func (m *Manager) GetClientModels() ModelsResponse {
	response := m.GetAllModels()
	m.mu.RLock()
	defer m.mu.RUnlock()
	response.Data = m.projectClientModelCatalogLocked(response.Data)
	return response
}

func (m *Manager) GetAllModelsScoped(visibility scope.Context) ModelsResponse {
	if response, ok := m.getCachedScopedAllModels(visibility); ok {
		return response
	}
	response := m.getAllModelsScoped(visibility)

	m.mu.Lock()
	defer m.mu.Unlock()

	visibleCreds := m.visibleCredentialNamesLocked(visibility)
	response = m.currentModelsLocked(response, visibleCreds)
	filtered := make([]Model, 0, len(response.Data))
	for _, model := range response.Data {
		if m.modelVisibleLocked(model.ID, visibleCreds, visibility) {
			filtered = append(filtered, model)
		}
	}
	scopedResponse := ModelsResponse{
		Object: response.Object,
		Data:   m.projectClientModelCatalogLocked(filtered),
	}
	m.scopedAllModelsCache.Add(m.scopedAllModelsCacheKeyLocked(visibility), allModelsCache{
		response:  copyModelsResponse(scopedResponse),
		expiresAt: utils.NowUTC().Add(allModelsCacheTTL),
	})
	return scopedResponse
}

func (m *Manager) getCachedScopedAllModels(visibility scope.Context) (ModelsResponse, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	cacheKey := m.scopedAllModelsCacheKeyLocked(visibility)
	cached, ok := m.scopedAllModelsCache.Get(cacheKey)
	if !ok || cached.expiresAt.IsZero() || !utils.NowUTC().Before(cached.expiresAt) {
		if ok {
			m.scopedAllModelsCache.Remove(cacheKey)
		}
		return ModelsResponse{}, false
	}
	return copyModelsResponse(cached.response), true
}

func (m *Manager) scopedAllModelsCacheKeyLocked(visibility scope.Context) string {
	credentialNames := make([]string, 0, len(m.credentials))
	for _, cred := range m.credentials {
		if cred.VisibleTo(visibility) {
			credentialNames = append(credentialNames, cred.Name)
		}
	}
	slices.Sort(credentialNames)
	return visibility.Key() + "|c:" + strings.Join(credentialNames, "\x00")
}

func copyModelsResponse(response ModelsResponse) ModelsResponse {
	return ModelsResponse{
		Object: response.Object,
		Data:   append([]Model(nil), response.Data...),
	}
}

func newScopedAllModelsCache() *lru.Cache[string, allModelsCache] {
	cache, err := lru.New[string, allModelsCache](scopedAllModelsCacheSize)
	if err != nil {
		panic(fmt.Sprintf("models: failed to create scoped models cache: %v", err))
	}
	return cache
}

func (m *Manager) invalidateScopedAllModelsCacheLocked() {
	m.scopedAllModelsCache.Purge()
}

func (m *Manager) currentModelsLocked(response ModelsResponse, visibleCredentials map[string]bool) ModelsResponse {
	metadata := make(map[string]Model, len(response.Data)+len(m.allModels))
	for _, model := range m.allModels {
		if model.ID != "" {
			metadata[model.ID] = model
		}
	}
	for _, model := range response.Data {
		if model.ID != "" {
			metadata[model.ID] = model
		}
	}
	candidateIDs := make(map[string]struct{}, len(metadata))
	appendCandidate := func(modelID string) {
		if modelID == "" {
			return
		}
		if m.credentialMappingsReady {
			visible := false
			for _, credentialName := range m.modelToCredentials[modelID] {
				if visibleCredentials[credentialName] {
					visible = true
					break
				}
			}
			if !visible {
				return
			}
		}
		candidateIDs[modelID] = struct{}{}
	}
	for modelID := range m.modelLimits {
		appendCandidate(modelID)
	}
	for _, model := range m.allModels {
		appendCandidate(model.ID)
	}
	for credentialName := range visibleCredentials {
		for _, modelID := range m.credentialModels[credentialName] {
			appendCandidate(modelID)
		}
	}
	modelIDs := make([]string, 0, len(candidateIDs))
	for modelID := range candidateIDs {
		modelIDs = append(modelIDs, modelID)
	}
	slices.Sort(modelIDs)
	models := make([]Model, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		model, ok := metadata[modelID]
		if !ok {
			model = Model{ID: modelID, Object: "model", Created: converterutil.GetCurrentTimestamp(), OwnedBy: "system"}
		}
		models = append(models, model)
	}
	return ModelsResponse{Object: response.Object, Data: models}
}

// projectClientModelCatalogLocked is the single public boundary used by both
// unscoped and key-scoped discovery. Its input must already contain only
// internal models visible through the selected credentials/scopes.
func (m *Manager) projectClientModelCatalogLocked(internalModels []Model) []Model {
	activePublicAliases := make(map[string]string, len(m.publicModelAliases))
	for alias, target := range m.publicModelAliases {
		if m.publicModelAliasTargetActiveLocked(target) {
			activePublicAliases[alias] = target
		}
	}
	var models []Model
	if m.clientModelSurfaceConfigured {
		models = projectConfiguredClientModelCatalog(
			internalModels,
			m.clientModelIDs,
			m.modelAliases,
			activePublicAliases,
		)
	} else {
		models = projectPublicModelCatalog(internalModels, m.modelAliases)
		models = projectCanonicalPublicAliases(models, activePublicAliases)
	}
	return hideAcceptedModelAliases(models, m.acceptedModelAliases)
}

func hideAcceptedModelAliases(models []Model, aliases map[string]string) []Model {
	if len(aliases) == 0 {
		return models
	}
	visible := make([]Model, 0, len(models))
	for _, model := range models {
		if _, hidden := aliases[model.ID]; !hidden {
			visible = append(visible, model)
		}
	}
	return visible
}

func (m *Manager) invalidateAllModelsCachesLocked() {
	m.allModelsCache = allModelsCache{}
	m.invalidateScopedAllModelsCacheLocked()
}

func (m *Manager) getAllModelsScoped(visibility scope.Context) ModelsResponse {
	m.mu.RLock()

	var models []Model
	modelMap := make(map[string]bool)
	allModelsSnapshot := append([]Model(nil), m.allModels...)
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
	for _, model := range allModelsSnapshot {
		if !modelMap[model.ID] {
			models = append(models, model)
			modelMap[model.ID] = true
		}
	}

	credentials := make([]config.CredentialConfig, 0, len(m.credentials))
	for _, cred := range m.credentials {
		if cred.VisibleTo(visibility) {
			credentials = append(credentials, cred)
		}
	}

	m.mu.RUnlock()

	for _, cred := range credentials {
		if !cred.IsProxyLike() {
			continue
		}
		remoteModels, err := m.GetRemoteModelsWithError(context.Background(), &cred)
		if err != nil {
			m.logger.Warn("Failed to fetch models from visible proxy during scoped model list refresh",
				"credential", cred.Name,
				"error", err,
			)
			continue
		}
		for _, model := range remoteModels {
			if !modelMap[model.ID] {
				models = append(models, model)
				modelMap[model.ID] = true
			}
		}
	}

	return ModelsResponse{Object: "list", Data: models}
}

// projectConfiguredClientModelCatalog emits only configured client IDs and active public aliases.
func projectConfiguredClientModelCatalog(
	internalModels []Model,
	clientModelIDs map[string]struct{},
	modelAliases map[string]string,
	publicModelAliases map[string]string,
) []Model {
	internalByID := make(map[string]Model, len(internalModels))
	availableTargets := make(map[string]struct{}, len(internalModels))
	for _, model := range internalModels {
		if model.ID == "" {
			continue
		}
		if _, exists := internalByID[model.ID]; !exists {
			internalByID[model.ID] = model
			availableTargets[model.ID] = struct{}{}
		}
	}
	activeAliases := activePublicModelAliases(availableTargets, modelAliases)
	publicByID := make(map[string]Model, len(clientModelIDs)+len(publicModelAliases))
	for modelID := range clientModelIDs {
		model, direct := internalByID[modelID]
		if !direct {
			target, active := activeAliases[modelID]
			if !active {
				continue
			}
			model = internalByID[target]
		}
		model.ID = modelID
		publicByID[modelID] = model
	}
	for alias, target := range publicModelAliases {
		targetModel, targetVisible := publicByID[target]
		if !targetVisible {
			continue
		}
		targetModel.ID = alias
		publicByID[alias] = targetModel
	}
	result := make([]Model, 0, len(publicByID))
	for _, model := range publicByID {
		result = append(result, model)
	}
	slices.SortFunc(result, func(left, right Model) int {
		return strings.Compare(left.ID, right.ID)
	})
	return result
}

func projectCanonicalPublicAliases(models []Model, aliases map[string]string) []Model {
	byID := make(map[string]Model, len(models)+len(aliases))
	for _, model := range models {
		if model.ID != "" {
			byID[model.ID] = model
		}
	}
	for alias, target := range aliases {
		if alias == "" || target == "" {
			continue
		}
		targetModel, targetVisible := byID[target]
		if !targetVisible {
			continue
		}
		if _, alreadyVisible := byID[alias]; alreadyVisible {
			continue
		}
		targetModel.ID = alias
		byID[alias] = targetModel
	}
	result := make([]Model, 0, len(byID))
	for _, model := range byID {
		result = append(result, model)
	}
	slices.SortFunc(result, func(left, right Model) int {
		return strings.Compare(left.ID, right.ID)
	})
	return result
}

// projectPublicModelCatalog augments routable model IDs with configured public
// aliases. Targets remain visible because LiteLLM exposes both a configured
// public model and its accepted short alias. Orphan aliases are ignored, so an
// alias cannot make an unrelated/unroutable model visible.
func projectPublicModelCatalog(internalModels []Model, aliases map[string]string) []Model {
	internalByID := make(map[string]Model, len(internalModels))
	availableTargets := make(map[string]struct{}, len(internalModels))
	for _, model := range internalModels {
		if model.ID == "" {
			continue
		}
		if _, exists := internalByID[model.ID]; !exists {
			internalByID[model.ID] = model
			availableTargets[model.ID] = struct{}{}
		}
	}

	activeAliases := activePublicModelAliases(availableTargets, aliases)

	publicByID := make(map[string]Model, len(internalByID)+len(aliases))
	for id, model := range internalByID {
		publicByID[id] = model
	}
	for alias, target := range activeAliases {
		targetModel := internalByID[target]
		targetModel.ID = alias
		publicByID[alias] = targetModel
	}

	publicModels := make([]Model, 0, len(publicByID))
	for _, model := range publicByID {
		publicModels = append(publicModels, model)
	}
	slices.SortFunc(publicModels, func(left, right Model) int {
		return strings.Compare(left.ID, right.ID)
	})
	return publicModels
}

// activePublicModelAliases applies the shared public-catalog rule used by both
// /v1/models variants: empty/orphan aliases are ignored, and only aliases for
// already-routable targets are activated.
func activePublicModelAliases(availableTargets map[string]struct{}, aliases map[string]string) map[string]string {
	activeAliases := make(map[string]string, len(aliases))
	for alias, target := range aliases {
		_, targetRoutable := availableTargets[target]
		if !isActiveModelAlias(alias, target, targetRoutable, aliases) {
			continue
		}
		activeAliases[alias] = target
	}
	return activeAliases
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
	return m.hasModelLocked(credentialName, modelID)
}

func (m *Manager) HasModelScoped(credentialName, modelID string, visibility scope.Context) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.hasModelLocked(credentialName, modelID) {
		return false
	}
	return m.modelScopeAllowsLocked(modelID, credentialName, visibility)
}

func (m *Manager) GetModelScopeExpressionForCredential(modelID, credentialName string) *scope.Expression {
	m.mu.RLock()
	defer m.mu.RUnlock()

	metadata, ok := m.dynamicModelScopes[modelID][credentialName]
	if !ok {
		return nil
	}
	if metadata.ScopeExpression != nil {
		return scope.NormalizeExpression(metadata.ScopeExpression)
	}
	return scope.FromScopes(metadata.Scopes, metadata.DeniedScopes)
}

func (m *Manager) hasModelLocked(credentialName, modelID string) bool {
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
	m.credentialMappingsReady = true
	m.invalidateScopedAllModelsCacheLocked()
}

// ReplaceModelsForCredential replaces the dynamic proxy-discovered model list
// for a credential with a fresh upstream snapshot. Static/DB model mappings are
// preserved so explicit local configuration still takes precedence.
func (m *Manager) ReplaceModelsForCredential(credentialName string, modelIDs []string) {
	if credentialName == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.replaceModelsForCredentialLocked(credentialName, modelIDs)
}

func (m *Manager) replaceModelsForCredentialLocked(credentialName string, modelIDs []string) {
	desiredSet := make(map[string]bool, len(modelIDs))
	desired := make([]string, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		if modelID == "" || desiredSet[modelID] {
			continue
		}
		desiredSet[modelID] = true
		desired = append(desired, modelID)
	}

	configured := make([]string, 0)
	for modelID := range m.modelLimits {
		if desiredSet[modelID] || !m.isConfiguredForCredentialLocked(modelID, credentialName) {
			continue
		}
		configured = append(configured, modelID)
	}
	slices.Sort(configured)

	replacement := append([]string(nil), desired...)
	replacement = append(replacement, configured...)
	m.credentialModels[credentialName] = replacement

	for modelID, creds := range m.modelToCredentials {
		kept := creds[:0]
		for _, cred := range creds {
			if cred != credentialName {
				kept = append(kept, cred)
			}
		}
		if len(kept) == 0 {
			delete(m.modelToCredentials, modelID)
		} else {
			m.modelToCredentials[modelID] = kept
		}
	}

	for _, modelID := range replacement {
		if !m.contains(m.modelToCredentials[modelID], credentialName) {
			m.modelToCredentials[modelID] = append(m.modelToCredentials[modelID], credentialName)
		}
	}
	m.credentialMappingsReady = true

	m.allModels = nil
	m.invalidateAllModelsCachesLocked()
}

// SetModelWeightForCredential stores a dynamic model-level weight learned from a proxy
// upstream. Static config/DB model weights still take precedence when present.
func (m *Manager) SetModelWeightForCredential(modelID, credentialName string, weight int) {
	if modelID == "" || credentialName == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if weight <= 0 {
		if weights, ok := m.dynamicModelWeights[modelID]; ok {
			delete(weights, credentialName)
			if len(weights) == 0 {
				delete(m.dynamicModelWeights, modelID)
			}
		}
		return
	}

	if m.dynamicModelWeights[modelID] == nil {
		m.dynamicModelWeights[modelID] = make(map[string]int)
	}
	m.dynamicModelWeights[modelID][credentialName] = weight
}

// ReplaceModelWeightsForCredential replaces all dynamic health-derived weights
// for a credential with a fresh upstream snapshot.
func (m *Manager) ReplaceModelWeightsForCredential(credentialName string, weights map[string]int) {
	if credentialName == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.replaceModelWeightsForCredentialLocked(credentialName, weights)
}

func (m *Manager) replaceModelWeightsForCredentialLocked(credentialName string, weights map[string]int) {
	for modelID, credentialWeights := range m.dynamicModelWeights {
		delete(credentialWeights, credentialName)
		if len(credentialWeights) == 0 {
			delete(m.dynamicModelWeights, modelID)
		}
	}

	for modelID, weight := range weights {
		if modelID == "" || weight <= 0 {
			continue
		}
		if m.dynamicModelWeights[modelID] == nil {
			m.dynamicModelWeights[modelID] = make(map[string]int)
		}
		m.dynamicModelWeights[modelID][credentialName] = weight
	}
}

func (m *Manager) ReplaceModelScopesForCredential(credentialName string, scopes map[string]ScopeMetadata) {
	if credentialName == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.replaceModelScopesForCredentialLocked(credentialName, scopes)
}

func (m *Manager) replaceModelScopesForCredentialLocked(credentialName string, scopes map[string]ScopeMetadata) {
	for modelID, credentialScopes := range m.dynamicModelScopes {
		delete(credentialScopes, credentialName)
		if len(credentialScopes) == 0 {
			delete(m.dynamicModelScopes, modelID)
		}
	}

	for modelID, metadata := range scopes {
		if modelID == "" {
			continue
		}
		metadata.Scopes = scope.NormalizeList(metadata.Scopes)
		metadata.DeniedScopes = scope.NormalizeList(metadata.DeniedScopes)
		metadata.ScopeExpression = scope.NormalizeExpression(metadata.ScopeExpression)
		if len(metadata.Scopes) == 0 && len(metadata.DeniedScopes) == 0 && metadata.ScopeExpression == nil {
			continue
		}
		if m.dynamicModelScopes[modelID] == nil {
			m.dynamicModelScopes[modelID] = make(map[string]ScopeMetadata)
		}
		m.dynamicModelScopes[modelID][credentialName] = metadata
	}
	m.invalidateScopedAllModelsCacheLocked()
}

func (m *Manager) UpdateProviderScopesForCredential(credentialName string, metadata ScopeMetadata) {
	if credentialName == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.updateProviderScopesForCredentialLocked(credentialName, metadata, true)
}

func (m *Manager) updateProviderScopesForCredentialLocked(credentialName string, metadata ScopeMetadata, known bool) {
	for i := range m.credentials {
		if m.credentials[i].Name != credentialName {
			continue
		}
		m.credentials[i].ProviderScopes = scope.NormalizeList(metadata.Scopes)
		m.credentials[i].ProviderDeniedScopes = scope.NormalizeList(metadata.DeniedScopes)
		m.credentials[i].ProviderScopeExpression = scope.NormalizeExpression(metadata.ScopeExpression)
		m.credentials[i].ProviderScopeKnown = known
		m.invalidateScopedAllModelsCacheLocked()
		return
	}
}

// CopyProviderScopeMetadata copies learned scope metadata into a credential snapshot.
func (m *Manager) CopyProviderScopeMetadata(cred *config.CredentialConfig) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for i := range m.credentials {
		current := &m.credentials[i]
		if current.Name != cred.Name || !cred.SameProviderIdentity(*current) {
			continue
		}
		cred.ProviderScopes = append([]string(nil), current.ProviderScopes...)
		cred.ProviderDeniedScopes = append([]string(nil), current.ProviderDeniedScopes...)
		cred.ProviderScopeExpression = scope.NormalizeExpression(current.ProviderScopeExpression)
		cred.ProviderScopeKnown = current.ProviderScopeKnown
		return true
	}
	return false
}

func (m *Manager) applyRemoteScopeSnapshot(
	cred *config.CredentialConfig,
	models []Model,
	snapshot *remoteScopeSnapshot,
) bool {
	if snapshot == nil {
		return true
	}
	cloned := cloneRemoteScopeSnapshot(snapshot)
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.applyRemoteScopeSnapshotLocked(cred, models, cloned)
}

func (m *Manager) applyRemoteScopeSnapshotAndCache(
	cred *config.CredentialConfig,
	models []Model,
	snapshot *remoteScopeSnapshot,
) bool {
	cloned := cloneRemoteScopeSnapshot(snapshot)
	cachedModels := append([]Model(nil), models...)
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.applyRemoteScopeSnapshotLocked(cred, cachedModels, cloned) {
		return false
	}
	m.remoteModelsCache[cred.Name] = remoteModelCache{
		credential:    *cred,
		models:        cachedModels,
		scopeSnapshot: cloneRemoteScopeSnapshot(cloned),
		expiresAt:     utils.NowUTC().Add(m.cacheExpiration),
	}
	return true
}

func (m *Manager) applyRemoteScopeSnapshotLocked(
	cred *config.CredentialConfig,
	models []Model,
	snapshot *remoteScopeSnapshot,
) bool {
	found := false
	for i := range m.credentials {
		if m.credentials[i].Name != cred.Name {
			continue
		}
		found = true
		if !cred.SameProviderIdentity(m.credentials[i]) {
			return false
		}
	}
	if m.credentialsConfigured && !found {
		return false
	}
	modelIDs := remoteModelIDs(models)
	m.updateProviderScopesForCredentialLocked(cred.Name, snapshot.providerScopes, snapshot.scopeKnown)
	m.replaceModelScopesForCredentialLocked(cred.Name, snapshot.modelScopes)
	m.replaceModelsForCredentialLocked(cred.Name, modelIDs)
	m.replaceModelWeightsForCredentialLocked(cred.Name, snapshot.modelWeights)
	return true
}

func cloneRemoteScopeSnapshot(snapshot *remoteScopeSnapshot) *remoteScopeSnapshot {
	if snapshot == nil {
		return nil
	}
	return &remoteScopeSnapshot{
		providerScopes: cloneScopeMetadata(snapshot.providerScopes),
		modelScopes:    cloneModelScopes(snapshot.modelScopes),
		modelWeights:   cloneModelWeights(snapshot.modelWeights),
		scopeKnown:     snapshot.scopeKnown,
	}
}

func cloneScopeMetadata(metadata ScopeMetadata) ScopeMetadata {
	return ScopeMetadata{
		Scopes:          append([]string(nil), metadata.Scopes...),
		DeniedScopes:    append([]string(nil), metadata.DeniedScopes...),
		ScopeExpression: scope.NormalizeExpression(metadata.ScopeExpression),
	}
}

func cloneModelScopes(modelScopes map[string]ScopeMetadata) map[string]ScopeMetadata {
	cloned := make(map[string]ScopeMetadata, len(modelScopes))
	for modelID, metadata := range modelScopes {
		cloned[modelID] = cloneScopeMetadata(metadata)
	}
	return cloned
}

func cloneModelWeights(modelWeights map[string]int) map[string]int {
	cloned := make(map[string]int, len(modelWeights))
	for modelID, weight := range modelWeights {
		cloned[modelID] = weight
	}
	return cloned
}

func (m *Manager) isConfiguredForCredentialLocked(modelID, credentialName string) bool {
	for _, limit := range m.modelLimits[modelID] {
		if limit.Credential == "" || limit.Credential == credentialName {
			return true
		}
	}
	return false
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

// findWeightLimit searches for a configured weighted round-robin weight with optional
// credential filtering. Weight 0 means unset, so it must not block fallback to a global
// model weight.
func findWeightLimit(limits []ModelLimits, credentialName string) (int, bool) {
	if credentialName != "" {
		for i := range limits {
			if limits[i].Credential == credentialName && limits[i].Weight > 0 {
				return limits[i].Weight, true
			}
		}
	}

	for i := range limits {
		if limits[i].Credential == "" && limits[i].Weight > 0 {
			return limits[i].Weight, true
		}
	}

	if credentialName == "" {
		for i := range limits {
			if limits[i].Weight > 0 {
				return limits[i].Weight, true
			}
		}
	}

	return 0, false
}

// GetModelWeightForCredential returns the configured weight for a (model, credential) pair.
// Returns 0 when no model-level weight is configured.
func (m *Manager) GetModelWeightForCredential(modelID, credentialName string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if limits, ok := m.modelLimits[modelID]; ok {
		if weight, found := findWeightLimit(limits, credentialName); found {
			return weight
		}
	}

	if weights, ok := m.dynamicModelWeights[modelID]; ok {
		if weight := weights[credentialName]; weight > 0 {
			return weight
		}
	}

	return 0
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
	config.ProviderTypeCometAPI:  "cometapi",
	config.ProviderTypeProMan:    "proman",
	config.ProviderTypeBedrock:   "bedrock",
	config.ProviderTypeProxy:     "openai",
	config.ProviderTypeAIR:       "openai",
}

// GetAllModelsWithAccessGroups returns all models in "provider/model-id" format,
// used when the caller requests ?include_model_access_groups=True.
// Each (provider, model) pair appears at most once in the response.
func (m *Manager) GetAllModelsWithAccessGroups() ModelsResponse {
	return m.GetAllModelsWithAccessGroupsScoped(scope.AdminContext())
}

func (m *Manager) GetAllModelsWithAccessGroupsScoped(visibility scope.Context) ModelsResponse {
	m.mu.RLock()
	explicitClientSurface := m.clientModelSurfaceConfigured
	m.mu.RUnlock()
	if explicitClientSurface {
		// Provider access-group projection is an administrative view over
		// internal routes. Once a product surface is explicit, returning that
		// projection would re-introduce backend IDs through a query parameter.
		return m.GetAllModelsScoped(visibility)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	credProvider := make(map[string]string, len(m.credentials))
	for _, cred := range m.credentials {
		if !cred.VisibleTo(visibility) {
			continue
		}
		prefix, ok := providerTypeLiteLLMPrefix[cred.Type]
		if !ok {
			prefix = string(cred.Type)
		}
		credProvider[cred.Name] = prefix
	}

	availableTargets := make(map[string]struct{}, len(m.modelToCredentials))
	for modelID, credentialNames := range m.modelToCredentials {
		if len(credentialNames) > 0 {
			availableTargets[modelID] = struct{}{}
		}
	}
	activeAliases := activePublicModelAliases(availableTargets, m.modelAliases)
	activeDeploymentAliases := make(map[string]string, len(m.publicModelAliases))
	for alias, target := range m.publicModelAliases {
		if m.publicModelAliasTargetActiveLocked(target) {
			activeDeploymentAliases[alias] = target
		}
	}

	// An alias is listed for a key only when the key could see the alias target
	// itself: the alias must not widen the scoped view onto foreign models.
	aliasTargetVisible := func(target string) bool {
		routeTarget, active := m.clientCanonicalRouteTargetLocked(target)
		if !active {
			return false
		}
		for _, credName := range m.modelToCredentials[routeTarget] {
			if _, visible := credProvider[credName]; !visible {
				continue
			}
			if m.modelScopeAllowsLocked(routeTarget, credName, visibility) {
				return true
			}
		}
		return false
	}

	seen := make(map[string]bool)
	result := make([]Model, 0, len(m.modelToCredentials)+len(activeAliases)+len(activeDeploymentAliases))

	for modelID, creds := range m.modelToCredentials {
		for _, credName := range creds {
			prefix, ok := credProvider[credName]
			if !ok {
				continue
			}
			if !m.modelScopeAllowsLocked(modelID, credName, visibility) {
				continue
			}
			prefixedID := modelID
			if !strings.HasPrefix(modelID, prefix+"/") {
				prefixedID = prefix + "/" + modelID
			}
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
	for alias, target := range activeAliases {
		if seen[alias] || !aliasTargetVisible(target) {
			continue
		}
		seen[alias] = true
		result = append(result, Model{
			ID:      alias,
			Object:  "model",
			Created: converterutil.GetCurrentTimestamp(),
			OwnedBy: "system",
		})
	}
	for alias, target := range activeDeploymentAliases {
		if seen[alias] || !aliasTargetVisible(target) {
			continue
		}
		seen[alias] = true
		result = append(result, Model{
			ID:      alias,
			Object:  "model",
			Created: converterutil.GetCurrentTimestamp(),
			OwnedBy: "system",
		})
	}
	slices.SortFunc(result, func(left, right Model) int {
		return strings.Compare(left.ID, right.ID)
	})

	return ModelsResponse{Object: "list", Data: result}
}

func (m *Manager) visibleCredentialNamesLocked(visibility scope.Context) map[string]bool {
	result := make(map[string]bool, len(m.credentials))
	for _, cred := range m.credentials {
		if cred.VisibleTo(visibility) {
			result[cred.Name] = true
		}
	}
	return result
}

func (m *Manager) modelVisibleLocked(modelID string, visibleCreds map[string]bool, visibility scope.Context) bool {
	if len(visibleCreds) == 0 {
		return false
	}
	creds, ok := m.modelToCredentials[modelID]
	if !ok {
		return true
	}
	for _, cred := range creds {
		if visibleCreds[cred] && m.modelScopeAllowsLocked(modelID, cred, visibility) {
			return true
		}
	}
	return false
}

func (m *Manager) modelScopeAllowsLocked(modelID, credentialName string, visibility scope.Context) bool {
	if scopedCreds, ok := m.dynamicModelScopes[modelID]; ok {
		if metadata, ok := scopedCreds[credentialName]; ok {
			if metadata.ScopeExpression != nil {
				return visibility.AllowsExpression(metadata.ScopeExpression)
			}
			return visibility.Allows(metadata.Scopes, metadata.DeniedScopes)
		}
	}
	return true
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
//
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
	if !cred.IsProxyLike() {
		return nil, nil
	}

	// Check cache first
	m.mu.RLock()
	if cached, ok := m.remoteModelsCache[cred.Name]; ok && cred.SameProviderIdentity(cached.credential) &&
		!cached.expiresAt.IsZero() && utils.NowUTC().Before(cached.expiresAt) {
		cachedModels := append([]Model(nil), cached.models...)
		cachedSnapshot := cloneRemoteScopeSnapshot(cached.scopeSnapshot)
		cachedCount := len(cachedModels)
		expiresIn := time.Until(cached.expiresAt).Seconds()
		m.mu.RUnlock()
		if !m.applyRemoteScopeSnapshot(cred, cachedModels, cachedSnapshot) {
			return nil, errProxyCredentialChanged
		}
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

	models, snapshot, err := m.fetchRemoteModelsFromHealth(ctx, cred)
	if err != nil {
		if !isLegacyProxyHealthError(err) || !m.providerScopeAllowsLegacyFallback(cred) {
			m.failClosedUnknownRemoteScope(cred)
			return nil, err
		}

		m.logger.Debug("Proxy health metadata unavailable; falling back to /v1/models",
			"credential", cred.Name,
			"error", err,
		)
		var modelsResp ModelsResponse
		if err := httputil.FetchJSONFromProxy(ctx, cred, "/v1/models", m.logger, &modelsResp); err != nil {
			m.failClosedUnknownRemoteScope(cred)
			m.logger.Error("Failed to fetch remote models",
				"credential", cred.Name,
				"error", err,
			)
			return nil, err
		}
		models = modelsResp.Data
		snapshot = &remoteScopeSnapshot{
			providerScopes: scopeMetadataFromExpression(scope.FromScopes(nil, nil)),
			modelScopes:    map[string]ScopeMetadata{},
			modelWeights:   map[string]int{},
			scopeKnown:     true,
		}
	}

	if !m.applyRemoteScopeSnapshotAndCache(cred, models, snapshot) {
		return nil, errProxyCredentialChanged
	}

	m.logger.Debug("Cached remote models",
		"credential", cred.Name,
		"models_count", len(models),
		"expires_in", m.cacheExpiration.Seconds(),
	)

	return models, nil
}

func (m *Manager) fetchRemoteModelsFromHealth(
	ctx context.Context,
	cred *config.CredentialConfig,
) ([]Model, *remoteScopeSnapshot, error) {
	var health httputil.ProxyHealthResponse
	if err := httputil.FetchJSONFromProxy(ctx, cred, "/health", m.logger, &health); err != nil {
		return nil, nil, err
	}

	if health.Credentials == nil || health.Models == nil {
		return nil, nil, errProxyHealthModelMetadataUnavailable
	}

	providerScopes := AggregateProviderScopesFromHealth(&health, cred.IsFallback)
	modelScopes := AggregateModelScopesFromHealth(&health, cred.IsFallback)

	modelsByID := make(map[string]Model)
	modelWeightsByID := make(map[string]int)
	for _, modelStats := range health.Models {
		credStats, ok := health.Credentials[modelStats.Credential]
		if !ok {
			continue
		}
		// For a non-fallback connection: skip upstream credentials marked as fallback
		// (they are reserved for fallback traffic and must not serve primary requests).
		// For a fallback connection: include ALL upstream credentials regardless of their
		// fallback status — use whatever the upstream can offer as a last resort.
		if !cred.IsFallback && credStats.IsFallback {
			continue
		}
		if modelStats.Model == "" {
			continue
		}
		modelWeightsByID[modelStats.Model] += httputil.EffectiveHealthWeight(modelStats, credStats)
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

	return models, &remoteScopeSnapshot{
		providerScopes: providerScopes,
		modelScopes:    modelScopes,
		modelWeights:   modelWeightsByID,
		scopeKnown:     true,
	}, nil
}

func isLegacyProxyHealthError(err error) bool {
	if errors.Is(err, errProxyHealthModelMetadataUnavailable) {
		return true
	}
	var statusErr *httputil.ProxyStatusError
	if !errors.As(err, &statusErr) {
		return false
	}
	return statusErr.StatusCode == http.StatusNotFound || statusErr.StatusCode == http.StatusMethodNotAllowed
}

func (m *Manager) providerScopeAllowsLegacyFallback(cred *config.CredentialConfig) bool {
	current := *cred
	m.CopyProviderScopeMetadata(&current)
	if !current.ProviderScopeKnown {
		return true
	}
	expression := scope.NormalizeExpression(current.ProviderScopeExpression)
	if expression == nil {
		return len(current.ProviderScopes) == 0 && len(current.ProviderDeniedScopes) == 0
	}
	for _, alternative := range expression.Alternatives {
		if len(alternative.Requirements) == 0 && len(alternative.DeniedScopes) == 0 {
			return true
		}
	}
	return false
}

func (m *Manager) failClosedUnknownRemoteScope(cred *config.CredentialConfig) {
	current := *cred
	m.CopyProviderScopeMetadata(&current)
	if current.ProviderScopeKnown {
		return
	}
	m.applyRemoteScopeSnapshot(cred, nil, &remoteScopeSnapshot{
		providerScopes: scopeMetadataFromExpression(scope.FalseExpression()),
		modelScopes:    map[string]ScopeMetadata{},
		modelWeights:   map[string]int{},
	})
}

func remoteModelIDs(models []Model) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		if model.ID != "" {
			ids = append(ids, model.ID)
		}
	}
	return ids
}
