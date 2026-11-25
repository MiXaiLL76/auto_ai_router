package responses

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// ResponsesRequestMode holds context for a native Responses API conversion session.
type ResponsesRequestMode struct {
	ModelID        string // real provider model name
	DisplayModelID string // alias to echo in responses; falls back to ModelID when empty
	IsStreaming    bool
	// ImageBudget bounds the image downloads performed while converting this
	// request. A nil budget means every download runs with its own timeout.
	ImageBudget *converterutil.ImageFetchBudget
}

// DisplayModel returns the model name to embed in responses.
// Uses DisplayModelID (alias) when set so the client sees the name it requested.
func (m ResponsesRequestMode) DisplayModel() string {
	if m.DisplayModelID != "" {
		return m.DisplayModelID
	}
	return m.ModelID
}

type providerResponsesEntry struct {
	factory     func(ResponsesRequestMode) ProviderResponses
	modelFilter func(modelID string) bool // nil = all models supported
}

// providerResponsesFactories maps provider type → registered entry.
// Populated by provider packages via init() to avoid import cycles.
var providerResponsesFactories = map[config.ProviderType]providerResponsesEntry{}

// RegisterProviderResponses registers a factory for all models of a provider type.
// Called from provider-specific packages via init().
func RegisterProviderResponses(providerType config.ProviderType, factory func(ResponsesRequestMode) ProviderResponses) {
	providerResponsesFactories[providerType] = providerResponsesEntry{factory: factory}
}

// RegisterProviderResponsesForModel registers a factory for a provider type with a per-model
// filter. modelFilter(modelID) must return true for models that support native Responses
// conversion; other models fall through to the Chat Completions converted path.
func RegisterProviderResponsesForModel(providerType config.ProviderType, factory func(ResponsesRequestMode) ProviderResponses, modelFilter func(string) bool) {
	providerResponsesFactories[providerType] = providerResponsesEntry{factory: factory, modelFilter: modelFilter}
}

// NewProviderResponses returns a ProviderResponses for the given provider type,
// or nil if the provider does not have a native Responses converter.
func NewProviderResponses(providerType config.ProviderType, mode ResponsesRequestMode) ProviderResponses {
	entry, ok := providerResponsesFactories[providerType]
	if !ok {
		return nil
	}
	return entry.factory(mode)
}

// HasNativeResponsesForModel reports whether the given provider type has a registered native
// Responses API converter that supports the specified model ID.
func HasNativeResponsesForModel(providerType config.ProviderType, modelID string) bool {
	entry, ok := providerResponsesFactories[providerType]
	if !ok {
		return false
	}
	if entry.modelFilter == nil {
		return true
	}
	return entry.modelFilter(modelID)
}

// HasNativeResponses reports whether the given provider type has a registered native
// Responses API converter for all models (no per-model filter).
// Use HasNativeResponsesForModel for model-aware routing decisions.
func HasNativeResponses(providerType config.ProviderType) bool {
	entry, ok := providerResponsesFactories[providerType]
	if !ok {
		return false
	}
	return entry.modelFilter == nil
}
