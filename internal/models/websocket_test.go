package models

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestWebSocketResponsesAliases(t *testing.T) {
	manager := New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{{Name: "astra", Model: "gpt-6-astra", WebSocketResponses: true}, {Name: "gpt-5.6-sol"}})
	manager.LoadModelsFromConfig([]config.CredentialConfig{{Name: "upstream", Type: config.ProviderTypeOpenAI, BaseURL: "http://example.invalid", APIKey: "test"}})
	manager.SetModelAliases(map[string]string{"fast": "astra"})
	manager.SetPublicModelAliases(map[string]string{"public": "astra"})
	manager.SetAcceptedModelAliases(map[string]string{"hidden": "astra"})
	for _, name := range []string{"astra", "fast", "public", "hidden"} {
		assert.True(t, manager.IsWebSocketResponses(name), name)
	}
	for _, name := range []string{"gpt-5.6-sol", "unknown"} {
		assert.False(t, manager.IsWebSocketResponses(name), name)
	}
}
