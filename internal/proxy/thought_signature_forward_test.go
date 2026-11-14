package proxy

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestStripThoughtSignaturesForCredential(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1__thought__c2ln","type":"function"}]},{"role":"tool","tool_call_id":"call_1__thought__c2ln"}]}`)
	for _, providerType := range []config.ProviderType{
		config.ProviderTypeVertexAI, config.ProviderTypeGemini, config.ProviderTypeAIR,
	} {
		assert.Equal(t, body, stripThoughtSignaturesForCredential(body, providerType), providerType)
	}
	for _, providerType := range []config.ProviderType{
		config.ProviderTypeOpenAI, config.ProviderTypeVLLM, config.ProviderTypeAnthropic,
		config.ProviderTypeBedrock, config.ProviderTypeProxy,
	} {
		cleaned := stripThoughtSignaturesForCredential(body, providerType)
		assert.NotContains(t, string(cleaned), "__thought__", providerType)
		assert.Contains(t, string(cleaned), `"tool_call_id":"call_1"`, providerType)
	}
}
