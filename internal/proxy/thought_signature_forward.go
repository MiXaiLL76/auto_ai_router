package proxy

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

func stripThoughtSignaturesForCredential(body []byte, providerType config.ProviderType) []byte {
	switch providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini, config.ProviderTypeAIR:
		return body
	default:
		return converterutil.StripThoughtSignaturesFromToolIDs(body)
	}
}
