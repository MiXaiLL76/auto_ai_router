package models

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"log/slog"
	"testing"
)

func TestGPT6ResponsesDetection(t *testing.T) {
	manager := New(slog.Default(), 100, nil)
	for _, model := range []string{"gpt-6-astra", "openai/gpt-6-astra", "GPT-6-ASTRA"} {
		assert.True(t, manager.IsPassthroughResponses(model), model)
		assert.True(t, manager.IsPassthroughResponsesForProvider(model, config.ProviderTypeOpenAI), model)
		assert.False(t, manager.IsPassthroughResponsesForProvider(model, config.ProviderTypeAnthropic), model)
	}
}
