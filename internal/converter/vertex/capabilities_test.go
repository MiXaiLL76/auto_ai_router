package vertex

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSupportsPenalty(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		// Penalty survived only up to Gemini 2.0.
		{"gemini-2.0-flash", true},
		{"gemini-1.5-pro", true},
		// 2.5 and 3.x reject it with 400 "Penalty is not enabled for this model".
		{"gemini-2.5-flash", false},
		{"gemini-2.5-pro", false},
		{"gemini-3-flash-preview", false},
		{"gemini-3.5-flash-lite", false},
		{"gemini-3.1-pro-preview", false},
		{"Gemini-3.7-Flash", false},
		// Unknown future Gemini models default to "not supported" — an allow-list
		// costs an optional knob, a deny-list would cost a failed request.
		{"gemini-4-flash", false},
		// Non-Gemini models routed through this converter keep prior behaviour.
		{"claude-sonnet-4", true},
		{"llama-3.1-70b", true},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.want, supportsPenalty(tt.model))
		})
	}
}

func TestBuildGenerationConfig_PenaltyGating(t *testing.T) {
	newReq := func(model string) *openai.OpenAIRequest {
		freq := 0.5
		pres := 0.3
		return &openai.OpenAIRequest{
			Model:            model,
			FrequencyPenalty: &freq,
			PresencePenalty:  &pres,
		}
	}

	t.Run("dropped_for_gemini3", func(t *testing.T) {
		cfg := buildGenerationConfig(newReq("gemini-3.5-flash-lite"), "gemini-3.5-flash-lite")
		require.NotNil(t, cfg)
		assert.Nil(t, cfg.FrequencyPenalty)
		assert.Nil(t, cfg.PresencePenalty)
	})

	t.Run("dropped_for_gemini25", func(t *testing.T) {
		cfg := buildGenerationConfig(newReq("gemini-2.5-pro"), "gemini-2.5-pro")
		require.NotNil(t, cfg)
		assert.Nil(t, cfg.FrequencyPenalty)
		assert.Nil(t, cfg.PresencePenalty)
	})

	t.Run("kept_for_gemini20", func(t *testing.T) {
		cfg := buildGenerationConfig(newReq("gemini-2.0-flash"), "gemini-2.0-flash")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.FrequencyPenalty)
		require.NotNil(t, cfg.PresencePenalty)
		assert.InDelta(t, 0.5, *cfg.FrequencyPenalty, 1e-6)
		assert.InDelta(t, 0.3, *cfg.PresencePenalty, 1e-6)
	})
}
