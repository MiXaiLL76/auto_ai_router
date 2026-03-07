package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestIsGemini3Model(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-3-pro", true},
		{"gemini-3-flash", true},
		{"Gemini-3-Pro", true},
		{"gemini-2.5-flash", false},
		{"gemini-2.5-pro", false},
		{"gpt-4", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.want, isGemini3Model(tt.model))
		})
	}
}

func TestIsFlashModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-3-flash", true},
		{"gemini-2.5-flash-lite", true},
		{"GEMINI-3-FLASH", true},
		{"gemini-3-pro", false},
		{"gemini-2.5-pro", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.want, isFlashModel(tt.model))
		})
	}
}

func TestMapReasoningEffort_Gemini25(t *testing.T) {
	tests := []struct {
		name   string
		effort string
		model  string
		budget int32
		think  bool
	}{
		{"minimal_flash_lite", "minimal", "gemini-2.5-flash-lite", 1000, true},
		{"minimal_pro", "minimal", "gemini-2.5-pro", 5000, true},
		{"minimal_flash", "minimal", "gemini-2.5-flash", 3000, true},
		{"low", "low", "gemini-2.5-flash", 5000, true},
		{"medium", "medium", "gemini-2.5-flash", 15000, true},
		{"high", "high", "gemini-2.5-flash", 30000, true},
		{"disable", "disable", "gemini-2.5-flash", 0, false},
		{"none", "none", "gemini-2.5-flash", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapReasoningEffort(tt.effort, tt.model)
			require.NotNil(t, result)
			assert.Equal(t, tt.think, result.IncludeThoughts)
			require.NotNil(t, result.ThinkingBudget)
			assert.Equal(t, tt.budget, *result.ThinkingBudget)
		})
	}
}

func TestMapReasoningEffort_Gemini3(t *testing.T) {
	tests := []struct {
		name   string
		effort string
		model  string
		level  genai.ThinkingLevel
		think  bool
	}{
		{"minimal_flash", "minimal", "gemini-3-flash", genai.ThinkingLevelMinimal, true},
		{"minimal_pro", "minimal", "gemini-3-pro", genai.ThinkingLevelLow, true},
		{"low_flash", "low", "gemini-3-flash", genai.ThinkingLevelMinimal, true},
		{"low_pro", "low", "gemini-3-pro", genai.ThinkingLevelLow, true},
		{"medium_flash", "medium", "gemini-3-flash", genai.ThinkingLevelMedium, true},
		{"medium_pro", "medium", "gemini-3-pro", genai.ThinkingLevelHigh, true},
		{"high_any", "high", "gemini-3-pro", genai.ThinkingLevelHigh, true},
		{"disable", "disable", "gemini-3-pro", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapReasoningEffort(tt.effort, tt.model)
			require.NotNil(t, result)
			assert.Equal(t, tt.think, result.IncludeThoughts)
			if tt.think {
				assert.Equal(t, tt.level, result.ThinkingLevel)
			}
		})
	}
}

func TestMapAnthropicThinking(t *testing.T) {
	t.Run("disabled", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "disabled",
			"budget_tokens": float64(0),
		}
		result := mapAnthropicThinking(thinking, "gemini-2.5-flash")
		assert.False(t, result.IncludeThoughts)
	})

	t.Run("enabled_zero_budget", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(0),
		}
		result := mapAnthropicThinking(thinking, "gemini-2.5-flash")
		assert.False(t, result.IncludeThoughts)
	})

	t.Run("enabled_gemini25_budget", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(15000),
		}
		result := mapAnthropicThinking(thinking, "gemini-2.5-flash")
		assert.True(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(15000), *result.ThinkingBudget)
	})

	t.Run("enabled_gemini3_high", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(20000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_low", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(5000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelLow, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_minimal", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(1000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, result.ThinkingLevel)
	})
}

func TestMapReasoningToThinkingConfig(t *testing.T) {
	t.Run("nil thinking and empty effort returns nil", func(t *testing.T) {
		result := mapReasoningToThinkingConfig(nil, "", "gemini-2.5-flash")
		assert.Nil(t, result)
	})

	t.Run("with thinking map uses anthropic-style mapping", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(10000),
		}
		result := mapReasoningToThinkingConfig(thinking, "", "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(10000), *result.ThinkingBudget)
	})

	t.Run("with reasoning_effort only", func(t *testing.T) {
		result := mapReasoningToThinkingConfig(nil, "high", "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(30000), *result.ThinkingBudget)
	})

	t.Run("thinking map takes precedence over reasoning_effort", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(5000),
		}
		// Even though reasoning_effort is "high", thinking map should take precedence
		result := mapReasoningToThinkingConfig(thinking, "high", "gemini-2.5-flash")
		require.NotNil(t, result)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(5000), *result.ThinkingBudget)
	})

	t.Run("non-map thinking falls through to reasoning_effort", func(t *testing.T) {
		// thinking is not nil but not a map, so falls through
		result := mapReasoningToThinkingConfig("invalid", "medium", "gemini-2.5-flash")
		require.NotNil(t, result)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(15000), *result.ThinkingBudget)
	})

	t.Run("gemini-3 model with reasoning_effort", func(t *testing.T) {
		result := mapReasoningToThinkingConfig(nil, "high", "gemini-3-pro")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})
}
