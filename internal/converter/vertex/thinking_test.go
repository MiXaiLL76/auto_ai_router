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

func TestIsThinkingCapableModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-2.5-flash", true},
		{"gemini-2.5-pro", true},
		{"gemini-2.5-flash-lite", true},
		{"gemini-2.5-flash-image", false},
		{"gemini-3-pro-image-preview", false},
		{"gemini-3-flash-preview", true},
		{"gemini-3-pro", true},
		{"gemini-2.0-flash", false},
		{"gemini-1.5-pro", false},
		{"gpt-4", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.want, isThinkingCapableModel(tt.model))
		})
	}
}

func TestDisableThinkingConfig(t *testing.T) {
	t.Run("gemini25_flash_sets_budget_zero", func(t *testing.T) {
		cfg := disableThinkingConfig("gemini-2.5-flash")
		require.NotNil(t, cfg)
		assert.False(t, cfg.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingBudget)
		assert.Equal(t, int32(0), *cfg.ThinkingBudget)
	})

	t.Run("gemini25_pro_returns_dynamic_budget", func(t *testing.T) {
		cfg := disableThinkingConfig("gemini-2.5-pro")
		require.NotNil(t, cfg, "gemini-2.5-pro must return dynamic config, not nil")
		assert.False(t, cfg.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingBudget)
		assert.Equal(t, int32(-1), *cfg.ThinkingBudget, "must use dynamic (-1) budget")
	})

	t.Run("gemini3_flash_sets_level_minimal", func(t *testing.T) {
		cfg := disableThinkingConfig("gemini-3-flash-preview")
		require.NotNil(t, cfg)
		assert.False(t, cfg.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, cfg.ThinkingLevel)
	})

	t.Run("gemini3_pro_sets_level_low", func(t *testing.T) {
		cfg := disableThinkingConfig("gemini-3.1-pro-preview")
		require.NotNil(t, cfg)
		assert.False(t, cfg.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelLow, cfg.ThinkingLevel)
	})
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
		// Official budget values: minimal/low=1024, medium=8192, high=24576
		{"minimal_flash_lite", "minimal", "gemini-2.5-flash-lite", 1024, false},
		{"minimal_pro", "minimal", "gemini-2.5-pro", 1024, false},
		{"minimal_flash", "minimal", "gemini-2.5-flash", 1024, false},
		{"low", "low", "gemini-2.5-flash", 1024, false},
		{"medium", "medium", "gemini-2.5-flash", 8192, false},
		{"high", "high", "gemini-2.5-flash", 24576, false},
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

func TestMapReasoningEffort_Gemini25Pro_Disable(t *testing.T) {
	// gemini-2.5-pro cannot disable thinking → returns dynamic budget (-1)
	t.Run("disable_returns_dynamic", func(t *testing.T) {
		result := mapReasoningEffort("disable", "gemini-2.5-pro")
		require.NotNil(t, result, "gemini-2.5-pro: must return dynamic config, not nil")
		assert.Equal(t, int32(-1), *result.ThinkingBudget, "must use dynamic (-1) budget")
	})
	t.Run("none_returns_dynamic", func(t *testing.T) {
		result := mapReasoningEffort("none", "gemini-2.5-pro")
		require.NotNil(t, result, "gemini-2.5-pro: must return dynamic config, not nil")
		assert.Equal(t, int32(-1), *result.ThinkingBudget, "must use dynamic (-1) budget")
	})
}

func TestMapReasoningEffort_Gemini3(t *testing.T) {
	tests := []struct {
		name   string
		effort string
		model  string
		level  genai.ThinkingLevel
		think  bool
	}{
		// "minimal": flash→Minimal, non-flash→Low (Minimal not supported on pro)
		{"minimal_flash", "minimal", "gemini-3-flash", genai.ThinkingLevelMinimal, false},
		{"minimal_pro", "minimal", "gemini-3-pro", genai.ThinkingLevelLow, false},
		// "low": all variants → Low (not Minimal, even for flash)
		{"low_flash", "low", "gemini-3-flash", genai.ThinkingLevelLow, false},
		{"low_pro", "low", "gemini-3-pro", genai.ThinkingLevelLow, false},
		// "medium": flash→Medium, pro→High (pro doesn't support MEDIUM)
		{"medium_flash", "medium", "gemini-3-flash", genai.ThinkingLevelMedium, false},
		{"medium_pro", "medium", "gemini-3-pro", genai.ThinkingLevelHigh, false},
		{"high_any", "high", "gemini-3-pro", genai.ThinkingLevelHigh, false},
		// disable/none: flash→Minimal, pro→Low (minimum supported)
		{"disable_pro", "disable", "gemini-3-pro", genai.ThinkingLevelLow, false},
		{"none_flash", "none", "gemini-3-flash", genai.ThinkingLevelMinimal, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapReasoningEffort(tt.effort, tt.model)
			require.NotNil(t, result)
			assert.Equal(t, tt.think, result.IncludeThoughts)
			assert.Equal(t, tt.level, result.ThinkingLevel)
		})
	}
}

func TestMapAnthropicThinking(t *testing.T) {
	t.Run("disabled_gemini25", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "disabled",
			"budget_tokens": float64(0),
		}
		result := mapAnthropicThinking(thinking, "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(0), *result.ThinkingBudget)
	})

	t.Run("disabled_gemini3", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "disabled",
			"budget_tokens": float64(0),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-flash-preview")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, result.ThinkingLevel)
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
		assert.False(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(15000), *result.ThinkingBudget)
	})

	t.Run("enabled_gemini3_high", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(20000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_medium_flash", func(t *testing.T) {
		// budget >= 5000 on flash → Medium
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(5000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-flash")
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMedium, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_medium_pro_uses_high", func(t *testing.T) {
		// budget >= 5000 on pro → High (pro doesn't support MEDIUM)
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(5000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_low_pro", func(t *testing.T) {
		// budget < 5000 + non-flash → Low (Minimal not supported on pro)
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(1000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-pro")
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelLow, result.ThinkingLevel)
	})

	t.Run("enabled_gemini3_minimal_flash", func(t *testing.T) {
		// budget < 5000 + flash → Minimal
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(1000),
		}
		result := mapAnthropicThinking(thinking, "gemini-3-flash")
		assert.False(t, result.IncludeThoughts)
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
		assert.False(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(10000), *result.ThinkingBudget)
	})

	t.Run("with reasoning_effort only", func(t *testing.T) {
		result := mapReasoningToThinkingConfig(nil, "high", "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(24576), *result.ThinkingBudget)
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
		assert.Equal(t, int32(8192), *result.ThinkingBudget)
	})

	t.Run("gemini-3 model with reasoning_effort", func(t *testing.T) {
		result := mapReasoningToThinkingConfig(nil, "high", "gemini-3-pro")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})
}

func TestMapNativeThinkingConfig(t *testing.T) {
	t.Run("gemini25_with_budget", func(t *testing.T) {
		tc := map[string]interface{}{"thinking_budget": float64(8192), "include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(8192), *result.ThinkingBudget)
	})

	t.Run("gemini25_with_budget_and_include_thoughts_false", func(t *testing.T) {
		tc := map[string]interface{}{"thinking_budget": float64(8192), "include_thoughts": false}
		result := mapNativeThinkingConfig(tc, "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(8192), *result.ThinkingBudget)
	})

	t.Run("gemini25_budget_zero_flash_allowed", func(t *testing.T) {
		// budget=0 disables thinking on flash; include_thoughts must be false
		tc := map[string]interface{}{"thinking_budget": float64(0)}
		result := mapNativeThinkingConfig(tc, "gemini-2.5-flash")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts, "include_thoughts must be false when budget=0")
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(0), *result.ThinkingBudget)
	})

	t.Run("gemini25pro_budget_zero_uses_dynamic", func(t *testing.T) {
		tc := map[string]interface{}{"thinking_budget": float64(0)}
		result := mapNativeThinkingConfig(tc, "gemini-2.5-pro")
		require.NotNil(t, result, "gemini-2.5-pro: budget=0 must use dynamic (-1), not nil")
		require.NotNil(t, result.ThinkingBudget)
		assert.Equal(t, int32(-1), *result.ThinkingBudget, "must use dynamic (-1) budget")
	})

	t.Run("gemini3_flash_with_level_minimal", func(t *testing.T) {
		tc := map[string]interface{}{"thinking_level": "minimal", "include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-3-flash")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, result.ThinkingLevel)
	})

	t.Run("gemini3_flash_with_level_minimal_and_include_thoughts_false", func(t *testing.T) {
		tc := map[string]interface{}{"thinking_level": "minimal", "include_thoughts": false}
		result := mapNativeThinkingConfig(tc, "gemini-3-flash")
		require.NotNil(t, result)
		assert.False(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, result.ThinkingLevel)
	})

	t.Run("gemini3_pro_with_level_minimal_uses_low", func(t *testing.T) {
		// "minimal" is not supported on pro; falls back to Low
		tc := map[string]interface{}{"thinking_level": "minimal"}
		result := mapNativeThinkingConfig(tc, "gemini-3-pro")
		require.NotNil(t, result)
		assert.Equal(t, genai.ThinkingLevelLow, result.ThinkingLevel)
	})

	t.Run("gemini3_pro_with_level_medium_uses_high", func(t *testing.T) {
		// Pro variants don't support MEDIUM; falls back to High
		tc := map[string]interface{}{"thinking_level": "medium", "include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-3.1-pro-preview")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, result.ThinkingLevel)
	})

	t.Run("gemini3_flash_with_level_medium", func(t *testing.T) {
		// Flash supports MEDIUM
		tc := map[string]interface{}{"thinking_level": "medium", "include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-3-flash")
		require.NotNil(t, result)
		assert.True(t, result.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMedium, result.ThinkingLevel)
	})

	t.Run("gemini3_no_level_flash_defaults_minimal", func(t *testing.T) {
		tc := map[string]interface{}{"include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-3-flash")
		require.NotNil(t, result)
		assert.Equal(t, genai.ThinkingLevelMinimal, result.ThinkingLevel)
	})

	t.Run("gemini3_no_level_pro_defaults_low", func(t *testing.T) {
		tc := map[string]interface{}{"include_thoughts": true}
		result := mapNativeThinkingConfig(tc, "gemini-3-pro")
		require.NotNil(t, result)
		assert.Equal(t, genai.ThinkingLevelLow, result.ThinkingLevel)
	})
}

func TestLowestThinkingLevel(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  genai.ThinkingLevel
	}{
		// Flash variants that still accept MINIMAL keep the cheaper floor.
		{"flash preview", "gemini-3-flash-preview", genai.ThinkingLevelMinimal},
		{"flash lite", "gemini-3.5-flash-lite", genai.ThinkingLevelMinimal},
		{"flash 3.6", "gemini-3.6-flash", genai.ThinkingLevelMinimal},
		// Observed in production rejecting MINIMAL with 400 INVALID_ARGUMENT.
		{"flash 3.7", "gemini-3.7-flash", genai.ThinkingLevelLow},
		{"flash 3.8", "gemini-3.8-flash", genai.ThinkingLevelLow},
		{"flash 3.7 dated", "gemini-3.7-flash-001", genai.ThinkingLevelLow},
		{"flash 3.7 mixed case", "Gemini-3.7-Flash", genai.ThinkingLevelLow},
		// Pro variants never supported MINIMAL.
		{"pro", "gemini-3.1-pro-preview", genai.ThinkingLevelLow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, lowestThinkingLevel(tt.model))
		})
	}
}

func TestDisableThinkingConfig_MinimalRejectingFlash(t *testing.T) {
	for _, model := range []string{"gemini-3.7-flash", "gemini-3.8-flash"} {
		t.Run(model, func(t *testing.T) {
			cfg := disableThinkingConfig(model)
			require.NotNil(t, cfg)
			assert.Equal(t, genai.ThinkingLevelLow, cfg.ThinkingLevel)
			assert.False(t, cfg.IncludeThoughts)
		})
	}
}

func TestMapReasoningEffort_MinimalOnRejectingFlash(t *testing.T) {
	cfg := mapReasoningEffort("minimal", "gemini-3.7-flash")
	require.NotNil(t, cfg)
	assert.Equal(t, genai.ThinkingLevelLow, cfg.ThinkingLevel)

	cfg = mapReasoningEffort("minimal", "gemini-3-flash-preview")
	require.NotNil(t, cfg)
	assert.Equal(t, genai.ThinkingLevelMinimal, cfg.ThinkingLevel)
}
