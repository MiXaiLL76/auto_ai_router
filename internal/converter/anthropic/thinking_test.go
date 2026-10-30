package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMapReasoningEffortToBudget(t *testing.T) {
	tests := []struct {
		effort string
		want   int
	}{
		{"minimal", 1024},
		{"low", 5000},
		{"medium", 15000},
		{"high", 30000},
		{"disable", 0},
		{"none", 0},
		{"unknown", 0},
		{"", 0},
	}
	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			assert.Equal(t, tt.want, mapReasoningEffortToBudget(tt.effort))
		})
	}
}

func TestEnsureMaxTokensForThinking(t *testing.T) {
	thinking := &AnthropicThinking{Type: "enabled", BudgetTokens: 5000}

	assert.Equal(t, 6000, EnsureMaxTokensForThinking(6000, thinking))
	assert.Equal(t, 6024, EnsureMaxTokensForThinking(5000, thinking))
	assert.Equal(t, 6024, EnsureMaxTokensForThinking(4096, thinking))
	assert.Equal(t, 4096, EnsureMaxTokensForThinking(4096, nil))
	assert.Equal(t, 4096, EnsureMaxTokensForThinking(4096, &AnthropicThinking{Type: "adaptive"}))
}

// Claude 3.x: expects type="enabled" + budget_tokens, no output_config.
func TestMapThinkingConfig_Classic(t *testing.T) {
	const model = "claude-3-7-sonnet-20250219"

	t.Run("nil_thinking_no_effort", func(t *testing.T) {
		tc, oc := mapThinkingConfig(nil, "", model)
		assert.Nil(t, tc)
		assert.Nil(t, oc)
	})

	t.Run("enabled_with_budget", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(10000),
		}
		tc, oc := mapThinkingConfig(thinking, "", model)
		assert.NotNil(t, tc)
		assert.Equal(t, "enabled", tc.Type)
		assert.Equal(t, 10000, tc.BudgetTokens)
		assert.Nil(t, oc)
	})

	t.Run("disabled", func(t *testing.T) {
		thinking := map[string]interface{}{"type": "disabled"}
		tc, oc := mapThinkingConfig(thinking, "", model)
		assert.Nil(t, tc)
		assert.Nil(t, oc)
	})

	t.Run("enabled_no_budget_falls_to_reasoning_effort", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(0),
		}
		tc, oc := mapThinkingConfig(thinking, "medium", model)
		assert.NotNil(t, tc)
		assert.Equal(t, "enabled", tc.Type)
		assert.Equal(t, 15000, tc.BudgetTokens)
		assert.Nil(t, oc)
	})

	t.Run("reasoning_effort_only", func(t *testing.T) {
		tc, oc := mapThinkingConfig(nil, "high", model)
		assert.NotNil(t, tc)
		assert.Equal(t, "enabled", tc.Type)
		assert.Equal(t, 30000, tc.BudgetTokens)
		assert.Nil(t, oc)
	})

	t.Run("reasoning_effort_disable", func(t *testing.T) {
		tc, oc := mapThinkingConfig(nil, "disable", model)
		assert.Nil(t, tc)
		assert.Nil(t, oc)
	})

	t.Run("adaptive_converts_to_budget_keeps_block_binding", func(t *testing.T) {
		thinking := map[string]interface{}{"type": "adaptive", "block_binding": map[string]interface{}{}}
		tc, _ := mapThinkingConfig(thinking, "", model)
		require.NotNil(t, tc)
		assert.Equal(t, "enabled", tc.Type)
		assert.NotNil(t, tc.BlockBinding)
	})
}

// Claude Opus 4+ and mythos-preview: expects type="adaptive" + output_config.effort, no budget_tokens.
func TestMapThinkingConfig_Adaptive(t *testing.T) {
	models := []string{
		"claude-opus-4-8",
		"claude-opus-4.7",
		"claude-opus-4-6",
		"claude-opus-5",
		"claude-sonnet-4-6",
		"claude-sonnet-5",
		"claude-fable-5",
		"claude-mythos-preview",
		"global.anthropic.claude-opus-4-8",
	}

	for _, model := range models {
		t.Run(model+"/reasoning_effort_high", func(t *testing.T) {
			tc, oc := mapThinkingConfig(nil, "high", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.Equal(t, 0, tc.BudgetTokens)
			assert.Equal(t, "summarized", tc.Display)
			assert.NotNil(t, oc)
			assert.Equal(t, "high", oc.Effort)
		})

		t.Run(model+"/reasoning_effort_medium", func(t *testing.T) {
			tc, oc := mapThinkingConfig(nil, "medium", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.NotNil(t, oc)
			assert.Equal(t, "medium", oc.Effort)
		})

		t.Run(model+"/reasoning_effort_minimal_maps_to_low", func(t *testing.T) {
			tc, oc := mapThinkingConfig(nil, "minimal", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.NotNil(t, oc)
			assert.Equal(t, "low", oc.Effort)
		})

		t.Run(model+"/enabled_budget_converts_to_effort", func(t *testing.T) {
			thinking := map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": float64(20000),
			}
			tc, oc := mapThinkingConfig(thinking, "", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.Equal(t, "summarized", tc.Display)
			assert.NotNil(t, oc)
			assert.Equal(t, "high", oc.Effort)
		})

		t.Run(model+"/adaptive_type_passthrough", func(t *testing.T) {
			thinking := map[string]interface{}{
				"type":   "adaptive",
				"effort": "xhigh",
			}
			tc, oc := mapThinkingConfig(thinking, "", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.NotNil(t, oc)
			assert.Equal(t, "xhigh", oc.Effort)
		})

		t.Run(model+"/adaptive_display_passthrough", func(t *testing.T) {
			thinking := map[string]interface{}{
				"type":    "adaptive",
				"effort":  "high",
				"display": "summarized",
			}
			tc, oc := mapThinkingConfig(thinking, "", model)
			assert.NotNil(t, tc)
			assert.Equal(t, "adaptive", tc.Type)
			assert.Equal(t, "summarized", tc.Display)
			assert.NotNil(t, oc)
			assert.Equal(t, "high", oc.Effort)
		})

		t.Run(model+"/disabled", func(t *testing.T) {
			thinking := map[string]interface{}{"type": "disabled"}
			tc, oc := mapThinkingConfig(thinking, "", model)
			assert.Nil(t, tc)
			assert.Nil(t, oc)
		})

		t.Run(model+"/no_effort_nil", func(t *testing.T) {
			tc, oc := mapThinkingConfig(nil, "none", model)
			assert.Nil(t, tc)
			assert.Nil(t, oc)
		})
	}
}

func TestIsAdaptiveThinkingModel(t *testing.T) {
	tests := []struct {
		model    string
		adaptive bool
	}{
		// Claude 4.6+ and Claude 5 → adaptive
		{"claude-opus-5", true},
		{"claude-sonnet-5", true},
		{"claude-fable-5", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-6", true},
		{"global.anthropic.claude-opus-4-8", true},
		{"claude-mythos-preview", true},
		// Claude 4.5 and earlier → legacy enabled thinking
		{"claude-opus-4-5-20251101", false},
		{"claude-opus-4-20250514", false},
		{"claude-sonnet-4-5-20250929", false},
		{"claude-haiku-4-5-20251001", false},
		{"claude-3-7-sonnet-20250219", false},
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-opus-20240229", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.adaptive, isAdaptiveThinkingModel(tt.model))
		})
	}
}

func TestMapThinkingConfig_AdaptiveBlockBindingAndDefaultEffort(t *testing.T) {
	const model = "claude-opus-4-8"
	binding := map[string]interface{}{"prefix_mismatch_behavior": "drop_block"}
	tc, oc := mapThinkingConfig(map[string]interface{}{"type": "adaptive", "block_binding": binding}, "", model)
	require.NotNil(t, tc)
	assert.Equal(t, binding, tc.BlockBinding)
	assert.Nil(t, oc, "no effort given: the model default applies")

	_, oc = mapThinkingConfig(map[string]interface{}{"type": "adaptive"}, "high", model)
	assert.Equal(t, &AnthropicOutputConfig{Effort: "high"}, oc, "reasoning_effort fills a missing effort")

	_, betas, _ := applyThinkingSideEffects(tc, nil, 100, []string{"x"})
	assert.Equal(t, []string{"x", thinkingBindingBeta}, betas)
}

func TestMapThinkingConfig_ExplicitEffortWinsOverBudget(t *testing.T) {
	budget := map[string]interface{}{"type": "enabled", "budget_tokens": float64(2000)}
	_, oc := mapThinkingConfig(budget, "xhigh", "claude-opus-4-7")
	assert.Equal(t, &AnthropicOutputConfig{Effort: "xhigh"}, oc)
	_, oc = mapThinkingConfig(budget, "", "claude-opus-4-7")
	assert.Equal(t, &AnthropicOutputConfig{Effort: "low"}, oc, "no explicit effort: derived from the budget")
}

// Claude Sonnet 5.5 rejects "disabled": every way of asking for no thinking becomes
// between_tools. Other models keep their rules.
func TestMapThinkingConfig_Sonnet55ThinkingOff(t *testing.T) {
	const model = "policy/claude-sonnet-5.5"
	for _, effort := range []string{"none", "disable", "disabled", "off"} {
		tc, oc := mapThinkingConfig(nil, effort, model)
		assert.Equal(t, &AnthropicThinking{Type: "between_tools"}, tc, effort)
		assert.Nil(t, oc, effort)
	}
	for _, thinking := range []string{"disabled", "between_tools"} {
		tc, oc := mapThinkingConfig(map[string]interface{}{"type": thinking}, "", model)
		assert.Equal(t, &AnthropicThinking{Type: "between_tools"}, tc, thinking)
		assert.Nil(t, oc, thinking)
	}
	tc, _ := mapThinkingConfig(map[string]interface{}{"type": "between_tools"}, "", "claude-sonnet-5")
	assert.Nil(t, tc, "between_tools is Sonnet 5.5's")
	tc, _ = mapThinkingConfig(nil, "none", "claude-opus-5-5")
	assert.Nil(t, tc, "Opus 5.5 can't turn thinking off")
}

func TestThinkingOffModeFor(t *testing.T) {
	tests := []struct {
		model string
		mode  thinkingOffMode
	}{
		{"claude-sonnet-5-5", thinkingOffBetweenTools},
		{"policy/claude-sonnet-5.5", thinkingOffBetweenTools},
		{"us.anthropic.claude-sonnet-5-5-v1:0", thinkingOffBetweenTools},
		{"Claude-Sonnet-5-5@20261001", thinkingOffBetweenTools},
		{"claude-opus-5-5", thinkingAlwaysOn},
		{"claude-fable-5-1", thinkingAlwaysOn},
		{"claude-fable-5", thinkingAlwaysOn},
		{"claude-mythos-preview", thinkingAlwaysOn},
		{"claude-opus-5", thinkingOffDisabledUpToHigh},
		{"claude-opus-5-20260101", thinkingOffDisabledUpToHigh},
		{"claude-haiku-5-5", thinkingOffDisabledUpToHigh},
		{"claude-sonnet-5", thinkingOffDisabled},
		{"claude-sonnet-5-20260101", thinkingOffDisabled},
		{"claude-sonnet-5-50", thinkingOffDisabled},
		{"claude-opus-4-8", thinkingOffWhenOmitted},
		{"claude-haiku-4-5", thinkingOffWhenOmitted},
		{"claude-3-5-sonnet-20241022", thinkingOffWhenOmitted},
		{"gemini-3.5-flash", thinkingOffWhenOmitted},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.mode, thinkingOffModeFor(tt.model))
		})
	}
}

func TestThinkingOff(t *testing.T) {
	disabled := &AnthropicThinking{Type: "disabled"}
	betweenTools := &AnthropicThinking{Type: "between_tools"}
	tests := []struct {
		model, effort string
		want          *AnthropicThinking
	}{
		{"claude-sonnet-5-5", "", betweenTools},
		{"claude-sonnet-5-5", "high", betweenTools},
		{"claude-sonnet-5-5", "xhigh", nil},
		{"claude-opus-5", "high", disabled},
		{"claude-opus-5", "max", nil},
		{"claude-haiku-5-5", "xhigh", nil},
		{"claude-opus-5-5", "", nil},
		{"claude-fable-5-1", "low", nil},
		{"claude-mythos-5-1", "", nil},
		{"claude-sonnet-5", "max", disabled},
		{"claude-opus-4-7", "", disabled},
		{"gemini-3.5-flash", "", disabled},
	}
	for _, tt := range tests {
		t.Run(tt.model+"/"+tt.effort, func(t *testing.T) {
			assert.Equal(t, tt.want, thinkingOff(tt.model, tt.effort))
		})
	}
}
