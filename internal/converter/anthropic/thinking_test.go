package anthropic

import (
	"encoding/json"
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
			if model == "claude-opus-5" {
				// Opus 5 takes an explicit "disabled" to turn thinking off
				// (up to effort high) and thinks by default, so the field
				// has to reach it -- without it the model reasons anyway.
				// The other models here either always think (no field is
				// right) or think only when asked (no field means off).
				require.NotNil(t, tc)
				assert.Equal(t, "disabled", tc.Type)
				assert.Nil(t, oc)
				return
			}
			assert.Nil(t, tc)
			assert.Nil(t, oc)
		})

		t.Run(model+"/no_effort_nil", func(t *testing.T) {
			tc, oc := mapThinkingConfig(nil, "none", model)
			if model == "claude-opus-5" {
				require.NotNil(t, tc)
				assert.Equal(t, "disabled", tc.Type)
				assert.Nil(t, oc)
				return
			}
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

	_, betas, _ := applyThinkingSideEffects(tc, nil, model, 100, []string{"x"})
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

func TestIsKnownClaudeModel(t *testing.T) {
	for _, model := range []string{
		"claude-opus-5-5",
		"claude-sonnet-5-5",
		"claude-haiku-4-5",
		"claude-fable-5-1",
		"claude-opus-4-5-20251101",
		"claude-sonnet-4-20250514",
		"us.anthropic.claude-opus-4-7-v1:0",
		"anthropic.claude-sonnet-4-6",
		"claude-mythos-preview",
		"Claude-Opus-5.5",
	} {
		assert.True(t, IsKnownClaudeModel(model), model)
	}
	for _, model := range []string{
		"gemini-3.5-flash",
		"gpt-5",
		"qwen3.8-max",
		"claude-3-haiku-20240307",
		"claude-3-5-sonnet-20241022",
		"claude",
		"",
	} {
		assert.False(t, IsKnownClaudeModel(model), model)
	}
}

// Fable 5.2 needs no model-specific code: the version parser classifies every provider
// spelling as adaptive thinking that can't be turned off, with sampling params removed,
// and the effort a client sets reaches the upstream on both the chat and the Messages path.
// Without an explicit effort none is injected, so the model's own default applies.
func TestFable52ThinkingFromVersionParser(t *testing.T) {
	for _, model := range []string{"claude-fable-5-2", "claude-fable-5.2", "policy/claude-fable-5.2", "global.anthropic.claude-fable-5-2"} {
		t.Run(model, func(t *testing.T) {
			assert.True(t, IsKnownClaudeModel(model))
			assert.True(t, isAdaptiveThinkingModel(model))
			assert.True(t, SamplingRemoved(model))
			assert.Equal(t, thinkingAlwaysOn, thinkingOffModeFor(model))

			for _, effort := range []string{"low", "high", "xhigh", "max"} {
				body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"` + effort + `","temperature":0,"top_p":0.5}`
				out, err := OpenAIToAnthropic([]byte(body), model, true, nil)
				require.NoError(t, err)
				var req map[string]interface{}
				require.NoError(t, json.Unmarshal(out, &req))
				assert.Equal(t, "adaptive", req["thinking"].(map[string]interface{})["type"], effort)
				assert.Equal(t, effort, req["output_config"].(map[string]interface{})["effort"], effort)
				assert.NotContains(t, req, "temperature")
				assert.NotContains(t, req, "top_p")

				got, err := normalizePassthrough(t, model, true, `,"thinking":{"type":"adaptive"},"output_config":{"effort":"`+effort+`"}`)
				require.NoError(t, err)
				assert.Equal(t, map[string]interface{}{"effort": effort}, got["output_config"], effort)
			}

			out, err := OpenAIToAnthropic([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`), model, true, nil)
			require.NoError(t, err)
			var req map[string]interface{}
			require.NoError(t, json.Unmarshal(out, &req))
			assert.NotContains(t, req, "output_config", "no effort injected over the model default")
		})
	}
}

// TestMapThinkingConfig_ThinkingOffOnDefaultThinkingModels covers the models
// that THINK BY DEFAULT and take an explicit "disabled" to turn thinking off
// (Claude Haiku 5.5, Claude Opus 5).
//
// The bug these cases pin: "disabled" and reasoning_effort "none" produced NO
// thinking field for these models, and since they think by default that meant
// they kept reasoning -- the client asked for no reasoning and got reasoning.
// In production, Opus 5 requests with reasoning_effort "none" came back with
// 104,672 prompt tokens against 59 completion tokens: the whole prompt
// budget went into reasoning the client had asked not to spend.
//
// The wording differs per model and the difference is load-bearing:
//   - Sonnet 5.5 takes "between_tools" -- thinking only between tool calls,
//     NOT thinking off. It is not the same thing and must not be conflated.
//   - A model that always thinks (Opus 5.5, Fable, Mythos) gets nothing: the
//     request cannot be satisfied.
//   - A model that thinks only when asked (Claude 4.x and older) gets nothing
//     too -- that IS thinking off, and it is what it always did.
func TestMapThinkingConfig_ThinkingOffOnDefaultThinkingModels(t *testing.T) {
	disabled := &AnthropicThinking{Type: "disabled"}
	betweenTools := &AnthropicThinking{Type: "between_tools"}

	tests := []struct {
		name      string
		model     string
		thinking  interface{}
		effort    string
		wantType  string
		wantNil   bool
		wantOCEff string
	}{
		{"haiku 5.5 thinking disabled", "claude-haiku-5-5", map[string]interface{}{"type": "disabled"}, "", "disabled", false, ""},
		{"haiku 5.5 effort none", "claude-haiku-5-5", nil, "none", "disabled", false, ""},
		{"haiku 5.5 effort none spelled off", "claude-haiku-5-5", nil, "off", "disabled", false, ""},
		{"haiku 5.5 effort none with high", "claude-haiku-5-5", nil, "none", "disabled", false, ""},
		{"opus 5 thinking disabled", "claude-opus-5", map[string]interface{}{"type": "disabled"}, "", "disabled", false, ""},
		{"opus 5 effort none", "claude-opus-5", nil, "none", "disabled", false, ""},
		// Sonnet 5.5 takes between_tools -- thinking only between tool
		// calls, which is not thinking off.
		{"sonnet 5.5 thinking disabled", "claude-sonnet-5-5", map[string]interface{}{"type": "disabled"}, "", "between_tools", false, ""},
		{"sonnet 5.5 effort none", "claude-sonnet-5-5", nil, "none", "between_tools", false, ""},
		// Models that always think get nothing: the request cannot be served.
		{"opus 5.5 thinking disabled", "claude-opus-5-5", map[string]interface{}{"type": "disabled"}, "", "", true, ""},
		{"opus 5.5 effort none", "claude-opus-5-5", nil, "none", "", true, ""},
		{"fable 5.1 effort none", "claude-fable-5-1", nil, "none", "", true, ""},
		// Models that think only when asked: no field means off, as before.
		{"opus 4.5 thinking disabled", "claude-opus-4-5", map[string]interface{}{"type": "disabled"}, "", "", true, ""},
		{"opus 4.5 effort none", "claude-opus-4-5", nil, "none", "", true, ""},
		{"sonnet 4.5 effort none", "claude-sonnet-4-5", nil, "none", "", true, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc, oc := mapThinkingConfig(tt.thinking, tt.effort, tt.model)
			if tt.wantNil {
				assert.Nil(t, tc)
				assert.Nil(t, oc)
				return
			}
			require.NotNil(t, tc)
			switch tt.wantType {
			case "disabled":
				assert.Equal(t, disabled.Type, tc.Type)
			case "between_tools":
				assert.Equal(t, betweenTools.Type, tc.Type)
			}
			if tt.wantOCEff == "" {
				assert.Nil(t, oc)
			} else {
				require.NotNil(t, oc)
				assert.Equal(t, tt.wantOCEff, oc.Effort)
			}
		})
	}
}

// TestValidateThinkingOffEffort covers the effort ceiling on a model that
// takes "disabled": Claude Haiku 5.5 and Claude Opus 5 turn thinking off only
// up to effort "high". Anthropic answers the rest itself --
// "output_config.effort 'xhigh' is not supported when thinking is disabled on
// this model" -- and the router answers it instead, naming the field and the
// fix, exactly as it already does for Sonnet 5.5's between_tools rules.
func TestValidateThinkingOffEffort(t *testing.T) {
	disabled := &AnthropicThinking{Type: "disabled"}
	adaptive := &AnthropicThinking{Type: "adaptive"}

	require.Error(t, validateThinkingOffEffort(disabled, "xhigh", "claude-haiku-5-5"))
	require.Error(t, validateThinkingOffEffort(disabled, "max", "claude-haiku-5-5"))
	require.Error(t, validateThinkingOffEffort(disabled, "xhigh", "claude-opus-5"))
	require.NoError(t, validateThinkingOffEffort(disabled, "high", "claude-haiku-5-5"))
	require.NoError(t, validateThinkingOffEffort(disabled, "medium", "claude-haiku-5-5"))
	require.NoError(t, validateThinkingOffEffort(disabled, "", "claude-haiku-5-5"))

	// A model that thinks only when asked has no such ceiling: there, no
	// thinking field at all IS thinking off, at any effort.
	require.NoError(t, validateThinkingOffEffort(disabled, "max", "claude-opus-4-5"))
	// And a thinking config that thinks is not constrained.
	require.NoError(t, validateThinkingOffEffort(adaptive, "xhigh", "claude-haiku-5-5"))
}
