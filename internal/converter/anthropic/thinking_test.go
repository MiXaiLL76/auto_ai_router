package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMapReasoningEffortToBudget(t *testing.T) {
	tests := []struct {
		effort string
		want   int
	}{
		{"minimal", 1000},
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

func TestMapThinkingConfig(t *testing.T) {
	t.Run("nil_thinking_no_effort", func(t *testing.T) {
		result := mapThinkingConfig(nil, "")
		assert.Nil(t, result)
	})

	t.Run("enabled_with_budget", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(10000),
		}
		result := mapThinkingConfig(thinking, "")
		assert.NotNil(t, result)
		assert.Equal(t, "enabled", result.Type)
		assert.Equal(t, 10000, result.BudgetTokens)
	})

	t.Run("disabled", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type": "disabled",
		}
		result := mapThinkingConfig(thinking, "")
		assert.Nil(t, result)
	})

	t.Run("enabled_no_budget_falls_to_reasoning_effort", func(t *testing.T) {
		thinking := map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": float64(0),
		}
		result := mapThinkingConfig(thinking, "medium")
		assert.NotNil(t, result)
		assert.Equal(t, "enabled", result.Type)
		assert.Equal(t, 15000, result.BudgetTokens)
	})

	t.Run("reasoning_effort_only", func(t *testing.T) {
		result := mapThinkingConfig(nil, "high")
		assert.NotNil(t, result)
		assert.Equal(t, "enabled", result.Type)
		assert.Equal(t, 30000, result.BudgetTokens)
	})

	t.Run("reasoning_effort_disable", func(t *testing.T) {
		result := mapThinkingConfig(nil, "disable")
		assert.Nil(t, result)
	})
}
