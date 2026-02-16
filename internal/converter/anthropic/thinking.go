package anthropic

// mapThinkingConfig maps OpenAI thinking / reasoning_effort parameters to an Anthropic
// ThinkingConfig.  Returns nil when thinking should not be included in the request.
//
// Priority:
//  1. Anthropic-style thinking param: {"type": "enabled", "budget_tokens": N}
//  2. OpenAI reasoning_effort string → mapped to budget_tokens
func mapThinkingConfig(thinking interface{}, reasoningEffort string) *AnthropicThinking {
	if thinking != nil {
		if thinkingMap, ok := thinking.(map[string]interface{}); ok {
			thinkingType, _ := thinkingMap["type"].(string)
			switch thinkingType {
			case "enabled":
				budgetTokens, _ := thinkingMap["budget_tokens"].(float64)
				if budgetTokens > 0 {
					return &AnthropicThinking{
						Type:         "enabled",
						BudgetTokens: int(budgetTokens),
					}
				}
				// "enabled" without a positive budget — fall through to reasoning_effort
			case "disabled":
				return nil
			}
		}
	}

	// Map OpenAI reasoning_effort → Anthropic budget_tokens
	if reasoningEffort != "" {
		budget := mapReasoningEffortToBudget(reasoningEffort)
		if budget > 0 {
			return &AnthropicThinking{
				Type:         "enabled",
				BudgetTokens: budget,
			}
		}
	}

	return nil
}

// mapReasoningEffortToBudget maps an OpenAI reasoning_effort string to an Anthropic
// budget_tokens value.  Returns 0 for unknown or disabled values.
func mapReasoningEffortToBudget(effort string) int {
	switch effort {
	case "minimal":
		return 1000
	case "low":
		return 5000
	case "medium":
		return 15000
	case "high":
		return 30000
	case "disable", "none":
		return 0
	default:
		return 0
	}
}
