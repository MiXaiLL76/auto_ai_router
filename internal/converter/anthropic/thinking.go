package anthropic

import (
	"regexp"
	"strconv"
	"strings"
)

const minTextTokensWithThinking = 1024

var claudeVersionPattern = regexp.MustCompile(`claude-(?:opus|sonnet|haiku|fable)-([0-9]+)(?:[.-]([0-9]+))?`)

// mapThinkingConfig maps OpenAI thinking / reasoning_effort parameters to an Anthropic
// ThinkingConfig (and optional OutputConfig for Claude 4+ adaptive thinking).
//
// Priority:
//  1. Anthropic-style thinking param: {"type": "enabled"/"adaptive", ...}
//  2. OpenAI reasoning_effort string → mapped accordingly
//
// Claude 3.x: returns (*AnthropicThinking{Type:"enabled", BudgetTokens:N}, nil)
// Claude 4+:  returns (*AnthropicThinking{Type:"adaptive"}, *AnthropicOutputConfig{Effort:E})
func mapThinkingConfig(thinking interface{}, reasoningEffort string, modelName string) (*AnthropicThinking, *AnthropicOutputConfig) {
	adaptive := isAdaptiveThinkingModel(modelName)

	if thinking != nil {
		if thinkingMap, ok := thinking.(map[string]interface{}); ok {
			thinkingType, _ := thinkingMap["type"].(string)
			display, _ := thinkingMap["display"].(string)
			switch thinkingType {
			case "enabled":
				budgetTokens, _ := thinkingMap["budget_tokens"].(float64)
				if budgetTokens > 0 {
					if adaptive {
						return &AnthropicThinking{Type: "adaptive", Display: "summarized"},
							&AnthropicOutputConfig{Effort: budgetTokensToEffort(int(budgetTokens))}
					}
					return &AnthropicThinking{Type: "enabled", BudgetTokens: int(budgetTokens)}, nil
				}
				// "enabled" without positive budget — fall through to reasoning_effort
			case "adaptive":
				effort, _ := thinkingMap["effort"].(string)
				if effort == "" {
					effort = "medium"
				}
				if adaptive {
					return &AnthropicThinking{Type: "adaptive", Display: display}, &AnthropicOutputConfig{Effort: effort}
				}
				// Caller passed adaptive but model uses legacy format — convert effort→budget
				budget := effortToBudgetTokens(effort)
				if budget > 0 {
					return &AnthropicThinking{Type: "enabled", BudgetTokens: budget}, nil
				}
			case "disabled":
				return nil, nil
			}
		}
	}

	if reasoningEffort != "" {
		if adaptive {
			effort := mapReasoningEffortToEffort(reasoningEffort)
			if effort != "" {
				return &AnthropicThinking{Type: "adaptive", Display: "summarized"}, &AnthropicOutputConfig{Effort: effort}
			}
		} else {
			budget := mapReasoningEffortToBudget(reasoningEffort)
			if budget > 0 {
				return &AnthropicThinking{Type: "enabled", BudgetTokens: budget}, nil
			}
		}
	}

	return nil, nil
}

// applyThinkingSideEffects computes the max_tokens/anthropic_beta/temperature updates a
// resolved thinking config (tc, oc as returned by mapThinkingConfig) requires. Shared by
// OpenAIToAnthropic (messages.go) and NormalizeMessagesForPassthrough (messages_api.go) —
// both already share mapThinkingConfig's resolution logic; this shares the "what follows
// from that resolution" logic too, so a future change to Anthropic's thinking requirements
// (a new required beta, a different temperature exception, ...) only needs updating here
// instead of in two independently-maintained copies. Callers are responsible for writing
// the returned values into whatever request representation they use (typed struct vs raw
// JSON map) and for actually setting thinking/output_config themselves.
func applyThinkingSideEffects(tc *AnthropicThinking, oc *AnthropicOutputConfig, maxTokens int, betas []string) (newMaxTokens int, newBetas []string, temperature float64) {
	newMaxTokens = EnsureMaxTokensForThinking(maxTokens, tc)
	newBetas = betas
	if oc != nil {
		// effort-based adaptive thinking requires the effort beta header
		newBetas = appendBetaUnique(betas, "effort-2025-11-24")
	}
	// Anthropic requires temperature=1.0 when thinking is enabled.
	return newMaxTokens, newBetas, 1.0
}

// appendBetaUnique appends s to slice only if it is not already present.
func appendBetaUnique(slice []string, s string) []string {
	for _, v := range slice {
		if v == s {
			return slice
		}
	}
	return append(slice, s)
}

// MapThinkingConfigFromEffort is the public entry point for packages (e.g. responses converter)
// that only have an effort string and model name, without a raw thinking param.
func MapThinkingConfigFromEffort(effort, modelName string) (*AnthropicThinking, *AnthropicOutputConfig) {
	return mapThinkingConfig(nil, effort, modelName)
}

// EnsureMaxTokensForThinking raises max_tokens when legacy thinking uses a
// budget_tokens value. Anthropic requires max_tokens to be greater than the
// thinking budget; otherwise the provider rejects the request before generation.
func EnsureMaxTokensForThinking(maxTokens int, thinking *AnthropicThinking) int {
	if thinking == nil || thinking.Type != "enabled" || thinking.BudgetTokens <= 0 {
		return maxTokens
	}
	if maxTokens > thinking.BudgetTokens {
		return maxTokens
	}
	return thinking.BudgetTokens + minTextTokensWithThinking
}

// isAdaptiveThinkingModel reports whether the model uses adaptive thinking.
func isAdaptiveThinkingModel(model string) bool {
	lower := strings.ToLower(model)
	if strings.Contains(lower, "mythos") {
		return true
	}

	match := claudeVersionPattern.FindStringSubmatch(lower)
	if len(match) == 0 {
		return false
	}
	major, err := strconv.Atoi(match[1])
	if err != nil {
		return false
	}
	if major >= 5 {
		return true
	}
	if major != 4 || len(match[2]) == 0 || len(match[2]) > 2 {
		return false
	}
	minor, err := strconv.Atoi(match[2])
	return err == nil && minor >= 6
}

// mapReasoningEffortToBudget maps an OpenAI reasoning_effort string to an Anthropic
// budget_tokens value for Claude 3.x models. Returns 0 for unknown or disabled values.
func mapReasoningEffortToBudget(effort string) int {
	switch effort {
	case "minimal":
		return 1024
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

// mapReasoningEffortToEffort maps an OpenAI reasoning_effort string to an Anthropic
// output_config.effort value for Claude 4+ models. Returns "" for disabled values.
func mapReasoningEffortToEffort(effort string) string {
	switch effort {
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high":
		return "high"
	case "xhigh":
		return "xhigh"
	case "max":
		return "max"
	case "disable", "none":
		return ""
	default:
		return ""
	}
}

// budgetTokensToEffort converts a legacy budget_tokens value to the nearest
// output_config.effort level for Claude 4+ adaptive thinking.
func budgetTokensToEffort(budget int) string {
	switch {
	case budget <= 5000:
		return "low"
	case budget <= 15000:
		return "medium"
	case budget <= 30000:
		return "high"
	default:
		return "high"
	}
}

// effortToBudgetTokens converts an output_config.effort string to a legacy
// budget_tokens value for Claude 3.x models.
func effortToBudgetTokens(effort string) int {
	switch effort {
	case "low":
		return 5000
	case "medium":
		return 15000
	case "high", "xhigh", "max":
		return 30000
	default:
		return 0
	}
}
