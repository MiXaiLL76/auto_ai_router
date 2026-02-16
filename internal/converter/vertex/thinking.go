package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// mapReasoningToThinkingConfig maps OpenAI reasoning params to Vertex ThinkingConfig.
// Checks Anthropic-style thinking first, then falls back to reasoning_effort.
func mapReasoningToThinkingConfig(thinking interface{}, reasoningEffort string, model string) *genai.ThinkingConfig {
	if thinking != nil {
		if thinkingMap, ok := thinking.(map[string]interface{}); ok {
			return mapAnthropicThinking(thinkingMap, model)
		}
	}
	if reasoningEffort != "" {
		return mapReasoningEffort(reasoningEffort, model)
	}
	return nil
}

// isGemini3Model returns true for Gemini 3+ models (use ThinkingLevel API)
func isGemini3Model(model string) bool {
	return strings.Contains(strings.ToLower(model), "gemini-3")
}

// isFlashModel returns true for flash variants
func isFlashModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "flash")
}

// mapReasoningEffort maps OpenAI reasoning_effort to Vertex ThinkingConfig.
// Gemini 2.5 uses ThinkingBudget (tokens), Gemini 3+ uses ThinkingLevel (enum).
func mapReasoningEffort(effort string, model string) *genai.ThinkingConfig {
	config := &genai.ThinkingConfig{IncludeThoughts: true}

	if isGemini3Model(model) {
		switch effort {
		case "minimal":
			if isFlashModel(model) {
				config.ThinkingLevel = genai.ThinkingLevelMinimal
			} else {
				config.ThinkingLevel = genai.ThinkingLevelLow
			}
		case "low":
			config.ThinkingLevel = genai.ThinkingLevelLow
		case "medium":
			if isFlashModel(model) {
				config.ThinkingLevel = genai.ThinkingLevelMedium
			} else {
				config.ThinkingLevel = genai.ThinkingLevelHigh
			}
		case "high":
			config.ThinkingLevel = genai.ThinkingLevelHigh
		case "disable", "none":
			config.IncludeThoughts = false
		}
	} else {
		// Gemini 2.5: ThinkingBudget (tokens)
		var budget int32
		switch effort {
		case "minimal":
			if strings.Contains(strings.ToLower(model), "2.5-flash-lite") {
				budget = 1000
			} else if strings.Contains(strings.ToLower(model), "2.5-pro") {
				budget = 5000
			} else {
				budget = 3000
			}
		case "low":
			budget = 5000
		case "medium":
			budget = 15000
		case "high":
			budget = 30000
		case "disable", "none":
			config.IncludeThoughts = false
			zero := int32(0)
			config.ThinkingBudget = &zero
			return config
		}
		config.ThinkingBudget = &budget
	}

	return config
}

// mapAnthropicThinking maps Anthropic-style thinking param to Vertex ThinkingConfig.
// Format: {"type": "enabled", "budget_tokens": 15000}
func mapAnthropicThinking(thinking map[string]interface{}, model string) *genai.ThinkingConfig {
	thinkingType, _ := thinking["type"].(string)
	budgetTokens, _ := thinking["budget_tokens"].(float64)

	config := &genai.ThinkingConfig{}
	if thinkingType != "enabled" || budgetTokens <= 0 {
		config.IncludeThoughts = false
		return config
	}

	config.IncludeThoughts = true

	if isGemini3Model(model) {
		switch {
		case budgetTokens >= 15000:
			config.ThinkingLevel = genai.ThinkingLevelHigh
		case budgetTokens >= 5000:
			config.ThinkingLevel = genai.ThinkingLevelLow
		default:
			config.ThinkingLevel = genai.ThinkingLevelMinimal
		}
	} else {
		budget := int32(budgetTokens)
		config.ThinkingBudget = &budget
	}

	return config
}
