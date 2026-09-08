package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// mapReasoningToThinkingConfig maps OpenAI reasoning params to Vertex ThinkingConfig.
// Checks Anthropic-style thinking first, then falls back to reasoning_effort.
func mapReasoningToThinkingConfig(thinking interface{}, reasoningEffort string, model string) *genai.ThinkingConfig {
	if isGeminiImageModel(model) {
		return nil
	}
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

// modelsRejectingMinimalThinking lists Gemini 3 flash models that reject
// ThinkingLevel=MINIMAL with 400 INVALID_ARGUMENT ("Thinking level MINIMAL is not
// supported for this model" on the AI Studio endpoint, "Thinking level is
// unsupported: THINKING_LEVEL_MINIMAL" on Vertex). MINIMAL is the documented floor
// for Gemini 3 flash in general and is still accepted by the earlier flash variants,
// so this is a deny-list of the models observed to reject it rather than an
// allow-list — a new flash model keeps the cheaper MINIMAL default until it is
// proven to refuse it.
//
// Matched as substrings so that dated/suffixed ids (e.g. "gemini-3.7-flash-001")
// resolve to the same floor as the base name.
var modelsRejectingMinimalThinking = []string{
	"gemini-3.7-flash",
	"gemini-3.8-flash",
}

// lowestThinkingLevel returns the lowest ThinkingLevel the given Gemini 3 model
// actually accepts: MINIMAL for flash variants that support it, LOW otherwise
// (pro variants never supported MINIMAL, and some flash variants dropped it).
//
// Every "minimal thinking" decision routes through here so the floor is defined
// in one place instead of being re-derived at each call site.
func lowestThinkingLevel(model string) genai.ThinkingLevel {
	if !isFlashModel(model) {
		// Pro variants: MINIMAL is not supported, LOW is the minimum.
		return genai.ThinkingLevelLow
	}
	lower := strings.ToLower(model)
	for _, rejected := range modelsRejectingMinimalThinking {
		if strings.Contains(lower, rejected) {
			return genai.ThinkingLevelLow
		}
	}
	return genai.ThinkingLevelMinimal
}

// isGeminiImageModel returns true for Gemini image generation/editing models.
// These models do not support ThinkingConfig.
func isGeminiImageModel(model string) bool {
	lower := strings.ToLower(model)
	return strings.Contains(lower, "gemini") && strings.Contains(lower, "image")
}

// isThinkingCapableModel returns true for models that support dynamic thinking
// (Gemini 2.5+ and Gemini 3+). These models think autonomously when ThinkingConfig
// is not set, causing unpredictable latency.
func isThinkingCapableModel(model string) bool {
	if isGeminiImageModel(model) {
		return false
	}
	lower := strings.ToLower(model)
	return strings.Contains(lower, "gemini-2.5") || strings.Contains(lower, "gemini-3")
}

// isGemini25ProModel returns true for Gemini 2.5 Pro variants.
// These models require thinking to always be enabled (ThinkingBudget=0 is invalid).
func isGemini25ProModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "gemini-2.5-pro")
}

// disableThinkingConfig returns a ThinkingConfig that minimizes thinking computation.
//
// Gemini 2.5 flash: ThinkingBudget=0 (full disable supported).
// Gemini 2.5 pro: ThinkingBudget=-1 (dynamic) — budget=0 is not supported by this model,
//
//	so we use dynamic mode which lets the model decide the budget.
//
// Gemini 3: the lowest level the specific model accepts — see lowestThinkingLevel.
func disableThinkingConfig(model string) *genai.ThinkingConfig {
	if isGemini3Model(model) {
		return &genai.ThinkingConfig{
			IncludeThoughts: false,
			ThinkingLevel:   lowestThinkingLevel(model),
		}
	}
	// Gemini 2.5 pro: budget=0 is not supported; use dynamic (-1) instead.
	if isGemini25ProModel(model) {
		dynamic := int32(-1)
		return &genai.ThinkingConfig{
			IncludeThoughts: false,
			ThinkingBudget:  &dynamic,
		}
	}
	// Gemini 2.5 flash: budget=0 fully disables thinking.
	zero := int32(0)
	return &genai.ThinkingConfig{
		IncludeThoughts: false,
		ThinkingBudget:  &zero,
	}
}

// mapReasoningEffort maps OpenAI reasoning_effort to Vertex ThinkingConfig.
// Gemini 2.5 uses ThinkingBudget (tokens), Gemini 3+ uses ThinkingLevel (enum).
func mapReasoningEffort(effort string, model string) *genai.ThinkingConfig {
	config := &genai.ThinkingConfig{IncludeThoughts: false}

	if isGemini3Model(model) {
		// Gemini 3+: ThinkingLevel enum.
		// Flash supports LOW/MEDIUM/HIGH plus, on most variants, MINIMAL.
		// Pro supports LOW/HIGH only (MINIMAL and MEDIUM are unsupported).
		switch effort {
		case "minimal":
			config.ThinkingLevel = lowestThinkingLevel(model)
		case "low":
			config.ThinkingLevel = genai.ThinkingLevelLow
		case "medium":
			if isFlashModel(model) {
				config.ThinkingLevel = genai.ThinkingLevelMedium
			} else {
				// Pro variants don't support MEDIUM; use HIGH.
				config.ThinkingLevel = genai.ThinkingLevelHigh
			}
		case "high":
			config.ThinkingLevel = genai.ThinkingLevelHigh
		case "disable", "none":
			return disableThinkingConfig(model)
		}
	} else {
		// Gemini 2.5: ThinkingBudget (tokens).
		// Official budget values per reasoning_effort level:
		//   minimal/low → 1,024  |  medium → 8,192  |  high → 24,576
		var budget int32
		switch effort {
		case "minimal", "low":
			budget = 1024
		case "medium":
			budget = 8192
		case "high":
			budget = 24576
		case "disable", "none":
			return disableThinkingConfig(model)
		}
		config.ThinkingBudget = &budget
	}

	return config
}

// MapReasoningEffortToThinkingConfig maps a Responses API reasoning.effort to a Vertex
// ThinkingConfig. Exported for use by sub-packages (e.g. vertex/responses).
func MapReasoningEffortToThinkingConfig(effort, model string) *genai.ThinkingConfig {
	return mapReasoningEffort(effort, model)
}

// DefaultThinkingConfig returns the ThinkingConfig for a model when no explicit thinking
// params are requested. For thinking-capable models this disables autonomous reasoning
// for predictable latency; for other models it returns nil.
// Exported for use by sub-packages (e.g. vertex/responses).
func DefaultThinkingConfig(model string) *genai.ThinkingConfig {
	if isThinkingCapableModel(model) {
		return disableThinkingConfig(model)
	}
	return nil
}

// mapNativeThinkingConfig maps Gemini-native thinking_config from extra_body to ThinkingConfig.
// Format: {"thinking_budget": 1024, "thinking_level": "medium", "include_thoughts": true}
// thinking_budget is used for Gemini 2.5; thinking_level is used for Gemini 3+.
// If both are present, thinking_budget takes precedence for Gemini 2.5 and
// thinking_level takes precedence for Gemini 3+.
func mapNativeThinkingConfig(tcMap map[string]interface{}, model string) *genai.ThinkingConfig {
	config := &genai.ThinkingConfig{IncludeThoughts: false}

	if include, ok := tcMap["include_thoughts"].(bool); ok {
		config.IncludeThoughts = include
	}

	if isGemini3Model(model) {
		if levelStr, ok := tcMap["thinking_level"].(string); ok {
			switch levelStr {
			case "minimal":
				config.ThinkingLevel = lowestThinkingLevel(model)
			case "low":
				config.ThinkingLevel = genai.ThinkingLevelLow
			case "medium":
				if isFlashModel(model) {
					config.ThinkingLevel = genai.ThinkingLevelMedium
				} else {
					// Pro variants don't support MEDIUM; use HIGH.
					config.ThinkingLevel = genai.ThinkingLevelHigh
				}
			case "high":
				config.ThinkingLevel = genai.ThinkingLevelHigh
			default:
				config.ThinkingLevel = genai.ThinkingLevelLow
			}
		} else {
			// No thinking_level specified: use default for model type.
			config.ThinkingLevel = lowestThinkingLevel(model)
		}
	} else {
		// Gemini 2.5: use thinking_budget if provided.
		if budgetRaw, ok := tcMap["thinking_budget"]; ok {
			var budget float64
			budgetSet := false
			switch b := budgetRaw.(type) {
			case float64:
				budget, budgetSet = b, true
			case int32:
				budget, budgetSet = float64(b), true
			case int64:
				budget, budgetSet = float64(b), true
			case int:
				budget, budgetSet = float64(b), true
				// Non-numeric (e.g. string "high"/"auto") — skip.
			}
			if budgetSet {
				if budget == 0 && isGemini25ProModel(model) {
					// 2.5-pro cannot disable thinking; use dynamic (-1) instead.
					dynamic := int32(-1)
					config.ThinkingBudget = &dynamic
				} else {
					v := int32(budget)
					config.ThinkingBudget = &v
					if budget == 0 {
						// include_thoughts requires thinking to be enabled.
						// When disabling thinking (budget=0), force false.
						config.IncludeThoughts = false
					}
				}
			}
		}
	}

	return config
}

// mapAnthropicThinking maps Anthropic-style thinking param to Vertex ThinkingConfig.
// Format: {"type": "enabled", "budget_tokens": 15000}
func mapAnthropicThinking(thinking map[string]interface{}, model string) *genai.ThinkingConfig {
	thinkingType, _ := thinking["type"].(string)
	budgetTokens, _ := thinking["budget_tokens"].(float64)

	if thinkingType != "enabled" || budgetTokens <= 0 {
		return disableThinkingConfig(model)
	}

	config := &genai.ThinkingConfig{}
	config.IncludeThoughts = false

	if isGemini3Model(model) {
		// Map Anthropic budget_tokens to Gemini 3 ThinkingLevel.
		// Flash supports LOW/MEDIUM/HIGH plus, on most variants, MINIMAL.
		// Pro supports LOW/HIGH only (MEDIUM is unsupported on pro variants).
		switch {
		case budgetTokens >= 15000:
			config.ThinkingLevel = genai.ThinkingLevelHigh
		case budgetTokens >= 5000:
			if isFlashModel(model) {
				config.ThinkingLevel = genai.ThinkingLevelMedium
			} else {
				// Pro variants don't support MEDIUM; use HIGH.
				config.ThinkingLevel = genai.ThinkingLevelHigh
			}
		default:
			config.ThinkingLevel = lowestThinkingLevel(model)
		}
	} else {
		budget := int32(budgetTokens)
		config.ThinkingBudget = &budget
	}

	return config
}
