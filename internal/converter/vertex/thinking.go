package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// mapReasoningToThinkingConfig maps OpenAI reasoning params to Vertex ThinkingConfig.
// Checks Anthropic-style thinking first, then falls back to reasoning_effort.
// Image models get none, except those whose profile pins their thinking levels.
func mapReasoningToThinkingConfig(thinking interface{}, reasoningEffort string, model string) *genai.ThinkingConfig {
	if isImageModel(model) && !lookupGeminiModelProfile(model).hasThinkingLevels() {
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
// A model without thinking (Gemini 2.0, image models) gets nil: it has nothing to
// disable, and a zero budget there could be rejected.
func disableThinkingConfig(model string) *genai.ThinkingConfig {
	if !isThinkingCapableModel(model) {
		return nil
	}
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
// A profiled model gets the nearest level it supports; an unknown effort leaves the
// depth to the model (no level, or a dynamic budget).
func mapReasoningEffort(effort string, model string) *genai.ThinkingConfig {
	if profile := lookupGeminiModelProfile(model); profile.hasThinkingLevels() {
		return profile.thinkingConfig(effort, false)
	}
	name := thinkingName(effort)
	if name == "none" {
		return disableThinkingConfig(model)
	}
	config := &genai.ThinkingConfig{IncludeThoughts: false}
	if isGemini3Model(model) {
		config.ThinkingLevel, _ = gemini3ThinkingLevel(name, model)
	} else {
		budget := gemini25ThinkingBudget(name)
		config.ThinkingBudget = &budget
	}
	return config
}

// MapReasoningEffortToThinkingConfig maps a Responses API reasoning.effort to a Vertex
// ThinkingConfig exactly like the chat route's reasoning_effort.
// Exported for use by sub-packages (e.g. vertex/responses).
func MapReasoningEffortToThinkingConfig(effort, model string) *genai.ThinkingConfig {
	return mapReasoningToThinkingConfig(nil, effort, model)
}

// DefaultThinkingConfig returns the ThinkingConfig for a model when no explicit thinking
// params are requested: a profiled model's own default level, otherwise disabled
// autonomous reasoning (for predictable latency) on thinking-capable models, else nil.
// Exported for use by sub-packages (e.g. vertex/responses).
func DefaultThinkingConfig(model string) *genai.ThinkingConfig {
	if profile := lookupGeminiModelProfile(model); profile.hasThinkingLevels() {
		return profile.thinkingConfig("", false)
	}
	return disableThinkingConfig(model)
}

// mapNativeThinkingConfig maps Gemini-native thinking_config from extra_body to ThinkingConfig.
// Format: {"thinking_budget": 1024, "thinking_level": "medium", "include_thoughts": true}
// (camelCase keys too). A level-based model reads a lone budget as the level of the
// same depth, Gemini 2.5 a lone level as its budget; given both, the model's own kind
// wins. A config that asks for nothing — no level, no numeric budget, include_thoughts
// not true (an empty object, a budget that is not a number) — returns nil, so the
// caller goes on to the next source and, failing that, to the model default.
func mapNativeThinkingConfig(tcMap map[string]interface{}, model string) *genai.ThinkingConfig {
	includeThoughts, _ := thinkingConfigField(tcMap, "include_thoughts", "includeThoughts").(bool)
	levelName, _ := thinkingConfigField(tcMap, "thinking_level", "thinkingLevel").(string)
	levelName = thinkingName(levelName)
	budget, hasBudget := thinkingBudgetValue(thinkingConfigField(tcMap, "thinking_budget", "thinkingBudget"))
	if !includeThoughts && levelName == "" && !hasBudget {
		return nil
	}

	profile := lookupGeminiModelProfile(model)
	if profile.hasThinkingLevels() || isGemini3Model(model) {
		dynamic := false
		if levelName == "" && hasBudget {
			levelName = thinkingLevelForBudget(budget)
			dynamic = levelName == ""
		}
		if profile.hasThinkingLevels() {
			return profile.thinkingConfig(levelName, includeThoughts)
		}
		config := &genai.ThinkingConfig{IncludeThoughts: includeThoughts}
		switch {
		case dynamic:
			// A dynamic budget leaves the depth to the model.
		case levelName == "":
			config.ThinkingLevel = lowestThinkingLevel(model)
		default:
			level, ok := gemini3ThinkingLevel(levelName, model)
			if !ok {
				level = genai.ThinkingLevelLow
			}
			config.ThinkingLevel = level
		}
		return config
	}

	// Gemini 2.5: thinking_budget, or the requested level's budget on a thinking model.
	config := &genai.ThinkingConfig{IncludeThoughts: includeThoughts}
	switch {
	case hasBudget && budget == 0 && isGemini25ProModel(model):
		// 2.5-pro cannot disable thinking; use dynamic (-1) instead.
		dynamic := int32(-1)
		config.ThinkingBudget = &dynamic
	case hasBudget:
		v := int32(budget)
		config.ThinkingBudget = &v
	case levelName == "none" && isThinkingCapableModel(model):
		config.ThinkingBudget = disableThinkingConfig(model).ThinkingBudget
	case levelName != "" && isThinkingCapableModel(model):
		v := gemini25ThinkingBudget(levelName)
		config.ThinkingBudget = &v
	}
	if config.ThinkingBudget != nil && *config.ThinkingBudget == 0 {
		// include_thoughts requires thinking to be enabled.
		config.IncludeThoughts = false
	}
	return config
}

// thinkingConfigField reads a thinking_config field in snake_case, then camelCase.
func thinkingConfigField(tcMap map[string]interface{}, snake, camel string) interface{} {
	if value, ok := tcMap[snake]; ok {
		return value
	}
	return tcMap[camel]
}

// mapAnthropicThinking maps Anthropic-style thinking param to Vertex ThinkingConfig.
// Format: {"type": "enabled", "budget_tokens": 15000}
func mapAnthropicThinking(thinking map[string]interface{}, model string) *genai.ThinkingConfig {
	thinkingType, _ := thinking["type"].(string)
	budgetTokens, _ := thinking["budget_tokens"].(float64)

	if profile := lookupGeminiModelProfile(model); profile.hasThinkingLevels() {
		// "adaptive" leaves the depth to the model: its default level.
		switch {
		case thinkingType == "adaptive":
			return profile.thinkingConfig("", false)
		case thinkingType != "enabled" || budgetTokens <= 0:
			return profile.thinkingConfig("none", false)
		default:
			return profile.thinkingConfig(thinkingLevelForBudget(budgetTokens), false)
		}
	}

	if thinkingType != "enabled" || budgetTokens <= 0 {
		return disableThinkingConfig(model)
	}

	config := &genai.ThinkingConfig{}
	config.IncludeThoughts = false

	if isGemini3Model(model) {
		// Map Anthropic budget_tokens to the Gemini 3 ThinkingLevel of the same depth.
		config.ThinkingLevel, _ = gemini3ThinkingLevel(thinkingLevelForBudget(budgetTokens), model)
	} else {
		budget := int32(budgetTokens)
		config.ThinkingBudget = &budget
	}

	return config
}

// thinkingBudgetValue reads a numeric thinking budget.
func thinkingBudgetValue(raw interface{}) (float64, bool) {
	switch b := raw.(type) {
	case float64:
		return b, true
	case int32:
		return float64(b), true
	case int64:
		return float64(b), true
	case int:
		return float64(b), true
	default:
		return 0, false
	}
}

// thinkingLevelForBudget maps a token budget to the level name of the same depth:
// 0 is "none", a negative (dynamic) budget "" (the model's default).
func thinkingLevelForBudget(budget float64) string {
	switch {
	case budget < 0:
		return ""
	case budget == 0:
		return "none"
	case budget >= 15000:
		return "high"
	case budget >= 5000:
		return "medium"
	default:
		return "minimal"
	}
}

// thinkingName normalizes a level or effort name: case-insensitive, THINKING_LEVEL_
// prefix dropped, "disable" → "none", xhigh/max → "high".
func thinkingName(name string) string {
	name = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(name)), "thinking_level_")
	switch name {
	case "disable", "disabled":
		return "none"
	case "xhigh", "max":
		return "high"
	}
	return name
}

// thinkingLevelRank orders the normalized level names (see thinkingName).
var thinkingLevelRank = map[string]int{
	"minimal": 0,
	"low":     1,
	"medium":  2,
	"high":    3,
}

// gemini3ThinkingLevel maps a normalized level name to one the Gemini 3 model accepts:
// pro has no MEDIUM (→ HIGH), "none"/"minimal" get its floor. ok is false otherwise.
func gemini3ThinkingLevel(name, model string) (genai.ThinkingLevel, bool) {
	switch name {
	case "none", "minimal":
		return lowestThinkingLevel(model), true
	case "low":
		return genai.ThinkingLevelLow, true
	case "medium":
		if isFlashModel(model) {
			return genai.ThinkingLevelMedium, true
		}
		return genai.ThinkingLevelHigh, true
	case "high":
		return genai.ThinkingLevelHigh, true
	}
	return "", false
}

// gemini25ThinkingBudget returns the official Gemini 2.5 budget for a normalized level
// name; any other gets dynamic (-1), never the 0 that 2.5-pro rejects. "none" is the
// caller's (see disableThinkingConfig).
func gemini25ThinkingBudget(name string) int32 {
	switch name {
	case "minimal", "low":
		return 1024
	case "medium":
		return 8192
	case "high":
		return 24576
	}
	return -1
}
