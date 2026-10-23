package anthropic

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

const minTextTokensWithThinking = 1024

// thinkingBindingBeta lets a request carry thinking.block_binding (without it the
// API rejects the field) and adds input_transformations to the response.
const thinkingBindingBeta = "thinking-binding-controls-2026-08-01"

var claudeVersionPattern = regexp.MustCompile(`claude-(opus|sonnet|haiku|fable)-([0-9]+)(?:[.-]([0-9]+))?`)

// claudeVersion parses the family and version out of a Claude model ID under any
// provider spelling (claude-sonnet-5-5, us.anthropic.claude-opus-4-7-v1:0,
// policy/claude-sonnet-5.5, claude-opus-5@20260101). minor is -1 when the ID has
// none: a trailing number longer than two digits is a date suffix. Mythos IDs carry
// no version and do not match.
func claudeVersion(model string) (family string, major, minor int, ok bool) {
	match := claudeVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return "", 0, 0, false
	}
	major, err := strconv.Atoi(match[2])
	if err != nil {
		return "", 0, 0, false
	}
	minor = -1
	if len(match[3]) > 0 && len(match[3]) <= 2 {
		minor, _ = strconv.Atoi(match[3])
	}
	return match[1], major, minor, true
}

// thinkingOffMode is how a model runs without extended thinking.
type thinkingOffMode int

const (
	// thinkingOffWhenOmitted: no thinking unless asked for; "disabled" is accepted too
	// (Claude 4.x and older, and any non-Claude model behind a gateway).
	thinkingOffWhenOmitted thinkingOffMode = iota
	// thinkingOffDisabled: thinks when thinking is omitted; "disabled" turns it off.
	thinkingOffDisabled
	// thinkingOffDisabledUpToHigh: as thinkingOffDisabled, at effort high or below
	// only (Claude Opus 5, Claude Haiku 5.5).
	thinkingOffDisabledUpToHigh
	// thinkingOffBetweenTools: rejects "disabled" and takes "between_tools", at effort
	// high or below only (Claude Sonnet 5.5).
	thinkingOffBetweenTools
	// thinkingAlwaysOn: thinking can't be turned off, "disabled" is a 400 at any effort
	// (Claude Opus 5.5, Fable, Mythos).
	thinkingAlwaysOn
)

func thinkingOffModeFor(model string) thinkingOffMode {
	if strings.Contains(strings.ToLower(model), "mythos") {
		return thinkingAlwaysOn
	}
	family, major, minor, ok := claudeVersion(model)
	switch {
	case !ok || major < 5:
		return thinkingOffWhenOmitted
	case family == "fable", family == "opus" && major == 5 && minor == 5:
		return thinkingAlwaysOn
	case family == "sonnet" && major == 5 && minor == 5:
		return thinkingOffBetweenTools
	case family == "opus" && major == 5 && minor <= 0, family == "haiku" && major == 5 && minor == 5:
		return thinkingOffDisabledUpToHigh
	default:
		return thinkingOffDisabled
	}
}

// thinkingOff is the explicit "no extended thinking" config for model at effort, or nil
// when the model can't run without thinking there (the request then omits thinking and
// the model runs adaptive): Opus 5.5, Fable and Mythos never can; Sonnet 5.5, Opus 5 and
// Haiku 5.5 not above effort high. Sonnet 5.5 rejects "disabled" and takes
// "between_tools" in its place.
func thinkingOff(model, effort string) *AnthropicThinking {
	aboveHigh := effort == "xhigh" || effort == "max"
	switch thinkingOffModeFor(model) {
	case thinkingAlwaysOn:
		return nil
	case thinkingOffBetweenTools:
		if aboveHigh {
			return nil
		}
		return &AnthropicThinking{Type: "between_tools"}
	case thinkingOffDisabledUpToHigh:
		if aboveHigh {
			return nil
		}
	}
	return &AnthropicThinking{Type: "disabled"}
}

// thinksByDefault reports whether model runs adaptive thinking when a request omits
// thinking (Claude 5 models), unlike Claude 4.x, where omitting it means no thinking.
func thinksByDefault(model string) bool {
	return thinkingOffModeFor(model) != thinkingOffWhenOmitted
}

// reasoningEffortOff reports whether an OpenAI reasoning effort asks for no reasoning.
func reasoningEffortOff(effort string) bool {
	switch effort {
	case "none", "disable", "disabled", "off":
		return true
	}
	return false
}

// mapThinkingConfig maps OpenAI thinking / reasoning_effort parameters to an Anthropic
// ThinkingConfig (and optional OutputConfig for Claude 4+ adaptive thinking).
//
// Priority:
//  1. Anthropic-style thinking param: {"type": "enabled"/"adaptive", ...}
//  2. OpenAI reasoning_effort string → mapped accordingly
//
// Claude 3.x: returns (*AnthropicThinking{Type:"enabled", BudgetTokens:N}, nil)
// Claude 4+:  returns (*AnthropicThinking{Type:"adaptive"}, *AnthropicOutputConfig{Effort:E}),
// with a nil OutputConfig when no effort was given, so the model default applies.
// Claude Sonnet 5.5 asked for no thinking: returns (*AnthropicThinking{Type:"between_tools"}, nil).
func mapThinkingConfig(thinking interface{}, reasoningEffort string, modelName string) (*AnthropicThinking, *AnthropicOutputConfig) {
	adaptive := isAdaptiveThinkingModel(modelName)
	betweenTools := thinkingOffModeFor(modelName) == thinkingOffBetweenTools

	if thinking != nil {
		if thinkingMap, ok := thinking.(map[string]interface{}); ok {
			thinkingType, _ := thinkingMap["type"].(string)
			display, _ := thinkingMap["display"].(string)
			budgetTokens, _ := thinkingMap["budget_tokens"].(float64)
			blockBinding := thinkingMap["block_binding"]
			switch thinkingType {
			case "enabled":
				if budgetTokens > 0 {
					if adaptive {
						// An effort the client set explicitly wins over the one derived from the budget.
						effort := mapReasoningEffortToEffort(reasoningEffort)
						if effort == "" {
							effort = budgetTokensToEffort(int(budgetTokens))
						}
						return &AnthropicThinking{Type: "adaptive", Display: "summarized", BlockBinding: blockBinding},
							&AnthropicOutputConfig{Effort: effort}
					}
					return &AnthropicThinking{Type: "enabled", BudgetTokens: int(budgetTokens), BlockBinding: blockBinding}, nil
				}
				// "enabled" without positive budget — fall through to reasoning_effort
			case "adaptive":
				effort, _ := thinkingMap["effort"].(string)
				if effort == "" {
					effort = mapReasoningEffortToEffort(reasoningEffort)
				}
				if adaptive {
					tc := &AnthropicThinking{Type: "adaptive", Display: display, BlockBinding: blockBinding}
					if effort == "" {
						return tc, nil
					}
					return tc, &AnthropicOutputConfig{Effort: effort}
				}
				// Caller passed adaptive but model uses legacy format — convert effort→budget
				if effort == "" {
					effort = "medium"
				}
				budget := effortToBudgetTokens(effort)
				if budget > 0 {
					return &AnthropicThinking{Type: "enabled", BudgetTokens: budget, BlockBinding: blockBinding}, nil
				}
			case "disabled", "between_tools":
				if !betweenTools {
					return nil, nil
				}
				// The other fields are kept so validateBetweenTools reports them.
				tc := &AnthropicThinking{Type: "between_tools", Display: display, BudgetTokens: int(budgetTokens), BlockBinding: blockBinding}
				if effort := mapReasoningEffortToEffort(reasoningEffort); effort != "" {
					return tc, &AnthropicOutputConfig{Effort: effort}
				}
				return tc, nil
			}
		}
	}

	if reasoningEffort != "" {
		if betweenTools && reasoningEffortOff(reasoningEffort) {
			return thinkingOff(modelName, ""), nil
		}
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
		newBetas = appendBetaUnique(newBetas, "effort-2025-11-24")
	}
	if tc.BlockBinding != nil {
		newBetas = appendBetaUnique(newBetas, thinkingBindingBeta)
	}
	// Anthropic requires temperature=1.0 when thinking is enabled.
	return newMaxTokens, newBetas, 1.0
}

// validateBetweenTools rejects what the API rejects with between_tools: any other
// thinking field, and effort above high. Upstream 400 bodies reach the client
// masked as "Invalid request" and are retried on other credentials, so the router
// answers these itself.
func validateBetweenTools(tc *AnthropicThinking, effort string) error {
	if tc == nil || tc.Type != "between_tools" {
		return nil
	}
	if tc.Display != "" || tc.BudgetTokens > 0 || tc.BlockBinding != nil {
		return converterutil.NewRequestValidationError("thinking",
			`thinking off on claude-sonnet-5.5 (thinking.type "between_tools", which "disabled" maps to) takes no other fields: remove display, budget_tokens and block_binding, or use thinking.type "adaptive"`)
	}
	if effort == "xhigh" || effort == "max" {
		return converterutil.NewRequestValidationError("output_config.effort",
			`thinking off on claude-sonnet-5.5 (thinking.type "disabled" or "between_tools") supports effort "low", "medium" or "high"; use thinking.type "adaptive" for "xhigh" or "max"`)
	}
	return nil
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
	if strings.Contains(strings.ToLower(model), "mythos") {
		return true
	}
	_, major, minor, ok := claudeVersion(model)
	return ok && (major >= 5 || major == 4 && minor >= 6)
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
