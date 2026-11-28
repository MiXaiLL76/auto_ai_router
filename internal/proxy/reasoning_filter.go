package proxy

import (
	"bytes"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

const (
	thinkingModeAdaptive    = "adaptive"
	thinkingModeDisabled    = "disabled"
	thinkingModeEnabled     = "enabled"
	thinkingModeUnknown     = "unknown"
	thinkingModeUnspecified = "unspecified"
)

func (p *Proxy) reasoningOnlyExclusions(reasoningRequested bool) map[string]bool {
	if reasoningRequested {
		return nil
	}

	excluded := make(map[string]bool)
	for _, cred := range p.balancer.GetCredentialsSnapshot() {
		if cred.ReasoningOnly {
			excluded[cred.Name] = true
			monitoring.CredentialSelectionRejected.WithLabelValues("reasoning_required").Inc()
		}
	}
	return excluded
}

func requestUsesReasoning(body []byte) bool {
	requested, _ := requestReasoning(body)
	return requested
}

func requestReasoning(body []byte) (bool, string) {
	requested, source, _ := requestReasoningDetails(body)
	return requested, source
}

func requestReasoningDetails(body []byte) (bool, string, string) {
	// Every key read below contains "reasoning" or "thinking". A key can also
	// be spelled with escaped ASCII characters, so their absence rules a
	// request out only when the body has no such escape either.
	if !bytes.Contains(body, []byte("reasoning")) &&
		!bytes.Contains(body, []byte("thinking")) &&
		!jsonscan.HasEscapedASCII(body) {
		return false, "", thinkingModeUnspecified
	}
	top, _ := jsonscan.ScanTop(body)
	return requestReasoningDetailsTop(top)
}

// reasoningTopLevelKeys are the only top-level keys mapReasoningSource and
// mapThinkingMode read; everything else they look at is nested under these.
var reasoningTopLevelKeys = []string{
	"reasoning_effort", "reasoning", "thinking",
	"thinking_budget", "thinkingBudget", "thinking_level", "thinkingLevel",
	"thinking_config", "thinkingConfig", "extra_body",
}

// requestReasoningDetailsTop is requestReasoningDetails for a body that has
// already been scanned (nil for a body that is not a JSON object). Only the
// reasoning fields are decoded, never the conversation.
func requestReasoningDetailsTop(top *jsonscan.Top) (bool, string, string) {
	if top == nil {
		return false, "", thinkingModeUnspecified
	}
	request, ok := top.Project(reasoningTopLevelKeys...)
	if !ok {
		return false, "", thinkingModeUnspecified
	}
	requested, source := mapReasoningSource(request, "")
	return requested, source, mapThinkingMode(request)
}

func mapThinkingMode(fields map[string]interface{}) string {
	if value, ok := fields["thinking"]; ok {
		return normalizeThinkingMode(value)
	}
	if nested, ok := fields["extra_body"].(map[string]interface{}); ok {
		return mapThinkingMode(nested)
	}
	return thinkingModeUnspecified
}

func normalizeThinkingMode(value interface{}) string {
	if fields, ok := value.(map[string]interface{}); ok {
		value, ok = fields["type"]
		if !ok {
			return thinkingModeUnknown
		}
	}
	kind, ok := value.(string)
	if !ok {
		return thinkingModeUnknown
	}
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case thinkingModeAdaptive:
		return thinkingModeAdaptive
	case thinkingModeDisabled, "between_tools": // Sonnet 5.5's "disabled"
		return thinkingModeDisabled
	case thinkingModeEnabled:
		return thinkingModeEnabled
	default:
		return thinkingModeUnknown
	}
}

func mapReasoningSource(fields map[string]interface{}, prefix string) (bool, string) {
	if value, ok := fields["reasoning_effort"]; ok && reasoningValueEnabled(value) {
		return true, prefix + "reasoning_effort"
	}
	if value, ok := fields["reasoning"]; ok && reasoningValueEnabled(value) {
		return true, prefix + "reasoning"
	}
	if value, ok := fields["thinking"]; ok && reasoningValueEnabled(value) {
		return true, prefix + "thinking"
	}
	// Gemini's camelCase spellings are accepted by the Gemini converter too.
	for _, key := range []string{"thinking_budget", "thinkingBudget"} {
		if value, ok := fields[key]; ok && reasoningBudgetEnabled(value) {
			return true, prefix + key
		}
	}
	for _, key := range []string{"thinking_level", "thinkingLevel"} {
		if value, ok := fields[key]; ok && reasoningValueEnabled(value) {
			return true, prefix + key
		}
	}
	if requested, source := thinkingConfigReasoningSource(fields, prefix); requested {
		return true, source
	}
	if nested, ok := fields["extra_body"].(map[string]interface{}); ok {
		if requested, source := mapReasoningSource(nested, prefix+"extra_body."); requested {
			return true, source
		}
		// The Gemini converter also reads extra_body.generation_config.thinking_config.
		if gc, ok := nested["generation_config"].(map[string]interface{}); ok {
			if requested, source := thinkingConfigReasoningSource(gc, prefix+"extra_body.generation_config."); requested {
				return true, source
			}
		}
	}
	return false, ""
}

// thinkingConfigReasoningSource checks fields' thinking_config / thinkingConfig.
func thinkingConfigReasoningSource(fields map[string]interface{}, prefix string) (bool, string) {
	for _, key := range []string{"thinking_config", "thinkingConfig"} {
		if nested, ok := fields[key].(map[string]interface{}); ok {
			if requested, source := mapReasoningSource(nested, prefix+key+"."); requested {
				return true, source
			}
		}
	}
	return false, ""
}

func reasoningValueEnabled(value interface{}) bool {
	switch typed := value.(type) {
	case string:
		return reasoningStringEnabled(typed)
	case bool:
		return typed
	case float64:
		return typed != 0
	case map[string]interface{}:
		if effort, ok := typed["effort"]; ok {
			if effort != nil {
				return reasoningValueEnabled(effort)
			}
		}
		if kind, ok := typed["type"]; ok {
			if kind != nil {
				return reasoningValueEnabled(kind)
			}
		}
		if budget, ok := typed["budget_tokens"]; ok {
			return reasoningBudgetEnabled(budget)
		}
		if budget, ok := typed["thinking_budget"]; ok {
			return reasoningBudgetEnabled(budget)
		}
		if level, ok := typed["thinking_level"]; ok {
			return reasoningValueEnabled(level)
		}
		return len(typed) > 0
	default:
		return false
	}
}

func reasoningBudgetEnabled(value interface{}) bool {
	switch typed := value.(type) {
	case float64:
		return typed != 0
	case string:
		budget, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return err == nil && budget != 0
	default:
		return false
	}
}

func reasoningStringEnabled(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "0", "false", "none", "disable", "disabled", "off", "between_tools":
		return false
	default:
		return true
	}
}
