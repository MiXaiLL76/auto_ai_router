package proxy

import (
	"encoding/json"
	"strconv"
	"strings"

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
	var request map[string]interface{}
	if json.Unmarshal(body, &request) != nil {
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
	case thinkingModeDisabled:
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
	if value, ok := fields["thinking_budget"]; ok && reasoningBudgetEnabled(value) {
		return true, prefix + "thinking_budget"
	}
	if value, ok := fields["thinking_level"]; ok && reasoningValueEnabled(value) {
		return true, prefix + "thinking_level"
	}
	if nested, ok := fields["thinking_config"].(map[string]interface{}); ok {
		if requested, source := mapReasoningSource(nested, prefix+"thinking_config."); requested {
			return true, source
		}
	}
	if nested, ok := fields["extra_body"].(map[string]interface{}); ok {
		if requested, source := mapReasoningSource(nested, prefix+"extra_body."); requested {
			return true, source
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
	case "", "0", "false", "none", "disable", "disabled", "off":
		return false
	default:
		return true
	}
}
