package proxy

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

func (p *Proxy) reasoningOnlyExclusions(body []byte) map[string]bool {
	if requestUsesReasoning(body) {
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
	var request map[string]interface{}
	if json.Unmarshal(body, &request) != nil {
		return false
	}
	return mapUsesReasoning(request)
}

func mapUsesReasoning(fields map[string]interface{}) bool {
	if value, ok := fields["reasoning_effort"]; ok && reasoningValueEnabled(value) {
		return true
	}
	if value, ok := fields["reasoning"]; ok && reasoningValueEnabled(value) {
		return true
	}
	if value, ok := fields["thinking"]; ok && reasoningValueEnabled(value) {
		return true
	}
	if value, ok := fields["thinking_budget"]; ok && reasoningBudgetEnabled(value) {
		return true
	}
	if value, ok := fields["thinking_level"]; ok && reasoningValueEnabled(value) {
		return true
	}
	if nested, ok := fields["thinking_config"].(map[string]interface{}); ok && mapUsesReasoning(nested) {
		return true
	}
	if nested, ok := fields["extra_body"].(map[string]interface{}); ok && mapUsesReasoning(nested) {
		return true
	}
	return false
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
