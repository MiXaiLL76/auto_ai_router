package proxy

import (
	"encoding/json"
	"strings"
)

// providerErrorMessage extracts a human-readable message from a provider's
// OpenAI-compatible error body.
//
// It is shared by the router-run tool adapters: the provider's own message
// is what the model is told, so it can react to it instead of the loop
// failing outright.
func providerErrorMessage(raw []byte) string {
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error.Message != "" {
		return envelope.Error.Message
	}
	if trimmed := strings.TrimSpace(string(raw)); trimmed != "" {
		return trimmed
	}
	return "the provider returned an error"
}
