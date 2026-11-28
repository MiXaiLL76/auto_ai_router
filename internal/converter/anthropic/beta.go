// Package anthropic converts requests and responses between the router's internal form and the Anthropic API.
package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

// ExtractBetaHeader takes the anthropic_beta field out of a request body and
// returns its values for the anthropic-beta header. The body is scanned for
// the field, not decoded; a body without it is returned as it is.
func ExtractBetaHeader(body []byte) ([]byte, []string) {
	obj, ok := jsonscan.ParseObject(body)
	if !ok {
		return body, nil
	}
	request := obj.Fields
	raw, ok := request["anthropic_beta"]
	if !ok {
		return body, nil
	}

	var values []string
	if err := json.Unmarshal(raw, &values); err != nil {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return body, nil
		}
		values = []string{value}
	}

	seen := make(map[string]struct{}, len(values))
	betas := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		betas = append(betas, value)
	}

	delete(request, "anthropic_beta")
	return obj.Encode(), betas
}
