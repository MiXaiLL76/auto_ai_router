// Package anthropic converts requests and responses between the router's internal form and the Anthropic API.
package anthropic

import (
	"encoding/json"
	"strings"
)

func ExtractBetaHeader(body []byte) ([]byte, []string) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return body, nil
	}
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
	cleaned, err := json.Marshal(request)
	if err != nil {
		return body, nil
	}
	return cleaned, betas
}
