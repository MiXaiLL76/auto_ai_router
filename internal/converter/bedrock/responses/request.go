package bedrockresponses

import (
	"encoding/json"
	"strings"

	anthropicresponses "github.com/mixaill76/auto_ai_router/internal/converter/anthropic/responses"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// ResponsesRequestToBedrock converts a Responses API request body to Bedrock Anthropic format.
// It reuses the Anthropic Responses converter then applies Bedrock-specific adjustments:
//   - Removes "model" (supplied in the URL path instead)
//   - Removes "stream" (endpoint choice controls streaming)
//   - Adds "anthropic_version": "bedrock-2023-05-31"
func ResponsesRequestToBedrock(body []byte, model string, budget *converterutil.ImageFetchBudget) ([]byte, error) {
	anthropicBody, err := anthropicresponses.ResponsesRequestToAnthropic(body, model, budget)
	if err != nil {
		return nil, err
	}

	var req map[string]interface{}
	if err := json.Unmarshal(anthropicBody, &req); err != nil {
		return nil, err
	}

	delete(req, "model")
	delete(req, "stream")
	req["anthropic_version"] = "bedrock-2023-05-31"

	return json.Marshal(req)
}

// isAnthropicBedrockModel returns true for Anthropic Claude models on Bedrock.
// These use the Anthropic Messages format; other Bedrock models (Llama, Titan, etc.) do not.
func isAnthropicBedrockModel(modelID string) bool {
	return strings.HasPrefix(modelID, "anthropic.") || strings.Contains(modelID, ".anthropic.")
}
