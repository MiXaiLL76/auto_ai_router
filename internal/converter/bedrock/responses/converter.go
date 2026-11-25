// Package bedrockresponses implements the ProviderResponses interface for AWS Bedrock
// Anthropic models. It converts Responses API requests/responses directly to/from the
// Bedrock Anthropic format without going through Chat Completions as an intermediate.
package bedrockresponses

import (
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

func init() {
	responses.RegisterProviderResponsesForModel(
		config.ProviderTypeBedrock,
		func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
			return &BedrockResponses{mode: mode}
		},
		isAnthropicBedrockModel,
	)
}

// BedrockResponses converts between the Responses API and Bedrock Anthropic Messages format.
type BedrockResponses struct {
	mode responses.ResponsesRequestMode
}

// RequestFrom converts a Responses API request body to Bedrock Anthropic JSON.
func (b *BedrockResponses) RequestFrom(body []byte) ([]byte, string, error) {
	converted, err := ResponsesRequestToBedrock(body, b.mode.ModelID, b.mode.ImageBudget)
	if err != nil {
		return nil, "", err
	}
	return converted, "application/json", nil
}

// ResponseTo converts a Bedrock Anthropic response body to a responses.Response.
func (b *BedrockResponses) ResponseTo(body []byte, displayModelID string) (*responses.Response, error) {
	model := displayModelID
	if model == "" {
		model = b.mode.DisplayModel()
	}
	return BedrockToResponsesResponse(body, model)
}

// StreamTo reads a Bedrock EventStream binary response and writes Responses API SSE events.
func (b *BedrockResponses) StreamTo(
	reader io.Reader,
	writer io.Writer,
	displayModelID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	model := displayModelID
	if model == "" {
		model = b.mode.DisplayModel()
	}
	return TransformBedrockStreamToResponses(reader, writer, model, meta, onComplete)
}

// BuildURL constructs the Bedrock invoke endpoint URL.
// Model ID goes in the URL path; streaming uses invoke-with-response-stream.
func (b *BedrockResponses) BuildURL(cred *config.CredentialConfig) string {
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	if b.mode.IsStreaming {
		return baseURL + "/model/" + b.mode.ModelID + "/invoke-with-response-stream"
	}
	return baseURL + "/model/" + b.mode.ModelID + "/invoke"
}
