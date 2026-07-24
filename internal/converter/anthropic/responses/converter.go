// Package anthropicresponses implements the ProviderResponses interface for Anthropic.
// It converts Responses API requests/responses directly to/from the Anthropic Messages
// API without going through Chat Completions format.
package anthropicresponses

import (
	"io"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

func init() {
	responses.RegisterProviderResponses(config.ProviderTypeAnthropic, func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
		return &AnthropicResponses{mode: mode}
	})
	responses.RegisterProviderResponses(config.ProviderTypeCometAPI, func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
		return &AnthropicResponses{mode: mode}
	})
	responses.RegisterProviderResponses(config.ProviderTypeProMan, func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
		return &AnthropicResponses{mode: mode}
	})
}

// AnthropicResponses converts between the Responses API and Anthropic Messages API.
type AnthropicResponses struct {
	mode responses.ResponsesRequestMode
}

// RequestFrom converts a Responses API request body to Anthropic Messages API JSON.
func (a *AnthropicResponses) RequestFrom(body []byte) ([]byte, string, error) {
	converted, err := ResponsesRequestToAnthropic(body, a.mode.ModelID)
	if err != nil {
		return nil, "", err
	}
	return converted, "application/json", nil
}

// ResponseTo converts an Anthropic Messages API response to a responses.Response.
func (a *AnthropicResponses) ResponseTo(body []byte, displayModelID string) (*responses.Response, error) {
	model := displayModelID
	if model == "" {
		model = a.mode.DisplayModel()
	}
	return AnthropicToResponsesResponse(body, model, "", 0)
}

// StreamTo reads an Anthropic SSE stream and writes Responses API SSE events.
func (a *AnthropicResponses) StreamTo(
	reader io.Reader,
	writer io.Writer,
	displayModelID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	model := displayModelID
	if model == "" {
		model = a.mode.DisplayModel()
	}
	return TransformAnthropicStreamToResponses(reader, writer, model, "", meta, onComplete)
}

// BuildURL constructs the Anthropic Messages API endpoint URL.
func (a *AnthropicResponses) BuildURL(cred *config.CredentialConfig) string {
	return converterutil.BuildVersionedURL(cred.BaseURL, "/v1/messages")
}
