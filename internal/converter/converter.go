package converter

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
)

// RequestMode holds context parameters for a conversion session.
type RequestMode struct {
	IsImageGeneration bool   // true for /images/generations requests
	IsStreaming       bool   // true for streaming (stream: true) requests
	ModelID           string // e.g. "gemini-2.0-flash", "claude-opus-4-5"
}

// ProviderConverter performs request/response conversion for a specific provider.
// Initialize with New() and use RequestFrom/ResponseTo/StreamTo methods.
type ProviderConverter struct {
	providerType config.ProviderType
	mode         RequestMode
}

// New creates a ProviderConverter for the given provider and request mode.
func New(providerType config.ProviderType, mode RequestMode) *ProviderConverter {
	return &ProviderConverter{
		providerType: providerType,
		mode:         mode,
	}
}

// RequestFrom converts an OpenAI-format request body to the provider-specific format.
// Returns the original body unchanged for OpenAI-compatible providers (passthrough).
func (c *ProviderConverter) RequestFrom(body []byte) ([]byte, error) {
	switch c.providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return vertex.OpenAIToVertex(body, c.mode.IsImageGeneration, c.mode.ModelID)
	case config.ProviderTypeAnthropic:
		// Anthropic does not support image generation
		if c.mode.IsImageGeneration {
			return nil, errors.New("anthropic does not support image generation")
		}
		return anthropic.OpenAIToAnthropic(body, c.mode.ModelID)
	default:
		// ProviderTypeOpenAI, ProviderTypeProxy, and others: pass through unchanged
		return body, nil
	}
}

// ResponseTo converts a provider-specific response body to OpenAI format.
// Returns the original body unchanged for OpenAI-compatible providers (passthrough).
func (c *ProviderConverter) ResponseTo(body []byte) ([]byte, error) {
	switch c.providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		if c.mode.IsImageGeneration {
			if strings.Contains(strings.ToLower(c.mode.ModelID), "gemini") {
				// Gemini image generation goes through chat API
				return vertex.VertexChatResponseToOpenAIImage(body)
			}
			// Imagen: native image generation endpoint
			return vertex.VertexImageToOpenAI(body)
		}
		return vertex.VertexToOpenAI(body, c.mode.ModelID)
	case config.ProviderTypeAnthropic:
		return anthropic.AnthropicToOpenAI(body, c.mode.ModelID)
	default:
		return body, nil
	}
}

// StreamTo transforms a provider SSE stream into OpenAI-compatible SSE format,
// writing the result to writer. For passthrough providers, bytes are copied directly.
func (c *ProviderConverter) StreamTo(reader io.Reader, writer io.Writer) error {
	switch c.providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return vertex.TransformVertexStreamToOpenAI(reader, c.mode.ModelID, writer)
	case config.ProviderTypeAnthropic:
		return anthropic.TransformAnthropicStreamToOpenAI(reader, c.mode.ModelID, writer)
	default:
		_, err := io.Copy(writer, reader)
		return err
	}
}

// BuildURL constructs the upstream target URL for this provider and credential.
// Returns empty string for providers where URL construction is handled externally.
func (c *ProviderConverter) BuildURL(cred *config.CredentialConfig) string {
	switch c.providerType {
	case config.ProviderTypeVertexAI:
		if c.mode.IsImageGeneration && !strings.Contains(strings.ToLower(c.mode.ModelID), "gemini") {
			return vertex.BuildVertexImageURL(cred, c.mode.ModelID)
		}
		return vertex.BuildVertexURL(cred, c.mode.ModelID, c.mode.IsStreaming)
	case config.ProviderTypeGemini:
		return vertex.BuildGeminiURL(cred, c.mode.ModelID, c.mode.IsStreaming)
	case config.ProviderTypeAnthropic:
		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		return baseURL + "/v1/messages"
	default:
		// OpenAI and Proxy: URL constructed by proxy based on cred.BaseURL + path
		return ""
	}
}

// IsPassthrough returns true if this provider requires no request/response transformation.
// Passthrough providers use the OpenAI wire format natively.
func (c *ProviderConverter) IsPassthrough() bool {
	switch c.providerType {
	case config.ProviderTypeOpenAI, config.ProviderTypeProxy:
		return true
	default:
		return false
	}
}

// UsageFromResponse extracts token usage from an OpenAI-format response body.
// Should be called after ResponseTo() so the body is always in OpenAI format.
func (c *ProviderConverter) UsageFromResponse(body []byte) *TokenUsage {
	return ExtractTokenUsage(body)
}

// AnthropicUsageToTokenUsage converts Anthropic-specific usage to universal TokenUsage.
// Convenience function for callers using the converter package.
func AnthropicUsageToTokenUsage(inputTokens, outputTokens, cacheReadTokens int) *TokenUsage {
	return &TokenUsage{
		PromptTokens:      inputTokens,
		CompletionTokens:  outputTokens,
		CachedInputTokens: cacheReadTokens,
	}
}

// ExtractTokenUsage parses token usage from an OpenAI-format JSON response body.
// Handles both chat completion format (prompt_tokens/completion_tokens)
// and image generation format (input_tokens/output_tokens).
// Returns nil if body cannot be parsed or contains no usage data.
func ExtractTokenUsage(body []byte) *TokenUsage {
	if len(body) == 0 {
		return nil
	}

	var resp struct {
		Usage struct {
			// Chat completion format
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens,omitempty"`
				AudioTokens  int `json:"audio_tokens,omitempty"`
				TextTokens   int `json:"text_tokens,omitempty"`
			} `json:"prompt_tokens_details,omitempty"`
			CompletionTokensDetails struct {
				AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
				RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
				AudioTokens              int `json:"audio_tokens,omitempty"`
				ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
			// Image generation format
			InputTokens        int `json:"input_tokens"`
			OutputTokens       int `json:"output_tokens"`
			InputTokensDetails struct {
				ImageTokens int `json:"image_tokens,omitempty"`
				TextTokens  int `json:"text_tokens,omitempty"`
			} `json:"input_tokens_details,omitempty"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	// Prefer chat completion tokens; fall back to image generation tokens
	promptTokens := resp.Usage.PromptTokens
	if promptTokens == 0 {
		promptTokens = resp.Usage.InputTokens
	}
	completionTokens := resp.Usage.CompletionTokens
	if completionTokens == 0 {
		completionTokens = resp.Usage.OutputTokens
	}

	if promptTokens == 0 && completionTokens == 0 {
		return nil
	}

	return &TokenUsage{
		PromptTokens:             promptTokens,
		CompletionTokens:         completionTokens,
		CachedInputTokens:        resp.Usage.PromptTokensDetails.CachedTokens,
		AudioInputTokens:         resp.Usage.PromptTokensDetails.AudioTokens,
		ImageTokens:              resp.Usage.InputTokensDetails.ImageTokens,
		AcceptedPredictionTokens: resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: resp.Usage.CompletionTokensDetails.RejectedPredictionTokens,
		AudioOutputTokens:        resp.Usage.CompletionTokensDetails.AudioTokens,
		ReasoningTokens:          resp.Usage.CompletionTokensDetails.ReasoningTokens,
	}
}
