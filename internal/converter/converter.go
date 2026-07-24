package converter

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	openaiconv "github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
)

// RequestMode holds context parameters for a conversion session.
type RequestMode struct {
	IsImageGeneration bool   // true for /images/generations requests
	IsImageEdit       bool   // true for /images/edits requests
	IsEmbeddings      bool   // true for /embeddings requests
	IsStreaming       bool   // true for streaming (stream: true) requests
	ModelID           string // real provider model name (URL construction, format detection)
	DisplayModelID    string // alias to echo in responses; falls back to ModelID when empty
	ContentType       string // original request content type (needed for multipart endpoints)
}

// responseModel returns the model name to embed in response/streaming output.
// Uses DisplayModelID (alias) when set, so the client sees the name it requested.
func (m RequestMode) responseModel() string {
	if m.DisplayModelID != "" {
		return m.DisplayModelID
	}
	return m.ModelID
}

// responseModelAliasWriter rewrites complete SSE lines before forwarding them.
// It buffers incomplete lines so model replacement still works when the JSON payload
// is split across multiple upstream writes.
type responseModelAliasWriter struct {
	w        io.Writer
	oldModel string
	newModel string
	pending  []byte
}

func (w *responseModelAliasWriter) Write(p []byte) (int, error) {
	w.pending = append(w.pending, p...)
	for {
		lineEnd := bytes.IndexByte(w.pending, '\n')
		if lineEnd < 0 {
			break
		}
		line := w.pending[:lineEnd+1]
		if _, err := w.w.Write(openaiconv.ReplaceModelInBody(line, w.oldModel, w.newModel)); err != nil {
			return 0, err
		}
		w.pending = w.pending[lineEnd+1:]
	}
	return len(p), nil
}

func (w *responseModelAliasWriter) Flush() error {
	if len(w.pending) == 0 {
		return nil
	}
	_, err := w.w.Write(openaiconv.ReplaceModelInBody(w.pending, w.oldModel, w.newModel))
	w.pending = nil
	return err
}

// ProviderConverter performs request/response conversion for a specific provider.
// Initialize with New() and use RequestFrom/ResponseTo/StreamTo methods.
type ProviderConverter struct {
	providerType config.ProviderType
	mode         RequestMode
	// inputTexts caches the original embedding request texts so that
	// GeminiEmbeddingToOpenAI can estimate prompt_tokens when the upstream
	// API (Gemini batchEmbedContents) does not return token statistics.
	inputTexts []string
	// rewrittenContentType is set by RequestFrom when a multipart body is rewritten
	// (e.g. for image edits).  Empty string means the original Content-Type is still valid.
	rewrittenContentType string
}

// isAnthropicBedrockModel returns true for Anthropic Claude models accessed via Bedrock
// (model IDs starting with "anthropic."). Other Bedrock-hosted models (GLM, Llama, etc.)
// use OpenAI-compatible format and must not be wrapped in Anthropic's API envelope.
func isAnthropicBedrockModel(modelID string) bool {
	return strings.HasPrefix(modelID, "anthropic.") || strings.Contains(modelID, ".anthropic.")
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
	// Handle embeddings requests
	if c.mode.IsEmbeddings {
		switch c.providerType {
		case config.ProviderTypeVertexAI:
			return vertex.OpenAIEmbeddingToVertex(body)
		case config.ProviderTypeGemini:
			// Cache input texts for token estimation in ResponseTo.
			if texts, err := vertex.ExtractEmbeddingTexts(body); err == nil {
				c.inputTexts = texts
			}
			return vertex.OpenAIEmbeddingToGemini(body, c.mode.ModelID)
		case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
			return nil, errors.New(string(c.providerType) + " does not support embeddings")
		case config.ProviderTypeBedrock:
			return nil, errors.New("bedrock does not support embeddings")
		default:
			return body, nil
		}
	}

	switch c.providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return vertex.OpenAIToVertex(body, c.mode.IsImageGeneration, c.mode.IsImageEdit, c.mode.ModelID, c.mode.ContentType)
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
		// Anthropic-compatible providers do not support image generation
		if c.mode.IsImageGeneration {
			return nil, errors.New(string(c.providerType) + " does not support image generation")
		}
		return anthropic.OpenAIToAnthropic(body, c.mode.ModelID)
	case config.ProviderTypeBedrock:
		if c.mode.IsImageGeneration {
			return nil, errors.New("bedrock does not support image generation")
		}
		if isAnthropicBedrockModel(c.mode.ModelID) {
			return anthropic.OpenAIToBedrock(body, c.mode.ModelID)
		}
		return body, nil
	default:
		// ProviderTypeOpenAI, ProviderTypeProxy, and others: convert non-function tools
		// (web_search, web_search_preview) to web_search_options, then pass through.
		body = openaiconv.ConvertWebSearchTools(body)

		// gpt-image-1 family does not support the response_format parameter in
		// /v1/images/generations — strip it before forwarding to avoid a 400.
		if c.mode.IsImageGeneration && openaiconv.IsGptImage1Model(c.mode.ModelID) {
			body = openaiconv.StripResponseFormat(body)
		}

		// /v1/images/edits uses multipart/form-data.  Rewrite the multipart to:
		//   1. Replace model aliases with the provider-facing model name.
		//   2. Fix image parts sent as application/octet-stream (detect real MIME from magic bytes).
		//   3. Strip the response_format field for gpt-image-1 (JSON stripping won't work on multipart).
		if c.mode.IsImageEdit && strings.Contains(strings.ToLower(c.mode.ContentType), "multipart/form-data") {
			stripRF := openaiconv.IsGptImage1Model(c.mode.ModelID)
			newBody, newCT := openaiconv.RewriteImageEditMultipart(body, c.mode.ContentType, c.mode.ModelID, stripRF)
			// Only replace when something actually changed (boundary or content differs).
			if newCT != c.mode.ContentType {
				c.rewrittenContentType = newCT
			}
			body = newBody
		}

		return body, nil
	}
}

// ResponseTo converts a provider-specific response body to OpenAI format.
// Returns the original body unchanged for OpenAI-compatible providers (passthrough).
func (c *ProviderConverter) ResponseTo(body []byte) ([]byte, error) {
	// Handle embeddings responses
	if c.mode.IsEmbeddings {
		switch c.providerType {
		case config.ProviderTypeVertexAI:
			return vertex.VertexEmbeddingToOpenAI(body, c.mode.ModelID)
		case config.ProviderTypeGemini:
			return vertex.GeminiEmbeddingToOpenAI(body, c.mode.ModelID, c.inputTexts)
		default:
			return body, nil
		}
	}

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
		return vertex.VertexToOpenAI(body, c.mode.responseModel())
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
		return anthropic.AnthropicToOpenAI(body, c.mode.responseModel())
	case config.ProviderTypeBedrock:
		if isAnthropicBedrockModel(c.mode.ModelID) {
			return anthropic.AnthropicToOpenAI(body, c.mode.responseModel())
		}
		// OpenAI-compatible passthrough; fix provider's real model ID to the client alias.
		if c.mode.DisplayModelID != "" && c.mode.DisplayModelID != c.mode.ModelID {
			return openaiconv.ReplaceModelInBody(body, c.mode.ModelID, c.mode.DisplayModelID), nil
		}
		return body, nil
	default:
		return body, nil
	}
}

// StreamTo transforms a provider SSE stream into OpenAI-compatible SSE format,
// writing the result to writer. For passthrough providers, bytes are copied directly.
func (c *ProviderConverter) StreamTo(reader io.Reader, writer io.Writer) error {
	switch c.providerType {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return vertex.TransformVertexStreamToOpenAI(reader, c.mode.responseModel(), writer)
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
		return anthropic.TransformAnthropicStreamToOpenAI(reader, c.mode.responseModel(), writer)
	case config.ProviderTypeBedrock:
		// Bedrock uses AWS Event Stream binary framing instead of SSE.
		// Decode to SSE first, then route by model family.
		pr, pw := io.Pipe()
		go func() {
			pw.CloseWithError(DecodeEventStreamToSSE(reader, pw))
		}()
		if isAnthropicBedrockModel(c.mode.ModelID) {
			return anthropic.TransformAnthropicStreamToOpenAI(pr, c.mode.responseModel(), writer)
		}
		// Non-Anthropic models on Bedrock (GLM, Llama, etc.) return OpenAI-compatible
		// SSE chunks after event stream decoding. Replace provider's real model ID with alias.
		if c.mode.DisplayModelID != "" && c.mode.DisplayModelID != c.mode.ModelID {
			aliasWriter := &responseModelAliasWriter{
				w:        writer,
				oldModel: c.mode.ModelID,
				newModel: c.mode.DisplayModelID,
			}
			if _, err := io.Copy(aliasWriter, pr); err != nil {
				return err
			}
			return aliasWriter.Flush()
		}
		_, err := io.Copy(writer, pr)
		return err
	default:
		_, err := io.Copy(writer, reader)
		return err
	}
}

// BuildURL constructs the upstream target URL for this provider and credential.
// Returns empty string for providers where URL construction is handled externally.
func (c *ProviderConverter) BuildURL(cred *config.CredentialConfig) string {
	// Handle embeddings URLs
	if c.mode.IsEmbeddings {
		switch c.providerType {
		case config.ProviderTypeVertexAI:
			return vertex.BuildVertexEmbeddingURL(cred, c.mode.ModelID)
		case config.ProviderTypeGemini:
			return vertex.BuildGeminiEmbeddingURL(cred, c.mode.ModelID)
		default:
			return ""
		}
	}

	switch c.providerType {
	case config.ProviderTypeVertexAI:
		if c.mode.IsImageGeneration && !strings.Contains(strings.ToLower(c.mode.ModelID), "gemini") {
			return vertex.BuildVertexImageURL(cred, c.mode.ModelID)
		}
		return vertex.BuildVertexURL(cred, c.mode.ModelID, c.mode.IsStreaming)
	case config.ProviderTypeGemini:
		if c.mode.IsImageGeneration && !strings.Contains(strings.ToLower(c.mode.ModelID), "gemini") {
			return vertex.BuildGeminiImageURL(cred, c.mode.ModelID)
		}
		return vertex.BuildGeminiURL(cred, c.mode.ModelID, c.mode.IsStreaming)
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI:
		return converterutil.BuildVersionedURL(cred.BaseURL, "/v1/messages")
	case config.ProviderTypeBedrock:
		baseURL := strings.TrimSuffix(cred.BaseURL, "/")
		if c.mode.IsStreaming {
			return baseURL + "/model/" + c.mode.ModelID + "/invoke-with-response-stream"
		}
		return baseURL + "/model/" + c.mode.ModelID + "/invoke"
	default:
		// OpenAI and Proxy: URL constructed by proxy based on cred.BaseURL + path
		return ""
	}
}

// RewrittenContentType returns the new Content-Type header value when RequestFrom rewrote
// a multipart body (e.g. for image edits with fixed MIME types or stripped fields).
// Returns an empty string when the original Content-Type is still valid.
func (c *ProviderConverter) RewrittenContentType() string {
	return c.rewrittenContentType
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

// ExtractTokenUsage parses token usage from an OpenAI-format JSON response body.
// Handles both chat completion format (prompt_tokens/completion_tokens)
// and image generation format (input_tokens/output_tokens).
// Returns nil if body cannot be parsed or contains no usage data.
func ExtractTokenUsage(body []byte) *TokenUsage {
	if len(body) == 0 {
		return nil
	}

	type responsesUsageDetails struct {
		InputTokens        int `json:"input_tokens"`
		OutputTokens       int `json:"output_tokens"`
		InputTokensDetails struct {
			CachedTokens        int `json:"cached_tokens,omitempty"`
			CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
			CacheWriteTokens    int `json:"cache_write_tokens,omitempty"`
			ImageTokens         int `json:"image_tokens,omitempty"`
			TextTokens          int `json:"text_tokens,omitempty"`
			AudioTokens         int `json:"audio_tokens,omitempty"`
		} `json:"input_tokens_details,omitempty"`
		OutputTokensDetails struct {
			AudioTokens     int `json:"audio_tokens,omitempty"`
			CachedTokens    int `json:"cached_tokens,omitempty"`
			ImageTokens     int `json:"image_tokens,omitempty"`
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
			TextTokens      int `json:"text_tokens,omitempty"`
		} `json:"output_tokens_details,omitempty"`
	}

	var resp struct {
		Usage struct {
			// Chat Completions format
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens        int `json:"cached_tokens,omitempty"`
				CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
				CacheWriteTokens    int `json:"cache_write_tokens,omitempty"`
				AudioTokens         int `json:"audio_tokens,omitempty"`
				ImageTokens         int `json:"image_tokens,omitempty"`
				TextTokens          int `json:"text_tokens,omitempty"`
			} `json:"prompt_tokens_details,omitempty"`
			CompletionTokensDetails struct {
				AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
				RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
				AudioTokens              int `json:"audio_tokens,omitempty"`
				CachedTokens             int `json:"cached_tokens,omitempty"`
				ImageTokens              int `json:"image_tokens,omitempty"`
				ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
				TextTokens               int `json:"text_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
			// Responses API / Image generation format (input_tokens/output_tokens)
			responsesUsageDetails
		} `json:"usage"`
		// Responses API streaming event format: {"type":"response.completed","response":{"usage":{...}}}
		Response struct {
			Usage *responsesUsageDetails `json:"usage,omitempty"`
		} `json:"response,omitempty"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	// Prefer Chat Completions tokens; fall back to Responses API / image tokens
	promptTokens := resp.Usage.PromptTokens
	if promptTokens == 0 {
		promptTokens = resp.Usage.InputTokens
	}
	completionTokens := resp.Usage.CompletionTokens
	if completionTokens == 0 {
		completionTokens = resp.Usage.OutputTokens
	}

	// Fall back to nested Responses API streaming event format (response.completed event)
	if promptTokens == 0 && completionTokens == 0 && resp.Response.Usage != nil {
		promptTokens = resp.Response.Usage.InputTokens
		completionTokens = resp.Response.Usage.OutputTokens
	}

	if promptTokens == 0 && completionTokens == 0 {
		return nil
	}

	// Merge detail fields: Chat Completions uses completion_tokens_details,
	// Responses API uses output_tokens_details. Pick whichever is populated.
	cachedTokens := resp.Usage.PromptTokensDetails.CachedTokens
	if cachedTokens == 0 {
		cachedTokens = resp.Usage.InputTokensDetails.CachedTokens
	}
	cacheCreationTokens := resp.Usage.PromptTokensDetails.CacheCreationTokens
	if cacheCreationTokens == 0 {
		cacheCreationTokens = resp.Usage.PromptTokensDetails.CacheWriteTokens
	}
	if cacheCreationTokens == 0 {
		cacheCreationTokens = resp.Usage.InputTokensDetails.CacheCreationTokens
	}
	if cacheCreationTokens == 0 {
		cacheCreationTokens = resp.Usage.InputTokensDetails.CacheWriteTokens
	}
	audioIn := resp.Usage.PromptTokensDetails.AudioTokens
	if audioIn == 0 {
		audioIn = resp.Usage.InputTokensDetails.AudioTokens
	}
	inputImageTokens := resp.Usage.PromptTokensDetails.ImageTokens
	if inputImageTokens == 0 {
		inputImageTokens = resp.Usage.InputTokensDetails.ImageTokens
	}
	audioOut := resp.Usage.CompletionTokensDetails.AudioTokens
	if audioOut == 0 {
		audioOut = resp.Usage.OutputTokensDetails.AudioTokens
	}
	reasoning := resp.Usage.CompletionTokensDetails.ReasoningTokens
	if reasoning == 0 {
		reasoning = resp.Usage.OutputTokensDetails.ReasoningTokens
	}
	outputImageTokens := resp.Usage.CompletionTokensDetails.ImageTokens
	if outputImageTokens == 0 {
		outputImageTokens = resp.Usage.OutputTokensDetails.ImageTokens
	}
	cachedOutputTokens := resp.Usage.CompletionTokensDetails.CachedTokens
	if cachedOutputTokens == 0 {
		cachedOutputTokens = resp.Usage.OutputTokensDetails.CachedTokens
	}

	// If tokens came from the nested response.completed event, use its detail fields
	if resp.Usage.PromptTokens == 0 && resp.Usage.InputTokens == 0 && resp.Response.Usage != nil {
		u := resp.Response.Usage
		if cachedTokens == 0 {
			cachedTokens = u.InputTokensDetails.CachedTokens
		}
		if cacheCreationTokens == 0 {
			cacheCreationTokens = u.InputTokensDetails.CacheCreationTokens
		}
		if cacheCreationTokens == 0 {
			cacheCreationTokens = u.InputTokensDetails.CacheWriteTokens
		}
		if audioIn == 0 {
			audioIn = u.InputTokensDetails.AudioTokens
		}
		if inputImageTokens == 0 {
			inputImageTokens = u.InputTokensDetails.ImageTokens
		}
		if audioOut == 0 {
			audioOut = u.OutputTokensDetails.AudioTokens
		}
		if reasoning == 0 {
			reasoning = u.OutputTokensDetails.ReasoningTokens
		}
		if inputImageTokens == 0 {
			inputImageTokens = u.InputTokensDetails.ImageTokens
		}
		if outputImageTokens == 0 {
			outputImageTokens = u.OutputTokensDetails.ImageTokens
		}
		if cachedOutputTokens == 0 {
			cachedOutputTokens = u.OutputTokensDetails.CachedTokens
		}
	}

	return &TokenUsage{
		PromptTokens:             promptTokens,
		CompletionTokens:         completionTokens,
		CachedInputTokens:        cachedTokens,
		CachedOutputTokens:       cachedOutputTokens,
		CacheCreationTokens:      cacheCreationTokens,
		AudioInputTokens:         audioIn,
		ImageTokens:              inputImageTokens,
		OutputImageTokens:        outputImageTokens,
		AcceptedPredictionTokens: resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: resp.Usage.CompletionTokensDetails.RejectedPredictionTokens,
		AudioOutputTokens:        audioOut,
		ReasoningTokens:          reasoning,
	}
}
