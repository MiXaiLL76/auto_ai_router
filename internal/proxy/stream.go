package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// streamChunkWriteTimeout is the per-chunk write deadline for streaming responses.
// If no data flows for this duration, the connection is terminated.
const streamChunkWriteTimeout = 60 * time.Second

// streamDrainTimeout caps how long we wait for the upstream to finish after a
// client disconnects. The provider charges for the full generation regardless,
// so we keep reading to capture the real usage chunk.
const streamDrainTimeout = 60 * time.Second

// streamTTFTDetectionLimit caps how many bytes streamToClient accumulates while
// looking for the first real content delta (for CompletionStartTime/ttft_ms).
// Streams that never produce a detectable content delta within this many bytes
// (error-only or keep-alive-only streams) stop being scanned; CompletionStartTime
// stays zero, matching existing "never reached" semantics.
const streamTTFTDetectionLimit = 64 * 1024

var streamBufPool = sync.Pool{
	New: func() any {
		buf := make([]byte, 8192)
		return &buf
	},
}

// StreamUsageInfo holds extracted usage information from streaming responses.
// It provides a unified structure for token counts across all providers.
// Not all fields will be populated; some providers don't report certain metrics.
type StreamUsageInfo struct {
	PromptTokens        int // May be 0 if not provided in streaming response
	CompletionTokens    int
	CachedTokens        int // Tokens from cached prompt content (prompt_caching feature)
	AudioInputTokens    int // Audio tokens in the request
	AudioOutputTokens   int // Audio tokens in the response
	ImageTokens         int // Input image/video tokens (if reported)
	OutputImageTokens   int // Generated image/video tokens (if reported)
	ReasoningTokens     int // Reasoning/thoughts tokens (output)
	CacheCreationTokens int // Tokens created for cache (billed at different rate)
	CacheReadTokens     int // Tokens read from cache (billed at cheaper rate)
}

// StreamUsageExtractor provides a provider-agnostic interface for extracting
// usage information from streaming response chunks.
// Each provider may use different JSON structures and field names,
// so implementations handle provider-specific parsing.
type StreamUsageExtractor interface {
	// ExtractUsage attempts to extract usage information from the given chunk.
	// Returns nil if the chunk doesn't contain usage information.
	// Errors are logged internally; the function never returns error.
	ExtractUsage(chunk []byte) *StreamUsageInfo
}

// openAIStreamUsageExtractor implements StreamUsageExtractor for OpenAI format
type openAIStreamUsageExtractor struct{}

func (o *openAIStreamUsageExtractor) ExtractUsage(chunk []byte) *StreamUsageInfo {
	// Supports two OpenAI streaming formats:
	//
	// 1. Chat Completions API:
	//    {"choices":[...],"usage":{"prompt_tokens":100,"completion_tokens":50,...}}
	//
	// 2. Responses API (GPT-5, /v1/responses):
	//    {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":50,...}}}
	//    Usage fields use input_tokens/output_tokens and output_tokens_details instead of
	//    prompt_tokens/completion_tokens and completion_tokens_details.

	payloads := extractJSONPayloadsFromStreamChunk(chunk)
	for i := len(payloads) - 1; i >= 0; i-- {
		if info := o.extractChatCompletionUsage(payloads[i]); info != nil {
			return info
		}
		if info := o.extractResponsesAPIUsage(payloads[i]); info != nil {
			return info
		}
	}

	return nil
}

// extractChatCompletionUsage parses usage from Chat Completions streaming format.
// Format: {"usage":{"prompt_tokens":N,"completion_tokens":N,...}}
func (o *openAIStreamUsageExtractor) extractChatCompletionUsage(payload []byte) *StreamUsageInfo {
	var data struct {
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens        int `json:"cached_tokens,omitempty"`
				CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
				CacheWriteTokens    int `json:"cache_write_tokens,omitempty"`
				AudioTokens         int `json:"audio_tokens,omitempty"`
				ImageTokens         int `json:"image_tokens,omitempty"`
			} `json:"prompt_tokens_details,omitempty"`
			CompletionTokensDetails struct {
				AudioTokens     int `json:"audio_tokens,omitempty"`
				ReasoningTokens int `json:"reasoning_tokens,omitempty"`
				ImageTokens     int `json:"image_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(payload, &data); err != nil {
		return nil
	}

	if data.Usage.PromptTokens == 0 && data.Usage.CompletionTokens == 0 {
		return nil
	}

	cacheCreationTokens := data.Usage.PromptTokensDetails.CacheCreationTokens
	if cacheCreationTokens == 0 {
		cacheCreationTokens = data.Usage.PromptTokensDetails.CacheWriteTokens
	}

	return &StreamUsageInfo{
		PromptTokens:        data.Usage.PromptTokens,
		CompletionTokens:    data.Usage.CompletionTokens,
		CachedTokens:        data.Usage.PromptTokensDetails.CachedTokens,
		CacheCreationTokens: cacheCreationTokens,
		AudioInputTokens:    data.Usage.PromptTokensDetails.AudioTokens,
		AudioOutputTokens:   data.Usage.CompletionTokensDetails.AudioTokens,
		ImageTokens:         data.Usage.PromptTokensDetails.ImageTokens,
		OutputImageTokens:   data.Usage.CompletionTokensDetails.ImageTokens,
		ReasoningTokens:     data.Usage.CompletionTokensDetails.ReasoningTokens,
	}
}

// extractResponsesAPIUsage parses usage from Responses API streaming format.
// The usage can appear at two levels:
//   - Top-level: {"usage":{"input_tokens":N,"output_tokens":N,...}}
//   - Nested in response.completed: {"type":"response.completed","response":{"usage":{...}}}
func (o *openAIStreamUsageExtractor) extractResponsesAPIUsage(payload []byte) *StreamUsageInfo {
	var data struct {
		// Top-level usage (some Responses API events)
		Usage *responsesAPIUsage `json:"usage,omitempty"`
		// Nested usage in response.completed event
		Response struct {
			Usage *responsesAPIUsage `json:"usage,omitempty"`
		} `json:"response,omitempty"`
	}

	if err := json.Unmarshal(payload, &data); err != nil {
		return nil
	}

	// Prefer nested response.usage (response.completed event), fall back to top-level
	usage := data.Response.Usage
	if usage == nil {
		usage = data.Usage
	}
	if usage == nil || (usage.InputTokens == 0 && usage.OutputTokens == 0) {
		return nil
	}

	cacheCreationTokens := usage.InputTokensDetails.CacheCreationTokens
	if cacheCreationTokens == 0 {
		cacheCreationTokens = usage.InputTokensDetails.CacheWriteTokens
	}

	return &StreamUsageInfo{
		PromptTokens:        usage.InputTokens,
		CompletionTokens:    usage.OutputTokens,
		CachedTokens:        usage.InputTokensDetails.CachedTokens,
		CacheCreationTokens: cacheCreationTokens,
		AudioInputTokens:    usage.InputTokensDetails.AudioTokens,
		AudioOutputTokens:   usage.OutputTokensDetails.AudioTokens,
		ImageTokens:         usage.InputTokensDetails.ImageTokens,
		OutputImageTokens:   usage.OutputTokensDetails.ImageTokens,
		ReasoningTokens:     usage.OutputTokensDetails.ReasoningTokens,
	}
}

// responsesAPIUsage represents the usage object in OpenAI Responses API format.
type responsesAPIUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens        int `json:"cached_tokens,omitempty"`
		CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
		CacheWriteTokens    int `json:"cache_write_tokens,omitempty"`
		AudioTokens         int `json:"audio_tokens,omitempty"`
		ImageTokens         int `json:"image_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails struct {
		AudioTokens     int `json:"audio_tokens,omitempty"`
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
		ImageTokens     int `json:"image_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
}

// anthropicStreamUsageExtractor implements StreamUsageExtractor for Anthropic format
type anthropicStreamUsageExtractor struct{}

func (a *anthropicStreamUsageExtractor) ExtractUsage(chunk []byte) *StreamUsageInfo {
	// Anthropic streaming format (message_delta event):
	// {"type":"message_delta","delta":{...},"usage":{"input_tokens":100,"output_tokens":50}}
	// Usage appears in the message_delta event at the end of streaming

	var data struct {
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
		} `json:"usage"`
	}

	payloads := extractJSONPayloadsFromStreamChunk(chunk)
	for i := len(payloads) - 1; i >= 0; i-- {
		if err := json.Unmarshal(payloads[i], &data); err != nil {
			continue
		}

		// Check if usage info is present
		if data.Usage.InputTokens == 0 && data.Usage.OutputTokens == 0 {
			continue
		}

		return &StreamUsageInfo{
			PromptTokens:        data.Usage.InputTokens,
			CompletionTokens:    data.Usage.OutputTokens,
			CacheCreationTokens: data.Usage.CacheCreationInputTokens,
			CacheReadTokens:     data.Usage.CacheReadInputTokens,
			// Anthropic separates cache_creation (cached prompt tokens)
			// For logging purposes, we combine under CachedTokens
			CachedTokens: data.Usage.CacheReadInputTokens,
		}
	}

	return nil
}

// extractJSONPayloadsFromStreamChunk extracts JSON payload candidates from raw stream chunks.
// Supports both plain JSON chunks and SSE-formatted chunks (lines prefixed with "data: ").
func extractJSONPayloadsFromStreamChunk(chunk []byte) [][]byte {
	trimmed := strings.TrimSpace(string(chunk))
	if trimmed == "" {
		return nil
	}

	// Fast path: non-SSE plain JSON
	if !strings.Contains(trimmed, "data:") {
		return [][]byte{[]byte(trimmed)}
	}

	lines := strings.Split(trimmed, "\n")
	payloads := make([][]byte, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		payloads = append(payloads, []byte(payload))
	}

	return payloads
}

// getStreamUsageExtractor returns the appropriate usage extractor for a provider.
// This factory method ensures all providers use the correct parsing logic.
// If the provider is unknown, defaults to OpenAI extractor (most compatible fallback).
func getStreamUsageExtractor(providerName string) StreamUsageExtractor {
	switch strings.ToLower(strings.TrimSpace(providerName)) {
	case "openai":
		return &openAIStreamUsageExtractor{}
	case "anthropic":
		// Anthropic streaming goes through handleTransformedStreaming which converts
		// chunks to OpenAI format, so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{}
	case "vertex ai":
		// Vertex AI transforms to OpenAI format during streaming,
		// so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{}
	case "bedrock":
		// Bedrock transforms to OpenAI format during streaming (via Anthropic converter),
		// so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{}
	default:
		// Fallback: try OpenAI format first (most common)
		return &openAIStreamUsageExtractor{}
	}
}

func IsStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json") ||
		strings.Contains(contentType, "application/vnd.amazon.eventstream")
}

type streamTransformer func(io.Reader, string, io.Writer) error

func (p *Proxy) handleProviderStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	cred *config.CredentialConfig,
	realModelID, displayModelID string,
	logCtx *RequestLogContext,
) error {
	switch cred.Type {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return p.handleVertexStreaming(w, resp, cred.Name, realModelID, displayModelID, logCtx)
	case config.ProviderTypeAnthropic:
		return p.handleAnthropicCompatibleStreaming(w, resp, cred.Name, realModelID, displayModelID, cred.Type, "Anthropic", logCtx)
	case config.ProviderTypeCometAPI:
		return p.handleAnthropicCompatibleStreaming(w, resp, cred.Name, realModelID, displayModelID, cred.Type, "Comet API", logCtx)
	case config.ProviderTypeBedrock:
		return p.handleBedrockStreaming(w, resp, cred.Name, realModelID, displayModelID, logCtx)
	default:
		return p.handleStreamingWithTokens(w, resp, cred.Name, displayModelID, logCtx)
	}
}

func (p *Proxy) handleVertexStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID, displayModelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeVertexAI, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Vertex AI", transformer, logCtx)
}

func (p *Proxy) handleAnthropicCompatibleStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID, displayModelID string, providerType config.ProviderType, providerLabel string, logCtx *RequestLogContext) error {
	conv := converter.New(providerType, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, providerLabel, transformer, logCtx)
}

func (p *Proxy) handleBedrockStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID, displayModelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeBedrock, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Bedrock", transformer, logCtx)
}

type tokenCapturingWriter struct {
	writer     io.Writer
	tokens     *int
	completion *completionTokenAccumulator
	logger     *slog.Logger
	onChunk    func([]byte) // Callback invoked for each chunk (optional, for capturing last chunk)
}

func (tcw *tokenCapturingWriter) Write(p []byte) (n int, err error) {
	// Extract tokens from the data being written.
	// Use assignment (not +=) because Vertex/Gemini include cumulative total_tokens in every
	// streaming chunk. Accumulating across chunks would multiply the real count by the number
	// of chunks (e.g. 50 chunks × 1000 tokens = 50 000 instead of 1 000).
	// OpenAI only emits total_tokens in the final usage chunk, so assignment is equivalent there.
	tokens := extractTokensFromStreamingChunk(string(p))
	if tokens > 0 {
		*tcw.tokens = tokens
	}

	if tcw.completion != nil {
		tcw.completion.AddChunk(p)
	}

	// Invoke callback if provided (used to capture last chunk for usage extraction)
	if tcw.onChunk != nil {
		tcw.onChunk(p)
	}

	return tcw.writer.Write(p)
}

// rememberLastStreamDataChunk stores each chunk, keeping only the last one that contains actual data and isn't [DONE].
func rememberLastStreamDataChunk(dst *[]byte, chunk []byte) {
	trimmed := strings.TrimSpace(string(chunk))
	if trimmed == "" || trimmed == "data: [DONE]" || trimmed == "[DONE]" {
		return
	}
	*dst = append([]byte(nil), chunk...)
}

func (p *Proxy) handleTransformedStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	credName string,
	modelID string,
	providerName string,
	transformFunc streamTransformer,
	logCtx *RequestLogContext,
) error {
	p.logger.DebugContext(respCtx(resp), "Starting streaming response", "provider", providerName, "credential", credName)

	pr, pw := io.Pipe()
	defer func() {
		_ = pr.Close()
	}()
	var totalTokens int
	completion := newCompletionTokenAccumulator(modelID)

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte

	// WaitGroup ensures the transform goroutine completes before we read
	// lastChunk and totalTokens, preventing a data race.
	var wg sync.WaitGroup
	wg.Add(1)
	chunkCount := 0
	go func() {
		defer wg.Done()
		err := transformFunc(resp.Body, modelID, &tokenCapturingWriter{
			writer:     pw,
			tokens:     &totalTokens,
			completion: completion,
			logger:     p.logger,
			onChunk: func(chunk []byte) {
				chunkCount++

				if logCtx != nil {
					if usage := extractTokenUsageFromStreamingChunk(string(chunk)); usage != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						*logCtx.TokenUsage = *usage
					}
				}

				// Store each chunk, keeping only the last one that contains actual data and isn't [DONE]
				rememberLastStreamDataChunk(&lastChunk, chunk)
			},
		})
		if err != nil {
			p.logger.ErrorContext(respCtx(resp), "Transform goroutine error",
				"provider", providerName, "error", err, "chunks_written", chunkCount)
			_ = pw.CloseWithError(fmt.Errorf("%s transform: %w", providerName, err))
		} else {
			p.logger.DebugContext(respCtx(resp), "Transform goroutine completed OK",
				"provider", providerName, "chunks_written", chunkCount, "total_tokens", totalTokens)
			_ = pw.Close()
		}
	}()

	if err := p.streamToClient(respCtx(resp), w, pr, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), nil, func() { _ = pr.Close() }, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handleTransformedStreaming", err,
			"credential", credName, "provider", providerName, "model", modelID)
		wg.Wait()
		// Stream aborted before the final usage chunk — log with whatever tokens we have.
		// Fall back to local token counting when totalTokens is still 0.
		estimated := totalTokens
		if estimated == 0 {
			estimated = completion.TokenCount()
		}
		p.finalizeStreamingLog(logCtx, estimated, lastChunk, providerName, resp.StatusCode)
		return err
	}
	wg.Wait()

	p.logger.DebugContext(respCtx(resp), "handleTransformedStreaming completed",
		"provider", providerName, "total_tokens", totalTokens,
		"chunks_written", chunkCount, "last_chunk_len", len(lastChunk))

	// When no usage chunk arrived (provider disconnected without sending one),
	// fall back to local token counting from accumulated delta text.
	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
		p.logger.Debug("No usage chunk received; counted completion tokens from delta text",
			"tokens", logTokens, "provider", providerName, "model", modelID)
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, logTokens, lastChunk, providerName, resp.StatusCode)

	p.logger.DebugContext(respCtx(resp), "Streaming response completed", "provider", providerName, "credential", credName)
	return nil
}

func (p *Proxy) handleStreamingWithTokens(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	p.logger.DebugContext(respCtx(resp), "Starting streaming response with token tracking (passthrough)",
		"credential", credName, "model", modelID,
		"content_type", resp.Header.Get("Content-Type"))

	var totalTokens int
	completion := newCompletionTokenAccumulator(modelID)
	chunkCount := 0

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte

	onChunk := func(chunk []byte) {
		chunkCount++

		if logCtx != nil {
			if usage := extractTokenUsageFromStreamingChunk(string(chunk)); usage != nil {
				if logCtx.TokenUsage == nil {
					logCtx.TokenUsage = &converter.TokenUsage{}
				}
				*logCtx.TokenUsage = *usage
			}
		}

		tokens := extractTokensFromStreamingChunk(string(chunk))
		if tokens > 0 {
			totalTokens += tokens
		}
		completion.AddChunk(chunk)

		// Don't let a bare [DONE] sentinel or empty chunks overwrite a lastChunk that carries usage data.
		rememberLastStreamDataChunk(&lastChunk, chunk)
	}

	if err := p.streamToClient(respCtx(resp), w, resp.Body, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), onChunk, nil, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handleStreamingWithTokens", err,
			"credential", credName, "model", modelID, "chunks_received", chunkCount)
		if p.drainUpstreamOnAbort {
			// Keep reading upstream to capture the real usage chunk.
			// The provider charges for the full generation regardless of client disconnect.
			drainCtx, cancel := context.WithTimeout(context.Background(), streamDrainTimeout)
			defer cancel()
			p.drainUpstream(drainCtx, resp.Body, onChunk, credName)
		}
		estimated := totalTokens
		if estimated == 0 {
			estimated = completion.TokenCount()
		}
		p.finalizeStreamingLog(logCtx, estimated, lastChunk, "openai", resp.StatusCode)
		return err
	}

	p.logger.DebugContext(respCtx(resp), "handleStreamingWithTokens completed",
		"credential", credName, "model", modelID,
		"chunks_received", chunkCount, "total_tokens", totalTokens,
		"last_chunk_len", len(lastChunk))

	// When no usage chunk arrived (provider disconnected without sending one),
	// fall back to local token counting from accumulated delta text.
	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
		p.logger.Debug("No usage chunk received; counted completion tokens from delta text",
			"tokens", logTokens, "credential", credName, "model", modelID)
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, logTokens, lastChunk, "openai", resp.StatusCode)

	p.logger.DebugContext(respCtx(resp), "Streaming response completed", "credential", credName)
	return nil
}

// finalizeStreamingLog extracts usage info from the last streaming chunk and logs spend to LiteLLM DB.
func (p *Proxy) finalizeStreamingLog(logCtx *RequestLogContext, totalTokens int, lastChunk []byte, providerName string, statusCode int) {
	if logCtx == nil || logCtx.Logged {
		return
	}

	if logCtx.TokenUsage == nil {
		logCtx.TokenUsage = &converter.TokenUsage{}
	}

	fallbackPrompt := logCtx.PromptTokensEstimate
	fallbackCompletion := totalTokens

	if len(lastChunk) > 0 {
		extractor := getStreamUsageExtractor(providerName)
		if usageInfo := extractor.ExtractUsage(lastChunk); usageInfo != nil {
			if usageInfo.PromptTokens > 0 {
				logCtx.TokenUsage.PromptTokens = usageInfo.PromptTokens
			}
			if usageInfo.CompletionTokens > 0 {
				logCtx.TokenUsage.CompletionTokens = usageInfo.CompletionTokens
			}

			if usageInfo.CachedTokens > 0 {
				logCtx.TokenUsage.CachedInputTokens = usageInfo.CachedTokens
			}
			if usageInfo.AudioInputTokens > 0 {
				logCtx.TokenUsage.AudioInputTokens = usageInfo.AudioInputTokens
			}
			if usageInfo.AudioOutputTokens > 0 {
				logCtx.TokenUsage.AudioOutputTokens = usageInfo.AudioOutputTokens
			}
			if usageInfo.ImageTokens > 0 {
				logCtx.TokenUsage.ImageTokens = usageInfo.ImageTokens
			}
			if usageInfo.OutputImageTokens > 0 {
				logCtx.TokenUsage.OutputImageTokens = usageInfo.OutputImageTokens
			}
			if usageInfo.ReasoningTokens > 0 {
				logCtx.TokenUsage.ReasoningTokens = usageInfo.ReasoningTokens
			}

			if usageInfo.CacheCreationTokens > 0 {
				logCtx.TokenUsage.CacheCreationTokens = usageInfo.CacheCreationTokens
			}

			p.logger.DebugContext(logCtx.Context(), "Extracted usage from streaming response",
				"provider", providerName,
				"prompt_tokens", usageInfo.PromptTokens,
				"completion_tokens", usageInfo.CompletionTokens,
				"cached_tokens", usageInfo.CachedTokens,
				"audio_input_tokens", usageInfo.AudioInputTokens,
				"audio_output_tokens", usageInfo.AudioOutputTokens,
				"image_tokens", usageInfo.ImageTokens,
				"output_image_tokens", usageInfo.OutputImageTokens,
				"reasoning_tokens", usageInfo.ReasoningTokens,
			)
		}
	}

	if logCtx.TokenUsage.PromptTokens == 0 {
		logCtx.TokenUsage.PromptTokens = fallbackPrompt
	}
	if logCtx.TokenUsage.CompletionTokens == 0 {
		logCtx.TokenUsage.CompletionTokens = fallbackCompletion
	}

	logCtx.HTTPStatus = statusCode
	if statusCode >= 400 {
		logCtx.Status = "failure"
		if logCtx.ErrorMsg == "" {
			logCtx.ErrorMsg = extractErrorMessage(lastChunk)
		}
	} else if streamErr := extractStreamErrorEvent(lastChunk); streamErr != "" {
		// Provider returned HTTP 2xx but sent an error event inside the stream
		// (e.g. `data: {"error":...}`, `event: error`, response.failed). Without
		// this check such requests are logged as success and never hit ERROR.
		logCtx.Status = "failure"
		logCtx.ErrorMsg = streamErr
		p.logUpstreamError(logCtx.Context(), "Provider sent error event in stream", statusCode,
			logCtx.Credential, logCtx.ModelID, []byte(streamErr),
			"request_id", logCtx.RequestID)
	} else {
		logCtx.Status = "success"
		if logCtx.Credential != nil {
			p.metrics.RecordTokenUsage(logCtx.Credential.Name, logCtx.ModelID,
				logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
				logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
		}
	}
	logCtx.Logged = true
	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue streaming spend log",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
}

// extractStreamErrorEvent scans an SSE chunk (or plain JSON payload) for an
// error event sent by the provider inside an HTTP 2xx stream. It returns the
// error payload as a string, or "" when the chunk carries no error.
// Recognized shapes:
//   - data: {"error": {...}}              (OpenAI-style mid-stream error)
//   - data: {"type":"error", ...}         (Anthropic `event: error`)
//   - data: {"type":"response.failed"...} (Responses API failed event)
//   - bare JSON object with the same shapes (non-SSE final chunk)
func extractStreamErrorEvent(chunk []byte) string {
	if len(chunk) == 0 {
		return ""
	}

	checkPayload := func(payload string) string {
		var evt struct {
			Type  string          `json:"type"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			return ""
		}
		if evt.Type == "error" || evt.Type == "response.error" || evt.Type == "response.failed" {
			return payload
		}
		if len(evt.Error) > 0 && string(evt.Error) != "null" {
			return payload
		}
		return ""
	}

	found := ""
	sawDataLine := false
	for _, line := range strings.Split(string(chunk), "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		sawDataLine = true
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		if errPayload := checkPayload(payload); errPayload != "" {
			found = errPayload
		}
	}
	if found != "" {
		return found
	}

	// Non-SSE chunk: try the whole payload as a single JSON object.
	if !sawDataLine {
		return checkPayload(strings.TrimSpace(string(chunk)))
	}
	return ""
}

// drainUpstream reads from body until EOF, ctx expiry, or an error, calling
// onChunk for every read. Invoked after a client disconnects so that the usage
// chunk emitted by the provider at the end of the stream is captured for
// accurate spend logging — the provider charges for the full generation
// regardless of whether the client waited for it.
// Must be called from the same goroutine that already owns body (no concurrent reads).
func (p *Proxy) drainUpstream(ctx context.Context, body io.Reader, onChunk func([]byte), credName string) {
	buf := make([]byte, 4096)
	drained := 0
	for {
		select {
		case <-ctx.Done():
			p.logger.Debug("Upstream drain timeout", "credential", credName, "bytes_drained", drained)
			return
		default:
		}
		n, err := body.Read(buf)
		if n > 0 {
			drained += n
			if onChunk != nil {
				onChunk(buf[:n])
			}
		}
		if err != nil {
			p.logger.Debug("Upstream drain complete", "credential", credName, "bytes_drained", drained)
			return
		}
	}
}

func (p *Proxy) streamToClient(
	ctx context.Context,
	w http.ResponseWriter,
	reader io.Reader,
	credName string,
	modelID string,
	endpoint string,
	onChunk func([]byte),
	onWriteErr func(),
	logCtx *RequestLogContext,
) error {
	_, ok := w.(http.Flusher)
	if !ok {
		p.logger.ErrorContext(ctx, "Streaming not supported", "credential", credName)
		WriteErrorInternal(w, "Streaming Not Supported")
		return fmt.Errorf("streaming not supported")
	}
	controller := http.NewResponseController(w)

	buf := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(buf)

	// TTFT detection buffer: separate from the bytes forwarded to the client via
	// onChunk (forwarding must stay byte-for-byte immediate for latency). A real
	// content/tool/reasoning delta can straddle multiple Read calls, or be
	// preceded by content-free SSE events (ping, role-only delta) that arrive in
	// their own Read — so bytes accumulate here across reads until
	// extractCompletionDeltaText finds actual content, rather than stamping
	// CompletionStartTime on the first non-empty Read regardless of content.
	var ttftBuf []byte
	ttftPending := logCtx != nil && logCtx.CompletionStartTime.IsZero()

	for {
		n, err := reader.Read(*buf)
		if n > 0 {
			if ttftPending {
				ttftBuf = append(ttftBuf, (*buf)[:n]...)
				if extractCompletionDeltaText(ttftBuf) != "" {
					logCtx.CompletionStartTime = time.Now()
					ttftPending = false
					ttftBuf = nil
				} else if len(ttftBuf) > streamTTFTDetectionLimit {
					ttftPending = false
					ttftBuf = nil
				}
			}
			if onChunk != nil {
				onChunk((*buf)[:n])
			}
			// Set write deadline before each write — keeps active streams alive,
			// terminates if client stops reading for streamChunkWriteTimeout.
			_ = controller.SetWriteDeadline(time.Now().Add(streamChunkWriteTimeout))
			if _, writeErr := w.Write((*buf)[:n]); writeErr != nil {
				if isClientDisconnectError(writeErr) {
					p.logger.DebugContext(ctx, "Client disconnected during streaming", "error", writeErr, "credential", credName)
					p.recordAbortedRequest(credName, endpoint, modelID)
				} else {
					p.logger.ErrorContext(ctx, "Failed to write streaming chunk", "error", writeErr, "credential", credName)
				}
				if onWriteErr != nil {
					onWriteErr()
				}
				return writeErr
			}
			if flushErr := p.flushStreaming(ctx, controller, credName); flushErr != nil {
				if isClientDisconnectError(flushErr) {
					p.logger.DebugContext(ctx, "Client disconnected during streaming flush", "error", flushErr, "credential", credName)
					p.recordAbortedRequest(credName, endpoint, modelID)
				}
				if onWriteErr != nil {
					onWriteErr()
				}
				return flushErr
			}
		}
		if err != nil {
			if err != io.EOF {
				p.logStreamHandlerError(ctx, "Streaming read error", err, "credential", credName)
			}
			break
		}
	}
	return nil
}

func (p *Proxy) flushStreaming(ctx context.Context, controller *http.ResponseController, credName string) (retErr error) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.ErrorContext(ctx, "Flusher panic", "panic", r, "credential", credName)
			retErr = fmt.Errorf("flusher panic: %v", r)
		}
	}()
	if err := controller.Flush(); err != nil {
		switch {
		case isClientDisconnectError(err):
			// Disconnects are logged once by the caller; don't double-log here.
		case errors.Is(err, http.ErrNotSupported):
			p.logger.ErrorContext(ctx, "Streaming not supported", "credential", credName)
		default:
			p.logger.ErrorContext(ctx, "Flusher error", "error", err, "credential", credName)
		}
		return err
	}
	return nil
}

// handleResponsesAPIStreaming handles streaming for Responses API requests.
// It first converts the provider stream to OpenAI Chat Completions SSE format,
// then converts that to Responses API SSE format.
// The optional onComplete callback is invoked with the fully-built Response
// once the stream finishes (used for store persistence).
// meta (may be nil) is used to echo store/previous_response_id/metadata fields
// back in every emitted SSE response object.
func (p *Proxy) handleResponsesAPIStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	cred *config.CredentialConfig,
	modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
	meta ...*responses.ResponsesMetadata,
) error {
	p.logger.DebugContext(respCtx(resp), "Starting Responses API streaming", "credential", cred.Name, "provider", cred.Type)

	// For providers that need transformation (Vertex, Anthropic, Bedrock),
	// first transform to OpenAI Chat Completions SSE, then to Responses API SSE.
	// For OpenAI (passthrough), the stream is already in Chat Completions SSE format.

	// Use realModelID for provider dispatch (e.g. isAnthropicBedrockModel check),
	// keep modelID (alias) as DisplayModelID so response chunks show the alias.
	converterModelID := modelID
	if logCtx != nil && logCtx.RealModelID != "" {
		converterModelID = logCtx.RealModelID
	}
	conv := converter.New(cred.Type, converter.RequestMode{
		ModelID:        converterModelID,
		DisplayModelID: modelID,
		IsStreaming:    true,
	})

	// Extract optional metadata for field echoing (nil is fine — echoing is skipped)
	var reqMeta *responses.ResponsesMetadata
	if len(meta) > 0 {
		reqMeta = meta[0]
	}

	// Create a wrapper transformer that chains:
	// Provider SSE -> Chat Completions SSE -> Responses API SSE
	transformer := func(r io.Reader, id string, w io.Writer) error {
		if conv.IsPassthrough() {
			p.logger.DebugContext(respCtx(resp), "Responses API streaming: passthrough mode (Chat Completions SSE → Responses SSE)",
				"model", modelID, "provider", cred.Type)
			return responses.TransformChatStreamToResponsesWithMeta(r, w, modelID, reqMeta, onComplete)
		}

		p.logger.DebugContext(respCtx(resp), "Responses API streaming: converted mode (Provider SSE → Chat Completions SSE → Responses SSE)",
			"model", modelID, "provider", cred.Type)

		// Non-passthrough providers: first convert to Chat Completions SSE via pipe
		pr, pw := io.Pipe()
		var transformErr error
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			transformErr = conv.StreamTo(r, pw)
			if transformErr != nil {
				p.logger.ErrorContext(respCtx(resp), "Responses API streaming: provider→ChatCompletions transform failed",
					"error", transformErr, "provider", cred.Type)
				_ = pw.CloseWithError(transformErr)
			} else {
				p.logger.DebugContext(respCtx(resp), "Responses API streaming: provider→ChatCompletions transform completed OK",
					"provider", cred.Type)
				_ = pw.Close()
			}
		}()

		// Then convert Chat Completions SSE to Responses API SSE
		err := responses.TransformChatStreamToResponsesWithMeta(pr, w, modelID, reqMeta, onComplete)
		_ = pr.Close()
		wg.Wait() // ensure goroutine completes before reading transformErr
		if err != nil {
			p.logger.ErrorContext(respCtx(resp), "Responses API streaming: ChatCompletions→Responses transform failed",
				"error", err, "provider", cred.Type)
			return err
		}
		if transformErr != nil {
			p.logger.ErrorContext(respCtx(resp), "Responses API streaming: provider transform error after Responses transform",
				"error", transformErr, "provider", cred.Type)
		}
		return transformErr
	}

	return p.handleTransformedStreaming(w, resp, cred.Name, modelID, string(cred.Type), transformer, logCtx)
}

// handleNativeResponsesStreaming handles Responses API streaming via the Phase 4
// ProviderResponses converter (Vertex AI, Anthropic). The provider SSE is converted
// directly to Responses API SSE format by the provider-specific StreamTo implementation.
func (p *Proxy) handleNativeResponsesStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	provResponses responses.ProviderResponses,
	modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
	meta *responses.ResponsesMetadata,
) error {
	credName := ""
	if logCtx.Credential != nil {
		credName = logCtx.Credential.Name
	}
	transformer := func(r io.Reader, _ string, ww io.Writer) error {
		return provResponses.StreamTo(r, ww, modelID, meta, onComplete)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "native_responses", transformer, logCtx)
}

// handlePassthroughResponsesStreaming handles Responses API streaming for codex models
// that natively support the /v1/responses endpoint. The provider SSE stream is forwarded
// to the client as-is.
//
// Token counts and the optional save callback are driven by the response.completed SSE
// event. Because the event payload can be very large (full response JSON with reasoning),
// it often spans multiple 8 KB buffer reads. This function maintains a line-level
// accumulator (partialSSELine) so that a data: line that arrives in pieces is reassembled
// before JSON parsing, avoiding the silent json.Unmarshal failures that would otherwise
// leave totalTokens = 0 and the store callback never invoked.
func (p *Proxy) handlePassthroughResponsesStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	credName, modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
) error {
	p.logger.DebugContext(respCtx(resp), "Starting passthrough Responses API streaming",
		"credential", credName, "model", modelID)

	var (
		totalTokens           int
		chunkCount            int
		lastRawChunk          []byte // last raw buffer for fallback in finalizeStreamingLog
		completedEventPayload []byte // JSON payload of response.completed (used instead of lastRawChunk)
		partialSSELine        string // partial SSE line accumulator across buffer reads
		completion            = newCompletionTokenAccumulator(modelID)
	)

	onChunk := func(chunk []byte) {
		chunkCount++
		completion.AddChunk(chunk)
		// Same [DONE]-skip as handleStreamingWithTokens: don't overwrite a useful
		// lastRawChunk with the bare sentinel — keeps the usage event accessible.
		rememberLastStreamDataChunk(&lastRawChunk, chunk)

		// Combine the partial line buffered from the previous read with the new chunk.
		// SSE data: lines can be arbitrarily long (e.g. response.completed with reasoning)
		// and will be split across multiple 8 KB buffer reads.
		combined := partialSSELine + string(chunk)
		partialSSELine = ""

		lastNL := strings.LastIndex(combined, "\n")
		if lastNL < 0 {
			// No newline yet — entire content is an incomplete line.
			partialSSELine = combined
			return
		}
		if lastNL < len(combined)-1 {
			// Characters after the last newline are an incomplete line.
			partialSSELine = combined[lastNL+1:]
		}

		// Walk every complete line in this chunk.
		for _, line := range strings.Split(combined[:lastNL+1], "\n") {
			line = strings.TrimRight(line, "\r")
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "" || jsonData == "[DONE]" {
				continue
			}

			var event struct {
				Type     string             `json:"type"`
				Response responses.Response `json:"response"`
			}
			if json.Unmarshal([]byte(jsonData), &event) == nil && event.Type == "response.completed" {
				if event.Response.Usage != nil {
					totalTokens = event.Response.Usage.TotalTokens
					if logCtx != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						logCtx.TokenUsage.PromptTokens = event.Response.Usage.InputTokens
						logCtx.TokenUsage.CompletionTokens = event.Response.Usage.OutputTokens
						logCtx.TokenUsage.CachedInputTokens = event.Response.Usage.InputTokensDetails.CachedTokens
						logCtx.TokenUsage.AudioInputTokens = event.Response.Usage.InputTokensDetails.AudioTokens
						logCtx.TokenUsage.AudioOutputTokens = event.Response.Usage.OutputTokensDetails.AudioTokens
						logCtx.TokenUsage.ReasoningTokens = event.Response.Usage.OutputTokensDetails.ReasoningTokens
					}
				}
				completedEventPayload = []byte(jsonData) // plain JSON; extractResponsesAPIUsage handles it
				if onComplete != nil {
					onComplete(&event.Response)
				}
			}
		}
	}

	if err := p.streamToClient(respCtx(resp), w, resp.Body, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), onChunk, nil, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handlePassthroughResponsesStreaming", err,
			"credential", credName, "model", modelID, "chunks_received", chunkCount)
		if p.drainUpstreamOnAbort {
			drainCtx, cancel := context.WithTimeout(context.Background(), streamDrainTimeout)
			defer cancel()
			p.drainUpstream(drainCtx, resp.Body, onChunk, credName)
		}
		finalChunk := lastRawChunk
		if len(completedEventPayload) > 0 {
			finalChunk = completedEventPayload
		}
		logTokens := totalTokens
		if logTokens == 0 {
			logTokens = completion.TokenCount()
		}
		p.finalizeStreamingLog(logCtx, logTokens, finalChunk, "openai", resp.StatusCode)
		return err
	}

	p.logger.DebugContext(respCtx(resp), "handlePassthroughResponsesStreaming completed",
		"credential", credName, "model", modelID,
		"chunks_received", chunkCount, "total_tokens", totalTokens)

	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded",
			"credential", credName, "model", modelID, "tokens", totalTokens)
	}

	// Prefer the parsed response.completed payload for detailed token extraction.
	// The raw lastRawChunk may contain only `data: [DONE]` with no usage info.
	// completedEventPayload is plain JSON; extractJSONPayloadsFromStreamChunk handles it
	// via the non-SSE fast path, so extractResponsesAPIUsage works correctly.
	finalChunk := lastRawChunk
	if len(completedEventPayload) > 0 {
		finalChunk = completedEventPayload
	}

	p.finalizeStreamingLog(logCtx, logTokens, finalChunk, "openai", resp.StatusCode)
	return nil
}
