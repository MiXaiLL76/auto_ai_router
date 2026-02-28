package proxy

import (
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
)

// streamChunkWriteTimeout is the per-chunk write deadline for streaming responses.
// If no data flows for this duration, the connection is terminated.
const streamChunkWriteTimeout = 60 * time.Second

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
	ImageTokens         int // Image/video tokens (if reported)
	ReasoningTokens     int // Reasoning/thoughts tokens (output)
	CacheCreationTokens int // Anthropic: tokens created for cache (billed at different rate)
	CacheReadTokens     int // Anthropic: tokens read from cache (billed at cheaper rate)
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
	// OpenAI streaming format:
	// {"choices":[...],"usage":{"prompt_tokens":100,"completion_tokens":50}}
	// Usage may appear in any chunk but typically in the final chunk

	payloads := extractJSONPayloadsFromStreamChunk(chunk)
	for i := len(payloads) - 1; i >= 0; i-- {
		var data struct {
			Usage struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				PromptTokensDetails struct {
					CachedTokens int `json:"cached_tokens,omitempty"`
					AudioTokens  int `json:"audio_tokens,omitempty"`
				} `json:"prompt_tokens_details,omitempty"`
				CompletionTokensDetails struct {
					AudioTokens     int `json:"audio_tokens,omitempty"`
					ReasoningTokens int `json:"reasoning_tokens,omitempty"`
				} `json:"completion_tokens_details,omitempty"`
			} `json:"usage"`
		}

		if err := json.Unmarshal(payloads[i], &data); err != nil {
			continue
		}

		// Check if usage info is present (both fields can't be 0 for valid usage)
		if data.Usage.PromptTokens == 0 && data.Usage.CompletionTokens == 0 {
			continue
		}

		return &StreamUsageInfo{
			PromptTokens:      data.Usage.PromptTokens,
			CompletionTokens:  data.Usage.CompletionTokens,
			CachedTokens:      data.Usage.PromptTokensDetails.CachedTokens,
			AudioInputTokens:  data.Usage.PromptTokensDetails.AudioTokens,
			AudioOutputTokens: data.Usage.CompletionTokensDetails.AudioTokens,
			ReasoningTokens:   data.Usage.CompletionTokensDetails.ReasoningTokens,
		}
	}

	return nil
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
	default:
		// Fallback: try OpenAI format first (most common)
		return &openAIStreamUsageExtractor{}
	}
}

func IsStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json")
}

type streamTransformer func(io.Reader, string, io.Writer) error

func (p *Proxy) handleVertexStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeVertexAI, converter.RequestMode{ModelID: modelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Vertex AI", transformer, logCtx)
}

func (p *Proxy) handleAnthropicStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeAnthropic, converter.RequestMode{ModelID: modelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Anthropic", transformer, logCtx)
}

type tokenCapturingWriter struct {
	writer  io.Writer
	tokens  *int
	logger  *slog.Logger
	onChunk func([]byte) // Callback invoked for each chunk (optional, for capturing last chunk)
}

func (tcw *tokenCapturingWriter) Write(p []byte) (n int, err error) {
	// Extract tokens from the data being written
	tokens := extractTokensFromStreamingChunk(string(p))
	if tokens > 0 {
		*tcw.tokens += tokens
	}

	// Invoke callback if provided (used to capture last chunk for usage extraction)
	if tcw.onChunk != nil {
		tcw.onChunk(p)
	}

	return tcw.writer.Write(p)
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
	p.logger.Debug("Starting streaming response", "provider", providerName, "credential", credName)

	pr, pw := io.Pipe()
	defer func() {
		_ = pr.Close()
	}()
	var totalTokens int

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte

	// WaitGroup ensures the transform goroutine completes before we read
	// lastChunk and totalTokens, preventing a data race.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := transformFunc(resp.Body, modelID, &tokenCapturingWriter{
			writer: pw,
			tokens: &totalTokens,
			logger: p.logger,
			onChunk: func(chunk []byte) {
				// Store each chunk, keeping only the last one
				// This allows us to extract usage info that typically appears in final chunks
				lastChunk = make([]byte, len(chunk))
				copy(lastChunk, chunk)
			},
		})
		if err != nil {
			// Don't log here — CloseWithError propagates the error to the
			// reader side where streamToClient logs it as "Streaming read error".
			_ = pw.CloseWithError(fmt.Errorf("%s transform: %w", providerName, err))
		} else {
			_ = pw.Close()
		}
	}()

	if err := p.streamToClient(w, pr, credName, nil, func() { _ = pr.Close() }); err != nil {
		wg.Wait()
		return err
	}
	wg.Wait()

	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, totalTokens, lastChunk, providerName, resp.StatusCode)

	p.logger.Debug("Streaming response completed", "provider", providerName, "credential", credName)
	return nil
}

func (p *Proxy) handleStreamingWithTokens(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	p.logger.Debug("Starting streaming response with token tracking", "credential", credName)

	var totalTokens int

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte

	onChunk := func(chunk []byte) {
		tokens := extractTokensFromStreamingChunk(string(chunk))
		if tokens > 0 {
			totalTokens += tokens
		}

		// Store each chunk, keeping only the last one
		// This allows us to extract usage info that typically appears in final chunks
		lastChunk = make([]byte, len(chunk))
		copy(lastChunk, chunk)
	}

	if err := p.streamToClient(w, resp.Body, credName, onChunk, nil); err != nil {
		return err
	}

	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, totalTokens, lastChunk, "openai", resp.StatusCode)

	p.logger.Debug("Streaming response completed", "credential", credName)
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

	logCtx.TokenUsage.PromptTokens = logCtx.PromptTokensEstimate
	logCtx.TokenUsage.CompletionTokens = totalTokens

	if len(lastChunk) > 0 {
		extractor := getStreamUsageExtractor(providerName)
		if usageInfo := extractor.ExtractUsage(lastChunk); usageInfo != nil {
			if usageInfo.PromptTokens > 0 {
				logCtx.TokenUsage.PromptTokens = usageInfo.PromptTokens
			}
			if usageInfo.CompletionTokens > 0 {
				logCtx.TokenUsage.CompletionTokens = usageInfo.CompletionTokens
			}

			logCtx.TokenUsage.CachedInputTokens = usageInfo.CachedTokens
			logCtx.TokenUsage.AudioInputTokens = usageInfo.AudioInputTokens
			logCtx.TokenUsage.AudioOutputTokens = usageInfo.AudioOutputTokens
			logCtx.TokenUsage.ImageTokens = usageInfo.ImageTokens
			logCtx.TokenUsage.ReasoningTokens = usageInfo.ReasoningTokens

			if usageInfo.CacheCreationTokens > 0 {
				logCtx.TokenUsage.CacheCreationTokens = usageInfo.CacheCreationTokens
			}

			p.logger.Debug("Extracted usage from streaming response",
				"provider", providerName,
				"prompt_tokens", usageInfo.PromptTokens,
				"completion_tokens", usageInfo.CompletionTokens,
				"cached_tokens", usageInfo.CachedTokens,
				"audio_input_tokens", usageInfo.AudioInputTokens,
				"audio_output_tokens", usageInfo.AudioOutputTokens,
				"image_tokens", usageInfo.ImageTokens,
				"reasoning_tokens", usageInfo.ReasoningTokens,
			)
		}
	}

	logCtx.HTTPStatus = statusCode
	if statusCode >= 400 {
		logCtx.Status = "failure"
	} else {
		logCtx.Status = "success"
	}
	logCtx.Logged = true
	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.Warn("Failed to queue streaming spend log",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
}

func (p *Proxy) streamToClient(
	w http.ResponseWriter,
	reader io.Reader,
	credName string,
	onChunk func([]byte),
	onWriteErr func(),
) error {
	_, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return fmt.Errorf("streaming not supported")
	}
	controller := http.NewResponseController(w)

	buf := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(buf)
	for {
		n, err := reader.Read(*buf)
		if n > 0 {
			if onChunk != nil {
				onChunk((*buf)[:n])
			}
			// Set write deadline before each write — keeps active streams alive,
			// terminates if client stops reading for streamChunkWriteTimeout.
			_ = controller.SetWriteDeadline(time.Now().Add(streamChunkWriteTimeout))
			if _, writeErr := w.Write((*buf)[:n]); writeErr != nil {
				if isClientDisconnectError(writeErr) {
					p.logger.Warn("Client disconnected during streaming", "error", writeErr, "credential", credName)
				} else {
					p.logger.Error("Failed to write streaming chunk", "error", writeErr, "credential", credName)
				}
				if onWriteErr != nil {
					onWriteErr()
				}
				return writeErr
			}
			p.flushStreaming(controller, credName)
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error", "error", err, "credential", credName)
			}
			break
		}
	}
	return nil
}

func (p *Proxy) flushStreaming(controller *http.ResponseController, credName string) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.Error("Flusher panic", "panic", r, "credential", credName)
		}
	}()
	if err := controller.Flush(); err != nil {
		if errors.Is(err, http.ErrNotSupported) {
			p.logger.Error("Streaming not supported", "credential", credName)
		} else {
			p.logger.Error("Flusher error", "error", err, "credential", credName)
		}
	}
}
