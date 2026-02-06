package proxy

import (
	_ "embed"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/transform/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/transform/vertex"
)

var streamBufPool = sync.Pool{
	New: func() any {
		buf := make([]byte, 8192)
		return &buf
	},
}

func IsStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json")
}

type streamTransformer func(io.Reader, string, io.Writer) error

func (p *Proxy) handleVertexStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Vertex AI", vertex.TransformVertexStreamToOpenAI, logCtx)
}

func (p *Proxy) handleAnthropicStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Anthropic", anthropic.TransformAnthropicStreamToOpenAI, logCtx)
}

func (p *Proxy) handleTransformedStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	credName string,
	modelID string,
	providerName string,
	transform streamTransformer,
	logCtx *RequestLogContext,
) error {
	p.logger.Debug("Starting streaming response", "provider", providerName, "credential", credName)

	pr, pw := io.Pipe()
	defer func() {
		_ = pr.Close()
	}()
	var totalTokens int

	go func() {
		defer func() {
			_ = pw.Close()
		}()
		err := transform(resp.Body, modelID, &tokenCapturingWriter{
			writer: pw,
			tokens: &totalTokens,
			logger: p.logger,
		})
		if err != nil {
			p.logger.Error("Failed to transform streaming response", "provider", providerName, "error", err)
		}
	}()

	if err := p.streamToClient(w, pr, credName, nil, func() { _ = pr.Close() }); err != nil {
		return err
	}

	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	// Log streaming response to LiteLLM DB if logCtx provided
	if logCtx != nil && !logCtx.Logged {
		logCtx.CompletionTokens = totalTokens
		logCtx.Status = "success"
		logCtx.HTTPStatus = 200
		logCtx.Logged = true
		if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
			p.logger.Warn("Failed to queue streaming spend log",
				"error", err,
				"request_id", logCtx.RequestID,
			)
		}
	}

	p.logger.Debug("Streaming response completed", "provider", providerName, "credential", credName)
	return nil
}

func (p *Proxy) handleStreamingWithTokens(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	p.logger.Debug("Starting streaming response with token tracking", "credential", credName)

	var totalTokens int
	onChunk := func(chunk []byte) {
		tokens := extractTokensFromStreamingChunk(string(chunk))
		if tokens > 0 {
			totalTokens += tokens
		}
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

	// Log streaming response to LiteLLM DB if logCtx provided
	if logCtx != nil && !logCtx.Logged {
		logCtx.CompletionTokens = totalTokens
		logCtx.Status = "success"
		logCtx.HTTPStatus = 200
		logCtx.Logged = true
		if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
			p.logger.Warn("Failed to queue streaming spend log",
				"error", err,
				"request_id", logCtx.RequestID,
			)
		}
	}

	p.logger.Debug("Streaming response completed", "credential", credName)
	return nil
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
			if _, writeErr := w.Write((*buf)[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk", "error", writeErr, "credential", credName)
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
