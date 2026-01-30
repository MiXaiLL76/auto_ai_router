package proxy

import (
	_ "embed"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/transform/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/transform/vertex"
)

func IsStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json")
}

func (p *Proxy) handleVertexStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return fmt.Errorf("streaming not supported")
	}

	p.logger.Debug("Starting Vertex AI streaming response", "credential", credName)

	// Create a pipe to capture the streaming data
	pr, pw := io.Pipe()
	var totalTokens int

	// Start goroutine to transform and capture tokens
	go func() {
		defer func() {
			_ = pw.Close()
		}()
		err := vertex.TransformVertexStreamToOpenAI(resp.Body, modelID, &tokenCapturingWriter{
			writer: pw,
			tokens: &totalTokens,
			logger: p.logger,
		})
		if err != nil {
			p.logger.Error("Failed to transform Vertex streaming", "error", err)
		}
	}()

	// Copy transformed data to response
	buf := make([]byte, 8192)
	for {
		n, err := pr.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk", "error", writeErr, "credential", credName)
				return writeErr
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error", "error", err, "credential", credName)
			}
			break
		}
	}

	// Record token usage after streaming completes
	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.logger.Debug("Vertex AI streaming response completed", "credential", credName)
	return nil
}

func (p *Proxy) handleAnthropicStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID string) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return fmt.Errorf("streaming not supported")
	}

	p.logger.Debug("Starting Anthropic streaming response", "credential", credName)

	// Create a pipe to capture the streaming data
	pr, pw := io.Pipe()
	var totalTokens int

	// Start goroutine to transform and capture tokens
	go func() {
		defer func() {
			_ = pw.Close()
		}()
		err := anthropic.TransformAnthropicStreamToOpenAI(resp.Body, modelID, &tokenCapturingWriter{
			writer: pw,
			tokens: &totalTokens,
			logger: p.logger,
		})
		if err != nil {
			p.logger.Error("Failed to transform Anthropic streaming", "error", err)
		}
	}()

	// Copy transformed data to response
	buf := make([]byte, 8192)
	for {
		n, err := pr.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk", "error", writeErr, "credential", credName)
				return writeErr
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error", "error", err, "credential", credName)
			}
			break
		}
	}

	// Record token usage after streaming completes
	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.logger.Debug("Anthropic streaming response completed", "credential", credName)
	return nil
}

func (p *Proxy) handleStreamingWithTokens(w http.ResponseWriter, resp *http.Response, credName, modelID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.logger.Error("Streaming not supported", "credential", credName)
		http.Error(w, "Streaming Not Supported", http.StatusInternalServerError)
		return
	}

	p.logger.Debug("Starting streaming response with token tracking", "credential", credName)

	var totalTokens int
	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			// Extract tokens from streaming chunks
			chunkData := string(buf[:n])
			tokens := extractTokensFromStreamingChunk(chunkData)
			if tokens > 0 {
				totalTokens += tokens
			}

			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				p.logger.Error("Failed to write streaming chunk", "error", writeErr, "credential", credName)
				return
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				p.logger.Error("Streaming read error", "error", err, "credential", credName)
			}
			break
		}
	}

	// Record token usage after streaming completes
	if totalTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, totalTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, totalTokens)
		}
		p.logger.Debug("Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.logger.Debug("Streaming response completed", "credential", credName)
}
