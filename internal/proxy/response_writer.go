package proxy

import (
	"io"
	"net/http"
)

// writeProxyResponse writes raw upstream proxy response to client.
// Respects the client's Accept-Encoding header to compress the response appropriately.
// Used by both primary proxy path and fallback retry path to avoid duplication.
func (p *Proxy) writeProxyResponse(w http.ResponseWriter, resp *ProxyResponse, clientReq *http.Request) {
	if resp == nil {
		return
	}

	// Determine target encoding based on client's Accept-Encoding
	acceptEncoding := clientReq.Header.Get("Accept-Encoding")
	acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
	targetEncoding := SelectBestEncoding(acceptedEncodings)

	p.logger.Debug("Proxy response encoding decision",
		"accept_encoding_header", acceptEncoding,
		"target_encoding", targetEncoding,
		"body_size", len(resp.Body),
	)

	// Compress body if needed (Go's http.Client already decompressed upstream response)
	responseBody := resp.Body
	contentEncoding := ""

	if targetEncoding != "identity" && len(resp.Body) > 0 {
		compressedBody, usedEncoding, err := CompressBody(resp.Body, targetEncoding)
		if err != nil {
			p.logger.Warn("Failed to compress response body",
				"encoding", targetEncoding,
				"error", err,
			)
			// Continue with uncompressed body on error
		} else {
			p.logger.Debug("Response body compressed",
				"encoding", usedEncoding,
				"original_size", len(resp.Body),
				"compressed_size", len(compressedBody),
			)
			responseBody = compressedBody
			contentEncoding = usedEncoding
		}
	}

	// Copy response headers
	for key, values := range resp.Headers {
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length, Transfer-Encoding, and Content-Encoding
		// We'll set Content-Encoding based on our compression, and Content-Length based on actual body size
		if key == "Content-Length" || key == "Transfer-Encoding" || key == "Content-Encoding" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Set Content-Encoding if we compressed the response
	if contentEncoding != "identity" {
		w.Header().Set("Content-Encoding", contentEncoding)
	}

	w.Header().Set("Content-Length", itoa(len(responseBody)))
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(responseBody); err != nil {
		if isClientDisconnectError(err) {
			p.logger.Debug("Client disconnected during proxy response write", "error", err)
		} else {
			p.logger.Error("Failed to write proxy response body", "error", err)
		}
	}
}

// writeProxyStreamingResponseWithTokens streams proxy response and captures token usage from stream chunks.
// Note: For streaming responses, we don't compress the body as it would break the streaming protocol.
// The client's Accept-Encoding preference is respected by not adding Content-Encoding header if compression isn't applied.
func (p *Proxy) writeProxyStreamingResponseWithTokens(
	w http.ResponseWriter,
	resp *ProxyResponse,
	clientReq *http.Request,
	credName string,
) (int, error) {
	if resp == nil || resp.StreamBody == nil {
		return 0, nil
	}
	defer func() {
		if closeErr := resp.StreamBody.Close(); closeErr != nil {
			p.logger.Error("Failed to close proxy streaming response body", "error", closeErr)
		}
	}()

	for key, values := range resp.Headers {
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length, Transfer-Encoding, and Content-Encoding
		// For streaming responses, we don't re-compress since it would break the stream protocol.
		// We remove Content-Encoding from upstream since Go's http.Client already decompressed it.
		if key == "Content-Length" || key == "Transfer-Encoding" || key == "Content-Encoding" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.WriteHeader(resp.StatusCode)

	totalTokens := 0
	onChunk := func(chunk []byte) {
		tokens := extractTokensFromStreamingChunk(string(chunk))
		if tokens > 0 {
			totalTokens += tokens
		}
	}

	if _, ok := w.(http.Flusher); ok {
		if err := p.streamToClient(w, resp.StreamBody, credName, onChunk, nil); err != nil {
			return totalTokens, err
		}
		return totalTokens, nil
	}

	// Non-flushing fallback: copy as-is (token usage cannot be parsed reliably here).
	if _, err := io.Copy(w, resp.StreamBody); err != nil {
		return totalTokens, err
	}
	return totalTokens, nil
}

// itoa avoids fmt.Sprintf for a hot path.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}

	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
