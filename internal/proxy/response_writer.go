package proxy

import (
	"io"
	"net/http"
)

// writeProxyResponse writes raw upstream proxy response to client.
// Used by both primary proxy path and fallback retry path to avoid duplication.
func (p *Proxy) writeProxyResponse(w http.ResponseWriter, resp *ProxyResponse) {
	if resp == nil {
		return
	}

	for key, values := range resp.Headers {
		if isHopByHopHeader(key) {
			continue
		}
		// Skip Content-Length and Transfer-Encoding - we'll set them correctly
		if key == "Content-Length" || key == "Transfer-Encoding" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.Header().Set("Content-Length", itoa(len(resp.Body)))
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(resp.Body); err != nil {
		p.logger.Error("Failed to write proxy response body", "error", err)
	}
}

// writeProxyStreamingResponseWithTokens streams proxy response and captures token usage from stream chunks.
func (p *Proxy) writeProxyStreamingResponseWithTokens(
	w http.ResponseWriter,
	resp *ProxyResponse,
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
		if key == "Content-Length" || key == "Transfer-Encoding" {
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
