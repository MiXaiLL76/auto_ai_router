package proxy

import (
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

// writeProxyResponse writes raw upstream proxy response to client.
// Respects the client's Accept-Encoding header to compress the response appropriately.
// Used by both primary proxy path and fallback retry path to avoid duplication.
func (p *Proxy) writeProxyResponse(w http.ResponseWriter, resp *ProxyResponse, clientReq *http.Request, credName, modelID string) {
	if resp == nil {
		return
	}

	responseBody := resp.Body
	responseBodyChanged := false
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		if normalizedBody, changed := modelutils.NormalizeCompletionUsage(responseBody, modelID); changed {
			responseBody = normalizedBody
			responseBodyChanged = true
		}
	}

	// Determine target encoding based on client's Accept-Encoding
	acceptEncoding := clientReq.Header.Get("Accept-Encoding")
	acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
	targetEncoding := SelectBestEncoding(acceptedEncodings)

	p.logger.DebugContext(clientReq.Context(), "Proxy response encoding decision",
		"accept_encoding_header", acceptEncoding,
		"target_encoding", targetEncoding,
		"body_size", len(responseBody),
	)

	// Compress body if needed (Go's http.Client already decompressed upstream response)
	contentEncoding := ""

	if targetEncoding != "identity" && len(responseBody) > 0 {
		uncompressedSize := len(responseBody)
		compressedBody, usedEncoding, err := CompressBody(responseBody, targetEncoding)
		if err != nil {
			p.logger.WarnContext(clientReq.Context(), "Failed to compress response body",
				"encoding", targetEncoding,
				"error", err,
			)
			// Continue with uncompressed body on error
		} else {
			p.logger.DebugContext(clientReq.Context(), "Response body compressed",
				"encoding", usedEncoding,
				"original_size", uncompressedSize,
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
		if responseBodyChanged && isRepresentationIntegrityHeader(key) {
			continue
		}
		// Skip Content-Length, Transfer-Encoding, and Content-Encoding
		// We'll set Content-Encoding based on our compression, and Content-Length based on actual body size
		// Skip X-Credential-Name — internal header for proxy-to-proxy routing, not exposed to end clients
		if key == "Content-Length" || key == "Transfer-Encoding" || key == "Content-Encoding" || key == "X-Credential-Name" {
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
			p.logger.DebugContext(clientReq.Context(), "Client disconnected during proxy response write", "error", err)
			p.recordAbortedRequest(credName, endpointFromRequest(clientReq), modelID)
		} else {
			p.logger.ErrorContext(clientReq.Context(), "Failed to write proxy response body", "error", err)
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
	modelID string,
	tokenizerModelID string,
	logCtx *RequestLogContext,
) (*converter.TokenUsage, error) {
	if resp == nil || resp.StreamBody == nil {
		return nil, nil
	}

	streamBody := resp.StreamBody
	normalizeStream := false
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		if normalizedStreamBody, wrapped := modelutils.NewUsageNormalizingReadCloser(streamBody, modelID); wrapped {
			streamBody = normalizedStreamBody
			normalizeStream = true
		}
	}
	defer func() {
		if closeErr := streamBody.Close(); closeErr != nil {
			p.logger.WarnContext(clientReq.Context(), "Failed to close proxy streaming response body", "error", closeErr)
		}
	}()

	for key, values := range resp.Headers {
		if isHopByHopHeader(key) {
			continue
		}
		if normalizeStream && isRepresentationIntegrityHeader(key) {
			continue
		}
		// Skip Content-Length, Transfer-Encoding, and Content-Encoding
		// For streaming responses, we don't re-compress since it would break the stream protocol.
		// We remove Content-Encoding from upstream since Go's http.Client already decompressed it.
		// Skip X-Credential-Name — internal header for proxy-to-proxy routing, not exposed to end clients
		if key == "Content-Length" || key == "Transfer-Encoding" || key == "Content-Encoding" || key == "X-Credential-Name" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.WriteHeader(resp.StatusCode)

	var lastUsage *converter.TokenUsage
	completion := newCompletionTokenAccumulator(tokenizerModelID)
	onChunk := func(chunk []byte) {
		if usage := extractTokenUsageFromStreamingChunk(string(chunk)); usage != nil {
			lastUsage = usage
		}
		completion.AddChunk(chunk)
	}

	buildFallbackUsage := func() *converter.TokenUsage {
		if lastUsage != nil {
			return lastUsage
		}
		if tokens := completion.TokenCount(); tokens > 0 {
			return &converter.TokenUsage{CompletionTokens: tokens}
		}
		return nil
	}

	if _, ok := w.(http.Flusher); ok {
		err := p.streamToClient(
			clientReq.Context(),
			w,
			streamBody,
			credName,
			modelID,
			endpointFromRequest(clientReq),
			onChunk,
			nil,
			logCtx,
		)
		if err != nil && p.drainUpstreamOnAbort {
			// Drain upstream so the usage chunk arrives even though the client left.
			drainCtx, cancel := context.WithTimeout(context.Background(), streamDrainTimeout)
			defer cancel()

			p.drainUpstream(
				drainCtx,
				streamBody,
				onChunk,
				credName,
			)
		}

		return buildFallbackUsage(), err
	}

	// Non-flushing fallback: copy as-is (token usage cannot be parsed reliably here).
	if _, err := io.Copy(w, streamBody); err != nil {
		if isClientDisconnectError(err) {
			p.recordAbortedRequest(credName, endpointFromRequest(clientReq), modelID)
		}
		return buildFallbackUsage(), err
	}
	return buildFallbackUsage(), nil
}

func isRepresentationIntegrityHeader(key string) bool {
	switch strings.ToLower(key) {
	case "etag", "content-md5", "digest", "content-digest", "repr-digest":
		return true
	default:
		return false
	}
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
