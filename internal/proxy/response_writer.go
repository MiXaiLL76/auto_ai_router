package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

const maxSanitizingSSELineBytes = 1 << 20

func shouldSanitizeUpstreamSurface(cred *config.CredentialConfig) bool {
	return isProManCredential(cred)
}

func clientResponseBodyForCredential(statusCode int, body []byte, cred *config.CredentialConfig, displayModel string) ([]byte, bool, bool) {
	if statusCode >= 400 && shouldMaskUpstreamErrors(cred) {
		masked := maskedUpstreamErrorBody(statusCode)
		return masked, !bytes.Equal(body, masked), true
	}
	if statusCode >= 200 && statusCode < 300 && shouldSanitizeUpstreamSurface(cred) {
		sanitized, changed := sanitizeUpstreamJSONBody(body, displayModel)
		return sanitized, changed, false
	}
	return body, false, false
}

func sanitizeUpstreamJSONBody(body []byte, displayModel string) ([]byte, bool) {
	if len(bytes.TrimSpace(body)) == 0 {
		return body, false
	}

	var value any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return body, false
	}

	changed := sanitizeJSONValue(&value, displayModel)
	if !changed {
		return body, false
	}

	sanitized, err := json.Marshal(value)
	if err != nil {
		return body, false
	}
	return sanitized, true
}

func sanitizeJSONValue(value *any, displayModel string) bool {
	switch typed := (*value).(type) {
	case map[string]any:
		changed := false
		if eventType, _ := typed["type"].(string); eventType == "error" {
			if errObj, ok := typed["error"].(map[string]any); ok {
				if _, hasMessage := errObj["message"]; hasMessage {
					errObj["message"] = "Upstream provider error"
					changed = true
				}
			}
		}
		for key, nested := range typed {
			if isInternalJSONField(key) {
				delete(typed, key)
				changed = true
				continue
			}
			lowerKey := strings.ToLower(key)
			if lowerKey == "model" && displayModel != "" {
				if s, ok := nested.(string); ok && isInternalModelString(s) {
					typed[key] = displayModel
					changed = true
					continue
				}
			}
			nestedValue := nested
			if sanitizeJSONValue(&nestedValue, displayModel) {
				typed[key] = nestedValue
				changed = true
			}
		}
		return changed
	case []any:
		changed := false
		for i, nested := range typed {
			nestedValue := nested
			if sanitizeJSONValue(&nestedValue, displayModel) {
				typed[i] = nestedValue
				changed = true
			}
		}
		return changed
	default:
		return false
	}
}

func isInternalJSONField(key string) bool {
	lower := strings.ToLower(key)
	switch lower {
	case "provider_specific_fields",
		"caller",
		"litellm_metadata",
		"litellm_params",
		"litellm_call_id",
		"llm_provider",
		"llm_provider_id":
		return true
	}
	return strings.HasPrefix(lower, "litellm_") ||
		strings.HasPrefix(lower, "x-litellm") ||
		strings.HasPrefix(lower, "llm_provider")
}

func isInternalModelString(value string) bool {
	lower := strings.ToLower(value)
	if strings.HasPrefix(lower, "anthropic/") {
		return true
	}
	internalMarkers := []string{
		"anthropic-direct-client",
		"litellm",
		"llm_provider",
		"proman",
		"bedrock",
		"amazonaws",
		"aws/",
	}
	for _, marker := range internalMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

type sanitizingSSEReadCloser struct {
	source io.Closer
	reader *sanitizingSSEReader
}

type sanitizingSSEReader struct {
	reader       *bufio.Reader
	pending      []byte
	line         []byte
	passthrough  bool
	displayModel string
	terminalErr  error
}

func newSanitizingSSEReader(source io.Reader, displayModel string) io.Reader {
	return &sanitizingSSEReader{
		reader:       bufio.NewReader(source),
		displayModel: displayModel,
	}
}

func newSanitizingSSEReadCloser(source io.ReadCloser, displayModel string) io.ReadCloser {
	return &sanitizingSSEReadCloser{
		source: source,
		reader: newSanitizingSSEReader(source, displayModel).(*sanitizingSSEReader),
	}
}

func (r *sanitizingSSEReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	for len(r.pending) == 0 {
		if r.terminalErr != nil {
			err := r.terminalErr
			if err != io.EOF {
				r.terminalErr = io.EOF
			}
			return 0, err
		}

		r.readNextSSEFragment()
	}

	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *sanitizingSSEReader) readNextSSEFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.reader.ReadSlice('\n')

		if r.passthrough {
			if len(fragment) > 0 {
				r.pending = fragment
			}
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
			}
		} else {
			r.consumeSanitizingSSEFragment(fragment, err)
		}

		if err != nil && !errors.Is(err, bufio.ErrBufferFull) {
			r.terminalErr = err
		}
	}
}

func (r *sanitizingSSEReader) consumeSanitizingSSEFragment(fragment []byte, readErr error) {
	if errors.Is(readErr, bufio.ErrBufferFull) {
		if len(r.line)+len(fragment) <= maxSanitizingSSELineBytes {
			r.line = append(r.line, fragment...)
			return
		}

		r.pending = append(r.line, fragment...)
		r.line = nil
		r.passthrough = true
		return
	}

	line := append(r.line, fragment...)
	r.line = nil
	if len(line) == 0 {
		return
	}

	if readErr == nil || errors.Is(readErr, io.EOF) {
		if len(line) <= maxSanitizingSSELineBytes {
			r.pending = sanitizeSSELine(line, r.displayModel)
			return
		}
	}

	r.pending = line
}

func (r *sanitizingSSEReadCloser) Read(p []byte) (int, error) {
	return r.reader.Read(p)
}

func (r *sanitizingSSEReadCloser) Close() error {
	return r.source.Close()
}

func sanitizeSSELine(line []byte, displayModel string) []byte {
	content, ending := splitSSELineEnding(line)
	if !bytes.HasPrefix(content, []byte("data:")) {
		return line
	}

	rawPayload := content[len("data:"):]
	payload := bytes.TrimSpace(rawPayload)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || payload[0] != '{' {
		return line
	}

	sanitizedPayload, changed := sanitizeUpstreamJSONBody(payload, displayModel)
	if !changed {
		return line
	}

	payloadStart := bytes.Index(rawPayload, payload)
	payloadEnd := payloadStart + len(payload)
	sanitized := make([]byte, 0, len(line)-len(payload)+len(sanitizedPayload))
	sanitized = append(sanitized, content[:len("data:")]...)
	sanitized = append(sanitized, rawPayload[:payloadStart]...)
	sanitized = append(sanitized, sanitizedPayload...)
	sanitized = append(sanitized, rawPayload[payloadEnd:]...)
	sanitized = append(sanitized, ending...)
	return sanitized
}

func splitSSELineEnding(line []byte) (content, ending []byte) {
	if len(line) == 0 || line[len(line)-1] != '\n' {
		return line, nil
	}
	if len(line) >= 2 && line[len(line)-2] == '\r' {
		return line[:len(line)-2], line[len(line)-2:]
	}
	return line[:len(line)-1], line[len(line)-1:]
}

// writeProxyResponse writes raw upstream proxy response to client.
// Respects the client's Accept-Encoding header to compress the response appropriately.
// Used by both primary proxy path and fallback retry path to avoid duplication.
func (p *Proxy) writeProxyResponse(w http.ResponseWriter, resp *ProxyResponse, clientReq *http.Request, cred *config.CredentialConfig, modelID string) {
	if resp == nil {
		return
	}

	credName := ""
	if cred != nil {
		credName = cred.Name
	}

	responseBody, responseBodyChanged, responseBodyMasked := clientResponseBodyForCredential(resp.StatusCode, resp.Body, cred, modelID)
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
		if shouldSkipResponseHeaderForClient(key, cred) {
			continue
		}
		if responseBodyChanged && isRepresentationIntegrityHeader(key) {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	if responseBodyMasked {
		w.Header().Set("Content-Type", "application/json")
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
	cred *config.CredentialConfig,
	modelID string,
	tokenizerModelID string,
	logCtx *RequestLogContext,
) (*converter.TokenUsage, error) {
	if resp == nil || resp.StreamBody == nil {
		return nil, nil
	}

	credName := ""
	if cred != nil {
		credName = cred.Name
	}

	streamBody := resp.StreamBody
	normalizeStream := false
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		if normalizedStreamBody, wrapped := modelutils.NewUsageNormalizingReadCloser(streamBody, modelID); wrapped {
			streamBody = normalizedStreamBody
			normalizeStream = true
		}
	}
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices && shouldSanitizeUpstreamSurface(cred) {
		streamBody = newSanitizingSSEReadCloser(streamBody, modelID)
	}
	defer func() {
		if closeErr := streamBody.Close(); closeErr != nil {
			p.logger.WarnContext(clientReq.Context(), "Failed to close proxy streaming response body", "error", closeErr)
		}
	}()

	for key, values := range resp.Headers {
		if shouldSkipResponseHeaderForClient(key, cred) {
			continue
		}
		if normalizeStream && isRepresentationIntegrityHeader(key) {
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
