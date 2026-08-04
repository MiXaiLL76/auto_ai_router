package proxy

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

func clientResponseBodyForCredential(statusCode int, body []byte, _ *config.CredentialConfig, displayModel string) ([]byte, bool, bool) {
	if statusCode >= 400 {
		masked := maskedUpstreamErrorBody(statusCode)
		return masked, !bytes.Equal(body, masked), true
	}
	if statusCode >= 200 && statusCode < 300 {
		sanitized, changed := promanutils.SanitizeUpstreamJSONBody(body, displayModel)
		return sanitized, changed, false
	}
	return body, false, false
}

const maxProxyStreamErrorCaptureBytes = 256 * 1024

// proxyProviderStreamError reports an error event carried by an otherwise
// successful HTTP/SSE response. The response has already been forwarded to the
// client, so callers must not try to replace it with another HTTP response; the
// error exists to drive terminal logging and session semantics.
type proxyProviderStreamError struct {
	payload string
}

func (e proxyProviderStreamError) Error() string {
	return "provider sent terminal error event in stream: " + e.payload
}

// proxyStreamErrorCapture reconstructs SSE frames across arbitrary network
// reads. Keeping only the unfinished frame bounds memory for long successful
// streams while still recognizing an error JSON object split across reads.
type proxyStreamErrorCapture struct {
	pending []byte
	payload string
}

type proxyStreamErrorObserver struct {
	capture *proxyStreamErrorCapture
}

type readerReadCloser struct {
	io.Reader
	io.Closer
}

func (w proxyStreamErrorObserver) Write(chunk []byte) (int, error) {
	w.capture.Observe(chunk)
	return len(chunk), nil
}

func (c *proxyStreamErrorCapture) Observe(chunk []byte) string {
	if c == nil || c.payload != "" || len(chunk) == 0 {
		if c == nil {
			return ""
		}
		return c.payload
	}

	c.pending = append(c.pending, chunk...)
	for {
		frameEnd := nextSSEFrameEnd(c.pending)
		if frameEnd < 0 {
			break
		}
		frame := c.pending[:frameEnd]
		c.pending = c.pending[frameEnd:]
		if frameMayCarryStreamError(frame) {
			if payload := extractStreamErrorEvent(frame); payload != "" {
				c.payload = payload
				c.pending = nil
				return payload
			}
		}
	}

	if len(c.pending) > maxProxyStreamErrorCaptureBytes {
		start := len(c.pending) - maxProxyStreamErrorCaptureBytes
		c.pending = append([]byte(nil), c.pending[start:]...)
	}
	return ""
}

func (c *proxyStreamErrorCapture) Finalize() string {
	if c == nil || c.payload != "" {
		if c == nil {
			return ""
		}
		return c.payload
	}
	if frameMayCarryStreamError(c.pending) {
		if payload := extractStreamErrorEvent(c.pending); payload != "" {
			c.payload = payload
		}
	}
	c.pending = nil
	return c.payload
}

func nextSSEFrameEnd(data []byte) int {
	lf := bytes.Index(data, []byte("\n\n"))
	crlf := bytes.Index(data, []byte("\r\n\r\n"))
	switch {
	case lf < 0 && crlf < 0:
		return -1
	case crlf < 0 || (lf >= 0 && lf < crlf):
		return lf + len("\n\n")
	default:
		return crlf + len("\r\n\r\n")
	}
}

func markProxyProviderStreamError(logCtx *RequestLogContext, statusCode int, payload string) {
	if logCtx == nil || payload == "" {
		return
	}
	logCtx.Status = "failure"
	logCtx.HTTPStatus = statusCode
	logCtx.ErrorMsg = payload
}

// resolveCapturedProviderStreamError finalizes one or more bounded stream
// observers after every upstream byte has already been forwarded. Raw provider
// and transformed-output observers may both be supplied; the first decoded
// terminal payload is retained as the authoritative error detail. A pre-existing
// transport/client error keeps precedence for StreamOutcome classification.
func resolveCapturedProviderStreamError(
	logCtx *RequestLogContext,
	statusCode int,
	streamErr error,
	captures ...*proxyStreamErrorCapture,
) error {
	if statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return streamErr
	}

	payload := ""
	for _, capture := range captures {
		if capture == nil {
			continue
		}
		if captured := capture.Finalize(); payload == "" && captured != "" {
			payload = captured
		}
	}
	if payload == "" {
		return streamErr
	}

	markProxyProviderStreamError(logCtx, statusCode, payload)
	if streamErr == nil {
		return proxyProviderStreamError{payload: payload}
	}
	return streamErr
}

// writeProxyResponse writes raw upstream proxy response to client.
// Respects the client's Accept-Encoding header to compress the response appropriately.
// Used by both primary proxy path and fallback retry path to avoid duplication.
func (p *Proxy) writeProxyResponse(w http.ResponseWriter, resp *ProxyResponse, clientReq *http.Request, cred *config.CredentialConfig, modelID string, logCtx *RequestLogContext, usageOptions ...converter.TokenUsageExtractionOptions) {
	if resp == nil {
		return
	}

	tokenUsageOptions := converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
	if len(usageOptions) > 0 {
		tokenUsageOptions = usageOptions[0]
	}

	credName := ""
	if cred != nil {
		credName = cred.Name
	}

	responseBody, responseBodyChanged, responseBodyMasked := clientResponseBodyForCredential(resp.StatusCode, resp.Body, cred, modelID)
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		endpoint := endpointFromRequest(clientReq)
		if logCtx != nil && logCtx.Request != nil {
			endpoint = endpointFromRequest(logCtx.Request)
		}
		normalizedModelBody := normalizeSuccessfulResponseModel(
			responseBody,
			endpoint,
			clientVisibleResponseModel(logCtx, modelID),
		)
		if !bytes.Equal(normalizedModelBody, responseBody) {
			responseBody = normalizedModelBody
			responseBodyChanged = true
		}
		if normalizedBody, changed := modelutils.NormalizeCompletionUsage(
			responseBody,
			clientVisibleResponseModel(logCtx, modelID),
		); changed {
			responseBody = normalizedBody
			responseBodyChanged = true
		}
	}
	logCtx.captureClientResponseID(responseBody)

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

	p.copyResponseHeaders(w, resp.Headers, cred, responseBodyChanged)
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		p.markAudioUsageContractForClient(w, logCtx, tokenUsageOptions.AudioInputIncludesCachedAudio)
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
	usageOptions ...converter.TokenUsageExtractionOptions,
) (*converter.TokenUsage, error) {
	if resp == nil || resp.StreamBody == nil {
		return nil, nil
	}

	credName := ""
	if cred != nil {
		credName = cred.Name
	}

	detectProviderStreamError := resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices
	providerStreamError := &proxyStreamErrorCapture{}
	streamBody := io.ReadCloser(resp.StreamBody)
	if detectProviderStreamError {
		streamBody = readerReadCloser{
			Reader: io.TeeReader(streamBody, proxyStreamErrorObserver{capture: providerStreamError}),
			Closer: streamBody,
		}
	}
	normalizeUsageStream := false
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		streamBody, normalizeUsageStream = modelutils.NewUsageNormalizingReadCloser(streamBody, modelID)
	}
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		streamBody = promanutils.NewSanitizingSSEReadCloser(streamBody, modelID)
	}
	normalizeResponseModel := resp.StatusCode >= http.StatusOK &&
		resp.StatusCode < http.StatusMultipleChoices && streamRouteReturnsModel(logCtx)
	defer func() {
		if closeErr := streamBody.Close(); closeErr != nil {
			p.logger.WarnContext(clientReq.Context(), "Failed to close proxy streaming response body", "error", closeErr)
		}
	}()

	tokenUsageOptions := converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
	if len(usageOptions) > 0 {
		tokenUsageOptions = usageOptions[0]
	}

	p.copyResponseHeaders(w, resp.Headers, cred, normalizeUsageStream || normalizeResponseModel)
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		p.markAudioUsageContractForClient(w, logCtx, tokenUsageOptions.AudioInputIncludesCachedAudio)
	}
	setSuccessfulSSEHeaders(w.Header(), resp.StatusCode)

	w.WriteHeader(resp.StatusCode)

	var lastUsage *converter.TokenUsage
	completion := p.newCompletionTokenAccumulator(tokenizerModelID)
	var payloadBuf [][]byte
	onChunk := func(chunk []byte) {
		// One split per chunk (plan item C), reused for both usage
		// extraction and the completion accumulator.
		payloadBuf = splitSSEPayloads(chunk, payloadBuf)
		if chunkMayCarryTokenUsage(chunk) {
			if usage := extractTokenUsageFromPayloads(payloadBuf, tokenUsageOptions); usage != nil {
				// Merge rather than replace: a web-search-only chunk (no
				// prompt/completion tokens) arriving separately from the
				// usage chunk must not have its WebSearchRequests clobbered
				// back to zero by a later chunk's usage read.
				if lastUsage == nil {
					lastUsage = &converter.TokenUsage{}
				}
				lastUsage.MergeNonZero(usage)
				if logCtx != nil {
					logCtx.UsageSource = "provider"
				}
			}
		}
		completion.AddPayloads(payloadBuf)
	}

	buildFallbackUsage := func() *converter.TokenUsage {
		if lastUsage != nil {
			return lastUsage
		}
		if tokens := completion.TokenCount(); tokens > 0 {
			if logCtx != nil && logCtx.UsageSource == "" {
				logCtx.UsageSource = "estimated"
			}
			return &converter.TokenUsage{CompletionTokens: tokens}
		}
		return nil
	}
	finalize := func(streamErr error) (*converter.TokenUsage, error) {
		if detectProviderStreamError {
			streamErr = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, streamErr, providerStreamError)
		}
		updateProxyStreamOutcome(logCtx, streamErr)
		return buildFallbackUsage(), streamErr
	}
	clientReader := normalizeSuccessfulResponseModelStream(
		streamBody,
		resp.StatusCode,
		logCtx,
		modelID,
	)

	if _, ok := w.(http.Flusher); ok {
		err := p.streamToClient(
			clientReq.Context(),
			w,
			clientReader,
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
				clientReader,
				onChunk,
				credName,
			)
		}
		return finalize(err)
	}

	// Non-flushing fallback: copy as-is (token usage cannot be parsed reliably here).
	var responseIDScanner clientResponseIDScanner
	observedReader := io.TeeReader(clientReader, clientResponseIDObserver{scanner: &responseIDScanner, logCtx: logCtx})
	if _, err := io.Copy(w, observedReader); err != nil {
		if isClientDisconnectError(err) {
			p.recordAbortedRequest(credName, endpointFromRequest(clientReq), modelID)
		}
		return finalize(err)
	}
	return finalize(nil)
}

func updateProxyStreamOutcome(logCtx *RequestLogContext, err error) {
	if logCtx == nil {
		return
	}
	if err != nil {
		markStreamFailure(logCtx, err)
		return
	}
	if logCtx.StreamOutcome == "" {
		logCtx.StreamOutcome = "completed"
	}
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
