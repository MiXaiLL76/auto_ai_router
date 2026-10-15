package proxy

// Simulated non-stream responses for the force_stream_for_ttft toggle.
//
// When the toggle is on and a client asks for a non-streaming response, the
// outbound request is sent with "stream": true (see applyForceStream), the
// upstream answers with SSE, and handleSimulatedNonStream consumes that stream,
// measures TTFT from the first real content delta (CompletionStartTime — the
// same detector the live streaming path uses), reassembles the provider's
// Chat-Completions-shaped SSE back into a single non-stream JSON body
// (reassembleChatCompletion), then runs the exact same client-format
// conversions and logging the plain non-stream path would have used
// (ChatToMessages / ChatToResponse / normalization / finalizeStreamingLog).
//
// Scope is deliberately chat-shaped requests only: /v1/chat/completions,
// /v1/messages and converted /v1/responses. Native and passthrough Responses
// API requests (provider-native wire formats without an OpenAI-shaped
// reassembly here) are never force-streamed, so their behavior is unchanged.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

// simulatedNonStreamSupported reports whether force_stream_for_ttft can serve
// this request: the response must be reassemblable into the non-stream JSON the
// client expects. Native and passthrough Responses API formats are excluded —
// their provider-native wire formats have no reconstruction path here yet.
func (p *Proxy) simulatedNonStreamSupported(prepared *orchestratedRequest) bool {
	return !prepared.nativeResponses && !prepared.passthroughResponses && !prepared.convertedToResponses
}

// handleSimulatedNonStream consumes an upstream SSE response that was produced
// because applyForceStream set "stream": true, reassembles it into the non-stream
// JSON body the client actually asked for, and writes that body to w. It owns
// the whole client-facing outcome: headers, compression, spend logging and the
// TTFT metric. The caller must return immediately afterwards.
func (p *Proxy) handleSimulatedNonStream(
	w http.ResponseWriter,
	r *http.Request,
	resp *http.Response,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	logCtx *RequestLogContext,
	saveResponseFn func(*responses.Response),
	targetURL string,
	requestBody []byte,
) {
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			p.logger.WarnContext(respCtx(resp), "Failed to close simulated non-stream response body", "error", closeErr)
		}
	}()

	publicModel := clientVisibleResponseModel(logCtx, modelID)
	converterModelID := modelID
	if logCtx != nil && logCtx.RealModelID != "" {
		converterModelID = logCtx.RealModelID
	}
	conv := converter.New(cred.EffectiveProviderType(), converter.RequestMode{
		ModelID:        converterModelID,
		DisplayModelID: publicModel,
		IsStreaming:    true,
	})
	providerLabel := streamingProviderLabel(cred, prepared.convertedMessages)
	tokenUsageOptions := tokenUsageExtractionOptionsForResponse(cred, resp.Header)

	pr, pw := io.Pipe()
	defer func() { _ = pr.Close() }()
	var totalTokens int
	completion := p.newCompletionTokenAccumulator(modelID, logCtx)
	// Capture the last real chunk for usage extraction (same hybrid approach as
	// handleTransformedStreaming).
	var lastChunk []byte
	rawProviderStreamError := &proxyStreamErrorCapture{}
	outputStreamError := &proxyStreamErrorCapture{}
	providerReader := io.Reader(resp.Body)
	// Same as handleProviderStreaming's default branch: OpenAI-compatible
	// providers get their usage deltas normalized before conversion.
	if conv.IsPassthrough() {
		providerReader, _ = modelutils.NewUsageNormalizingReadCloser(resp.Body, publicModel)
	}
	providerReader = io.TeeReader(providerReader, proxyStreamErrorObserver{capture: rawProviderStreamError})

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		transformer := p.simulatedTransformer(prepared, cred, conv, publicModel)
		err := transformer(providerReader, modelID, &tokenCapturingWriter{
			writer:     pw,
			tokens:     &totalTokens,
			completion: completion,
			logger:     p.logger,
			onChunk: func(chunk []byte, payloads [][]byte, hasUsage bool) {
				outputStreamError.Observe(chunk)
				if logCtx != nil && hasUsage {
					// The same header-derived options the non-streaming path's
					// token extraction uses (Kimi/Moonshot cache-write TTL
					// splits, audio-cached-input flags) — a zero-value options
					// struct would silently drop those from the billing merge.
					if usage := extractTokenUsageFromPayloads(payloads, tokenUsageOptions); usage != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						// Merge rather than replace, same as handleTransformedStreaming:
						// a web-search-only chunk's fields must not be clobbered back to
						// zero by a later chunk's usage read that doesn't carry them.
						logCtx.TokenUsage.MergeNonZero(usage)
					}
				}
				rememberLastStreamDataChunk(&lastChunk, chunk)
			},
		})
		if err != nil {
			p.logger.ErrorContext(respCtx(resp), "Simulated non-stream transform goroutine error",
				"provider", providerLabel, "error", err)
			_ = pw.CloseWithError(fmt.Errorf("%s transform: %w", providerLabel, err))
		} else {
			_ = pw.Close()
		}
	}()

	// Consume the normalized Chat SSE, stamping TTFT on the first real content
	// delta with the same incremental scanner the live streaming path uses.
	clientReader := normalizeSuccessfulResponseModelStream(
		promanutils.NewSanitizingSSEReader(pr, publicModel),
		resp.StatusCode,
		logCtx,
		modelID,
	)
	var streamBuf bytes.Buffer
	streamBuf.Grow(64 * 1024)
	var ttftScan ttftScanState
	ttftPending := logCtx != nil && logCtx.CompletionStartTime.IsZero()
	readBuf := make([]byte, 8192)
	overflow := false
	var lastReadErr error
	for {
		n, readErr := clientReader.Read(readBuf)
		if n > 0 {
			chunk := readBuf[:n]
			if int64(streamBuf.Len()+n) > p.maxResponseBodySize {
				overflow = true
				break
			}
			streamBuf.Write(chunk)
			if ttftPending {
				if ttftScan.observe(chunk) {
					logCtx.CompletionStartTime = time.Now()
					ttftPending = false
				}
			}
		}
		if readErr != nil {
			lastReadErr = readErr
			break
		}
	}
	_ = pr.Close()
	wg.Wait()

	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
	}

	// A genuine mid-stream error (transport drop, converter failure) must be
	// classified as a failure, same as the live streaming path: EOF means the
	// provider finished its stream, anything else means it did not.
	readErrForResolve := lastReadErr
	if errors.Is(readErrForResolve, io.EOF) {
		readErrForResolve = nil
	}

	var streamErr error
	var chatBody []byte
	if streamErr = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, readErrForResolve, rawProviderStreamError, outputStreamError); streamErr != nil {
		markStreamFailure(logCtx, streamErr)
	} else if overflow {
		streamErr = ErrResponseBodyTooLarge
		markStreamFailure(logCtx, streamErr)
	} else {
		chatBody = reassembleChatCompletion(streamBuf.Bytes())
		if len(chatBody) == 0 {
			streamErr = fmt.Errorf("simulated non-stream response: no reassemblable content in upstream stream")
			markStreamFailure(logCtx, streamErr)
		}
	}

	if streamErr != nil {
		p.logger.ErrorContext(respCtx(resp), "Simulated non-stream response failed",
			"credential", cred.Name, "provider", providerLabel, "model", modelID,
			"error", streamErr, "request_id", logCtx.RequestID)
		p.setPromptTokensEstimate(logCtx, requestBody, realModelID)
		p.finalizeStreamingLog(logCtx, logTokens, lastChunk, providerLabel, resp.StatusCode, !tokenUsageOptions.AudioInputIncludesCachedAudio)
		p.writeSimulatedError(w, logCtx, http.StatusBadGateway, targetURL)
		return
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(cred.Name, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(cred.Name, modelID, logTokens)
		}
	}

	// Client-format conversion chain — mirrors the plain non-stream path
	// (proxy.go "Process final response"): the reassembled body is OpenAI
	// Chat-shaped, so the same converters run on it.
	finalBody := chatBody
	if conv.IsPassthrough() {
		finalBody = rewriteResponseModelAlias(finalBody, realModelID, modelID)
		finalBody = openai.StripServerToolCalls(finalBody)
	}
	if prepared.convertedMessages {
		messagesBody, convErr := anthropicconv.ChatToMessages(finalBody, prepared.messagesMetadata,
			tokenUsageOptions.CacheWriteTTLHeader5mTokens, tokenUsageOptions.CacheWriteTTLHeader1hTokens)
		if convErr != nil {
			p.logger.ErrorContext(respCtx(resp), "Failed to convert to Messages API format (simulated non-stream)",
				"credential", cred.Name, "model", modelID, "error", convErr,
				"request_id", logCtx.RequestID)
		} else {
			finalBody = messagesBody
			// Same contract reset the plain non-stream path applies after any
			// client-format conversion (proxy.go "Process final response").
			tokenUsageOptions.AudioInputIncludesCachedAudio = false
		}
	} else if prepared.convertedResp {
		responsesBody, convErr := responses.ChatToResponse(
			finalBody,
			responses.WithAudioInputIncludesCachedAudio(tokenUsageOptions.AudioInputIncludesCachedAudio),
		)
		if convErr != nil {
			p.logger.ErrorContext(respCtx(resp), "Failed to convert to Responses API format (simulated non-stream)",
				"credential", cred.Name, "model", modelID, "error", convErr,
				"request_id", logCtx.RequestID)
		} else {
			var respObj responses.Response
			if err := json.Unmarshal(responsesBody, &respObj); err == nil {
				applyResponsesMetadata(&respObj, prepared.responsesMetadata)
				if enriched, marshalErr := json.Marshal(&respObj); marshalErr == nil {
					responsesBody = enriched
				}
				if saveResponseFn != nil {
					saveResponseFn(&respObj)
				}
			} else if saveResponseFn != nil {
				var r2 responses.Response
				if json.Unmarshal(responsesBody, &r2) == nil {
					saveResponseFn(&r2)
				}
			}
			finalBody = responsesBody
			// Same contract reset as the plain non-stream path.
			tokenUsageOptions.AudioInputIncludesCachedAudio = false
		}
	}

	if normalizedBody := normalizeSuccessfulResponseModel(finalBody, endpointFromLogContext(logCtx), publicModel); !bytes.Equal(normalizedBody, finalBody) {
		finalBody = normalizedBody
		for key := range resp.Header {
			if isRepresentationIntegrityHeader(key) {
				resp.Header.Del(key)
			}
		}
	}
	if normalizedBody, changed := modelutils.NormalizeCompletionUsage(finalBody, publicModel); changed {
		finalBody = normalizedBody
		dropRepresentationIntegrityHeaders(resp.Header)
	}

	// web_search travels in the reassembled body unless the request hides it,
	// mirroring the plain non-stream path (strip only when HideWebSearchResults).
	if logCtx.HideWebSearchResults {
		if stripped, ok := stripWebSearchResults(finalBody); ok {
			finalBody = stripped
			dropRepresentationIntegrityHeaders(resp.Header)
		}
	}
	// Correlate billing/analytics rows with the provider's own completion id,
	// same as the plain non-stream and streaming paths.
	logCtx.captureClientResponseID(finalBody)
	audioInputAlreadyExcludesCachedAudio := !tokenUsageOptions.AudioInputIncludesCachedAudio

	// Stream-domain logging: usage extraction from the stream, spend log, TTFT
	// metric and kafka event — all keyed off CompletionStartTime / lastChunk.
	p.setPromptTokensEstimate(logCtx, requestBody, realModelID)
	p.finalizeStreamingLog(logCtx, logTokens, lastChunk, providerLabel, resp.StatusCode, audioInputAlreadyExcludesCachedAudio)

	logCtx.RequestCompleted = true
	p.setSessionBinding(logCtx.SessionID, modelID, cred.Name)

	// Client-facing headers: copy the provider's, then fix the ones that are
	// meaningless on a reassembled body. The upstream answered SSE, so its
	// Content-Type must not leak to a client that asked for JSON.
	p.copyResponseHeaders(w, resp.Header, cred, false)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Del("X-Accel-Buffering")
	if tokenUsageOptions.AudioInputIncludesCachedAudio {
		p.markAudioUsageIncludesCachedForClient(w, logCtx)
	} else {
		p.markAudioUsageExcludesCachedForClient(w, logCtx)
	}
	p.setCredentialResponseHeader(w, logCtx, cred.Name)

	p.writeSimulatedBody(w, r, resp, cred, modelID, logCtx, finalBody)
}

// writeSimulatedBody writes a buffered non-stream body to the client, mirroring
// the plain non-stream write path (compression negotiation, Content-Length,
// deadlines, aborted-request accounting).
func (p *Proxy) writeSimulatedBody(
	w http.ResponseWriter,
	r *http.Request,
	resp *http.Response,
	cred *config.CredentialConfig,
	modelID string,
	logCtx *RequestLogContext,
	body []byte,
) {
	acceptEncoding := r.Header.Get("Accept-Encoding")
	acceptedEncodings := ParseAcceptEncoding(acceptEncoding)
	targetEncoding := SelectBestEncoding(acceptedEncodings)

	outputBody := body
	if targetEncoding != "identity" && len(body) > 0 {
		compressedBody, usedEncoding, compErr := CompressBody(body, targetEncoding)
		if compErr != nil {
			p.logger.WarnContext(r.Context(), "Failed to compress response body",
				"credential", cred.Name, "encoding", targetEncoding, "error", compErr)
		} else {
			outputBody = compressedBody
			w.Header().Set("Content-Encoding", usedEncoding)
		}
	}

	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(outputBody)))

	rc := http.NewResponseController(w)
	_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
	w.WriteHeader(resp.StatusCode)

	_ = rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
	if _, err := p.streamResponseBody(w, bytes.NewReader(outputBody)); err != nil {
		if isClientDisconnectError(err) {
			p.logger.DebugContext(r.Context(), "Client disconnected during simulated non-stream response body copy", "error", err)
			p.recordAbortedRequest(cred.Name, r.URL.Path, modelID)
		} else {
			p.logger.ErrorContext(r.Context(), "Failed to copy simulated non-stream response body", "error", err)
		}
	}
}

// writeSimulatedError writes a masked JSON error for a failed simulated
// non-stream response (stream-level provider error or reassembly failure).
func (p *Proxy) writeSimulatedError(w http.ResponseWriter, logCtx *RequestLogContext, statusCode int, targetURL string) {
	body := maskedUpstreamErrorBody(statusCode, logCtx.RequestID)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
	w.WriteHeader(statusCode)
	_, _ = w.Write(body)
	logCtx.Status = "failure"
	logCtx.HTTPStatus = statusCode
	logCtx.TargetURL = targetURL
}

// streamingProviderLabel names the provider for finalizeStreamingLog's usage
// extractor selection — the same labels the live streaming handlers pass.
func streamingProviderLabel(cred *config.CredentialConfig, convertedMessages bool) string {
	if convertedMessages {
		return "messages"
	}
	switch cred.EffectiveProviderType() {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return "Vertex AI"
	case config.ProviderTypeAnthropic:
		return "Anthropic"
	case config.ProviderTypeCometAPI:
		return "Comet API"
	case config.ProviderTypeProMan:
		return "ProMan"
	case config.ProviderTypeBedrock:
		return "Bedrock"
	default:
		return "openai"
	}
}

// simulatedTransformer returns the provider-SSE → Chat-Completions-SSE
// transformer for the simulated path. It mirrors the per-provider streaming
// handlers: Anthropic-family providers sanitize input before conversion (their
// converters can fold a raw error message into delta.content, which the
// output-side sanitizer never sees), OpenAI-compatible providers are passed
// through unchanged, everything else goes through the provider converter.
func (p *Proxy) simulatedTransformer(
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	conv *converter.ProviderConverter,
	publicModel string,
) func(io.Reader, string, io.Writer) error {
	identity := func(r io.Reader, _ string, w io.Writer) error {
		_, err := io.Copy(w, r)
		return err
	}
	streamTo := func(r io.Reader, _ string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	anthropicStreamTo := func(r io.Reader, _ string, w io.Writer) error {
		r = promanutils.NewSanitizingSSEReader(r, publicModel)
		return conv.StreamTo(r, w)
	}

	if prepared.convertedMessages && conv.IsPassthrough() {
		return identity
	}
	switch cred.EffectiveProviderType() {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini, config.ProviderTypeBedrock:
		return streamTo
	case config.ProviderTypeAnthropic, config.ProviderTypeCometAPI, config.ProviderTypeProMan:
		return anthropicStreamTo
	default:
		return identity
	}
}

// ---------------------------------------------------------------------------
// Chat-Completions SSE → single non-stream JSON reassembly
// ---------------------------------------------------------------------------

// simToolCallState accumulates one tool call across streaming deltas.
type simToolCallState struct {
	id        string
	name      strings.Builder
	arguments strings.Builder
}

// simChoiceState accumulates one choice across streaming deltas.
type simChoiceState struct {
	index        int
	role         string
	content      strings.Builder
	refusal      strings.Builder
	reasoning    strings.Builder
	images       []openai.ImageData
	toolCalls    map[int]*simToolCallState
	toolOrder    []int
	finishReason *string
}

func newSimChoiceState(index int) *simChoiceState {
	return &simChoiceState{index: index, toolCalls: make(map[int]*simToolCallState)}
}

// reassembleChatCompletion folds an OpenAI Chat-Completions SSE body (as
// produced by the provider converters after conv.StreamTo) into a single
// Chat-Completions non-stream JSON body. Returns nil when nothing reassemblable
// was found. Deltas are accumulated per choice index; tool-call id/name/
// arguments are stitched from their fragment chunks; id/model/created/usage are
// taken from the chunk that carried them (usage: the last one, matching
// OpenAI's final-usage-chunk convention).
func reassembleChatCompletion(stream []byte) []byte {
	var (
		id, model, systemFingerprint, serviceTier string
		createdAt                                 int64
		usage                                     json.RawMessage
		webSearch                                 json.RawMessage
		choices                                   = map[int]*simChoiceState{}
		choiceOrder                               []int
	)

	for _, line := range strings.Split(string(stream), "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		// Top-level "web_search" (search citations) rides in a chunk of its
		// own, outside the OpenAIStreamingChunk struct (no such field there).
		// The cheap substring prefilter avoids an extra unmarshal on ordinary
		// content chunks; a false positive (the key in delta text) costs one
		// harmless unmarshal that finds nothing.
		if bytes.Contains([]byte(payload), []byte(`"web_search"`)) {
			var withSearch struct {
				WebSearch json.RawMessage `json:"web_search"`
			}
			if err := json.Unmarshal([]byte(payload), &withSearch); err == nil && len(withSearch.WebSearch) > 0 && string(withSearch.WebSearch) != "null" {
				webSearch = withSearch.WebSearch
			}
		}
		if chunk.ID != "" {
			id = chunk.ID
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.Created != 0 {
			createdAt = chunk.Created
		}
		if chunk.SystemFingerprint != "" {
			systemFingerprint = chunk.SystemFingerprint
		}
		if chunk.ServiceTier != "" {
			serviceTier = chunk.ServiceTier
		}
		for _, c := range chunk.Choices {
			st := choices[c.Index]
			if st == nil {
				st = newSimChoiceState(c.Index)
				choices[c.Index] = st
				choiceOrder = append(choiceOrder, c.Index)
			}
			if st.role == "" && c.Delta.Role != "" {
				st.role = c.Delta.Role
			}
			if c.Delta.Content != "" {
				st.content.WriteString(c.Delta.Content)
			}
			if c.Delta.Refusal != "" {
				st.refusal.WriteString(c.Delta.Refusal)
			}
			if c.Delta.ReasoningContent != "" {
				st.reasoning.WriteString(c.Delta.ReasoningContent)
			}
			if len(c.Delta.Images) > 0 {
				// Gemini image generation streams its images as deltas
				// (vertex/streaming.go) — mirror the non-stream message shape.
				st.images = append(st.images, c.Delta.Images...)
			}
			for _, tc := range c.Delta.ToolCalls {
				idx := tc.Index
				t := st.toolCalls[idx]
				if t == nil {
					t = &simToolCallState{}
					st.toolCalls[idx] = t
					st.toolOrder = append(st.toolOrder, idx)
				}
				if t.id == "" && tc.ID != "" {
					t.id = tc.ID
				}
				if tc.Function != nil {
					if tc.Function.Name != "" {
						t.name.WriteString(tc.Function.Name)
					}
					if tc.Function.Arguments != "" {
						t.arguments.WriteString(tc.Function.Arguments)
					}
				}
			}
			if c.FinishReason != nil {
				st.finishReason = c.FinishReason
			}
		}
		if chunk.Usage != nil {
			if b, err := json.Marshal(chunk.Usage); err == nil {
				usage = b
			}
		}
	}

	if len(choiceOrder) == 0 {
		return nil
	}
	sort.Ints(choiceOrder)

	type simToolCall struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	type simMessage struct {
		Role             string             `json:"role"`
		Content          string             `json:"content"`
		Refusal          *string            `json:"refusal,omitempty"`
		ReasoningContent *string            `json:"reasoning_content,omitempty"`
		Images           []openai.ImageData `json:"images,omitempty"`
		ToolCalls        []simToolCall      `json:"tool_calls,omitempty"`
	}
	type simChoice struct {
		Index   int        `json:"index"`
		Message simMessage `json:"message"`
		// finish_reason is omitted (never null) when the stream ended without
		// ever carrying one: the native non-stream contract is a string.
		FinishReason *string `json:"finish_reason,omitempty"`
	}
	type simCompletion struct {
		ID                string          `json:"id"`
		Object            string          `json:"object"`
		Created           int64           `json:"created"`
		Model             string          `json:"model"`
		SystemFingerprint string          `json:"system_fingerprint,omitempty"`
		ServiceTier       string          `json:"service_tier,omitempty"`
		WebSearch         json.RawMessage `json:"web_search,omitempty"`
		Choices           []simChoice     `json:"choices"`
		Usage             json.RawMessage `json:"usage,omitempty"`
	}

	out := simCompletion{
		ID:                id,
		Object:            "chat.completion",
		Created:           createdAt,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		ServiceTier:       serviceTier,
		WebSearch:         webSearch,
		Usage:             usage,
	}
	if out.ID == "" {
		out.ID = "chatcmpl-simulated"
	}
	for _, idx := range choiceOrder {
		st := choices[idx]
		msg := simMessage{
			Role:    st.role,
			Content: st.content.String(),
			Images:  st.images,
		}
		if msg.Role == "" {
			msg.Role = "assistant"
		}
		if refusal := st.refusal.String(); refusal != "" {
			msg.Refusal = &refusal
		}
		if reasoning := st.reasoning.String(); reasoning != "" {
			msg.ReasoningContent = &reasoning
		}
		if len(st.toolOrder) > 0 {
			sort.Ints(st.toolOrder)
			for _, tidx := range st.toolOrder {
				t := st.toolCalls[tidx]
				tc := simToolCall{ID: t.id, Type: "function"}
				tc.Function.Name = t.name.String()
				tc.Function.Arguments = t.arguments.String()
				msg.ToolCalls = append(msg.ToolCalls, tc)
			}
		}
		out.Choices = append(out.Choices, simChoice{
			Index:        st.index,
			Message:      msg,
			FinishReason: st.finishReason,
		})
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil
	}
	return b
}
