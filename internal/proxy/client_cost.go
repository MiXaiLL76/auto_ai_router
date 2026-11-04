package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// ==================== Client-facing cost exposure ====================
//
// Organizations whose policy sets response_cost_in_body (OrganizationPolicy.
// ReturnsCostInBody) opt into having AIR add the final, margin-adjusted
// per-request cost to the "usage" object of the response AIR returns to
// their own clients -- the same usage.cost convention OpenRouter uses.
// Independent of response_cost_mode (setResponseCostHeader's X-Total-Cost/
// X-Cost-Breakdown headers): body injection also works for streaming, since
// the cost becomes known exactly when the final usage chunk is read, not
// only after the whole response is already header-committed. This is purely
// additive: the field is only ever added, never required, and any failure
// to compute or inject it leaves the response byte-for-byte as it would
// otherwise have been.

// shouldExposeCostToClient reports whether logCtx's organization opted into
// usage.cost in client-facing responses.
func shouldExposeCostToClient(logCtx *RequestLogContext) bool {
	return logCtx != nil && logCtx.OrganizationPolicy.ReturnsCostInBody()
}

// clientFacingCost resolves the billing price and computes the final,
// margin-adjusted cost for usage, the same way logSpendToLiteLLMDB does for
// the spend log and the X-Total-Cost header (resolveBillingPrice ->
// CalculateCostsWithResolver -> applyCostMargin). It never mutates logCtx's
// cached billing-price fields beyond the normal resolveBillingPrice caching,
// so calling it ahead of logSpendToLiteLLMDB (as the streaming path must)
// does not change what gets logged or billed.
func (p *Proxy) clientFacingCost(logCtx *RequestLogContext, usage *converter.TokenUsage) (float64, bool) {
	if p == nil || logCtx == nil || usage == nil || usage.IsZero() {
		return 0, false
	}
	modelPrice := logCtx.ModelPrice
	if modelPrice == nil {
		_, modelPrice = p.resolveBillingPrice(logCtx, logCtx.PublicModelID, logCtx.ModelID, logCtx.RealModelID)
	}
	if modelPrice == nil {
		return 0, false
	}
	costs := modelPrice.CalculateCostsWithResolver(usage, p.billingPriceResolver(logCtx))
	if costs == nil {
		return 0, false
	}
	logCtx.applyCostMargin(costs)
	return costs.TotalCost, true
}

// injectCostIntoJSONUsage adds "cost" to the top-level "usage" object of a
// non-streaming JSON response body. Returns the original body unchanged
// (ok=false) for any shape it doesn't recognize -- a response with no
// "usage" key (e.g. some error bodies, binary/audio responses that never
// reach here as JSON) or malformed JSON is left exactly as-is.
func injectCostIntoJSONUsage(body []byte, cost float64) ([]byte, bool) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		return body, false
	}
	usageRaw, ok := doc["usage"]
	if !ok || bytes.Equal(bytes.TrimSpace(usageRaw), []byte("null")) {
		return body, false
	}
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usage); err != nil {
		return body, false
	}
	costJSON, err := json.Marshal(cost)
	if err != nil {
		return body, false
	}
	usage["cost"] = costJSON
	newUsage, err := json.Marshal(usage)
	if err != nil {
		return body, false
	}
	doc["usage"] = newUsage
	newBody, err := json.Marshal(doc)
	if err != nil {
		return body, false
	}
	return newBody, true
}

// injectCostIntoSSEUsageChunk adds "cost" to the "usage" object of a single
// SSE chunk, rebuilt as "data: <json>\n\n" the same way AIR's own stream
// writer frames chunks (see websocket_native.go). payloads is the chunk's
// already-split "data:" payload list (see splitSSEPayloads) -- only the
// unambiguous single-payload case is handled; anything else (no payload,
// several batched together, no "usage" key, malformed JSON) is left
// untouched so a corner case can never corrupt the stream.
func injectCostIntoSSEUsageChunk(payloads [][]byte, cost float64) ([]byte, bool) {
	if len(payloads) != 1 {
		return nil, false
	}
	newPayload, ok := injectCostIntoJSONUsage(payloads[0], cost)
	if !ok {
		return nil, false
	}
	chunk := make([]byte, 0, len(sseDataPrefix)+1+len(newPayload)+2)
	chunk = append(chunk, sseDataPrefix...)
	chunk = append(chunk, ' ')
	chunk = append(chunk, newPayload...)
	chunk = append(chunk, '\n', '\n')
	return chunk, true
}

// ==================== Raw passthrough streaming (chat-completion-shaped) ====================
//
// handleStreamingWithTokens forwards a provider's SSE bytes to the client as
// soon as they are read off the socket, deliberately without buffering past
// a complete line (TTFT) -- the chunk boundaries streamToClient reads are
// socket-read boundaries, not SSE-line boundaries, so a usage payload can be
// split across several reads. newCostInjectingSSEReader gives that path the
// same per-line rewrite its model-name normalization already does (see
// responseModelStreamReader in chat_response_compat.go): read exactly one
// physical line at a time via bufio.Reader.ReadSlice('\n'), rewrite only the
// one line that turns out to carry the stream's usage object, and pass every
// other line through byte-for-byte. Only ever constructed for a request
// whose organization opted into usage.cost (callers check
// shouldExposeCostToClient first) -- every other request's passthrough
// never touches this reader at all, so its latency is unchanged.

// newCostInjectingSSEReader wraps reader so the one SSE line carrying the
// stream's final usage object gets "cost" added before reaching the client.
// opts must match whatever TokenUsageExtractionOptions the caller's own
// usage bookkeeping uses, so the usage this computes cost from agrees with
// what finalizeStreamingLog will log.
func newCostInjectingSSEReader(reader io.Reader, proxy *Proxy, logCtx *RequestLogContext, opts converter.TokenUsageExtractionOptions) io.Reader {
	if reader == nil || proxy == nil || logCtx == nil {
		return reader
	}
	return &costInjectingSSEReader{
		source:       bufio.NewReader(reader),
		proxy:        proxy,
		logCtx:       logCtx,
		opts:         opts,
		maxLineBytes: maxSSEModelRewriteLineBytes,
	}
}

type costInjectingSSEReader struct {
	source       *bufio.Reader
	proxy        *Proxy
	logCtx       *RequestLogContext
	opts         converter.TokenUsageExtractionOptions
	pending      []byte
	pendingErr   error
	line         []byte
	passthrough  bool
	maxLineBytes int
	// done is set once cost has been injected into a line -- further lines
	// (typically just "data: [DONE]") are passed through unexamined.
	done       bool
	payloadBuf [][]byte
}

func (r *costInjectingSSEReader) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		if r.pendingErr != nil {
			err := r.pendingErr
			r.pendingErr = nil
			return 0, err
		}
		fragment, err := r.source.ReadSlice('\n')
		if len(fragment) == 0 {
			return 0, err
		}
		if r.done || r.passthrough {
			r.pending = append(r.pending, fragment...)
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
				r.pendingErr = err
			}
			continue
		}
		maxLineBytes := r.maxLineBytes
		if maxLineBytes <= 0 {
			maxLineBytes = maxSSEModelRewriteLineBytes
		}
		if len(r.line)+len(fragment) > maxLineBytes {
			// Pathologically long line (bigger than any real usage payload
			// should ever be) -- fall back to plain passthrough for it
			// rather than buffering without bound.
			r.pending = append(r.pending, r.line...)
			r.pending = append(r.pending, fragment...)
			r.line = r.line[:0]
			if errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = true
			} else {
				r.pendingErr = err
			}
			continue
		}
		r.line = append(r.line, fragment...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		r.pending = r.injectCostIntoLine(r.line)
		r.line = nil
		r.pendingErr = err
	}

	n := copy(dst, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

// injectCostIntoLine returns line unchanged unless it is the single SSE data
// line carrying the stream's usage object, in which case it returns a
// rewritten copy with "cost" added. usage is extracted straight from this
// line (not read back from logCtx.TokenUsage, which the caller's own
// onLine/onChunk observer -- running after Read returns this line -- hasn't
// merged it into yet) and merged into a throwaway copy of whatever
// logCtx.TokenUsage already holds from earlier lines (e.g. a web-search-only
// chunk), so the cost reflects the full request even when usage arrives
// split across lines.
func (r *costInjectingSSEReader) injectCostIntoLine(line []byte) []byte {
	if !chunkMayCarryTokenUsage(line) {
		return line
	}
	r.payloadBuf = splitSSEPayloads(line, r.payloadBuf)
	usage := extractTokenUsageFromPayloads(r.payloadBuf, r.opts)
	if usage == nil {
		return line
	}
	merged := converter.TokenUsage{}
	if r.logCtx.TokenUsage != nil {
		merged = *r.logCtx.TokenUsage
	}
	merged.MergeNonZero(usage)
	cost, ok := r.proxy.clientFacingCost(r.logCtx, &merged)
	if !ok {
		return line
	}
	withCost, injected := injectCostIntoSSELine(line, r.payloadBuf, cost)
	if !injected {
		return line
	}
	r.done = true
	return withCost
}

// injectCostIntoSSELine adds "cost" to the usage object of a single SSE
// "data: <json>" line, preserving the line's own trailing newline style
// (or its absence, on a final unterminated line) and bailing out (ok=false,
// line returned as-is by the caller) for anything that isn't unambiguously
// one data line with a non-null "usage" key.
func injectCostIntoSSELine(line []byte, payloads [][]byte, cost float64) ([]byte, bool) {
	if len(payloads) != 1 {
		return nil, false
	}
	var newline []byte
	body := line
	switch {
	case bytes.HasSuffix(body, []byte("\r\n")):
		newline = body[len(body)-2:]
		body = body[:len(body)-2]
	case bytes.HasSuffix(body, []byte("\n")):
		newline = body[len(body)-1:]
		body = body[:len(body)-1]
	}
	if !bytes.HasPrefix(bytes.TrimSpace(body), sseDataPrefix) {
		return nil, false
	}
	newPayload, ok := injectCostIntoJSONUsage(payloads[0], cost)
	if !ok {
		return nil, false
	}
	result := make([]byte, 0, len(sseDataPrefix)+1+len(newPayload)+len(newline))
	result = append(result, sseDataPrefix...)
	result = append(result, ' ')
	result = append(result, newPayload...)
	result = append(result, newline...)
	return result, true
}

// ==================== Native Responses API passthrough streaming ====================
//
// handlePassthroughResponsesStreaming forwards a codex-native /v1/responses
// provider's SSE bytes directly, the same raw byte-chunked way
// handleStreamingWithTokens does for chat-completion-shaped passthrough (see
// above) -- so it needs the same per-line rewrite technique. What differs is
// the usage shape: the Responses API reports usage once, nested as
// response.usage on the single terminal "response.completed" (or
// "response.incomplete") event line (see stream_responses_terminal.go),
// not as a flat top-level "usage" key on every chunk.
// responsesAPICostInjectingReader is the Responses-API-shaped counterpart of
// costInjectingSSEReader, built on the same terminal-event helpers
// handlePassthroughResponsesStreaming's own onChunk closure uses
// (mayBeTerminalResponsesEvent, parseResponsesTerminalEvent, typedUsage).

// newResponsesAPICostInjectingReader wraps reader so the one SSE line
// carrying "response.completed"/"response.incomplete" gets "cost" added to
// its nested response.usage before reaching the client.
// audioInputAlreadyExcludesCachedAudio and tokenUsageOptions must match
// whatever handlePassthroughResponsesStreaming's own usage bookkeeping uses,
// so the usage this computes cost from agrees with what
// finalizeStreamingLogParsed logs.
func newResponsesAPICostInjectingReader(
	reader io.Reader,
	proxy *Proxy,
	logCtx *RequestLogContext,
	tokenUsageOptions converter.TokenUsageExtractionOptions,
	audioInputAlreadyExcludesCachedAudio bool,
) io.Reader {
	if reader == nil || proxy == nil || logCtx == nil {
		return reader
	}
	return &responsesAPICostInjectingReader{
		source:                               bufio.NewReader(reader),
		proxy:                                proxy,
		logCtx:                               logCtx,
		tokenUsageOptions:                    tokenUsageOptions,
		audioInputAlreadyExcludesCachedAudio: audioInputAlreadyExcludesCachedAudio,
		maxLineBytes:                         maxSSEModelRewriteLineBytes,
	}
}

type responsesAPICostInjectingReader struct {
	source                               *bufio.Reader
	proxy                                *Proxy
	logCtx                               *RequestLogContext
	tokenUsageOptions                    converter.TokenUsageExtractionOptions
	audioInputAlreadyExcludesCachedAudio bool
	pending                              []byte
	pendingErr                           error
	line                                 []byte
	passthrough                          bool
	maxLineBytes                         int
	// done is set once cost has been injected -- the completion event is
	// terminal, so no later line needs examining.
	done bool
}

// Read has the identical bufio/ReadSlice/oversized-line-fallback mechanics as
// costInjectingSSEReader.Read (see its doc comment for why) -- only the
// per-line rewrite in injectCostIntoLine differs.
func (r *responsesAPICostInjectingReader) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		if r.pendingErr != nil {
			err := r.pendingErr
			r.pendingErr = nil
			return 0, err
		}
		fragment, err := r.source.ReadSlice('\n')
		if len(fragment) == 0 {
			return 0, err
		}
		if r.done || r.passthrough {
			r.pending = append(r.pending, fragment...)
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
				r.pendingErr = err
			}
			continue
		}
		maxLineBytes := r.maxLineBytes
		if maxLineBytes <= 0 {
			maxLineBytes = maxSSEModelRewriteLineBytes
		}
		if len(r.line)+len(fragment) > maxLineBytes {
			r.pending = append(r.pending, r.line...)
			r.pending = append(r.pending, fragment...)
			r.line = r.line[:0]
			if errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = true
			} else {
				r.pendingErr = err
			}
			continue
		}
		r.line = append(r.line, fragment...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		r.pending = r.injectCostIntoLine(r.line)
		r.line = nil
		r.pendingErr = err
	}

	n := copy(dst, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

// injectCostIntoLine returns line unchanged unless it is the single SSE data
// line carrying "response.completed"/"response.incomplete" with a non-nil
// response.usage, in which case it returns a rewritten copy with "cost"
// added under response.usage. The Responses API reports usage once, fully,
// on this one terminal event -- unlike the chat-completion case, there is no
// earlier-chunk usage to merge in.
func (r *responsesAPICostInjectingReader) injectCostIntoLine(line []byte) []byte {
	if !mayBeTerminalResponsesEvent(line) {
		return line
	}
	trimmed := bytes.TrimRight(line, "\r\n")
	payload := responsesStreamDataPayload(trimmed)
	if payload == nil {
		return line
	}
	event, ok := parseResponsesTerminalEvent(payload)
	if !ok {
		return line
	}
	rawUsage := event.typedUsage()
	if rawUsage == nil {
		return line
	}
	usage := tokenUsageFromResponsesAPIUsage(rawUsage, payload, r.tokenUsageOptions, r.audioInputAlreadyExcludesCachedAudio)
	cost, ok := r.proxy.clientFacingCost(r.logCtx, usage)
	if !ok {
		return line
	}
	withCost, injected := injectCostIntoResponsesSSELine(line, payload, cost)
	if !injected {
		return line
	}
	r.done = true
	return withCost
}

// tokenUsageFromResponsesAPIUsage converts a native Responses API usage
// object into converter.TokenUsage, mirroring the field mapping
// handlePassthroughResponsesStreaming's own onChunk closure uses for
// logCtx.TokenUsage -- including the tool-usage-extension merge
// (MergeUsageExtensions) that picks up xAI's counters, built-in image
// generation tool images, reasoning accounting and a provider's own
// reported cost. Kept independent (not shared with that closure) so the
// cost-exposure path -- which needs this usage before onChunk gets to run,
// see the package doc above -- never risks mutating logCtx.TokenUsage
// itself.
func tokenUsageFromResponsesAPIUsage(
	usage *responses.Usage,
	payload []byte,
	tokenUsageOptions converter.TokenUsageExtractionOptions,
	audioInputAlreadyExcludesCachedAudio bool,
) *converter.TokenUsage {
	if usage == nil {
		return nil
	}
	cachedTokens, cachedAudioTokens := converterutil.NormalizeCachedAudioBreakdown(
		usage.InputTokensDetails.CachedTokens,
		usage.InputTokensDetails.CachedAudioTokens,
	)
	tu := &converter.TokenUsage{
		PromptTokens:           usage.InputTokens,
		CompletionTokens:       usage.OutputTokens,
		CachedInputTokens:      cachedTokens,
		CachedAudioInputTokens: cachedAudioTokens,
		CacheCreationTokens:    usage.InputTokensDetails.CacheCreationTokens,
		AudioInputTokens: normalizeStreamAudioInput(
			usage.InputTokensDetails.AudioTokens,
			cachedTokens,
			cachedAudioTokens,
			audioInputAlreadyExcludesCachedAudio,
		),
		AudioOutputTokens: usage.OutputTokensDetails.AudioTokens,
		ReasoningTokens:   usage.OutputTokensDetails.ReasoningTokens,
	}
	if details := usage.InputTokensDetails.CacheCreationTokenDetails; details != nil {
		tu.CacheCreation5mTokens = details.Ephemeral5mInputTokens
		tu.CacheCreation1hTokens = details.Ephemeral1hInputTokens
	}
	if details := usage.InputTokensDetails.CacheWriteTokensDetails; details != nil {
		tu.CacheCreation12hTokens = details.TTL12hTokens
		if tu.CacheCreation5mTokens == 0 && tu.CacheCreation1hTokens == 0 {
			tu.CacheCreation5mTokens = details.TTL30mTokens
		}
	}
	if usage.ServerToolUse != nil {
		tu.WebSearchRequests = usage.ServerToolUse.WebSearchRequests
	}
	tu.MergeUsageExtensions(converter.ExtractTokenUsageWithOptions(payload, tokenUsageOptions))
	return tu
}

// injectCostIntoResponsesSSELine adds "cost" to the nested response.usage
// object of a single SSE "data: <json>" line, preserving the line's own
// trailing newline style. payload is the line's already-extracted JSON (see
// responsesStreamDataPayload). Bails out (ok=false) for anything that isn't
// unambiguously one data line with a "response" object carrying a non-null
// "usage".
func injectCostIntoResponsesSSELine(line []byte, payload []byte, cost float64) ([]byte, bool) {
	var newline []byte
	body := line
	switch {
	case bytes.HasSuffix(body, []byte("\r\n")):
		newline = body[len(body)-2:]
		body = body[:len(body)-2]
	case bytes.HasSuffix(body, []byte("\n")):
		newline = body[len(body)-1:]
		body = body[:len(body)-1]
	}
	if !bytes.HasPrefix(bytes.TrimSpace(body), sseDataPrefix) {
		return nil, false
	}
	newPayload, ok := injectCostIntoNestedResponseUsage(payload, cost)
	if !ok {
		return nil, false
	}
	result := make([]byte, 0, len(sseDataPrefix)+1+len(newPayload)+len(newline))
	result = append(result, sseDataPrefix...)
	result = append(result, ' ')
	result = append(result, newPayload...)
	result = append(result, newline...)
	return result, true
}

// injectCostIntoNestedResponseUsage adds "cost" to response.usage for a
// native Responses API event payload ({"type":...,"response":{"usage":
// {...}, ...}}) -- the Responses API's own one-level-deeper usage nesting,
// versus the top-level "usage" key injectCostIntoJSONUsage handles for chat
// completions. Bails out (ok=false) for anything it doesn't recognize.
func injectCostIntoNestedResponseUsage(payload []byte, cost float64) ([]byte, bool) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(payload, &doc); err != nil {
		return payload, false
	}
	responseRaw, ok := doc["response"]
	if !ok || bytes.Equal(bytes.TrimSpace(responseRaw), []byte("null")) {
		return payload, false
	}
	newResponse, injected := injectCostIntoJSONUsage(responseRaw, cost)
	if !injected {
		return payload, false
	}
	doc["response"] = newResponse
	newPayload, err := json.Marshal(doc)
	if err != nil {
		return payload, false
	}
	return newPayload, true
}
