package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"

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
//
// Every injector in this file splices the new "cost" field directly into
// the original bytes (see spliceBytes) rather than decode+remarshal the
// whole document: a remarshal would re-sort map keys and re-escape "<",
// ">" and "&" (encoding/json's default HTML-escaping) anywhere in the
// response, not just in "usage" -- a client comparing raw bytes, or any
// message content containing those characters, would see a changed
// response for reasons that have nothing to do with cost. Splicing touches
// only the bytes between "usage"'s (or response.usage's) braces.
//
// A known limitation: the cost injected here is computed from whatever
// usage this specific chunk/line/body carries at the moment it is read,
// before finalizeStreamingLog's own post-hoc corrections (the tiktoken
// fallback when a provider never reports prompt tokens, Normalize()'s
// clamping, the local completion-token fallback) can run. For the dominant
// case -- a provider that reports real, complete usage -- this matches the
// spend log exactly; for the local-fallback edge cases it can't, since
// those corrections only run once the whole stream (or retry) has already
// finished and the client has already seen this response.

// shouldExposeCostToClient reports whether logCtx's organization opted into
// usage.cost in client-facing responses.
func shouldExposeCostToClient(logCtx *RequestLogContext) bool {
	return logCtx != nil && logCtx.OrganizationPolicy.ReturnsCostInBody()
}

// clientFacingCost resolves the billing price and computes the final,
// margin-adjusted cost for usage, the same way logSpendToLiteLLMDB does for
// the spend log and the X-Total-Cost header (resolveBillingPrice ->
// CalculateCostsWithResolver -> applyCostMargin, applied only once the base
// cost is actually positive -- logSpendToLiteLLMDB's own fixed-margin-only-
// on-priced-responses rule, see its comment on applyCostMargin). It never
// mutates logCtx's cached billing-price fields beyond the normal
// resolveBillingPrice caching, so calling it ahead of logSpendToLiteLLMDB
// (as the streaming path must) does not change what gets logged or billed.
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
	if costs.TotalCost > 0 {
		logCtx.applyCostMargin(costs)
	}
	return costs.TotalCost, true
}

// appendCostField inserts `"cost":<value>` into a JSON object's raw bytes
// just before its closing brace, touching no other byte: no remarshal, no
// key reordering, no re-escaping. objectRaw must already be known to decode
// as a JSON object (callers check this via a map[string]json.RawMessage
// decode first); garbage that merely starts with '{' and ends with '}' is
// still rejected by the decode this does for safety.
func appendCostField(objectRaw []byte, cost float64) ([]byte, bool) {
	trimmed := bytes.TrimSpace(objectRaw)
	if len(trimmed) < 2 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return nil, false
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &probe); err != nil {
		return nil, false
	}
	costField := []byte(`"cost":` + strconv.FormatFloat(cost, 'f', -1, 64))
	inner := bytes.TrimSpace(trimmed[1 : len(trimmed)-1])
	result := make([]byte, 0, len(trimmed)+len(costField)+1)
	if len(inner) == 0 {
		result = append(result, trimmed[0])
	} else {
		result = append(result, trimmed[:len(trimmed)-1]...)
		result = append(result, ',')
	}
	result = append(result, costField...)
	result = append(result, '}')
	return result, true
}

// spliceBytes replaces the first occurrence of old within frame with
// replacement, or reports ok=false when old cannot be found verbatim in
// frame (it always can in practice here: every caller derives old as an
// actual sub-slice of frame, e.g. via encoding/json.RawMessage or
// splitSSEPayloads, never a value reconstructed independently).
func spliceBytes(frame, old, replacement []byte) ([]byte, bool) {
	offset := bytes.Index(frame, old)
	if offset < 0 {
		return nil, false
	}
	result := make([]byte, 0, len(frame)+len(replacement)-len(old))
	result = append(result, frame[:offset]...)
	result = append(result, replacement...)
	result = append(result, frame[offset+len(old):]...)
	return result, true
}

// injectCostIntoJSONUsage adds "cost" to the top-level "usage" object of a
// JSON document (a non-streaming response body, or a single chat-
// completion-shaped SSE payload). Returns the original bytes unchanged
// (ok=false) for any shape it doesn't recognize -- no "usage" key (e.g. some
// error bodies, binary/audio responses that never reach here as JSON), a
// null usage, or malformed JSON.
func injectCostIntoJSONUsage(body []byte, cost float64) ([]byte, bool) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		return body, false
	}
	usageRaw, ok := doc["usage"]
	if !ok || bytes.Equal(bytes.TrimSpace(usageRaw), []byte("null")) {
		return body, false
	}
	newUsage, ok := appendCostField(usageRaw, cost)
	if !ok {
		return body, false
	}
	return spliceBytes(body, usageRaw, newUsage)
}

// injectCostIntoNestedResponseUsage adds "cost" to response.usage for a
// native Responses API JSON document ({"type":...,"response":{"usage":
// {...}, ...}}) -- the Responses API's own one-level-deeper usage nesting,
// versus the top-level "usage" key injectCostIntoJSONUsage handles for chat
// completions. Implemented as injectCostIntoJSONUsage on the "response"
// sub-object (its own top-level key is "usage" too) spliced back into the
// original bytes, so the outer document's other fields and the response
// object's own field order/escaping are untouched either way. Bails out
// (ok=false) for anything it doesn't recognize.
func injectCostIntoNestedResponseUsage(payload []byte, cost float64) ([]byte, bool) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(payload, &doc); err != nil {
		return payload, false
	}
	responseRaw, ok := doc["response"]
	if !ok || bytes.Equal(bytes.TrimSpace(responseRaw), []byte("null")) {
		return payload, false
	}
	newResponse, ok := injectCostIntoJSONUsage(responseRaw, cost)
	if !ok {
		return payload, false
	}
	return spliceBytes(payload, responseRaw, newResponse)
}

// injectCostIntoSSEUsageChunk adds "cost" to the usage object carried by the
// one "data:" line among payloads -- top-level "usage" for chat-completion-
// shaped payloads, tried first, falling back to the Responses API's nested
// "response.usage" (the shape handleNativeResponsesStreaming/
// handleResponsesAPIStreaming produce when they run through
// handleTransformedStreaming, same as every other converted provider) --
// splicing the rewritten payload directly into frame's own bytes. Every
// other byte of frame survives untouched: a bundled "event: ...\n" line
// (fmt.Fprintf writes "event: %s\ndata: %s\n\n" as a single Write call for
// /v1/messages, see TransformChatStreamToMessages), a batched [DONE]
// sentinel, blank-line framing. Only the unambiguous single-payload case is
// handled; anything else (several payloads batched together, no "usage"
// key found either shape, malformed JSON) leaves frame untouched
// (ok=false).
func injectCostIntoSSEUsageChunk(frame []byte, payloads [][]byte, cost float64) ([]byte, bool) {
	if len(payloads) != 1 {
		return nil, false
	}
	newPayload, ok := injectCostIntoJSONUsage(payloads[0], cost)
	if !ok {
		newPayload, ok = injectCostIntoNestedResponseUsage(payloads[0], cost)
	}
	if !ok {
		return nil, false
	}
	return spliceBytes(frame, payloads[0], newPayload)
}

// ==================== Provider-cost snapshot/restore ====================
//
// extractTokenUsageFromPayloads/converter.ExtractTokenUsageWithOptions read
// a usage object's own "cost" field (OpenRouter/Requesty-style aggregators,
// or xAI's cost_in_usd_ticks/cost_in_nano_usd) as converter.TokenUsage.
// ProviderCostUSD -- the provider's own reported price, kept only for
// reconciliation/observability (provider_reported_cost in spend metadata,
// ProviderReportedCost in the Kafka event), never used to bill the client.
//
// The two line-buffering readers below (costInjectingSSEReader,
// responsesAPICostInjectingReader) sit between the provider and
// streamToClient: the caller's own onChunk/onLine observer -- which is what
// populates logCtx.TokenUsage.ProviderCostUSD -- reads the *forwarded*
// bytes, i.e. runs strictly after Read() has already returned this
// reader's rewritten line. There is no hook available to make that
// observer see the pre-injection bytes instead (it is driven entirely by
// streamToClient's own read loop, shared by every streaming handler, not
// something this one organization's opt-in should alter for everyone). So
// without correction, AIR's own injected usage.cost would be misread one
// Read() cycle later as the provider's reported cost, polluting both the
// spend log metadata and the Kafka event with the client's own bill.
//
// providerCostUSDGuard fixes this without touching streamToClient: snapshot
// logCtx.TokenUsage.ProviderCostUSD immediately before injecting (i.e.
// before the observer has had any chance to run on this line), then restore
// it once the observer has actually had its one chance to run. Embedded by
// value in both readers.
//
// The injected line is often returned to the caller across several small
// Read() calls (dst smaller than the rewritten line) before it is fully
// drained -- the real line-reassembling observer (streamUsageLines in both
// handleStreamingWithTokens and handlePassthroughResponsesStreaming) only
// fires once it has seen the line's whole trailing newline, i.e. on the
// return of whichever Read() call hands out the line's *last* byte. So a
// naive "restore on the very next Read()" fires too early while that line
// is still being drained, before the observer has run on it at all --
// drained() marks that the last byte has gone out, and restoreIfDue(),
// checked at the top of every subsequent Read(), only then reverts the
// corruption, on the first call after that.
type providerCostUSDGuard struct {
	state  providerCostUSDGuardState
	before float64
}

type providerCostUSDGuardState int

const (
	providerCostUSDGuardIdle providerCostUSDGuardState = iota
	// providerCostUSDGuardInjected: the rewritten line has been snapshotted
	// but not yet fully handed to the caller.
	providerCostUSDGuardInjected
	// providerCostUSDGuardDrained: the rewritten line's last byte has been
	// returned -- the caller's observer will run on it before the next
	// Read() call starts.
	providerCostUSDGuardDrained
)

// snapshot records the pre-injection ProviderCostUSD so restoreIfDue can
// revert whatever the caller's observer is about to (mis)write once this
// line has been fully returned.
func (g *providerCostUSDGuard) snapshot(usage *converter.TokenUsage) {
	g.state = providerCostUSDGuardInjected
	if usage != nil {
		g.before = usage.ProviderCostUSD
	} else {
		g.before = 0
	}
}

// drained marks that the injected line's last byte has just been returned
// to the caller -- a no-op unless snapshot armed the guard for this line.
func (g *providerCostUSDGuard) drained() {
	if g.state == providerCostUSDGuardInjected {
		g.state = providerCostUSDGuardDrained
	}
}

// restoreIfDue undoes the previous line's corruption, but only once the
// caller's observer has already had its one chance to run on it (state ==
// drained); a no-op otherwise, including while that line is still being
// handed out across several small Read() calls.
func (g *providerCostUSDGuard) restoreIfDue(usage *converter.TokenUsage) {
	if g.state != providerCostUSDGuardDrained {
		return
	}
	g.state = providerCostUSDGuardIdle
	if usage != nil {
		usage.ProviderCostUSD = g.before
	}
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
	done         bool
	payloadBuf   [][]byte
	providerCost providerCostUSDGuard
}

func (r *costInjectingSSEReader) Read(dst []byte) (int, error) {
	r.providerCost.restoreIfDue(r.logCtx.TokenUsage)
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
	if len(r.pending) == 0 {
		r.providerCost.drained()
	}
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
	withCost, injected := injectCostIntoSSEUsageChunk(line, r.payloadBuf, cost)
	if !injected {
		return line
	}
	r.providerCost.snapshot(r.logCtx.TokenUsage)
	r.done = true
	return withCost
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
	done         bool
	providerCost providerCostUSDGuard
}

// Read has the identical bufio/ReadSlice/oversized-line-fallback mechanics as
// costInjectingSSEReader.Read (see its doc comment for why) -- only the
// per-line rewrite in injectCostIntoLine differs.
func (r *responsesAPICostInjectingReader) Read(dst []byte) (int, error) {
	r.providerCost.restoreIfDue(r.logCtx.TokenUsage)
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
	if len(r.pending) == 0 {
		r.providerCost.drained()
	}
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
	newPayload, injected := injectCostIntoNestedResponseUsage(payload, cost)
	if !injected {
		return line
	}
	withCost, spliced := spliceBytes(line, payload, newPayload)
	if !spliced {
		return line
	}
	r.providerCost.snapshot(r.logCtx.TokenUsage)
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

// ==================== LiteLLM response-compatibility interop ====================
//
// response_compatibility: litellm (see response_compat.go) reshapes every
// response through compatlitellm.Transformer before it reaches the client,
// and that transform unconditionally strips any usage.cost/response.usage.
// cost it finds (dropProviderCost -- it exists specifically to hide a
// provider's own reported price from a client expecting LiteLLM's shape).
// It cannot tell AIR's own injected cost apart from a provider's, so
// injecting before the transform (as the direct, non-compat-mode paths
// above do) is silently undone by it.
//
// wireCostInjectionForCompatWriter arms the compatibility writer itself to
// re-add the cost *after* its transform has already run, mirroring how
// PR #334 preserved the X-Total-Cost/X-Cost-Breakdown headers through the
// same writer (preservedRouterHeaders) -- the router's own late-computed
// data has to survive a transform aimed at a provider's data, for body and
// headers alike. A no-op when w isn't that writer, or the organization
// didn't opt in (checked lazily, inside the closures: w is wrapped long
// before logCtx.OrganizationPolicy is resolved -- see its call site in
// proxyRequest).
func wireCostInjectionForCompatWriter(w http.ResponseWriter, proxy *Proxy, logCtx *RequestLogContext) {
	cw, ok := w.(*responseCompatibilityWriter)
	if !ok || cw == nil {
		return
	}
	cw.costInjector = func(body []byte) []byte {
		if !shouldExposeCostToClient(logCtx) || !logCtx.responseCostKnown {
			return body
		}
		if withCost, injected := injectCostIntoJSONUsage(body, logCtx.responseCostUSD); injected {
			return withCost
		}
		return body
	}
	cw.streamCostInjector = func(stream io.Reader) io.Reader {
		if !shouldExposeCostToClient(logCtx) {
			return stream
		}
		// The LiteLLM transform normalizes every provider's stream to a
		// single known shape per endpoint, so (unlike the direct,
		// non-compat-mode streaming paths) no provider-specific
		// TokenUsageExtractionOptions are available here -- a plain zero
		// value accepts the same small loss of precision on audio/cache
		// edge cases response_compatibility: litellm already trades away
		// elsewhere (see normalizeUsage's own token-count defaulting).
		if cw.compatContext().Endpoint == "/v1/responses" {
			return newResponsesAPICostInjectingReader(stream, proxy, logCtx, converter.TokenUsageExtractionOptions{}, true)
		}
		return newCostInjectingSSEReader(stream, proxy, logCtx, converter.TokenUsageExtractionOptions{})
	}
}
