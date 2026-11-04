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

// topLevelEntry is one key's value from topLevelEntries, plus where to start
// searching for its exact byte occurrence (see topLevelEntries).
type topLevelEntry struct {
	value      json.RawMessage
	searchFrom int
}

// topLevelEntries decodes a JSON object's own top-level key/value pairs
// (never descending into a value to look for the same key again), recording
// for each not just its raw bytes but the byte offset to search from to
// find that exact occurrence back in objectRaw.
//
// Searching from that offset (via spliceBytesFrom), instead of from the
// start of objectRaw (bytes.Index(objectRaw, value)), is what makes the
// splice land on the right occurrence: a map[string]json.RawMessage decode
// only gives the value's bytes, and a content-based search for those bytes
// can match an earlier, byte-identical span instead -- a sibling object
// with the same content (e.g. {"metadata":{"usage":{"prompt_tokens":10}},
// "usage":{"prompt_tokens":10}}) or a duplicate key (the last one, which is
// also what a standards-conforming decoder returns, since dec.Token()/
// dec.Decode() walk keys in document order and a later entry overwrites an
// earlier one here the same way). searchFrom is exactly where this decoder
// started reading this key's value, so an identical span anywhere before it
// can never be mistaken for it.
//
// ok=false if objectRaw isn't a well-formed JSON object.
func topLevelEntries(objectRaw []byte) (map[string]topLevelEntry, bool) {
	dec := json.NewDecoder(bytes.NewReader(objectRaw))
	tok, err := dec.Token()
	if err != nil {
		return nil, false
	}
	if delim, isDelim := tok.(json.Delim); !isDelim || delim != '{' {
		return nil, false
	}
	entries := make(map[string]topLevelEntry)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, false
		}
		key, isString := keyTok.(string)
		if !isString {
			return nil, false
		}
		searchFrom := int(dec.InputOffset())
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, false
		}
		entries[key] = topLevelEntry{value: raw, searchFrom: searchFrom}
	}
	if _, err := dec.Token(); err != nil { // closing '}'
		return nil, false
	}
	return entries, true
}

// spliceBytesFrom replaces the first occurrence of old found at or after
// byte offset from within frame, with replacement. Reports ok=false when
// old cannot be found there (from out of range, or old genuinely absent
// from frame[from:]).
func spliceBytesFrom(frame []byte, from int, old, replacement []byte) ([]byte, bool) {
	if from < 0 || from > len(frame) {
		return nil, false
	}
	offset := bytes.Index(frame[from:], old)
	if offset < 0 {
		return nil, false
	}
	offset += from
	result := make([]byte, 0, len(frame)+len(replacement)-len(old))
	result = append(result, frame[:offset]...)
	result = append(result, replacement...)
	result = append(result, frame[offset+len(old):]...)
	return result, true
}

// appendCostField sets "cost" on a JSON object's raw bytes to the given
// value: replaced in place via topLevelEntries+spliceBytesFrom if the key
// already exists (a provider-reported cost survives in non-compat mode --
// only the LiteLLM compatibility transform strips it, see dropProviderCost
// -- and an aggregator-style provider can send one), appended just before
// the closing brace otherwise. Never writes a second, duplicate "cost" key,
// which would be invalid JSON. Touches no other byte either way: no
// remarshal, no key reordering, no re-escaping.
func appendCostField(objectRaw []byte, cost float64) ([]byte, bool) {
	trimmed := bytes.TrimSpace(objectRaw)
	entries, ok := topLevelEntries(trimmed)
	if !ok {
		return nil, false
	}
	costField := []byte(strconv.FormatFloat(cost, 'f', -1, 64))
	if existing, has := entries["cost"]; has {
		return spliceBytesFrom(trimmed, existing.searchFrom, existing.value, costField)
	}
	inner := bytes.TrimSpace(trimmed[1 : len(trimmed)-1])
	result := make([]byte, 0, len(trimmed)+len(costField)+8)
	if len(inner) == 0 {
		result = append(result, trimmed[0])
	} else {
		result = append(result, trimmed[:len(trimmed)-1]...)
		result = append(result, ',')
	}
	result = append(result, `"cost":`...)
	result = append(result, costField...)
	result = append(result, '}')
	return result, true
}

// injectCostIntoJSONUsage adds "cost" to the top-level "usage" object of a
// JSON document (a non-streaming response body, or a single chat-
// completion-shaped SSE payload). Returns the original bytes unchanged
// (ok=false) for any shape it doesn't recognize -- no "usage" key (e.g. some
// error bodies, binary/audio responses that never reach here as JSON), a
// null usage, or malformed JSON.
func injectCostIntoJSONUsage(body []byte, cost float64) ([]byte, bool) {
	entries, ok := topLevelEntries(body)
	if !ok {
		return body, false
	}
	usage, ok := entries["usage"]
	if !ok || bytes.Equal(bytes.TrimSpace(usage.value), []byte("null")) {
		return body, false
	}
	newUsage, ok := appendCostField(usage.value, cost)
	if !ok {
		return body, false
	}
	return spliceBytesFrom(body, usage.searchFrom, usage.value, newUsage)
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
	entries, ok := topLevelEntries(payload)
	if !ok {
		return payload, false
	}
	response, ok := entries["response"]
	if !ok || bytes.Equal(bytes.TrimSpace(response.value), []byte("null")) {
		return payload, false
	}
	newResponse, ok := injectCostIntoJSONUsage(response.value, cost)
	if !ok {
		return payload, false
	}
	return spliceBytesFrom(payload, response.searchFrom, response.value, newResponse)
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
// (ok=false). Searching for payloads[0] from byte 0 of frame (rather than a
// tracked offset, unlike the two functions above) is safe here specifically
// because len(payloads)==1 is already required: there is exactly one
// "data:" line in frame, so no other span of frame can coincidentally equal
// payloads[0] -- a bundled "event: ..." line is a short type label, never a
// JSON object.
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
	offset := bytes.Index(frame, payloads[0])
	if offset < 0 {
		return nil, false
	}
	result := make([]byte, 0, len(frame)+len(newPayload)-len(payloads[0]))
	result = append(result, frame[:offset]...)
	result = append(result, newPayload...)
	result = append(result, frame[offset+len(payloads[0]):]...)
	return result, true
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
// providerCostUSDGuard fixes this without touching streamToClient: the
// caller computes, from this line's own pre-injection usage extraction (the
// same one it used to compute the cost being injected), the value
// ProviderCostUSD should hold once this line is fully processed --
// this line's own genuine provider-reported figure if it sent one (the same
// "last non-zero wins" precedence MergeNonZero/MergeUsageExtensions already
// use), otherwise whatever was accumulated from earlier lines. arm()
// records that target; once the observer has actually had its one chance to
// run on the forwarded (injected) bytes, restoreIfDue() overwrites whatever
// it just wrote with that target -- not simply "undoes the injection", which
// would also discard a genuine provider-reported cost this exact line
// carried, the same bug class as never fixing this at all, just in the
// other direction. Embedded by value in both readers.
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
// checked at the top of every subsequent Read(), only then applies the
// target, on the first call after that.
type providerCostUSDGuard struct {
	state  providerCostUSDGuardState
	target float64
}

type providerCostUSDGuardState int

const (
	providerCostUSDGuardIdle providerCostUSDGuardState = iota
	// providerCostUSDGuardInjected: the rewritten line has been armed but
	// not yet fully handed to the caller.
	providerCostUSDGuardInjected
	// providerCostUSDGuardDrained: the rewritten line's last byte has been
	// returned -- the caller's observer will run on it before the next
	// Read() call starts.
	providerCostUSDGuardDrained
)

// arm records the correct post-processing ProviderCostUSD (see the type's
// own doc comment for how the caller must compute it) so restoreIfDue can
// apply it once this line has been fully returned and the observer has run.
func (g *providerCostUSDGuard) arm(target float64) {
	g.state = providerCostUSDGuardInjected
	g.target = target
}

// drained marks that the injected line's last byte has just been returned
// to the caller -- a no-op unless arm set a target for this line.
func (g *providerCostUSDGuard) drained() {
	if g.state == providerCostUSDGuardInjected {
		g.state = providerCostUSDGuardDrained
	}
}

// restoreIfDue applies the previous line's armed target, but only once the
// caller's observer has already had its one chance to run on it (state ==
// drained); a no-op otherwise, including while that line is still being
// handed out across several small Read() calls.
func (g *providerCostUSDGuard) restoreIfDue(usage *converter.TokenUsage) {
	if g.state != providerCostUSDGuardDrained {
		return
	}
	g.state = providerCostUSDGuardIdle
	if usage != nil {
		usage.ProviderCostUSD = g.target
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
	// The correct ProviderCostUSD once this line is processed: this line's
	// own genuine provider-reported figure (OpenRouter/Requesty-style
	// usage.cost, xAI's cost_in_usd_ticks, ...) wins over whatever was
	// accumulated before, the same precedence MergeNonZero itself uses --
	// never the value AIR's own injection is about to overwrite it with on
	// the wire.
	correctProviderCost := merged.ProviderCostUSD
	if usage.ProviderCostUSD != 0 {
		correctProviderCost = usage.ProviderCostUSD
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
	r.providerCost.arm(correctProviderCost)
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
	// line has exactly one data: line (mayBeTerminalResponsesEvent/
	// responsesStreamDataPayload already established that), so -- same
	// reasoning as injectCostIntoSSEUsageChunk -- a plain content search
	// for payload within line cannot land on the wrong span.
	offset := bytes.Index(line, payload)
	if offset < 0 {
		return line
	}
	withCost := make([]byte, 0, len(line)+len(newPayload)-len(payload))
	withCost = append(withCost, line[:offset]...)
	withCost = append(withCost, newPayload...)
	withCost = append(withCost, line[offset+len(payload):]...)

	// The correct ProviderCostUSD once this line is processed: this line's
	// own genuine provider-reported figure (e.g. an xAI cost_in_usd_ticks
	// under response.usage) wins over whatever was accumulated before, the
	// same precedence MergeUsageExtensions itself uses -- never the value
	// AIR's own injection is about to overwrite it with on the wire.
	correctProviderCost := float64(0)
	if r.logCtx.TokenUsage != nil {
		correctProviderCost = r.logCtx.TokenUsage.ProviderCostUSD
	}
	if usage.ProviderCostUSD != 0 {
		correctProviderCost = usage.ProviderCostUSD
	}
	r.providerCost.arm(correctProviderCost)
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
