package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"maps"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	// Also used by request ingress sanitization below. RawMessage keeps large
	// numbers and provider-specific fields byte-for-byte while the request map
	// is inspected and selectively rebuilt.
	goccyjson "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

type usageTotalTokens struct {
	TotalTokens int `json:"total_tokens"`
}

type openAIUsageResponse struct {
	// Chat Completions: usage at top level
	Usage usageTotalTokens `json:"usage"`
	// Responses API: usage nested inside response object (response.completed event)
	Response struct {
		Usage usageTotalTokens `json:"usage"`
	} `json:"response"`
}

func extractOpenAITotalTokens(payload []byte) int {
	var openAIResp openAIUsageResponse
	if err := goccyjson.Unmarshal(payload, &openAIResp); err != nil {
		return 0
	}

	if openAIResp.Usage.TotalTokens > 0 {
		return openAIResp.Usage.TotalTokens
	}
	return openAIResp.Response.Usage.TotalTokens
}

// extractMessagesTotalTokens reads Anthropic Messages API streaming usage
// (message_delta event) for the rate limiter's crude per-chunk token count.
// Kept as a fallback behind extractOpenAITotalTokens in extractTokensFromPayloads
// so the extra unmarshal only runs for the minority Anthropic-shaped traffic,
// not on every chunk of the majority Chat Completions/Responses/Vertex/Bedrock
// hot path.
func extractMessagesTotalTokens(payload []byte) int {
	var event struct {
		Type  string `json:"type"`
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &event); err != nil || event.Type != "message_delta" {
		return 0
	}
	return event.Usage.InputTokens +
		event.Usage.OutputTokens +
		event.Usage.CacheReadInputTokens +
		event.Usage.CacheCreationInputTokens
}

// extractOpenAITokensAndUsage runs both non-streaming token-accounting
// consumers off a single shared decode of body (converter.ExtractTotalTokensAndUsageWithOptions):
// the plain "total_tokens" count consumed by the rate limiter (RPM/TPM
// enforcement) and the full converter.TokenUsage consumed by spend
// logging/metrics. Replaces two independent full-body decodes
// (extractTokensFromResponse + converter.ExtractTokenUsageWithOptions) at
// each non-streaming response call site (proxy.go's proxy-credential and
// direct-provider branches, retry.go's fallback branch) with one — see
// ExtractTotalTokensAndUsageWithOptions's doc comment for why the two
// returned numbers are computed independently (not guaranteed equal) even
// though the decode is shared. Only wired in for the OpenAI-shaped
// (config.ProviderTypeOpenAI) path, matching all three call sites this
// replaces — none of them ever passed a Vertex/Gemini credType to
// extractTokensFromResponse.
func extractOpenAITokensAndUsage(body []byte, opts converter.TokenUsageExtractionOptions) (int, *converter.TokenUsage) {
	return converter.ExtractTotalTokensAndUsageWithOptions(body, opts)
}

// extractTokensFromStreamingChunk splits chunk via the shared zero-copy
// splitSSEPayloads (plan item C — one parse, no string(chunk) copy) and looks
// for usage information in the resulting payloads. Not on any hot per-chunk
// path itself (callers that already split the chunk once use
// extractTokensFromPayloads directly with the shared sub-slices instead), but
// kept as a convenience wrapper for callers that only have a raw chunk.
func extractTokensFromStreamingChunk(chunk []byte) int {
	return extractTokensFromPayloads(splitSSEPayloads(chunk, nil))
}

// extractTokensFromPayloads is the split-once building block behind
// extractTokensFromStreamingChunk — callers that already hold
// splitSSEPayloads' result (e.g. tokenCapturingWriter.Write) call this
// directly to avoid re-splitting the same chunk.
func extractTokensFromPayloads(payloads [][]byte) int {
	for _, payload := range payloads {
		if tokens := extractOpenAITotalTokens(payload); tokens > 0 {
			return tokens
		}
		if tokens := extractMessagesTotalTokens(payload); tokens > 0 {
			return tokens
		}
	}
	return 0
}

func extractTokenUsageFromStreamingChunkWithOptions(chunk []byte, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	return extractTokenUsageFromPayloads(splitSSEPayloads(chunk, nil), opts)
}

// extractTokenUsageFromPayloads is the split-once building block behind
// extractTokenUsageFromStreamingChunkWithOptions — callers that already hold
// splitSSEPayloads' result call this directly instead of re-splitting the
// same chunk (plan item C).
//
// Merges usage across every payload in the batch (via MergeNonZero) rather
// than returning the first non-nil hit: a single Read can surface multiple
// SSE frames (e.g. a web-search-only annotation frame followed by the final
// usage frame), and returning early on the first would silently drop the
// real prompt/completion tokens carried by a later frame in the same batch.
func extractTokenUsageFromPayloads(payloads [][]byte, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	var merged *converter.TokenUsage
	for _, payload := range payloads {
		if usage := converter.ExtractTokenUsageWithOptions(payload, opts); usage != nil {
			if merged == nil {
				merged = &converter.TokenUsage{}
			}
			merged.MergeNonZero(usage)
		}
	}
	return merged
}

// rawJSONString mirrors a `.(string)` type assertion on a decoded
// interface{}, but on a json.RawMessage sub-slice: it only succeeds if the
// raw value is actually a JSON string, matching the exists+ok semantics the
// old map[string]interface{} code relied on (e.g. a JSON null or number
// under the same key must NOT be treated as a present string).
func rawJSONString(raw goccyjson.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '"' {
		return "", false
	}
	var s string
	if err := goccyjson.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

// rawJSONBool is the bool counterpart of rawJSONString.
func rawJSONBool(raw goccyjson.RawMessage) (bool, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || (trimmed[0] != 't' && trimmed[0] != 'f') {
		return false, false
	}
	var v bool
	if err := goccyjson.Unmarshal(raw, &v); err != nil {
		return false, false
	}
	return v, true
}

// extractSessionIDFromRawBody mirrors the extra_body/root-level session ID
// lookup that used to run against a fully-decoded map[string]interface{}.
// extra_body is small, so decoding it as its own map[string]RawMessage is
// cheap; the large fields of reqBody (messages/input) are never touched.
func extractSessionIDFromRawBody(reqBody map[string]goccyjson.RawMessage) string {
	if extraRaw, ok := reqBody["extra_body"]; ok {
		var extraBody map[string]goccyjson.RawMessage
		if err := goccyjson.Unmarshal(extraRaw, &extraBody); err == nil {
			if sid, ok := rawJSONString(extraBody["litellm_session_id"]); ok && sid != "" {
				return sid
			}
			if cid, ok := rawJSONString(extraBody["chat_id"]); ok && cid != "" {
				return cid
			}
			if sid, ok := rawJSONString(extraBody["session_id"]); ok && sid != "" {
				return sid
			}
		}
	}
	if sid, ok := rawJSONString(reqBody["litellm_session_id"]); ok && sid != "" {
		return sid
	}
	if sid, ok := rawJSONString(reqBody["session_id"]); ok && sid != "" {
		return sid
	}
	if pck, ok := rawJSONString(reqBody["prompt_cache_key"]); ok && pck != "" {
		return pck
	}
	if metadataRaw, ok := reqBody["metadata"]; ok {
		var metadata map[string]goccyjson.RawMessage
		if err := goccyjson.Unmarshal(metadataRaw, &metadata); err == nil {
			if sid, ok := rawJSONString(metadata["session_id"]); ok && sid != "" {
				return sid
			}
		}
	}
	return ""
}

// extractClientIDFromRawBody returns the per-client identifier
// (safety_identifier, then the user field it replaces, the precedence OpenAI
// applies). It is the last-resort session key, after the body session fields
// and the session headers: binding to it pins every conversation of a client
// to one credential instead of balancing each one.
func extractClientIDFromRawBody(reqBody map[string]goccyjson.RawMessage) string {
	id, _ := clientIDWithSource(reqBody)
	return id
}

// clientIDWithSource is extractClientIDFromRawBody plus which field actually
// matched. safety_identifier is a stable, router-caller-controlled client
// marker; user is the OpenAI field it replaces, and in practice is far more
// often an application-level end-user id a client already tracks for its
// own purposes than something sent to mean "this is one session" -- callers
// that log or reason about *why* a request became cache-eligible (not just
// whether) should keep the two apart rather than reporting both as one
// generic "client_id" source.
func clientIDWithSource(reqBody map[string]goccyjson.RawMessage) (id, source string) {
	if cid, ok := rawJSONString(reqBody["safety_identifier"]); ok && cid != "" {
		return cid, "safety_identifier"
	}
	if cid, ok := rawJSONString(reqBody["user"]); ok && cid != "" {
		return cid, "user_field"
	}
	return "", ""
}

// sessionIDHeaderPattern matches header names agents commonly use to carry a
// session identifier when they don't put one in the JSON body, e.g.
// "Session-Id", "X-Session-Id", "X-Codex-Session-Id".
var sessionIDHeaderPattern = regexp.MustCompile(`(?i)^(x-.+-)?session[-_]id$`)

// extractSessionIDFromHeaders is the fallback used when the request body
// carries no session identifier. Header iteration order is randomized by Go,
// so matching keys are sorted for a deterministic pick: a bare "Session-Id"
// sorts before any "X-...-Session-Id" variant and wins.
func extractSessionIDFromHeaders(header http.Header) string {
	var keys []string
	for k := range header {
		if sessionIDHeaderPattern.MatchString(k) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		if v := header.Get(k); v != "" {
			return v
		}
	}
	return ""
}

var rawJSONTrue = goccyjson.RawMessage("true")

// injectIncludeUsageRaw ensures reqBody["stream_options"]["include_usage"] is
// true, merging into whatever stream_options object the client sent rather
// than discarding it. This runs before the destination provider is resolved
// (sanitizeAndExtractRequestBody is called at request ingress, ahead of
// credential selection), so it cannot know yet whether a client-sent
// provider-specific key like vLLM's continuous_usage_stats should ultimately
// survive: that decision needs the resolved provider and is made later, in
// converter.ProviderConverter.RequestFrom (see shouldStripStreamOptionsExtras
// / RebuildStreamOptionsIncludeUsageOnly). Operates entirely on RawMessage
// sub-slices so the rest of reqBody (in particular messages/input) is never
// boxed into interface{}.
func injectIncludeUsageRaw(reqBody map[string]goccyjson.RawMessage) error {
	return setStreamOptionsRaw(reqBody, "include_usage")
}

// setStreamOptionsRaw sets each of keys to true in reqBody["stream_options"],
// merging into whatever stream_options object the client sent.
func setStreamOptionsRaw(reqBody map[string]goccyjson.RawMessage, keys ...string) error {
	streamOptions := streamOptionsRaw(reqBody)
	if streamOptions == nil {
		streamOptions = make(map[string]goccyjson.RawMessage, len(keys))
	}
	for _, key := range keys {
		streamOptions[key] = rawJSONTrue
	}

	marshaled, err := goccyjson.Marshal(streamOptions)
	if err != nil {
		return err
	}
	reqBody["stream_options"] = marshaled
	return nil
}

// streamOptionsRaw returns reqBody's stream_options object, nil when it is
// absent or not an object.
func streamOptionsRaw(reqBody map[string]goccyjson.RawMessage) map[string]goccyjson.RawMessage {
	raw, exists := reqBody["stream_options"]
	if !exists {
		return nil
	}
	var streamOptions map[string]goccyjson.RawMessage
	_ = goccyjson.Unmarshal(raw, &streamOptions)
	return streamOptions
}

// clientControlledBillingParams are request parameters through which a client
// could make the upstream bill us for more than the router charges back:
//   - service_tier: a more expensive service tier (e.g. xAI/OpenAI "priority",
//     billed at a premium the router does not price);
//   - deferred: xAI deferred completions, which return only a request_id —
//     the provider still runs and bills the generation, but the response the
//     router sees carries no usage to charge;
//   - search_parameters: xAI's legacy Live Search, billed per source by xAI
//     but not priced by the router.
//
// Responses background mode is the same kind of parameter; see
// deleteBackgroundMode.
var clientControlledBillingParams = []string{"service_tier", "deferred", "search_parameters"}

// isClientControlledBillingFormField reports whether a multipart field name
// sets one of clientControlledBillingParams, directly or through extra_body.
func isClientControlledBillingFormField(name string) bool {
	for _, param := range clientControlledBillingParams {
		if name == param || name == "extra_body["+param+"]" || name == "extra_body."+param {
			return true
		}
	}
	return false
}

// deleteClientControlledBillingParams removes clientControlledBillingParams
// and Responses background mode from one JSON object level, reporting whether
// anything was removed.
func deleteClientControlledBillingParams(object map[string]goccyjson.RawMessage) bool {
	changed := deleteBackgroundMode(object)
	for _, param := range clientControlledBillingParams {
		if _, exists := object[param]; exists {
			delete(object, param)
			changed = true
		}
	}
	return changed
}

// deleteBackgroundMode removes a Responses API "background": true from one
// JSON object level, reporting whether it was removed. A background response
// comes back "queued" with no usage while the provider still runs and bills
// the generation, which the router can neither fetch nor charge for; without
// the flag the request runs as a regular one. Only the boolean is removed:
// the Images API has its own string "background", so it cannot be a
// clientControlledBillingParams entry.
func deleteBackgroundMode(object map[string]goccyjson.RawMessage) bool {
	if background, ok := rawJSONBool(object["background"]); !ok || !background {
		return false
	}
	delete(object, "background")
	return true
}

// appendServiceTier appends object's service_tier to tiers when it is a
// non-empty string.
func appendServiceTier(tiers []string, object map[string]goccyjson.RawMessage) []string {
	if tier, _ := rawJSONString(object["service_tier"]); tier != "" {
		return append(tiers, tier)
	}
	return tiers
}

// stripClientControlledBillingParams removes clientControlledBillingParams
// and Responses background mode from the two request locations through which
// clients may set them: the top level and extra_body. It is intentionally not
// recursive: the same names in metadata, messages, or tool schemas are user
// data and must be preserved. It returns the service tiers it removed, so they
// can still be checked against the model they reach (see clientServiceTiers).
func stripClientControlledBillingParams(reqBody map[string]goccyjson.RawMessage) (bool, []string, error) {
	serviceTiers := appendServiceTier(nil, reqBody)
	changed := deleteClientControlledBillingParams(reqBody)

	extraBodyRaw, exists := reqBody["extra_body"]
	if !exists {
		return changed, serviceTiers, nil
	}
	trimmed := bytes.TrimSpace(extraBodyRaw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return changed, serviceTiers, nil
	}

	var extraBody map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(extraBodyRaw, &extraBody); err != nil {
		return changed, serviceTiers, err
	}
	serviceTiers = appendServiceTier(serviceTiers, extraBody)
	if !deleteClientControlledBillingParams(extraBody) {
		return changed, serviceTiers, nil
	}

	sanitizedExtraBody, err := goccyjson.Marshal(extraBody)
	if err != nil {
		return true, serviceTiers, err
	}
	reqBody["extra_body"] = sanitizedExtraBody
	return true, serviceTiers, nil
}

type sanitizedRequestBody struct {
	Body      []byte
	ModelID   string
	Streaming bool
	SessionID string
	ClientID  string
	// ClientIDSource is "safety_identifier" or "user_field", saying which of
	// the two actually produced ClientID; "" when ClientID is empty. See
	// clientIDWithSource's doc comment for why callers should keep these
	// apart rather than reporting both as one generic "client id" signal.
	ClientIDSource string
	Changed        bool
	// ServiceTiers are the client's service_tier values (top level, then
	// extra_body), removed from Body.
	ServiceTiers []string
	// scan is the top-level layout of Body when it is already known (a JSON
	// body the scanner accepted); nil otherwise.
	scan *jsonscan.Top
}

var errInvalidMultipartRequestBody = errors.New("invalid multipart request body")

type requestMultipartPart struct {
	header textproto.MIMEHeader
	data   []byte
}

func cloneMIMEHeader(header textproto.MIMEHeader) textproto.MIMEHeader {
	clone := make(textproto.MIMEHeader, len(header))
	for key, values := range header {
		clone[key] = append([]string(nil), values...)
	}
	return clone
}

func invalidMultipartError(err error) error {
	if err == nil {
		return errInvalidMultipartRequestBody
	}
	return fmt.Errorf("%w: %v", errInvalidMultipartRequestBody, err)
}

func declaredMediaType(contentType string) string {
	base, _, _ := strings.Cut(contentType, ";")
	return strings.TrimSpace(base)
}

// sanitizeAndExtractRequestBody sanitizes client-controlled request fields at
// the common ingress boundary before any provider, retry, or fallback body is
// derived. Multipart bodies are parsed to completion even when no rewrite is
// needed so malformed input cannot pass through after an early model part.
func sanitizeAndExtractRequestBody(body []byte, contentType string, isMessagesAPI bool) (sanitizedRequestBody, error) {
	result := sanitizedRequestBody{Body: body}
	mediaType, params, mediaTypeErr := mime.ParseMediaType(contentType)
	if mediaTypeErr != nil {
		if strings.EqualFold(mediaType, "multipart/form-data") ||
			strings.EqualFold(declaredMediaType(contentType), "multipart/form-data") {
			return sanitizedRequestBody{}, invalidMultipartError(mediaTypeErr)
		}
	} else if strings.EqualFold(mediaType, "multipart/form-data") {
		return sanitizeMultipartRequestBody(body, params)
	}
	if len(body) == 0 {
		return result, nil
	}

	return sanitizeJSONRequestBody(body, isMessagesAPI)
}

// sanitizeJSONRequestBody reads the handful of top-level fields the router
// acts on and removes or rewrites the ones it must. The body is scanned, not
// decoded: the fields are read in place, and a body that has to change is
// rebuilt by copying every untouched member as it is (jsonscan.Top.Rewrite)
// rather than decoding and re-encoding the conversation.
func sanitizeJSONRequestBody(body []byte, isMessagesAPI bool) (sanitizedRequestBody, error) {
	top, ok := jsonscan.ScanTop(body)
	if !ok {
		// Not a JSON object by the scanner's reading. The decode-based path
		// keeps deciding what happens to such a body, exactly as before.
		return sanitizeJSONRequestBodyDecode(body, isMessagesAPI)
	}
	reqBody := make(map[string]goccyjson.RawMessage, 16)
	top.Range(func(name string, raw []byte) { reqBody[name] = raw })
	original := maps.Clone(reqBody)

	result, changed, err := sanitizeRequestFields(body, reqBody, isMessagesAPI)
	if err != nil {
		return result, err
	}
	result.scan = top
	if !changed {
		return result, nil
	}

	edits := requestFieldEdits(original, reqBody)
	// A body that is sent on changed used to be encoded again from its
	// decoded members, which also keeps one of each duplicated name and
	// repairs a name that is not valid UTF-8; so is one that needs either.
	if len(edits) > 0 || top.HasDuplicates() || top.HasInvalidNames() {
		result.scan = top.RewriteTop(edits...)
		result.Body = result.scan.Bytes()
	}
	// Changed is reported exactly when the decode-based path reported it,
	// including the case where the fields already had the values the router
	// wants and the bytes above are the client's own: callers key header
	// clean-up off it, not only the body.
	result.Changed = true
	return result, nil
}

// requestFieldEdits lists what sanitizeRequestFields did to the fields:
// removals, replacements and additions, in a stable order.
func requestFieldEdits(original, current map[string]goccyjson.RawMessage) []jsonscan.Edit {
	var edits []jsonscan.Edit
	for _, name := range slices.Sorted(maps.Keys(original)) {
		now, kept := current[name]
		switch {
		case !kept:
			edits = append(edits, jsonscan.Edit{Name: name})
		case !bytes.Equal(now, original[name]):
			if name == "stream_options" && streamOptionsIncludeUsage(original[name]) && streamOptionsIncludeUsage(now) &&
				sameJSONObject(original[name], now) {
				continue // already what the router asks for: keep the client's bytes
			}
			edits = append(edits, jsonscan.Edit{Name: name, Raw: now})
		}
	}
	for _, name := range slices.Sorted(maps.Keys(current)) {
		if _, existed := original[name]; !existed {
			edits = append(edits, jsonscan.Edit{Name: name, Raw: current[name]})
		}
	}
	return edits
}

// streamOptionsIncludeUsage reports whether raw is an object whose
// include_usage is true.
func streamOptionsIncludeUsage(raw []byte) bool {
	var options map[string]goccyjson.RawMessage
	if goccyjson.Unmarshal(raw, &options) != nil {
		return false
	}
	on, isBool := rawJSONBool(options["include_usage"])
	return isBool && on
}

// sameJSONObject reports whether client, a JSON object the client sent, says
// exactly what rewritten says: the same value once decoded, and no repeated
// member that a decode would have silently dropped. Used only on
// stream_options, a few dozen bytes.
func sameJSONObject(client, rewritten []byte) bool {
	var cv, rv interface{}
	if json.Unmarshal(client, &cv) != nil || json.Unmarshal(rewritten, &rv) != nil {
		return false
	}
	// Invalid UTF-8 counts as a difference too: the rewritten object has it
	// replaced, and decoding hides that.
	top, ok := jsonscan.ScanTop(client)
	return ok && !top.HasDuplicates() && utf8.Valid(client) && reflect.DeepEqual(cv, rv)
}

// sanitizeJSONRequestBodyDecode is the decode-based implementation: the whole
// body is decoded into raw members and, when anything changes, encoded again.
// It handles the bodies the scanner refuses and is the reference the scanning
// path is tested against.
func sanitizeJSONRequestBodyDecode(body []byte, isMessagesAPI bool) (sanitizedRequestBody, error) {
	result := sanitizedRequestBody{Body: body}

	// Parse JSON body — RawMessage sub-slices instead of interface{} boxing:
	// only six scalar top-level fields are ever read, so the large
	// messages/input field never needs to be decoded (see doc comment above).
	var reqBody map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(body, &reqBody); err != nil {
		return result, nil // Existing invalid-body handling reports missing model.
	}

	result, changed, err := sanitizeRequestFields(body, reqBody, isMessagesAPI)
	if err != nil || !changed {
		return result, err
	}
	// Marshal back to JSON — untouched fields (messages/input, etc.) pass
	// through as raw bytes, no re-encoding cost.
	modifiedBody, err := goccyjson.Marshal(reqBody)
	if err != nil {
		return sanitizedRequestBody{}, err
	}

	result.Body = modifiedBody
	result.Changed = true
	return result, nil
}

// sanitizeRequestFields is the decision part shared by both paths: it reads
// what the router needs from reqBody (the request's top-level members) and
// applies the removals and rewrites to that map. changed reports whether the
// request has to be sent on in a different form than it arrived.
func sanitizeRequestFields(body []byte, reqBody map[string]goccyjson.RawMessage, isMessagesAPI bool) (sanitizedRequestBody, bool, error) {
	result := sanitizedRequestBody{Body: body}

	changed, serviceTiers, err := stripClientControlledBillingParams(reqBody)
	if err != nil {
		return sanitizedRequestBody{}, false, err
	}
	result.ServiceTiers = serviceTiers

	model, ok := rawJSONString(reqBody["model"])
	if !ok {
		return result, false, nil
	}
	result.ModelID = model

	// Extract session ID before removing LiteLLM-only request metadata.
	result.SessionID = extractSessionIDFromRawBody(reqBody)
	result.ClientID, result.ClientIDSource = clientIDWithSource(reqBody)
	// litellm_session_id/session_id only ever drive AIR's own sticky-routing
	// decision (already captured above into result.SessionID); no provider
	// understands either as a wire-protocol field, and both are gone from
	// reqBody before routing even happens, so removing them here can't affect
	// which credential gets picked. inference_geo and trace are the same
	// shape: internal-only metadata no provider needs, safe to drop for
	// everyone regardless of destination -- unlike cache_salt/stream_options,
	// there's no backend confirmed to actually use these.
	for _, key := range [...]string{"litellm_session_id", "session_id", "inference_geo", "trace"} {
		if _, exists := reqBody[key]; exists {
			delete(reqBody, key)
			changed = true
		}
	}

	// Check if this is a streaming request
	stream, _ := rawJSONBool(reqBody["stream"])
	result.Streaming = stream
	if !stream && !changed {
		return result, false, nil
	}

	// Responses API (/v1/responses) uses "input" instead of "messages" and does NOT
	// support stream_options — it always returns usage in streaming. The native
	// Anthropic Messages API (/v1/messages) also has a "messages" field but has no
	// stream_options concept at all — real api.anthropic.com rejects it outright
	// ("stream_options: Extra inputs are not permitted"), so it must be excluded here
	// too, not just detected by field shape.
	// Only inject stream_options for Chat Completions API requests.
	_, hasInput := reqBody["input"]
	_, hasMessages := reqBody["messages"]
	isResponsesAPI := hasInput && !hasMessages

	if stream && !isResponsesAPI && !isMessagesAPI {
		// Ensure stream_options exists and include_usage is true (Chat Completions only)
		if err := injectIncludeUsageRaw(reqBody); err != nil {
			return sanitizedRequestBody{}, false, err
		}
		changed = true
	}

	return result, changed, nil
}

func sanitizeMultipartRequestBody(body []byte, params map[string]string) (sanitizedRequestBody, error) {
	boundary := params["boundary"]
	if boundary == "" {
		return sanitizedRequestBody{}, invalidMultipartError(errors.New("missing boundary"))
	}
	boundaryValidator := multipart.NewWriter(io.Discard)
	if err := boundaryValidator.SetBoundary(boundary); err != nil {
		return sanitizedRequestBody{}, invalidMultipartError(err)
	}
	if !hasMultipartClosingBoundary(body, boundary) {
		return sanitizedRequestBody{}, invalidMultipartError(errors.New("missing closing boundary"))
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	result := sanitizedRequestBody{Body: body}
	parts := make([]requestMultipartPart, 0, 8)
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return sanitizedRequestBody{}, invalidMultipartError(err)
		}

		disposition, dispositionParams, err := mime.ParseMediaType(part.Header.Get("Content-Disposition"))
		if err != nil || !strings.EqualFold(disposition, "form-data") {
			_ = part.Close()
			if err == nil {
				err = errors.New("invalid content disposition")
			}
			return sanitizedRequestBody{}, invalidMultipartError(err)
		}
		name := dispositionParams["name"]
		_, hasFilename := dispositionParams["filename"]
		data, readErr := io.ReadAll(part)
		closeErr := part.Close()
		if readErr != nil {
			return sanitizedRequestBody{}, invalidMultipartError(readErr)
		}
		if closeErr != nil {
			return sanitizedRequestBody{}, invalidMultipartError(closeErr)
		}

		if isClientControlledBillingFormField(name) {
			result.Changed = true
			continue
		}

		if name == "extra_body" && !hasFilename {
			trimmed := bytes.TrimSpace(data)
			if len(trimmed) > 0 && trimmed[0] == '{' {
				var extraBody map[string]goccyjson.RawMessage
				if err := goccyjson.Unmarshal(data, &extraBody); err != nil {
					return sanitizedRequestBody{}, invalidMultipartError(err)
				}
				if deleteClientControlledBillingParams(extraBody) {
					data, err = goccyjson.Marshal(extraBody)
					if err != nil {
						return sanitizedRequestBody{}, invalidMultipartError(err)
					}
					result.Changed = true
				}
			}
		}

		if !hasFilename {
			value := strings.TrimSpace(string(data))
			switch name {
			case "model":
				result.ModelID = value
			case "session_id":
				result.SessionID = value
			case "user":
				result.ClientID = value
				result.ClientIDSource = "user_field"
			}
		}

		parts = append(parts, requestMultipartPart{
			header: cloneMIMEHeader(part.Header),
			data:   data,
		})
	}

	if !result.Changed {
		return result, nil
	}

	var rewritten bytes.Buffer
	writer := multipart.NewWriter(&rewritten)
	if err := writer.SetBoundary(boundary); err != nil {
		return sanitizedRequestBody{}, invalidMultipartError(err)
	}
	for _, part := range parts {
		dst, err := writer.CreatePart(part.header)
		if err != nil {
			return sanitizedRequestBody{}, invalidMultipartError(err)
		}
		if _, err := dst.Write(part.data); err != nil {
			return sanitizedRequestBody{}, invalidMultipartError(err)
		}
	}
	if err := writer.Close(); err != nil {
		return sanitizedRequestBody{}, invalidMultipartError(err)
	}
	result.Body = rewritten.Bytes()
	return result, nil
}

func replaceRequestModel(body []byte, contentType, modelID string) ([]byte, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err == nil && strings.EqualFold(mediaType, "multipart/form-data") {
		return replaceMultipartRequestModel(body, params["boundary"], modelID)
	}

	var request map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(body, &request); err != nil {
		return body, nil
	}
	currentModel, ok := rawJSONString(request["model"])
	if !ok || currentModel == modelID {
		return body, nil
	}
	return openai.ReplaceModelInBody(body, currentModel, modelID), nil
}

func replaceMultipartRequestModel(body []byte, boundary, modelID string) ([]byte, error) {
	if boundary == "" {
		return nil, invalidMultipartError(errors.New("missing boundary"))
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	parts := make([]requestMultipartPart, 0, 8)
	changed := false
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, invalidMultipartError(err)
		}

		name := part.FormName()
		filename := part.FileName()
		header := cloneMIMEHeader(part.Header)
		data, readErr := io.ReadAll(part)
		closeErr := part.Close()
		if readErr != nil {
			return nil, invalidMultipartError(readErr)
		}
		if closeErr != nil {
			return nil, invalidMultipartError(closeErr)
		}
		if name == "model" && filename == "" && strings.TrimSpace(string(data)) != modelID {
			data = []byte(modelID)
			changed = true
		}
		parts = append(parts, requestMultipartPart{header: header, data: data})
	}
	if !changed {
		return body, nil
	}

	var rewritten bytes.Buffer
	writer := multipart.NewWriter(&rewritten)
	if err := writer.SetBoundary(boundary); err != nil {
		return nil, invalidMultipartError(err)
	}
	for _, part := range parts {
		dst, err := writer.CreatePart(part.header)
		if err != nil {
			return nil, invalidMultipartError(err)
		}
		if _, err := dst.Write(part.data); err != nil {
			return nil, invalidMultipartError(err)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, invalidMultipartError(err)
	}
	return rewritten.Bytes(), nil
}

func hasMultipartClosingBoundary(body []byte, boundary string) bool {
	marker := []byte("--" + boundary + "--")
	index := bytes.LastIndex(body, marker)
	if index < 0 {
		return false
	}
	if index > 0 && (index < 2 || !bytes.Equal(body[index-2:index], []byte("\r\n"))) {
		return false
	}
	after := body[index+len(marker):]
	return len(after) == 0 || bytes.HasPrefix(after, []byte("\r\n"))
}

// webSearchRequestUsage is extractWebSearchRequestUsage reusing the scan the
// request already holds for this body, if it holds one.
func webSearchRequestUsage(logCtx *RequestLogContext, body []byte, contentType string) (bool, string) {
	if len(body) == 0 || strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		return false, ""
	}
	return webSearchRequestUsageFromScan(logCtx.scanBody(body))
}

func extractWebSearchRequestUsage(body []byte, contentType string) (bool, string) {
	if len(body) == 0 || strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		return false, ""
	}

	// Only three fields are read, so only they are decoded: the body is
	// scanned for its top-level layout and the conversation next to them is
	// skipped. Each field is filled from every key encoding/json would match
	// to it, in document order, which is what decoding the whole body into
	// a struct of the three does.
	top, _ := jsonscan.ScanTop(body)
	return webSearchRequestUsageFromScan(top)
}

// webSearchRequestUsageFromScan is extractWebSearchRequestUsage for a JSON
// body that has already been scanned (nil for one that is not a JSON object).
func webSearchRequestUsageFromScan(top *jsonscan.Top) (bool, string) {
	if top == nil {
		// Not a JSON object. A full decode accepts one more document, the
		// literal null, and finds nothing in it either.
		return false, ""
	}
	var req struct {
		WebSearchOptions map[string]interface{}
		Tools            []map[string]interface{}
		Plugins          []map[string]interface{}
	}
	decoded := true
	for _, field := range [...]struct {
		name string
		dst  interface{}
	}{
		{"web_search_options", &req.WebSearchOptions},
		{"tools", &req.Tools},
		{"plugins", &req.Plugins},
	} {
		for _, raw := range top.RawFoldAll(field.name) {
			if err := json.Unmarshal(raw, field.dst); err != nil {
				decoded = false
			}
		}
	}
	if !decoded {
		return false, ""
	}

	if req.WebSearchOptions != nil {
		return true, webSearchContextSizeFromMap(req.WebSearchOptions)
	}
	for _, tool := range req.Tools {
		if !isWebSearchTool(tool) {
			continue
		}
		return true, webSearchContextSizeFromMap(tool)
	}
	// OpenRouter's paid web-search plugin: a client asking for it (and only
	// preserved on the wire for genuine OpenRouter, see
	// converter.shouldStripOpenRouterOnlyFields) is a billable request the
	// same way web_search_options/tools are -- without this, pre-billing and
	// quota checks never see it. Convention per OpenRouter's own docs
	// (id: "web" enables the plugin); not independently verified against a
	// live OpenRouter response this session.
	for _, plugin := range req.Plugins {
		if !isWebSearchPlugin(plugin) {
			continue
		}
		return true, webSearchContextSizeFromMap(plugin)
	}
	return false, ""
}

func isWebSearchTool(tool map[string]interface{}) bool {
	toolType, _ := tool["type"].(string)
	// google_search is the Gemini route's alias (see vertex.GoogleSearchTypes).
	return toolType == "web_search" || strings.HasPrefix(toolType, "web_search_") || toolType == "google_search"
}

func isWebSearchPlugin(plugin map[string]interface{}) bool {
	id, _ := plugin["id"].(string)
	return id == "web"
}

func webSearchContextSizeFromMap(values map[string]interface{}) string {
	if values == nil {
		return converter.NormalizeWebSearchContextSize("")
	}
	size, _ := values["search_context_size"].(string)
	return converter.NormalizeWebSearchContextSize(size)
}

// decodeResponseBody decodes the response body based on Content-Encoding
func decodeResponseBody(body []byte, encoding string) string {
	lowerEncoding := strings.ToLower(encoding)

	// Check if response is gzip-encoded
	if strings.Contains(lowerEncoding, "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return string(body) // Return as-is if can't decode
		}
		defer func() {
			_ = reader.Close()
		}()

		decoded, err := io.ReadAll(reader)
		if err != nil {
			return string(body) // Return as-is if can't read
		}
		return string(decoded)
	}

	// Check if response is deflate-encoded
	if strings.Contains(lowerEncoding, "deflate") {
		reader := flate.NewReader(bytes.NewReader(body))
		defer func() {
			_ = reader.Close()
		}()

		decoded, err := io.ReadAll(reader)
		if err != nil {
			return string(body) // Return as-is if can't read
		}
		return string(decoded)
	}

	// Return as plain text
	return string(body)
}

func decodeResponseBodyPrefix(body []byte, encoding string, limit int64) []byte {
	if limit <= 0 {
		return nil
	}

	lowerEncoding := strings.ToLower(encoding)
	if strings.Contains(lowerEncoding, "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return truncateBytes(body, limit)
		}
		defer func() {
			_ = reader.Close()
		}()
		decoded, err := io.ReadAll(io.LimitReader(reader, limit))
		if err != nil {
			return truncateBytes(body, limit)
		}
		return decoded
	}

	if strings.Contains(lowerEncoding, "deflate") {
		reader := flate.NewReader(bytes.NewReader(body))
		defer func() {
			_ = reader.Close()
		}()
		decoded, err := io.ReadAll(io.LimitReader(reader, limit))
		if err != nil {
			return truncateBytes(body, limit)
		}
		return decoded
	}

	return truncateBytes(body, limit)
}

func truncateBytes(body []byte, limit int64) []byte {
	if int64(len(body)) <= limit {
		return body
	}
	return body[:limit]
}

// extractTokensFromResponse extracts total_tokens from the response body
// Supports both OpenAI format (usage.total_tokens) and Vertex AI format (usageMetadata.totalTokenCount)
// Takes []byte (not string) because every caller already holds the response
// body as []byte — a string param would force a full copy in and back out
// for no reason, doubling the cost of scanning large (e.g. embedding-vector)
// bodies just to pull out a usage count.
func extractTokensFromResponse(body []byte, credType config.ProviderType) int {
	if len(body) == 0 {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == config.ProviderTypeVertexAI || credType == config.ProviderTypeGemini {
		var vertexResp struct {
			UsageMetadata struct {
				TotalTokenCount int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}

		if err := goccyjson.Unmarshal(body, &vertexResp); err != nil {
			return 0
		}
		return vertexResp.UsageMetadata.TotalTokenCount
	}

	// For OpenAI and other providers, use standard format
	return extractOpenAITotalTokens(body)
}

func setStreamTrue(body []byte) []byte {
	return updateStreamRequest(body, true, false)
}

// injectStreamOptions requests usage while preserving provider-specific options.
func injectStreamOptions(body []byte) []byte {
	return updateStreamRequest(body, false, true)
}

func updateStreamRequest(body []byte, forceStream, includeUsage bool) []byte {
	var fields map[string]jsontext.Value
	if err := jsonv2.Unmarshal(body, &fields); err != nil {
		return updateStreamRequestLegacy(body, forceStream, includeUsage)
	}
	if fields == nil {
		return body
	}
	if forceStream {
		fields["stream"] = jsontext.Value("true")
	}
	if includeUsage {
		var options map[string]jsontext.Value
		if raw := fields["stream_options"]; raw.Kind() == '{' {
			if err := jsonv2.Unmarshal(raw, &options); err != nil {
				return updateStreamRequestLegacy(body, forceStream, includeUsage)
			}
		} else {
			options = make(map[string]jsontext.Value, 1)
		}
		options["include_usage"] = jsontext.Value("true")
		raw, err := jsonv2.Marshal(options, json.DefaultOptionsV1())
		if err != nil {
			return updateStreamRequestLegacy(body, forceStream, includeUsage)
		}
		fields["stream_options"] = raw
	}
	modified, err := jsonv2.Marshal(fields, json.DefaultOptionsV1())
	if err != nil {
		return updateStreamRequestLegacy(body, forceStream, includeUsage)
	}
	return modified
}

// Preserve legacy handling of duplicate keys, invalid UTF-8 and suffixes
// accepted by the original single-value decoder.
func updateStreamRequestLegacy(body []byte, forceStream, includeUsage bool) []byte {
	var fields map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&fields); err != nil || fields == nil {
		return body
	}
	if forceStream {
		fields["stream"] = true
	}
	if includeUsage {
		options, ok := fields["stream_options"].(map[string]any)
		if !ok {
			options = make(map[string]any, 1)
			fields["stream_options"] = options
		}
		options["include_usage"] = true
	}
	modified, err := json.Marshal(fields)
	if err != nil {
		return body
	}
	return modified
}
