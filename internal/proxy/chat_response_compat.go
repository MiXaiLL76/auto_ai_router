package proxy

import (
	"bufio"
	"bytes"
	"errors"
	"io"

	// goccy/go-json for every JSON call in this file: normalizeSuccessfulResponseModel
	// full-body-decodes+remarshals every non-streaming response (benchmarked
	// ~1.66x faster than encoding/json against a real embedding body, 1536-float
	// vector). normalizeSSEDataLineModel/decodeShallowJSONObject were originally
	// left on stdlib on the assumption that small per-SSE-chunk map decodes
	// wouldn't show the same win — re-benchmarked after the
	// token_estimator.go streaming fixes proved that assumption wrong even for
	// tiny typed-struct payloads; goccy measured ~1.5x faster here too
	// (map[string]interface{} decode+encode, not just typed structs).
	goccyjson "github.com/goccy/go-json"
)

const maxSSEModelRewriteLineBytes = 64 * 1024 * 1024

// existingModelOnlyResponseRoutes are surfaces where a top-level "model" is
// rewritten to the client alias when the backend already emits one, but is
// never added when absent. This keeps schema-less image passthrough shapes
// (OpenAI gpt-image-1, Imagen) byte-identical while still presenting the alias
// for backends that do echo a model (canonical Gemini image, and any other
// OpenAI-compatible image provider that returns "model") — matching the alias
// every other route reports. System One replies get the same treatment: only
// the model they name changes, every other field keeps its raw value.
var existingModelOnlyResponseRoutes = map[string]struct{}{
	"/v1/images/generations": {},
	"/v1/images/edits":       {},
	systemOnePath:            {},
}

// modelBearingResponseRoutes is the product surface whose successful response
// schema exposes a client-visible model and may receive one when absent. Images
// use existingModelOnlyResponseRoutes above so GPT passthrough shapes stay unchanged.
var modelBearingResponseRoutes = map[string]struct{}{
	"/v1/chat/completions": {},
	"/v1/completions":      {},
	"/v1/embeddings":       {},
	"/v1/responses":        {},
}

// Rewrites only the top-level "model" field, via map[string]json.RawMessage
// (same technique as rewriteJSONResponseModel in model_alias_response.go) so
// the rest of the body — e.g. a 1536-float embedding vector — is copied
// through as opaque bytes instead of being deep-decoded into
// map[string]interface{} (profiled: that was ~21% of total CPU under
// embeddings load, entirely to leave everything but "model" unchanged).
func normalizeSuccessfulResponseModel(body []byte, endpoint, publicModel string) []byte {
	if publicModel == "" {
		return body
	}
	_, modelBearing := modelBearingResponseRoutes[endpoint]
	_, existingModelOnly := existingModelOnlyResponseRoutes[endpoint]
	if !modelBearing && !existingModelOnly {
		return body
	}
	// existing-model-only routes never add "model"; skip the unmarshal entirely
	// when the body can't contain the field (gpt-image-1 / Imagen passthrough).
	if existingModelOnly && !modelBearing && !bytes.Contains(body, []byte(`"model"`)) {
		return body
	}

	var response map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(body, &response); err != nil || response == nil {
		return body
	}
	if rawModel, ok := response["model"]; ok {
		var currentModel string
		if err := goccyjson.Unmarshal(rawModel, &currentModel); err == nil && currentModel == publicModel {
			return body
		}
	} else if existingModelOnly {
		return body
	}
	modelJSON, err := goccyjson.Marshal(publicModel)
	if err != nil {
		return body
	}
	response["model"] = modelJSON
	normalized, err := goccyjson.Marshal(response)
	if err != nil {
		return body
	}
	return normalized
}

func clientVisibleResponseModel(logCtx *RequestLogContext, fallback string) string {
	if logCtx == nil {
		return fallback
	}
	if logCtx.PublicModelID != "" {
		return logCtx.PublicModelID
	}
	return fallback
}

func streamRouteReturnsModel(logCtx *RequestLogContext) bool {
	if logCtx == nil || logCtx.Request == nil || logCtx.Request.URL == nil {
		return false
	}
	_, ok := modelBearingResponseRoutes[logCtx.Request.URL.Path]
	return ok
}

// normalizeSuccessfulResponseModelStream rewrites only JSON carried by SSE
// data lines. Event names, blank-line framing, [DONE], malformed payloads and
// provider error events are retained byte-for-byte.
func normalizeSuccessfulResponseModelStream(
	reader io.Reader,
	statusCode int,
	logCtx *RequestLogContext,
	fallbackModel string,
) io.Reader {
	if reader == nil || statusCode < 200 || statusCode >= 300 || !streamRouteReturnsModel(logCtx) {
		return reader
	}
	publicModel := clientVisibleResponseModel(logCtx, fallbackModel)
	if publicModel == "" {
		return reader
	}
	return &responseModelStreamReader{
		source:                      bufio.NewReader(reader),
		publicModel:                 publicModel,
		preserveNestedResponseModel: responseCompatRequestFromContext(logCtx.Request.Context()) != nil,
		maxLineBytes:                maxSSEModelRewriteLineBytes,
	}
}

type responseModelStreamReader struct {
	source                      *bufio.Reader
	publicModel                 string
	preserveNestedResponseModel bool
	pending                     []byte
	pendingErr                  error
	line                        []byte
	passthrough                 bool
	maxLineBytes                int
}

func (r *responseModelStreamReader) Read(dst []byte) (int, error) {
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
		if r.passthrough {
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
		r.pending = normalizeSSEDataLineModelWithOptions(r.line, r.publicModel, r.preserveNestedResponseModel)
		r.line = nil
		r.pendingErr = err
	}

	n := copy(dst, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func normalizeSSEDataLineModel(line []byte, publicModel string) []byte {
	return normalizeSSEDataLineModelWithOptions(line, publicModel, false)
}

func normalizeSSEDataLineModelWithOptions(line []byte, publicModel string, preserveNestedResponseModel bool) []byte {
	newline := []byte(nil)
	body := line
	switch {
	case bytes.HasSuffix(body, []byte("\r\n")):
		newline = []byte("\r\n")
		body = body[:len(body)-2]
	case bytes.HasSuffix(body, []byte("\n")):
		newline = []byte("\n")
		body = body[:len(body)-1]
	}
	if !bytes.HasPrefix(body, []byte("data:")) {
		return line
	}

	rest := body[len("data:"):]
	payload := bytes.TrimSpace(rest)
	if len(payload) == 0 || payload[0] != '{' {
		return line
	}
	// A chunk can only carry a model to rewrite if it has a "model" field of its
	// own, or an "object" naming a shape the alias is ADDED to (a chunk
	// without a model), or a nested "response" object (the /v1/responses
	// shape). Nothing else in the payload matters here.
	//
	// The needles are the field names followed by their closing quote --
	// model", object", response" -- and not the bare names. A Responses API
	// delta carries the word "response" in the VALUE of "type"
	// ("response.output_text.delta"), so a bare "response" matched every
	// delta of every /v1/responses stream and made the gate decode them all
	// for nothing: the thousands of delta events of one response each paid
	// a decode and 15 allocations, and the rewrite still did not happen.
	// A delta event carries no "model", no "object" and no nested
	// "response" object, so the closing-quote needles skip it.
	//
	// The closing quote is also what keeps the needle off the slow path:
	// bytes.Contains scans with a first-byte filter, and a needle starting
	// with a quote has to compare the quote at every key of the payload,
	// while one starting with a letter does not. Measured on a
	// chat.completion.chunk: `"model"` costs ~73 ns, `model"` ~21 ns. The
	// length of the needle is not what matters -- `"modelxxxxx"` costs the
	// same ~73 ns as `"model"`.
	//
	// The gate deliberately does NOT look for "chat.completion.chunk" or
	// "text_completion": by the OpenAI spec every such event carries
	// "object":"chat.completion.chunk", so those needles were present in
	// every chat chunk and the gate never skipped anything — it only added
	// four scans per chunk to a path that then decoded the chunk anyway.
	// The object name is checked below, on the decoded event, where it is
	// needed to decide whether the alias is added.
	//
	// Short-circuited: a chunk with a "model" (the common case) costs one
	// scan.
	//
	// A needle may still match a VALUE rather than a key -- {"note":"model"}
	// matches model" -- which only means such a chunk is decoded when it
	// could have been skipped: a decode that finds no "model" field changes
	// nothing, so the client sees the same bytes either way.
	//
	// A backslash means something in the payload is escaped, and an escaped
	// KEY can spell "model" without those bytes being there
	// ("model" is "model"). Such a chunk must be decoded, so the
	// backslash check short-circuits the gate to "decode it" — the same
	// guard the previous gate had.
	if !bytes.Contains(payload, []byte(`model"`)) &&
		!bytes.Contains(payload, []byte(`object"`)) &&
		!bytes.Contains(payload, []byte(`response"`)) &&
		bytes.IndexByte(payload, '\\') < 0 {
		return line
	}
	// Shallow decode: every field but "model" (and, on /v1/responses, the
	// nested response object) is kept as opaque bytes so a multi-MB base64
	// image delta is never expanded into a map[string]interface{} tree and
	// re-marshaled — same technique as normalizeSuccessfulResponseModel.
	event, err := decodeShallowJSONObject(payload)
	if err != nil || event == nil {
		return line
	}
	rawError, hasError := event["error"]
	eventType := rawMessageString(event["type"])
	if isStreamErrorEvent(eventType, hasError && !isJSONNull(rawError)) {
		return line
	}

	changed := false
	if _, exists := event["model"]; exists {
		if rawMessageString(event["model"]) != publicModel {
			event["model"] = marshalJSONString(publicModel)
			changed = true
		}
	} else if object := rawMessageString(event["object"]); object == "chat.completion.chunk" || object == "text_completion" {
		event["model"] = marshalJSONString(publicModel)
		changed = true
	}
	if rawResponse, ok := event["response"]; ok && !preserveNestedResponseModel {
		if updated, nestedChanged := rewriteNestedResponseModel(rawResponse, publicModel); nestedChanged {
			event["response"] = updated
			changed = true
		}
	}
	if !changed {
		return line
	}

	normalized, err := goccyjson.Marshal(event)
	if err != nil {
		return line
	}
	prefixLen := len(rest) - len(bytes.TrimLeft(rest, " \t"))
	result := make([]byte, 0, len(body)+len(newline))
	result = append(result, []byte("data:")...)
	result = append(result, rest[:prefixLen]...)
	result = append(result, normalized...)
	result = append(result, newline...)
	return result
}

// decodeShallowJSONObject decodes exactly one JSON object, keeping each value as
// raw bytes, and rejects trailing content after it.
//
// The decode is a plain Unmarshal rather than a streaming Decoder: Unmarshal
// rejects trailing content the same way ("invalid character after
// top-level value"), so the outcome is unchanged, but it does not build a
// per-call RuntimeContext the way NewDecoder does — measured on a single
// chat.completion.chunk, ~630 ns and 18 allocations for the decoder against
// ~410 ns and 13 allocations for Unmarshal, on the path every streamed chunk
// of every response goes through. normalizeSuccessfulResponseModel already
// decodes this way.
//
// A JSON null decodes to no map and no error, exactly as the decoder left it;
// the caller treats a nil map as "nothing here" either way.
func decodeShallowJSONObject(data []byte) (map[string]goccyjson.RawMessage, error) {
	var object map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(data, &object); err != nil {
		return nil, err
	}
	return object, nil
}

// rewriteNestedResponseModel mirrors the /v1/responses handling of the old deep
// decode: if "response" is a JSON object it is given the client-visible model
// (added when absent), touching only that one nested field.
func rewriteNestedResponseModel(raw goccyjson.RawMessage, publicModel string) (goccyjson.RawMessage, bool) {
	var nested map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(raw, &nested); err != nil || nested == nil {
		return raw, false
	}
	if _, ok := nested["model"]; ok && rawMessageString(nested["model"]) == publicModel {
		return raw, false
	}
	nested["model"] = marshalJSONString(publicModel)
	updated, err := goccyjson.Marshal(nested)
	if err != nil {
		return raw, false
	}
	return updated, true
}

// rawMessageString returns the string value of a JSON raw message, or "" if it
// is absent or not a JSON string.
func rawMessageString(raw goccyjson.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := goccyjson.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

// isJSONNull reports whether a raw message is absent or the JSON literal null.
func isJSONNull(raw goccyjson.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// marshalJSONString encodes s as a JSON string, falling back to "" on the
// (practically impossible) marshal error.
func marshalJSONString(s string) goccyjson.RawMessage {
	encoded, err := goccyjson.Marshal(s)
	if err != nil {
		return goccyjson.RawMessage(`""`)
	}
	return encoded
}
