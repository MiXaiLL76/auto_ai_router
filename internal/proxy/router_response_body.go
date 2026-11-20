package proxy

import (
	"encoding/json"
	"strings"
)

// maxRouterResponseBodyBytes caps the captured response body. A chat
// completion is normally a few kilobytes, but a long one (a big tool
// argument, a long answer) can reach tens of kilobytes, and the Kafka
// producer's default message limit is 1 MB. The cap keeps one enormous
// response from filling the queue or pushing a batch over that limit.
const maxRouterResponseBodyBytes = 256 * 1024

// routerResponseRedactedPlaceholder replaces the model's own text when
// redaction is on. It is the same word the request-side masking uses, so an
// operator reads one vocabulary.
const routerResponseRedactedPlaceholder = "[REDACTED]"

// routerResponseTextFields are the keys whose values are the model's own
// output -- the text it wrote, the reasoning it produced, a refusal. They
// are replaced by the placeholder when redaction is on.
//
// A key is masked by its VALUE's shape, not by where it sits: "content" is
// the answer (a string) in a Chat completion but a container of parts (an
// array) in a Responses-API or Anthropic body, and "text" is the answer
// inside such a part. Deciding by position instead ("is this under a
// message-ish key?") missed the Anthropic-native body entirely -- its top
// level IS the message, with no "message"/"choices"/"output" key to be
// found -- and shipped the model's text and its tool-call arguments in the
// clear.
var routerResponseTextFields = map[string]struct{}{
	// Chat completions: message.content, delta.content.
	"content": {},
	// Chat completions: the model's reasoning, when the provider returns it
	// separately (deepseek's reasoning_content, OpenAI's reasoning).
	"reasoning_content": {},
	"reasoning":         {},
	// Responses API: an output part's text. Anthropic Messages: a text
	// block's text.
	"text": {},
	// Anthropic Messages: a thinking block.
	"thinking": {},
	// A refusal is the model's own words too.
	"refusal": {},
}

// routerResponseToolArgumentFields are the keys holding a tool call's
// arguments. Their VALUE is the model's own output -- the arguments it
// chose -- but the object's shape (which parameter, which type) is what an
// operator reads to understand the call, so the arguments are masked as a
// whole rather than field by field.
//
// "input" is Anthropic's spelling of the same thing ("arguments" is
// OpenAI's); "content" on a tool_result block is the tool's OUTPUT, which
// is the model's own text too, and is masked by the same rule.
var routerResponseToolArgumentFields = map[string]struct{}{
	"arguments": {},
	"input":     {},
}

// routerResponseLoggableFields are the keys whose values are NOT the
// model's output and are therefore kept verbatim even when redaction is
// on: what was called and what came back about the exchange.
//
// Keeping these is the point of the feature: with the text gone, the row
// still says which model answered, how many tokens it used, how it ended,
// and which id the client got -- which is what an error analysis or a cost
// reconciliation needs.
var routerResponseLoggableFields = map[string]struct{}{
	"id":                 {},
	"model":              {},
	"object":             {},
	"created":            {},
	"index":              {},
	"finish_reason":      {},
	"finish_reasons":     {},
	"role":               {},
	"type":               {},
	"status":             {},
	"stop_reason":        {},
	"stop_sequence":      {},
	"usage":              {},
	"system_fingerprint": {},
	"service_tier":       {},
	"provider":           {},
	"logprobs":           {},
}

// redactRouterResponseBody masks the model's own text in a response body,
// keeping the structure intact: the content fields become the placeholder,
// while the model, the usage, the ids and the finish reasons survive.
//
// It is deliberately different from the raw-bodies redaction, which
// replaces the WHOLE body with a "[REDACTED body bytes=N]" placeholder.
// That is right for a provider's error text, whose only value is in its
// message, and wrong for a response: the whole point of capturing it is to
// read its SHAPE -- which model, which tool, which finish reason, how many
// tokens -- and a placeholder for the whole body leaves nothing to read.
//
// Fails closed: a body that isn't valid JSON comes back as the placeholder
// for the whole body, never as the original text. That direction is the
// safe one: an unparseable body might be a plain-text answer, and shipping
// it unredacted would leak the model's output on the one path where this
// function could not tell output from structure.
func redactRouterResponseBody(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", true
	}
	var parsed any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", false
	}
	masked := maskRouterResponseValue(parsed)
	out, err := json.Marshal(masked)
	if err != nil {
		return "", false
	}
	return string(out), true
}

// maskRouterResponseValue walks a decoded response body, replacing the
// model's own text with the placeholder.
//
// What to mask is decided by the KEY and the VALUE's shape, never by where
// the value sits: "content" is the answer when it is a string and a
// container of parts when it is an array, and the same is true of "text"
// inside such a part. Deciding by position ("is this under a message-ish
// key?") left the Anthropic-native body -- whose top level IS the message,
// with no "message"/"choices"/"output" key anywhere -- unmasked, and it
// shipped the model's text and its tool-call arguments in the clear.
//
// The one thing the walk must not destroy is the STRUCTURE: the part types,
// the tool names, the ids, the usage. Those are what an operator reads.
func maskRouterResponseValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, child := range typed {
			switch {
			case routerResponseIsLoggableField(key):
				out[key] = child
			case routerResponseIsTextField(key) && routerResponseIsTextValue(child):
				// A text field whose value IS text: the model's own output,
				// replaced outright.
				out[key] = routerResponseRedactedPlaceholder
			case routerResponseIsToolArgumentField(key):
				// A tool call's arguments: the model's own output. The
				// object shape (which parameter, which type) survives --
				// only the values are dropped -- so an operator still
				// sees WHICH tool was called with WHAT shape.
				out[key] = routerResponseRedactedPlaceholder
			case routerResponseIsBinaryPayload(child):
				// A binary payload -- an image, a document, an audio
				// clip -- carried as a string (base64, or a data: URL
				// around one). It is not text, and it can be megabytes:
				// it is the client's own picture or scan, and it says
				// nothing about the exchange. Dropped whole.
				out[key] = routerResponseRedactedPlaceholder
			default:
				// A text field whose value is a container (the Responses
				// API's "content": an array of output parts, or Anthropic's
				// "content": an array of blocks) is NOT the answer -- it
				// is the answer's structure. Masking it wholesale would
				// destroy the part types an operator reads, so the walk
				// goes into it instead and masks the text inside each
				// part.
				out[key] = maskRouterResponseValue(child)
			}
		}
		return out
	case []any:
		out := make([]any, 0, len(typed))
		for _, item := range typed {
			out = append(out, maskRouterResponseValue(item))
		}
		return out
	default:
		return value
	}
}

// routerResponseIsTextField reports whether the key holds the model's own
// text, whatever it sits under.
func routerResponseIsTextField(key string) bool {
	_, ok := routerResponseTextFields[key]
	return ok
}

// routerResponseIsToolArgumentField reports whether the key holds a tool
// call's arguments.
func routerResponseIsToolArgumentField(key string) bool {
	_, ok := routerResponseToolArgumentFields[key]
	return ok
}

// routerResponseIsBinaryPayload reports whether the value is a binary
// payload carried as text: a base64 blob, or a data: URL around one.
//
// It is detected by the VALUE, not by the key, because the keys that carry
// one differ per provider and per shape -- Anthropic puts it under
// source.data, Gemini under inlineData.data, OpenAI's image extension under
// image_url.url -- and a key list would have to enumerate all of them, and
// would miss the next one. A base64 payload is long and base64-shaped, so
// the value itself says what it is.
//
// The payload is dropped whole rather than truncated: it is the client's own
// picture or scan, and half a base64 blob is worth nothing to an operator
// reading the row.
func routerResponseIsBinaryPayload(value any) bool {
	text, ok := value.(string)
	if !ok {
		return false
	}
	return isBase64Payload(text) || isDataURLPayload(text)
}

// isBase64Payload reports whether the text is a base64 blob: long enough
// that it cannot be prose, and made only of the base64 alphabet.
//
// The length threshold is what keeps prose out -- prose is not hundreds of
// characters of the base64 alphabet -- and the alphabet check is what keeps a
// long non-base64 string (a URL, an identifier) out.
func isBase64Payload(text string) bool {
	// 256 characters: long enough that prose cannot reach it (a sentence
	// is not 256 characters of the base64 alphabet), short enough to catch
	// a small picture or a one-page document. The real payloads measured in
	// production were 67 KB and up; the threshold only has to tell a
	// payload from a word.
	const minBinaryLength = 256
	if len(text) < minBinaryLength {
		return false
	}
	for i := 0; i < len(text); i++ {
		c := text[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9':
		case c == '+', c == '/', c == '=', c == '\n', c == '\r':
		default:
			return false
		}
	}
	return true
}

// isDataURLPayload reports whether the text is a data: URL carrying a BINARY
// payload -- "data:image/png;base64,...".
//
// Only the binary media types are caught. A data: URL with a text media type
// ("data:text/plain;base64,SGVsbG8=") carries prose, not a picture, and
// prose is the model's own output -- masked by the text rules, not by this
// one.
func isDataURLPayload(text string) bool {
	const prefix = "data:"
	if !strings.HasPrefix(text, prefix) {
		return false
	}
	comma := strings.IndexByte(text, ',')
	if comma < 0 {
		return false
	}
	header := strings.ToLower(text[:comma])
	if !strings.HasSuffix(header, ";base64") {
		return false
	}
	for _, mediaType := range []string{
		"image/", "audio/", "video/", "application/pdf",
		"application/octet-stream", "application/zip",
	} {
		if strings.Contains(header, mediaType) {
			return true
		}
	}
	return false
}

// routerResponseIsTextValue reports whether the value under a text field is
// actually text. A field like "content" holds the answer in a Chat
// completion (a string) and the answer's PARTS in a Responses API body (an
// array) -- only the first is text to mask.
func routerResponseIsTextValue(value any) bool {
	switch typed := value.(type) {
	case string:
		return true
	case []any:
		// An array of strings is a list of text values, not a container of
		// parts; a container of parts holds objects.
		for _, item := range typed {
			if _, isObject := item.(map[string]any); isObject {
				return false
			}
		}
		return len(typed) > 0
	default:
		return false
	}
}

// routerResponseIsLoggableField reports whether the key's value is
// metadata about the exchange rather than the model's output.
func routerResponseIsLoggableField(key string) bool {
	_, ok := routerResponseLoggableFields[key]
	return ok
}

// captureRouterResponseBody returns the body to publish for a router
// response, masked per redact and capped per the size limit.
//
// The cap is applied AFTER the masking, so a huge body is truncated the
// same way whether or not redaction is on -- and the truncation marker
// says so, rather than the reader believing the JSON is complete.
//
// An embedding or an image answer is NOT captured at all: see
// isBulkBinaryAnswer.
func captureRouterResponseBody(body []byte, redact bool) string {
	if len(body) == 0 {
		return ""
	}
	if isBulkBinaryAnswer(body) {
		// Not published: the body is the payload itself (a vector, a
		// base64 image), not the model's answer.
		return ""
	}
	if redact {
		masked, ok := redactRouterResponseBody(body)
		if !ok {
			// Fails closed: an unparseable body is not shipped unredacted.
			return routerResponseRedactedPlaceholder
		}
		body = []byte(masked)
	}
	if len(body) > maxRouterResponseBodyBytes {
		return string(body[:maxRouterResponseBodyBytes]) + "...[truncated]"
	}
	return string(body)
}

// isBulkBinaryAnswer reports whether the body is an embedding vector or an
// image -- the two answer shapes whose body IS the payload, not the model's
// words.
//
// Such a body is not published, for two reasons. It is the client's own data
// (the embedding of its input, the pixels it asked for) rather than the
// model's output, and it is enormous: an embedding answer is ~40 KB and an
// image answer is a base64 blob that hits the 256 KB cap every time.
// Together they were 140 MB of the 687 MB the stream carried in a day --
// a fifth of the volume, for rows that say nothing about the exchange.
//
// The detection is on the STRUCTURE, not on the words: an ordinary answer
// that merely mentions "embedding" in its text must not be dropped.
func isBulkBinaryAnswer(body []byte) bool {
	var doc map[string]any
	if err := json.Unmarshal(body, &doc); err != nil {
		return false
	}
	// Gemini's native shape: "embeddings": [{"values": [...]}] at the top
	// level, and an image part's "inlineData" inside a candidate.
	if _, ok := doc["embeddings"].([]any); ok {
		return true
	}
	if bodyContainsKey(doc, "inlineData") {
		return true
	}
	// OpenAI's shapes: "data": [{"embedding": [...]}] and "data": [{"b64_json":
	// "..."}] -- the same "data" array holds both, so they are told apart by
	// the key inside the element.
	items, ok := doc["data"].([]any)
	if !ok {
		return false
	}
	for _, item := range items {
		element, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if _, ok := element["embedding"].([]any); ok {
			return true
		}
		if _, ok := element["b64_json"].(string); ok {
			return true
		}
	}
	return false
}

// bodyContainsKey reports whether the decoded body holds the given key
// anywhere in it.
func bodyContainsKey(value any, key string) bool {
	switch typed := value.(type) {
	case map[string]any:
		for k, child := range typed {
			if k == key || bodyContainsKey(child, key) {
				return true
			}
		}
		return false
	case []any:
		for _, item := range typed {
			if bodyContainsKey(item, key) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// captureRouterResponse stores the response body on the log context, for
// the router-response write-path.
//
// It is a no-op when the path is off, so the capture costs nothing on a
// router that does not use it: the body's own text is never held in the
// log context unless the operator asked for it.
//
// The body is masked here rather than at publish time so the model's text
// never sits in memory in the clear between the response and the write.
func (p *Proxy) captureRouterResponse(logCtx *RequestLogContext, body []byte, streaming bool) {
	if p == nil || logCtx == nil || !p.routerResponsesEnabled {
		return
	}
	if len(body) == 0 {
		return
	}
	logCtx.RouterResponseBody = captureRouterResponseBody(body, p.routerResponsesRedact)
	logCtx.RouterResponseIsStreaming = streaming
}
