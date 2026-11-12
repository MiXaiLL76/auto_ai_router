package proxy

import (
	"encoding/json"
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
func captureRouterResponseBody(body []byte, redact bool) string {
	if len(body) == 0 {
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
