package httputil

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const (
	maxErrorBody    = 8 << 10 // how much of an error body is kept to read
	maxErrorMessage = 512     // bounds the message on the span
	maxAnswerWord   = 128     // bounds a type, a code, an id
)

// requestIDHeaders are where providers give a request its id.
var requestIDHeaders = []string{"X-Request-Id", "Request-Id", "Apim-Request-Id", "X-Amzn-Requestid"}

// spanName names an attempt's client span by its credential.
func spanName(_ string, r *http.Request) string {
	if number, current := AttemptsFrom(r.Context()).current(); number > 0 {
		return r.Method + " " + current.credential
	}
	return "HTTP " + r.Method
}

// answered puts the provider's request id and error on the client span.
func answered(span trace.Span, resp *http.Response) {
	for _, name := range requestIDHeaders {
		if id := resp.Header.Get(name); id != "" && len(id) <= maxAnswerWord {
			span.SetAttributes(attribute.String("aar.upstream_request_id", id))
			break
		}
	}
	// A body that the transport did not decode cannot be read here.
	if resp.StatusCode >= http.StatusBadRequest && resp.Header.Get("Content-Encoding") == "" {
		resp.Body = &errorBody{ReadCloser: resp.Body, span: span}
	}
}

// errorBody keeps the start of a failed answer's body and puts the provider's
// error on the span. It sits beneath otelhttp's reader, which ends the span.
type errorBody struct {
	io.ReadCloser
	span trace.Span
	kept []byte
	cut  bool // there was more than is kept
	said bool
}

func (b *errorBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if b.said {
		return n, err
	}
	room := maxErrorBody - len(b.kept)
	b.kept = append(b.kept, p[:min(n, room)]...)
	b.cut = n > room
	if err != nil || b.cut {
		b.said = true
		b.span.SetAttributes(providerError(b.kept, err == io.EOF && !b.cut)...)
		b.kept = nil
	}
	return n, err
}

// ProviderError is what a provider's error body or stream error event says,
// as span attributes.
func ProviderError(body []byte) []attribute.KeyValue {
	return providerError(body, true)
}

// providerError is what an error body says, as span attributes.
func providerError(body []byte, whole bool) []attribute.KeyValue {
	var message, kind, code string
	if whole {
		message, kind, code = errorFields(body)
	}
	if message == "" && kind == "" && code == "" {
		message = strings.TrimSpace(string(body))
	}
	attrs := make([]attribute.KeyValue, 0, 3)
	add := func(key, value string, limit int) {
		if len(value) > limit {
			value = value[:limit]
		}
		if value = strings.ToValidUTF8(value, ""); value != "" {
			attrs = append(attrs, attribute.String(key, value))
		}
	}
	add("aar.error.message", message, maxErrorMessage)
	add("aar.error.type", kind, maxAnswerWord)
	add("aar.error.code", code, maxAnswerWord)
	return attrs
}

// errorFields reads the providers' error shapes: an "error" object (OpenAI,
// Anthropic, Google) or string, else "message" or "detail".
func errorFields(body []byte) (message, kind, code string) {
	var doc any
	if err := json.Unmarshal(body, &doc); err != nil {
		return "", "", ""
	}
	if list, ok := doc.([]any); ok && len(list) > 0 {
		doc = list[0]
	}
	answer, ok := doc.(map[string]any)
	if !ok {
		return "", "", ""
	}
	switch said := answer["error"].(type) {
	case map[string]any:
		message = word(said["message"])
		if kind = word(said["type"]); kind == "" {
			kind = word(said["status"])
		}
		code = word(said["code"])
	case string:
		message = said
	}
	if message == "" {
		message = word(answer["message"])
	}
	if message == "" {
		message = word(answer["detail"])
	}
	return message, kind, code
}

// word is a JSON string or number as text, "" for anything else.
func word(v any) string {
	switch v := v.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	}
	return ""
}
