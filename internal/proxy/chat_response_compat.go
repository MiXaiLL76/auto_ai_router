package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"

	// Aliased separately (not a blanket replacement of this file's json import)
	// because only normalizeSuccessfulResponseModel below benefits: it
	// full-body-decodes+remarshals every non-streaming response, and
	// benchmarked against a real embedding body (1536-float vector) goccy is
	// ~1.66x faster there. normalizeSSEDataLineModel/decodeJSONObject further
	// down operate on small per-SSE-chunk payloads, not large arrays, so
	// there's no measured case for touching them too.
	goccyjson "github.com/goccy/go-json"
)

const maxSSEModelRewriteLineBytes = 1024 * 1024

// modelBearingResponseRoutes is the product surface whose successful response
// schema exposes a client-visible model. Image responses intentionally do not
// participate because their OpenAI schema has no model field.
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
	if _, ok := modelBearingResponseRoutes[endpoint]; !ok {
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
		source:      bufio.NewReader(reader),
		publicModel: publicModel,
	}
}

type responseModelStreamReader struct {
	source      *bufio.Reader
	publicModel string
	pending     []byte
	pendingErr  error
	line        []byte
	passthrough bool
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
		if len(r.line)+len(fragment) > maxSSEModelRewriteLineBytes {
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
		r.pending = normalizeSSEDataLineModel(r.line, r.publicModel)
		r.line = nil
		r.pendingErr = err
	}

	n := copy(dst, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func normalizeSSEDataLineModel(line []byte, publicModel string) []byte {
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
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}
	event, err := decodeJSONObject(payload)
	if err != nil || event == nil {
		return line
	}
	rawError, hasError := event["error"]
	eventType, _ := event["type"].(string)
	if isStreamErrorEvent(eventType, hasError && rawError != nil) {
		return line
	}

	changed := false
	if current, exists := event["model"]; exists {
		if current != publicModel {
			event["model"] = publicModel
			changed = true
		}
	} else if object, _ := event["object"].(string); object == "chat.completion.chunk" || object == "text_completion" {
		event["model"] = publicModel
		changed = true
	}
	if response, ok := event["response"].(map[string]interface{}); ok {
		if response["model"] != publicModel {
			response["model"] = publicModel
			changed = true
		}
	}
	if !changed {
		return line
	}

	normalized, err := json.Marshal(event)
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

func decodeJSONObject(data []byte) (map[string]interface{}, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var object map[string]interface{}
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	var trailing interface{}
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("multiple JSON values")
		}
		return nil, err
	}
	return object, nil
}
