package utils

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

const MaxSanitizingSSELineBytes = 1 << 20

func IsCredential(cred *config.CredentialConfig) bool {
	return cred != nil && cred.Type == config.ProviderTypeProMan
}

func ShouldSanitizeUpstreamSurface(cred *config.CredentialConfig) bool {
	return IsCredential(cred)
}

func IsProviderInternalResponseHeader(key string) bool {
	lower := strings.ToLower(key)
	switch lower {
	case "server",
		"via",
		"x-powered-by",
		"request-id",
		"x-request-id",
		"anthropic-organization-id",
		"cf-ray",
		"cf-cache-status",
		"traceresponse",
		"x-robots-tag":
		return true
	}

	internalPrefixes := []string{
		"x-litellm-",
		"llm_provider-",
		"x-provider-",
		"x-ratelimit-",
		"anthropic-ratelimit-",
		"x-amz-",
		"x-amzn-",
	}
	for _, prefix := range internalPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

func UnsupportedRequest(body []byte, _ string) string {
	if len(bytes.TrimSpace(body)) == 0 {
		return ""
	}

	var root any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return ""
	}

	obj, ok := root.(map[string]any)
	if !ok {
		return ""
	}
	if hasServerToolUseMessageBlock(obj) {
		return "server_tool_use"
	}
	return ""
}

func SanitizeUpstreamJSONBody(body []byte, displayModel string) ([]byte, bool) {
	if len(bytes.TrimSpace(body)) == 0 {
		return body, false
	}

	var value any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return body, false
	}

	changed := sanitizeJSONEnvelope(&value, displayModel)
	if !changed {
		return body, false
	}

	sanitized, err := json.Marshal(value)
	if err != nil {
		return body, false
	}
	return sanitized, true
}

type sanitizingSSEReadCloser struct {
	source io.Closer
	reader *sanitizingSSEReader
}

type sanitizingSSEReader struct {
	reader       *bufio.Reader
	pending      []byte
	line         []byte
	passthrough  bool
	displayModel string
	terminalErr  error
}

func NewSanitizingSSEReader(source io.Reader, displayModel string) io.Reader {
	return &sanitizingSSEReader{
		reader:       bufio.NewReader(source),
		displayModel: displayModel,
	}
}

func NewSanitizingSSEReadCloser(source io.ReadCloser, displayModel string) io.ReadCloser {
	return &sanitizingSSEReadCloser{
		source: source,
		reader: &sanitizingSSEReader{
			reader:       bufio.NewReader(source),
			displayModel: displayModel,
		},
	}
}

func (r *sanitizingSSEReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	for len(r.pending) == 0 {
		if r.terminalErr != nil {
			err := r.terminalErr
			if err != io.EOF {
				r.terminalErr = io.EOF
			}
			return 0, err
		}

		r.readNextSSEFragment()
	}

	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *sanitizingSSEReader) readNextSSEFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.reader.ReadSlice('\n')

		if r.passthrough {
			if len(fragment) > 0 {
				r.pending = fragment
			}
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
			}
		} else {
			r.consumeSanitizingSSEFragment(fragment, err)
		}

		if err != nil && !errors.Is(err, bufio.ErrBufferFull) {
			r.terminalErr = err
		}
	}
}

func (r *sanitizingSSEReader) consumeSanitizingSSEFragment(fragment []byte, readErr error) {
	if errors.Is(readErr, bufio.ErrBufferFull) {
		if len(r.line)+len(fragment) <= MaxSanitizingSSELineBytes {
			r.line = append(r.line, fragment...)
			return
		}

		r.pending = append(r.line, fragment...)
		r.line = nil
		r.passthrough = true
		return
	}

	line := append(r.line, fragment...)
	r.line = nil
	if len(line) == 0 {
		return
	}

	if readErr == nil || errors.Is(readErr, io.EOF) {
		if len(line) <= MaxSanitizingSSELineBytes {
			r.pending = sanitizeSSELine(line, r.displayModel)
			return
		}
	}

	r.pending = line
}

func (r *sanitizingSSEReadCloser) Read(p []byte) (int, error) {
	return r.reader.Read(p)
}

func (r *sanitizingSSEReadCloser) Close() error {
	return r.source.Close()
}

func sanitizeJSONEnvelope(value *any, displayModel string) bool {
	obj, ok := (*value).(map[string]any)
	if !ok {
		return false
	}
	return sanitizeEnvelopeMap(obj, displayModel)
}

func sanitizeEnvelopeMap(obj map[string]any, displayModel string) bool {
	changed := sanitizeEnvelopeFields(obj, displayModel)
	if errObj, ok := obj["error"].(map[string]any); ok {
		if _, hasMessage := errObj["message"]; hasMessage {
			errObj["message"] = "Request failed"
			changed = true
		}
	}

	for _, key := range []string{
		"response", "message", "delta", "usage", "error",
		"prompt_tokens_details", "completion_tokens_details",
		"input_tokens_details", "output_tokens_details",
	} {
		if child, ok := obj[key].(map[string]any); ok {
			if sanitizeEnvelopeMap(child, displayModel) {
				changed = true
			}
		}
	}

	if choices, ok := obj["choices"].([]any); ok {
		for _, choice := range choices {
			choiceObj, ok := choice.(map[string]any)
			if !ok {
				continue
			}
			if sanitizeEnvelopeMap(choiceObj, displayModel) {
				changed = true
			}
		}
	}
	return changed
}

func sanitizeEnvelopeFields(obj map[string]any, displayModel string) bool {
	changed := false
	for key, nested := range obj {
		if isInternalJSONField(key) {
			delete(obj, key)
			changed = true
			continue
		}
		lowerKey := strings.ToLower(key)
		if lowerKey == "model" && displayModel != "" {
			if s, ok := nested.(string); ok && isInternalModelString(s) {
				obj[key] = displayModel
				changed = true
			}
		}
	}
	return changed
}

func isInternalJSONField(key string) bool {
	lower := strings.ToLower(key)
	switch lower {
	case "provider_specific_fields",
		"caller",
		"provider",
		"provider_id",
		"provider_name",
		"credential",
		"credential_id",
		"credential_name",
		"api_base",
		"base_url",
		"router",
		"router_id",
		"route",
		"route_id",
		"deployment",
		"deployment_id",
		"deployment_name",
		"fallback",
		"fallbacks",
		"fallback_route",
		"selected_provider",
		"selected_credential",
		"upstream",
		"upstream_url",
		"litellm_metadata",
		"litellm_params",
		"litellm_call_id",
		"llm_provider",
		"llm_provider_id":
		return true
	}
	return strings.HasPrefix(lower, "litellm_") ||
		strings.HasPrefix(lower, "provider_") ||
		strings.HasPrefix(lower, "credential_") ||
		strings.HasPrefix(lower, "router_") ||
		strings.HasPrefix(lower, "route_") ||
		strings.HasPrefix(lower, "deployment_") ||
		strings.HasPrefix(lower, "fallback_") ||
		strings.HasPrefix(lower, "x-litellm") ||
		strings.HasPrefix(lower, "llm_provider")
}

func isInternalModelString(value string) bool {
	lower := strings.ToLower(value)
	if strings.HasPrefix(lower, "anthropic/") {
		return true
	}
	internalMarkers := []string{
		"anthropic-direct-client",
		"litellm",
		"llm_provider",
		"proman",
		"bedrock",
		"amazonaws",
		"aws/",
	}
	for _, marker := range internalMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func sanitizeSSELine(line []byte, displayModel string) []byte {
	content, ending := splitSSELineEnding(line)
	if !bytes.HasPrefix(content, []byte("data:")) {
		return line
	}

	rawPayload := content[len("data:"):]
	payload := bytes.TrimSpace(rawPayload)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || payload[0] != '{' {
		return line
	}

	sanitizedPayload, changed := SanitizeUpstreamJSONBody(payload, displayModel)
	if !changed {
		return line
	}

	payloadStart := bytes.Index(rawPayload, payload)
	if payloadStart < 0 {
		return line
	}
	payloadEnd := payloadStart + len(payload)
	sanitized := make([]byte, 0, len(line)-len(payload)+len(sanitizedPayload))
	sanitized = append(sanitized, content[:len("data:")]...)
	sanitized = append(sanitized, rawPayload[:payloadStart]...)
	sanitized = append(sanitized, sanitizedPayload...)
	sanitized = append(sanitized, rawPayload[payloadEnd:]...)
	sanitized = append(sanitized, ending...)
	return sanitized
}

func splitSSELineEnding(line []byte) (content, ending []byte) {
	if len(line) == 0 || line[len(line)-1] != '\n' {
		return line, nil
	}
	if len(line) >= 2 && line[len(line)-2] == '\r' {
		return line[:len(line)-2], line[len(line)-2:]
	}
	return line[:len(line)-1], line[len(line)-1:]
}

func hasServerToolUseMessageBlock(obj map[string]any) bool {
	return hasServerToolUseInMessages(obj["messages"]) ||
		hasServerToolUseInMessages(obj["input"])
}

func hasServerToolUseInMessages(value any) bool {
	messages, ok := value.([]any)
	if !ok {
		return false
	}
	for _, message := range messages {
		messageObj, ok := message.(map[string]any)
		if !ok {
			continue
		}
		if hasServerToolUseContentBlock(messageObj["content"]) {
			return true
		}
	}
	return false
}

func hasServerToolUseContentBlock(value any) bool {
	blocks, ok := value.([]any)
	if !ok {
		return false
	}
	for _, block := range blocks {
		blockObj, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(stringValue(blockObj["type"])), "server_tool_use") {
			return true
		}
	}
	return false
}

func stringValue(value any) string {
	s, _ := value.(string)
	return s
}
