package utils

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/url"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

const MaxSanitizingSSELineBytes = 1 << 20

func IsCredential(cred *config.CredentialConfig) bool {
	if cred == nil {
		return false
	}
	if cred.Type == config.ProviderTypeProMan {
		return true
	}
	name := strings.ToLower(cred.Name)
	return strings.Contains(name, "proman") ||
		strings.Contains(name, "pro-man") ||
		strings.Contains(name, "pro_man") ||
		isProviderHost(cred.BaseURL, "proman.ai")
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

func UnsupportedRequest(body []byte, selectedModel string) string {
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
	model := proManCanonicalModel(selectedModel)
	if model == "" {
		model = proManCanonicalModel(stringValue(obj["model"]))
	}
	caps, hasCaps := proManCapabilitiesByModel[model]
	if hasProManRecursiveType(root, "server_tool_use") {
		return "server_tool_use"
	}
	if hasCaps {
		if caps.blockLegacyThinking && hasProManUnsupportedThinking(obj) {
			return "thinking"
		}
		if caps.blockAssistantPrefill && hasProManAssistantPrefill(obj) {
			return "assistant_prefill"
		}
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

	changed := sanitizeJSONValue(&value, displayModel)
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

type proManModelCapabilities struct {
	blockAssistantPrefill bool
	blockLegacyThinking   bool
}

var proManCapabilitiesByModel = map[string]proManModelCapabilities{
	"claude-fable-5": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   true,
	},
	"claude-haiku-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.6": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.7": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-opus-4.8": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-4.5": {
		blockAssistantPrefill: false,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-4.6": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   false,
	},
	"claude-sonnet-5": {
		blockAssistantPrefill: true,
		blockLegacyThinking:   true,
	},
}

func sanitizeJSONValue(value *any, displayModel string) bool {
	switch typed := (*value).(type) {
	case map[string]any:
		changed := false
		if eventType, _ := typed["type"].(string); eventType == "error" {
			if errObj, ok := typed["error"].(map[string]any); ok {
				if _, hasMessage := errObj["message"]; hasMessage {
					errObj["message"] = "Upstream provider error"
					changed = true
				}
			}
		}
		for key, nested := range typed {
			if isInternalJSONField(key) {
				delete(typed, key)
				changed = true
				continue
			}
			lowerKey := strings.ToLower(key)
			if lowerKey == "model" && displayModel != "" {
				if s, ok := nested.(string); ok && isInternalModelString(s) {
					typed[key] = displayModel
					changed = true
					continue
				}
			}
			nestedValue := nested
			if sanitizeJSONValue(&nestedValue, displayModel) {
				typed[key] = nestedValue
				changed = true
			}
		}
		return changed
	case []any:
		changed := false
		for i, nested := range typed {
			nestedValue := nested
			if sanitizeJSONValue(&nestedValue, displayModel) {
				typed[i] = nestedValue
				changed = true
			}
		}
		return changed
	default:
		return false
	}
}

func isInternalJSONField(key string) bool {
	lower := strings.ToLower(key)
	switch lower {
	case "provider_specific_fields",
		"caller",
		"litellm_metadata",
		"litellm_params",
		"litellm_call_id",
		"llm_provider",
		"llm_provider_id":
		return true
	}
	return strings.HasPrefix(lower, "litellm_") ||
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

func hasProManUnsupportedThinking(obj map[string]any) bool {
	for _, container := range []map[string]any{obj, mapValue(obj["extra_body"])} {
		if container == nil {
			continue
		}
		if thinking, ok := container["thinking"]; ok {
			if hasEnabledThinkingConfig(thinking) {
				return true
			}
		}
		if nonNoneString(container["reasoning_effort"]) {
			return true
		}
	}

	if reasoning, ok := obj["reasoning"].(map[string]any); ok {
		return nonNoneString(reasoning["effort"])
	}
	return false
}

func hasEnabledThinkingConfig(value any) bool {
	thinking, ok := value.(map[string]any)
	if !ok {
		return true
	}
	typ := strings.ToLower(strings.TrimSpace(stringValue(thinking["type"])))
	return typ != "disabled"
}

func hasProManRecursiveType(value any, targetType string) bool {
	switch typed := value.(type) {
	case map[string]any:
		if typ, _ := typed["type"].(string); strings.EqualFold(strings.TrimSpace(typ), targetType) {
			return true
		}
		for _, nested := range typed {
			if hasProManRecursiveType(nested, targetType) {
				return true
			}
		}
	case []any:
		for _, nested := range typed {
			if hasProManRecursiveType(nested, targetType) {
				return true
			}
		}
	}
	return false
}

func hasProManAssistantPrefill(obj map[string]any) bool {
	messages, ok := obj["messages"].([]any)
	if !ok || len(messages) == 0 {
		if input, ok := obj["input"].([]any); ok {
			messages = input
		}
	}
	if len(messages) == 0 {
		return false
	}
	last, ok := messages[len(messages)-1].(map[string]any)
	if !ok {
		return false
	}
	role := strings.TrimSpace(strings.ToLower(stringValue(last["role"])))
	return role == "assistant" || role == "model"
}

func proManCanonicalModel(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	normalized = strings.ReplaceAll(normalized, "_", "-")
	normalized = strings.ReplaceAll(normalized, ".", "-")
	for _, prefix := range []string{"anthropic/", "anthropic-", "claude/"} {
		normalized = strings.TrimPrefix(normalized, prefix)
	}

	switch {
	case strings.Contains(normalized, "fable-5"):
		return "claude-fable-5"
	case strings.Contains(normalized, "haiku-4-5"):
		return "claude-haiku-4.5"
	case strings.Contains(normalized, "opus-4-5"):
		return "claude-opus-4.5"
	case strings.Contains(normalized, "opus-4-6"):
		return "claude-opus-4.6"
	case strings.Contains(normalized, "opus-4-7"):
		return "claude-opus-4.7"
	case strings.Contains(normalized, "opus-4-8"):
		return "claude-opus-4.8"
	case strings.Contains(normalized, "sonnet-4-5"):
		return "claude-sonnet-4.5"
	case strings.Contains(normalized, "sonnet-4-6"):
		return "claude-sonnet-4.6"
	case strings.Contains(normalized, "sonnet-5"):
		return "claude-sonnet-5"
	default:
		return ""
	}
}

func mapValue(value any) map[string]any {
	obj, _ := value.(map[string]any)
	return obj
}

func stringValue(value any) string {
	s, _ := value.(string)
	return s
}

func nonNoneString(value any) bool {
	s, ok := value.(string)
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "none", "disable", "disabled":
		return false
	default:
		return true
	}
}

func isProviderHost(rawBaseURL, domain string) bool {
	baseURL := strings.TrimSpace(rawBaseURL)
	domain = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(domain)), ".")
	if baseURL == "" || domain == "" {
		return false
	}
	u, err := url.Parse(baseURL)
	if err != nil || u.Hostname() == "" {
		u, err = url.Parse("https://" + baseURL)
		if err != nil {
			return false
		}
	}
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	return host == domain || strings.HasSuffix(host, "."+domain)
}
