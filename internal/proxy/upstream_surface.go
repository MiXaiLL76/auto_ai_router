package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/url"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

func shouldSanitizeUpstreamSurface(cred *config.CredentialConfig) bool {
	return isProManCredential(cred)
}

func isProManCredential(cred *config.CredentialConfig) bool {
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
		isProManHost(cred.BaseURL)
}

func isProManHost(rawBaseURL string) bool {
	baseURL := strings.TrimSpace(rawBaseURL)
	if baseURL == "" {
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
	return host == "proman.ai" || strings.HasSuffix(host, ".proman.ai")
}

func clientResponseBodyForCredential(statusCode int, body []byte, cred *config.CredentialConfig, displayModel string) ([]byte, bool) {
	if statusCode >= 400 && shouldMaskUpstreamErrors(cred) {
		masked := maskedUpstreamErrorBody(statusCode)
		return masked, !bytes.Equal(body, masked)
	}
	if statusCode >= 200 && statusCode < 300 && shouldSanitizeUpstreamSurface(cred) {
		return sanitizeUpstreamJSONBody(body, displayModel)
	}
	return body, false
}

func sanitizeUpstreamJSONBody(body []byte, displayModel string) ([]byte, bool) {
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

type sanitizingSSEReadCloser struct {
	source io.Closer
	reader *sanitizingSSEReader
}

type sanitizingSSEReader struct {
	scanner      *bufio.Scanner
	buffer       bytes.Buffer
	displayModel string
	done         bool
	err          error
}

func newSanitizingSSEReader(source io.Reader, displayModel string) io.Reader {
	scanner := bufio.NewScanner(source)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	return &sanitizingSSEReader{
		scanner:      scanner,
		displayModel: displayModel,
	}
}

func newSanitizingSSEReadCloser(source io.ReadCloser, displayModel string) io.ReadCloser {
	return &sanitizingSSEReadCloser{
		source: source,
		reader: newSanitizingSSEReader(source, displayModel).(*sanitizingSSEReader),
	}
}

func (r *sanitizingSSEReader) Read(p []byte) (int, error) {
	for r.buffer.Len() == 0 && !r.done {
		if !r.scanner.Scan() {
			r.done = true
			r.err = r.scanner.Err()
			break
		}
		line := r.scanner.Text()
		r.buffer.WriteString(sanitizeSSELine(line, r.displayModel))
		r.buffer.WriteByte('\n')
	}

	if r.buffer.Len() > 0 {
		return r.buffer.Read(p)
	}
	if r.err != nil {
		return 0, r.err
	}
	return 0, io.EOF
}

func (r *sanitizingSSEReadCloser) Read(p []byte) (int, error) {
	return r.reader.Read(p)
}

func (r *sanitizingSSEReadCloser) Close() error {
	return r.source.Close()
}

func sanitizeSSELine(line, displayModel string) string {
	if !strings.HasPrefix(line, "data: ") {
		return line
	}
	data := strings.TrimPrefix(line, "data: ")
	if data == "" || data == "[DONE]" {
		return line
	}
	if sanitized, changed := sanitizeUpstreamJSONBody([]byte(data), displayModel); changed {
		return "data: " + string(sanitized)
	}
	return line
}
