// Package upstreamerror classifies a raw 400 Bad Request body from an
// upstream LLM provider into a short, pre-vetted message plus (when
// available) the specific parameter/field name that was rejected — without
// ever echoing the provider's own free-text message back to the client.
// Shared by native mode (proxy.maskedUpstreamErrorBody) and litellm
// compatibility mode (responsecompat/litellm.normalizeError), so both
// surface the same classification instead of one of them discarding it.
package upstreamerror

import (
	"bytes"
	"encoding/json"
	"strings"
)

// BadRequest is the result of classifying a raw 400 body: always one of a
// small fixed set of messages/codes, plus an optional narrowly-scoped
// parameter/field name — never the provider's own message text.
type BadRequest struct {
	Message string
	Code    string
	Param   *string
}

// ClassifyBadRequest inspects rawBody for known signal phrases and returns a
// classified message/code/param. Returns the generic "Invalid request" /
// "invalid_request" (Param nil) when nothing recognizable is found.
func ClassifyBadRequest(rawBody []byte) BadRequest {
	result := BadRequest{Message: "Invalid request", Code: "invalid_request"}
	if len(rawBody) == 0 {
		return result
	}

	signals, providerParam := providerErrorSignalsFromBody(rawBody)
	joined := strings.ToLower(strings.Join(signals, " "))

	switch {
	case hasSignal(joined, "tool_choice", "tool choice", "toolchoice"):
		param := "tool_choice"
		result.Message = "Invalid tool_choice"
		result.Code = "invalid_tool_choice"
		result.Param = &param
	case hasSignal(joined, "max_completion_tokens", "max_output_tokens", "max_tokens", "max tokens", "maximum tokens", "output tokens"):
		param := inferBadRequestParam(joined, providerParam)
		if param == nil || !strings.Contains(*param, "token") {
			v := inferMaxTokensParam(joined)
			param = &v
		}
		result.Message = "Invalid " + *param
		result.Code = "invalid_max_tokens"
		result.Param = param
	case hasSignal(joined, "context length", "context window", "context limit", "too many tokens", "input too long", "prompt too long", "token limit"):
		result.Message = "Context length exceeded"
		result.Code = "context_length_exceeded"
		result.Param = inferBadRequestParam(joined, providerParam)
	case hasSignal(joined, "model group", "model not found", "model does not exist", "unsupported model", "invalid model", "model not supported"):
		param := "model"
		result.Message = "Invalid model"
		result.Code = "invalid_model"
		result.Param = &param
	// "invalid_parameter" (underscore) is this branch's own Code value below —
	// needed so re-classifying this package's own already-classified output
	// (e.g. litellm-compat mode's normalizeError running ClassifyBadRequest
	// on the router's own body: message "Invalid request parameter" doesn't
	// itself contain "invalid parameter"/"invalidparameter" as a substring,
	// "request" sits in between) still lands back in this same bucket
	// instead of falling through to the generic default.
	case hasSignal(joined, "invalid argument", "invalid parameter", "invalidparameter", "invalid_parameter", "invalid value", "unsupported parameter", "unknown parameter", "unrecognized parameter", "missing required", "required field", "must be", "should be", "does not support", "is not supported"):
		result.Message = "Invalid request parameter"
		result.Code = "invalid_parameter"
		// Precedence: an explicit "param" from the provider's own JSON is
		// authoritative; next, a field path quoted directly in the message
		// ("Invalid 'output[1].type': 'input_file'. Supported values are:
		// ...") is precise even for dynamic/nested paths a fixed list can't
		// cover; only fall back to the generic keyword list last — it does
		// broad substring matching (e.g. "input") that a quoted path like
		// "input_file" would otherwise shadow.
		param := providerParam
		if param == nil {
			param = extractQuotedInvalidField(signals)
		}
		if param == nil {
			param = inferBadRequestParam(joined, nil)
		}
		result.Param = param
	default:
		result.Param = providerParam
	}

	return result
}

const maxProviderErrorSignalBytes = 8 * 1024

func providerErrorSignalsFromBody(body []byte) ([]string, *string) {
	if len(body) > maxProviderErrorSignalBytes {
		body = body[:maxProviderErrorSignalBytes]
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil, nil
	}

	signals := make([]string, 0, 8)
	var param *string
	var value any
	if err := json.Unmarshal(trimmed, &value); err == nil {
		collectProviderErrorSignals(value, 0, &signals, &param)
	} else {
		signals = append(signals, string(trimmed))
	}
	return signals, param
}

func collectProviderErrorSignals(value any, depth int, signals *[]string, param **string) {
	if depth > 5 || len(*signals) >= 32 {
		return
	}
	switch v := value.(type) {
	case map[string]any:
		for _, key := range []string{"message", "type", "code", "status", "reason", "error"} {
			raw, exists := mapValueFold(v, key)
			if !exists {
				continue
			}
			text, ok := raw.(string)
			if ok {
				if s := cleanProviderErrorSignal(text); s != "" {
					*signals = append(*signals, s)
				}
			}
		}
		if *param == nil {
			for _, key := range []string{"param", "parameter", "field"} {
				raw, exists := mapValueFold(v, key)
				if !exists {
					continue
				}
				if text, ok := raw.(string); ok {
					if p := cleanProviderErrorParam(text); p != "" {
						*param = &p
						break
					}
				}
			}
		}
		for _, key := range []string{"error", "response", "details", "detail", "violations"} {
			raw, exists := mapValueFold(v, key)
			if exists {
				collectProviderErrorSignals(raw, depth+1, signals, param)
			}
		}
	case []any:
		for _, item := range v {
			collectProviderErrorSignals(item, depth+1, signals, param)
		}
	case string:
		if s := cleanProviderErrorSignal(v); s != "" {
			*signals = append(*signals, s)
		}
	}
}

func mapValueFold(values map[string]any, key string) (any, bool) {
	if raw, exists := values[key]; exists {
		return raw, true
	}
	for k, raw := range values {
		if strings.EqualFold(k, key) {
			return raw, true
		}
	}
	return nil, false
}

func cleanProviderErrorSignal(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if len(s) > 512 {
		s = s[:512]
	}
	return s
}

func cleanProviderErrorParam(s string) string {
	s = strings.TrimSpace(s)
	if s == "" || len(s) > 80 {
		return ""
	}
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			continue
		}
		switch r {
		case '_', '.', '-', '[', ']':
			continue
		default:
			return ""
		}
	}
	return s
}

func hasSignal(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func inferBadRequestParam(joined string, providerParam *string) *string {
	if providerParam != nil {
		return providerParam
	}
	for _, param := range []string{
		"max_completion_tokens",
		"max_output_tokens",
		"max_tokens",
		"tool_choice",
		"parallel_tool_calls",
		"response_format",
		"reasoning_effort",
		"service_tier",
		"stream_options",
		"temperature",
		"top_p",
		"logprobs",
		"messages",
		"input",
		"tools",
		"model",
		"stop",
		"metadata",
		"audio",
		"image",
		"prompt",
		"n",
	} {
		if strings.Contains(joined, param) {
			p := param
			return &p
		}
	}
	return nil
}

// extractQuotedInvalidField pulls the field path out of a provider message
// shaped like `Invalid 'output[1].type': 'input_file'. Supported values
// are: ...` — the quoted token right after "Invalid " is the offending
// field/parameter path. Fixed param-name lists can't cover these because
// the path is dynamic (array index, nested field), so this is checked as a
// fallback once the static list in inferBadRequestParam comes up empty.
// Tries each raw (pre-lowercased) signal in turn and returns the first hit.
func extractQuotedInvalidField(signals []string) *string {
	const marker = "invalid '"
	for _, s := range signals {
		idx := strings.Index(strings.ToLower(s), marker)
		if idx == -1 {
			continue
		}
		rest := s[idx+len(marker):]
		end := strings.IndexByte(rest, '\'')
		if end <= 0 {
			continue
		}
		field := rest[:end]
		return &field
	}
	return nil
}

func inferMaxTokensParam(joined string) string {
	switch {
	case strings.Contains(joined, "max_completion_tokens"):
		return "max_completion_tokens"
	case strings.Contains(joined, "max_output_tokens"):
		return "max_output_tokens"
	default:
		return "max_tokens"
	}
}
