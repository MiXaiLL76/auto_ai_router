package security

import (
	"encoding/json"
)

// redactedStringPlaceholder replaces a string that is not diagnostic. It is
// the same word the whole-body placeholder uses, so an operator reads one
// vocabulary.
const redactedStringPlaceholder = "[REDACTED]"

// maxResponseBodyRedactedBytes bounds a redacted provider response body, the
// same way the request side does: a body is kept whole up to this size and
// cut off with a marker beyond it, so one enormous answer cannot fill the
// log sink.
const maxResponseBodyRedactedBytes = 64 * 1024

// providerResponseDiagFields are the keys of a provider response whose values
// an operator reads to reconcile the exchange: what was called, what came
// back about it, and how many tokens it cost.
//
// They are an ALLOWLIST, not a denylist: a provider's error text can carry a
// credential in ordinary prose ("invalid api key sk-..."), so every string
// that is not on this list is masked. The request side works the same way
// (see maskClientParams) for the same reason.
//
// The list holds LEAF keys only. A key whose value is a container ("error",
// "usage", "choices") is not on it: the container is walked into, and the
// allowlist is applied to the keys inside it. Putting a container on the
// list would keep its whole value verbatim -- an "error" object whose
// "message" carries the credential would survive.
var providerResponseDiagFields = map[string]struct{}{
	// What was called, and what came back.
	"id":                 {},
	"model":              {},
	"object":             {},
	"created":            {},
	"index":              {},
	"finish_reason":      {},
	"finish_reasons":     {},
	"stop_reason":        {},
	"stop_sequence":      {},
	"role":               {},
	"type":               {},
	"status":             {},
	"error_type":         {},
	"error_code":         {},
	"provider":           {},
	"service_tier":       {},
	"system_fingerprint": {},
	"usage":              {},
	"logprobs":           {},
}

// providerResponseCountFields are the keys whose NUMERIC value is kept: the
// token counts and the sizes an error analysis needs. A number cannot carry
// a credential.
var providerResponseCountFields = map[string]struct{}{
	"prompt_tokens":                {},
	"completion_tokens":            {},
	"total_tokens":                 {},
	"cached_tokens":                {},
	"cache_creation_tokens":        {},
	"cache_write_tokens":           {},
	"cache_read_tokens":            {},
	"cache_creation_input_tokens":  {},
	"cache_read_input_tokens":      {},
	"ephemeral_5m_input_tokens":    {},
	"ephemeral_1h_input_tokens":    {},
	"reasoning_tokens":             {},
	"output_tokens":                {},
	"input_tokens":                 {},
	"max_tokens":                   {},
	"web_search_requests":          {},
	"web_search_count":             {},
	"duration_ms":                  {},
	"request_duration_ms":          {},
	"temperature":                  {},
	"top_p":                        {},
	"frequency_penalty":            {},
	"presence_penalty":             {},
	"seed":                         {},
	"stop_reason":                  {},
	"content_filter_results":       {},
	"input_token_details":          {},
	"output_token_details":         {},
	"prompt_tokens_details":        {},
	"completion_tokens_details":    {},
	"cache_creation_token_details": {},
	"cache_write_tokens_details":   {},
	"cache_creation":               {},
	"prompt_tokens_details_cache":  {},
	"logit_bias":                   {},
	"top_logprobs":                 {},
	"refusal":                      {},
	"reasoning":                    {},
	"reasoning_content":            {},
	"message_stopped":              {},
	"message_start":                {},
	"message_delta":                {},
	"content_block_start":          {},
	"content_block_delta":          {},
	"content_block_stop":           {},
	"message_stop":                 {},
	"content_block":                {},
	"thinking":                     {},
	"signature":                    {},
	"data":                         {},
}

// RedactProviderResponseBody masks a provider's response body before it
// reaches a log sink, keeping the diagnostic structure and dropping the
// text.
//
// It exists because the WHOLE-body placeholder the raw-bodies topic used
// left nothing to read: a row that says "[REDACTED body bytes=495]" cannot
// answer "which model answered, how many tokens, which finish reason, what
// did the provider complain about" -- which is exactly what an error
// analysis and a cost reconciliation need. The request side already keeps
// its structure (redactRequestBodyForLogging), and this brings the response
// side to the same level.
//
// ALLOWLIST, like the request side: a provider's message can carry a
// credential in ordinary prose, so every string that is not on the
// diagnostic list is masked, and only the diagnostic strings survive. The
// tool-call arguments and the model's own text are masked like the
// request side masks a prompt.
//
// Fails closed: a body that is not valid JSON comes back as the
// whole-body placeholder, never as the original text.
func RedactProviderResponseBody(body string) string {
	if body == "" {
		return ""
	}
	parsed, ok := parseJSONValue(body)
	if !ok {
		return RedactBodyForLog(body)
	}
	masked := maskProviderResponseValue(parsed)
	out, err := json.Marshal(masked)
	if err != nil {
		return RedactBodyForLog(body)
	}
	if len(out) > maxResponseBodyRedactedBytes {
		return string(out[:maxResponseBodyRedactedBytes]) + "...[truncated]"
	}
	return string(out)
}

// parseJSONValue decodes a body into a generic JSON value.
func parseJSONValue(body string) (any, bool) {
	var parsed any
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		return nil, false
	}
	return parsed, true
}

// maskProviderResponseValue walks a decoded provider response, keeping the
// diagnostic values and masking the rest.
//
// What survives is decided by the KEY and the VALUE's shape:
//
//   - a diagnostic field keeps its value verbatim;
//   - a string under any other key is masked -- the provider's prose, which
//     is where a credential would leak;
//   - a number or a boolean under any other key is kept, because neither
//     can carry a credential;
//   - an object or an array is walked into, so a container of parts keeps
//     its shape (the part types, the token counts) while the text inside
//     each part is masked.
func maskProviderResponseValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, child := range typed {
			switch {
			case providerResponseIsDiagField(key):
				out[key] = child
			case providerResponseIsStringValue(child):
				// A string that is not diagnostic: the provider's own
				// text, masked outright.
				out[key] = redactedStringPlaceholder
			case providerResponseIsContainer(child):
				// A container: walk into it, so its shape survives and
				// the text inside it does not.
				out[key] = maskProviderResponseValue(child)
			default:
				// A number or a boolean: kept. Neither can carry a
				// credential, and the counts are what an operator
				// reads.
				out[key] = child
			}
		}
		return out
	case []any:
		out := make([]any, 0, len(typed))
		for _, item := range typed {
			out = append(out, maskProviderResponseValue(item))
		}
		return out
	default:
		return value
	}
}

// providerResponseIsDiagField reports whether the key holds a diagnostic
// value that is kept verbatim.
func providerResponseIsDiagField(key string) bool {
	if _, ok := providerResponseDiagFields[key]; ok {
		return true
	}
	_, ok := providerResponseCountFields[key]
	return ok
}

// providerResponseIsStringValue reports whether the value is a string.
func providerResponseIsStringValue(value any) bool {
	_, ok := value.(string)
	return ok
}

// providerResponseIsContainer reports whether the value is an object or an
// array -- the two shapes that are walked into rather than masked whole.
func providerResponseIsContainer(value any) bool {
	switch value.(type) {
	case map[string]any, []any:
		return true
	default:
		return false
	}
}
