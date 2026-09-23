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
	"unicode"
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
	if classified, ok := classifyValidationError(joined, providerParam, signals); ok {
		return classified
	}

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
	case hasSignal(joined, "context length", "context window", "context limit", "too many tokens", "input too long", "prompt too long", "prompt is too long", "token limit"):
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
	case hasSignal(joined, "invalid argument", "invalid parameter", "invalidparameter", "invalid_parameter", "invalid value", "unsupported parameter", "unknown parameter", "unrecognized parameter", "argument not supported", "missing required", "required field", "must be", "should be", "does not support", "is not supported"):
		result.Message = "Invalid request parameter"
		result.Code = "invalid_parameter"
		// Precedence: an explicit "param" from the provider's own JSON is
		// authoritative; next, a field path quoted directly in the message
		// ("Invalid 'output[1].type': 'input_file'. Supported values are:
		// ...") is precise even for dynamic/nested paths a fixed list can't
		// cover; then a parameter the message names in prose ("The parameter
		// size specified in the request is not valid: image size must be
		// ..."); only fall back to the generic keyword list last — it matches
		// any listed word anywhere in the message, so unrelated wording such
		// as "image size" or "by the model" would otherwise win.
		param := providerParam
		if param == nil {
			param = extractQuotedInvalidField(signals)
		}
		if param == nil {
			param = extractNamedParameterField(signals)
		}
		if param == nil {
			param = inferBadRequestParam(joined, nil)
		}
		result.Param = param
	default:
		// providerParam comes from a structured "param"/"parameter"/"field" key
		// in the provider's own JSON (see collectProviderErrorSignals) — not a
		// text-substring guess — so it's trustworthy even when none of the
		// message-shape branches above recognize the provider's wording (e.g.
		// "temperature must be between 0 and 2" doesn't contain any of this
		// switch's keyword signals, yet the provider still told us exactly
		// which field was rejected). Surfacing "Invalid <param>" instead of
		// the fully generic "Invalid request" costs nothing in safety — it's
		// still one of a fixed, pre-vetted shape, never provider free text —
		// and gives callers something actionable instead of a blank message.
		if providerParam != nil {
			result.Message = "Invalid " + *providerParam
			// "invalid_field", not "invalid_value" — that string is
			// classifyValidationError's own signal for a different, more
			// specific bucket. A status code is itself fed back into the
			// joined signal text on re-classification (see
			// providerErrorSignalsFromBody collecting "code"), so reusing
			// that phrase would send this package's own re-classification of
			// its output into the wrong bucket — the exact idempotency
			// failure TestClassifyBadRequest_IsIdempotent guards against.
			// "invalid_field" matches no hasSignal list in this file, so a
			// second pass falls through to this same default branch again.
			result.Code = "invalid_field"
		}
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
		if containsWord(joined, param) {
			p := param
			return &p
		}
	}
	return nil
}

// containsWord reports whether word occurs in text as a whole identifier —
// not as part of a longer one. Plain substring matching made short names
// like "n" match almost any message, and "input" match inside "input_file".
func containsWord(text, word string) bool {
	for from := 0; from < len(text); {
		idx := strings.Index(text[from:], word)
		if idx < 0 {
			return false
		}
		start := from + idx
		end := start + len(word)
		if (start == 0 || !isIdentifierByte(text[start-1])) && (end == len(text) || !isIdentifierByte(text[end])) {
			return true
		}
		from = start + 1
	}
	return false
}

func isIdentifierByte(b byte) bool {
	return b == '_' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9'
}

// namedParameterStopwords are words that commonly follow "parameter" in
// provider prose without being a parameter name ("Invalid parameter value",
// "parameters specified in the request").
var namedParameterStopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "can": {}, "cannot": {}, "does": {},
	"error": {}, "for": {}, "format": {}, "has": {}, "have": {}, "in": {},
	"invalid": {}, "is": {}, "list": {}, "missing": {}, "must": {}, "name": {},
	"names": {}, "not": {}, "of": {}, "or": {}, "provided": {}, "required": {},
	"set": {}, "should": {}, "specified": {}, "supplied": {}, "that": {},
	"the": {}, "this": {}, "to": {}, "type": {}, "types": {}, "validation": {},
	"value": {}, "values": {}, "was": {}, "were": {}, "which": {}, "with": {},
}

// extractNamedParameterField pulls a parameter name out of provider messages
// that name it in prose instead of a structured "param" field:
//
//	The parameter size specified in the request is not valid: ...
//	The specified parameter `image_config.aspect_ratio` is invalid.
//	The parameter service_tier=flex specified in the request is not supported ...
//	Argument not supported: size
//
// Returns nil when the word after "parameter" is ordinary prose.
func extractNamedParameterField(signals []string) *string {
	for _, signal := range signals {
		lowerBytes := []byte(signal)
		for i, b := range lowerBytes {
			if b >= 'A' && b <= 'Z' {
				lowerBytes[i] = b + ('a' - 'A')
			}
		}
		lower := string(lowerBytes)

		for _, marker := range []string{"argument not supported:", "unsupported argument:"} {
			if idx := strings.Index(lower, marker); idx >= 0 {
				if field := leadingParameterName(signal[idx+len(marker):]); field != "" {
					return &field
				}
			}
		}

		const word = "parameter"
		for from := 0; from < len(lower); {
			idx := strings.Index(lower[from:], word)
			if idx < 0 {
				break
			}
			start := from + idx
			from = start + len(word)
			if start > 0 && isIdentifierByte(lower[start-1]) {
				continue // e.g. "invalidparameter"
			}
			rest := signal[from:]
			switch {
			case strings.HasPrefix(rest, "(s)"):
				rest = rest[len("(s)"):]
			case strings.HasPrefix(rest, "s"):
				rest = rest[len("s"):]
			}
			if rest == "" || (rest[0] != ' ' && rest[0] != ':') {
				continue // e.g. "parameterized"
			}
			if field := parameterNamedAfter(rest); field != "" {
				return &field
			}
		}
	}
	return nil
}

var namedParameterFollowers = map[string]struct{}{
	"are": {}, "can": {}, "cannot": {}, "does": {}, "has": {}, "have": {}, "is": {},
	"is/are": {}, "must": {}, "not": {}, "should": {}, "specified": {}, "value": {},
	"values": {}, "was": {}, "were": {},
}

// parameterNamedAfter reads the word following "parameter" and accepts it as a
// name only when the message marks it as one: quoted ("parameter `size`"),
// introduced by a colon ("Invalid parameter: size"), assigned
// ("service_tier=flex"), or followed by a verb about it ("parameter size
// specified ..."). Plain prose such as "parameter parsing failed" is rejected.
func parameterNamedAfter(rest string) string {
	colon := false
	i := 0
	for i < len(rest) && (rest[i] == ' ' || rest[i] == ':') {
		colon = colon || rest[i] == ':'
		i++
	}
	s := rest[i:]
	quoted := s != "" && strings.IndexByte("`'\"", s[0]) >= 0

	token := leadingParameterName(s)
	if token == "" {
		return ""
	}
	if quoted || colon {
		return token
	}
	after := strings.TrimPrefix(s[len(token):], ".")
	if strings.HasPrefix(after, "=") {
		return token
	}
	fields := strings.Fields(after)
	if len(fields) == 0 {
		return ""
	}
	if _, ok := namedParameterFollowers[strings.ToLower(strings.TrimRight(fields[0], ".,;:"))]; ok {
		return token
	}
	return ""
}

// leadingParameterName reads the parameter token at the start of s, skipping
// separators and an opening quote, and stopping at anything that can't be part
// of a parameter path ("size`", "service_tier=flex", "size.").
func leadingParameterName(s string) string {
	s = strings.TrimLeft(s, " :")
	s = strings.TrimLeft(s, "`'\"")
	end := 0
	for end < len(s) && (isIdentifierByte(s[end]) || s[end] == '.' || s[end] == '[' || s[end] == ']' || s[end] == '-') {
		end++
	}
	token := strings.TrimRight(s[:end], ".")
	if _, stop := namedParameterStopwords[strings.ToLower(token)]; stop {
		return ""
	}
	return cleanProviderErrorParam(token)
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

func classifyValidationError(joined string, param *string, signals []string) (BadRequest, bool) {
	result := BadRequest{Param: param}
	switch {
	case hasSignal(joined, "invalid_image_size", "invalid image size"):
		result.Message, result.Code = "Invalid image size", "invalid_image_size"
		if result.Param == nil {
			field := "size"
			result.Param = &field
		}
	case hasSignal(joined, "invalid_image", "invalid image", "could not decode image", "unable to decode image", "failed to decode image", "image is not valid", "image is invalid", "image_parse_error", "image file could not be processed", "unable to process input image", "unable to process the input image"):
		result.Message, result.Code = "Invalid image data", "invalid_image"
		if result.Param == nil {
			field := "image"
			result.Param = &field
		}
	case hasSignal(joined, "missing_required_parameter", "missing required parameter", "missing required argument"):
		result.Message, result.Code = "Missing required parameter", "missing_required_parameter"
	case hasSignal(joined, "invalid_type", "invalid type", "invalid parameter type"):
		result.Message, result.Code = "Invalid parameter type", "invalid_type"
	case hasSignal(joined, "invalid_json", "invalid json"):
		result.Message, result.Code = "Invalid JSON", "invalid_json"
	case hasSignal(joined, "invalid_multipart", "invalid multipart"):
		result.Message, result.Code = "Invalid multipart form data", "invalid_multipart"
	case hasSignal(joined, "invalid_value", "invalid value", "invalid parameter value"):
		result.Message, result.Code = "Invalid parameter value", "invalid_value"
	default:
		return BadRequest{}, false
	}
	if result.Param == nil {
		if field := extractQuotedInvalidField(signals); field != nil {
			if cleaned := cleanProviderErrorParam(*field); cleaned != "" {
				result.Param = &cleaned
			}
		}
	}
	if result.Param == nil {
		for _, word := range strings.FieldsFunc(joined, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
		}) {
			switch word {
			case "prompt", "size", "n", "image", "mask", "seed", "quality", "output_format", "output_compression", "background", "moderation", "response_format", "image_config", "aspect_ratio", "image_size":
				field := word
				result.Param = &field
				return result, true
			}
		}
	}
	return result, true
}
