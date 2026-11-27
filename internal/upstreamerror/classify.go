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
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
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
	result := BadRequest{Message: fixedErrorMessages["invalid_request"], Code: "invalid_request"}
	if len(rawBody) == 0 {
		return result
	}

	// A body this package's classifier already produced is returned as it
	// is. Without this, every caller that re-classifies a router-written
	// body (litellm compatibility mode, an AIR-to-AIR proxy chain) would
	// depend on each individual branch recognizing its own canonical
	// output -- an obligation that grew with every branch and had already
	// leaked once, when a branch whose message did not round-trip degraded
	// to the generic one on the second pass. Recognizing the shape makes
	// the classifier idempotent by construction instead.
	if classified, ok := extractRouterClassification(rawBody, "invalid_request_error"); ok {
		return classified
	}

	info := providerErrorInfoFromBody(rawBody)
	signals := info.Signals
	providerParam := info.Param
	joined := strings.ToLower(strings.Join(signals, " "))

	// A provider message the router has a rule for is answered with the
	// router's own wording: the provider's words are matched, and the
	// router's answer is what reaches the client (see
	// ClassifyProviderMessage in provider_messages.go).
	//
	// The status, the code and the message all come from ONE rule, so they
	// cannot disagree about which rule answered. It used to be two
	// independent lookups -- one for the status, one for the message -- and
	// the two could answer from different rules: the client was answered
	// with a status its message did not belong to, or with a message whose
	// status it never got.
	if classified, ok := ClassifyProviderMessage(info); ok {
		// The parameter the provider named comes with the answer: a client
		// told "the request carries a parameter this model does not support"
		// can act on it only if it is also told WHICH parameter.
		return BadRequest{Message: classified.Message, Code: classified.Code, Param: classified.Param}
	}

	// An unknown model is checked FIRST, before the keyword branches below.
	//
	// The message names the model explicitly ("Received Model Group=x",
	// "the model gpt-5 does not exist"), which is a specific diagnosis. The
	// keyword branches match on generic words, and a body can carry both:
	// error_type="invalid_value" beside a message about a model group is
	// one body, and whichever branch runs first decides the outcome. The
	// specific one has to win, so the model check comes before them -- the
	// generic "invalid value" wording is still there to be matched by the
	// branches below when the message says nothing about a model.
	if hasSignal(joined, "model group", "model not found", "model does not exist", "unsupported model", "invalid model", "model not supported") {
		param := "model"
		result.Code = "invalid_model"
		result.Message = fixedErrorMessages[result.Code]
		result.Param = &param
		return result
	}

	// A credential/session-level block is checked next, before
	// classifyValidationError's generic content-policy branch: production
	// carries this exact provider signal at BOTH 400 and 403 (classify_status.go
	// has the 403 case), and its message happens to say "...content policy" in
	// plain English. Without this check that wording falls into the generic
	// content_policy_violation bucket -- which means "every credential would
	// refuse the same prompt", and the retry policy (config.DefaultNonRetryableMarkers)
	// treats it as non-retryable for exactly that reason. This is the opposite:
	// the SESSION behind this one credential got blocked, not the request, so a
	// different credential answers it fine and the retry must go through.
	// Reporting the identical code/message regardless of which status the
	// upstream happened to use keeps the two statuses from disagreeing about
	// the same failure.
	if hasSignal(joined, sessionBlockedSignal) {
		result.Message = sessionBlockedMessage
		result.Code = sessionBlockedCode
		return result
	}

	if classified, ok := classifyValidationError(joined, providerParam, signals); ok {
		return classified
	}

	// Context-window overflow is checked before the max_tokens branch: vLLM's
	// message ("This model's maximum context length is N tokens. However, you
	// requested M output tokens and your prompt contains at least K input
	// tokens ...") also says "output tokens"/"max_tokens", and reporting it as
	// "Invalid input_tokens"/invalid_max_tokens hides the real cause.
	if hasSignal(joined, contextLengthSignals...) {
		result.Message = contextLengthMessage(joined)
		result.Code = "context_length_exceeded"
		result.Param = inferBadRequestParam(joined, providerParam)
		return result
	}

	switch {
	case hasSignal(joined, "tool_choice", "tool choice", "toolchoice"):
		param := "tool_choice"
		result.Code = "invalid_tool_choice"
		result.Message = fixedErrorMessages[result.Code]
		result.Param = &param
	case hasSignal(joined, "maxoutputtokens value of", "maxoutputtokens of"):
		// A provider that pins an output-token ceiling states the client's
		// value AND the range it will accept: "Unable to submit request
		// because it has a maxOutputTokens value of 200000 but the supported
		// range is from 1 (inclusive) to 65537 (exclusive). Update the value
		// and try again."
		//
		// The message is assembled here rather than echoed, because the
		// client's next action is a concrete number: "ask for 65536 instead
		// of 200000". "Invalid max_tokens" alone would not tell it what to
		// ask for. Only the NUMBERS are taken from the provider's text --
		// they are re-validated as digits and re-rendered by this package's
		// own wording, so no provider free text reaches the client.
		//
		// Same shape as the context-length message above (contextLengthMessage),
		// which already reports the provider's own numbers to the client for
		// the same reason.
		//
		// The parameter name is the provider's camelCase "maxOutputTokens",
		// normalized to the snake_case spelling the rest of the classifier
		// uses (and the one OpenAI's own API documents).
		param := "max_output_tokens"
		result.Message = maxOutputTokensMessage(joined)
		result.Code = "invalid_max_tokens"
		result.Param = &param
	case hasSignal(joined, "max_tokens_exceeded", "max tokens exceeded", "maxtokens exceeded"):
		// The generation ran into the output-token limit: the request was
		// accepted, the model started producing output, and the answer was
		// cut off. NOT the same thing as the client's own max_tokens value
		// being rejected -- that is "invalid_max_tokens", with the
		// parameter, because the client's fix is to lower the value.
		//
		// The two are told apart by the provider's own structured "param":
		// a provider that rejects a VALUE names the parameter it rejected
		// ("Max tokens exceeded: requested 100000, but the maximum allowed
		// is 8192." with param:"max_tokens"), while a provider that reports
		// a reached limit names no parameter -- in production all 71,764 of
		// those came from an agent loop that had not set max_tokens at all
		// (the string appeared only as the NAME of a tool-schema property),
		// so there was nothing for the client to change.
		//
		// Checked BEFORE the max_tokens branch below, which matches the
		// "max tokens" substring this name contains.
		if providerParam == nil || !isOutputTokenLimitParam(*providerParam) {
			result.Code = "max_tokens_exceeded"
			result.Message = fixedErrorMessages[result.Code]
			break
		}
		fallthrough
	case hasSignal(joined, "max_completion_tokens", "max_output_tokens", "max_tokens", "max tokens", "maximum tokens", "output tokens"):
		param := inferBadRequestParam(joined, providerParam)
		if param == nil || !isOutputTokenLimitParam(*param) {
			v := inferMaxTokensParam(joined)
			param = &v
		}
		result.Message = invalidFieldPrefix + *param
		result.Code = "invalid_max_tokens"
		result.Param = param
	case hasSignal(joined, "unrecognized request argument supplied",
		"unrecognized request arguments supplied"):
		// A provider that does not know a request field says which one:
		// "Unrecognized request argument supplied: reasoning_effort" (2,018
		// requests in production, 1,996 of them naming reasoning_effort --
		// clients sending reasoning_effort to a model that has no such
		// field).
		//
		// The wording was not recognized before: the switch's marker for a
		// rejected parameter is "unrecognized parameter", and this provider
		// writes "unrecognized request argument" -- the words in between
		// ("request") mean the marker does not match, so every one of those
		// requests fell through to the generic "Invalid request" and the
		// client was not told WHICH field to drop. It was told, in the
		// provider's own first line.
		//
		// The parameter name is read out of the message and accepted only
		// from the closed list of request parameters, so an unknown name
		// cannot smuggle provider free text into the client's "param" field.
		//
		// The code is unsupported_parameter, NOT invalid_field. "Invalid
		// <param>" reads as "the value you sent is not acceptable, change
		// it" -- but here the provider does not know the FIELD at all (a
		// client sending reasoning_effort to a model that has no such
		// field), and there is no value to change: the field has to go.
		// The two need different client actions, so they get different
		// codes.
		if param := extractUnrecognizedArgumentName(signals); param != nil {
			result.Message = unsupportedParameterPrefix + *param
			result.Code = "unsupported_parameter"
			result.Param = param
		} else {
			result.Code = "invalid_parameter"
			result.Message = fixedErrorMessages[result.Code]
		}
	case hasSignal(joined, "is allowed for this model", "not allowed for this model"):
		// A provider that pins a sampling parameter to one value ("invalid
		// temperature: only 1 is allowed for this model") names the
		// parameter in the first word of its message and then states the
		// constraint. The generic "invalid ..." branches never fire on this
		// wording -- it contains no "invalid parameter", no "must be",
		// nothing else the switch recognizes -- so the request fell through
		// to the fully generic "Invalid request" and the client could not
		// tell which parameter to change, which is the one thing the
		// provider stated plainly in the first two words.
		//
		// The parameter name is read out of the message itself, but only a
		// KNOWN parameter name is accepted, so an unknown wording cannot
		// smuggle provider free text into the client's "param" field.
		//
		// This case sits AFTER the model-group case on purpose. The two
		// phrasings overlap in the wild -- "gpt-5 is not allowed for this
		// model group" contains both "not allowed for this model" and
		// "model group" -- and the model-group diagnosis is the more
		// specific one: the client asked for a model that does not exist
		// here, which is a different fix from "this parameter has a fixed
		// value".
		if param := extractPinnedParameterName(signals); param != nil {
			result.Message = pinnedParameterMessage(*param, signals)
			result.Code = "invalid_field"
			result.Param = param
		} else {
			result.Code = "invalid_parameter"
			result.Message = fixedErrorMessages[result.Code]
		}
	// "invalid_parameter" (underscore) is this branch's own Code value below —
	// needed so re-classifying this package's own already-classified output
	// (e.g. litellm-compat mode's normalizeError running ClassifyBadRequest
	// on the router's own body: message "Invalid request parameter" doesn't
	// itself contain "invalid parameter"/"invalidparameter" as a substring,
	// "request" sits in between) still lands back in this same bucket
	// instead of falling through to the generic default.
	case hasSignal(joined, "invalid argument", "invalid parameter", "invalidparameter", "invalid_parameter", "invalid value", "unsupported parameter", "unknown parameter", "unrecognized parameter", "argument not supported", "missing required", "required field", "must be", "should be", "does not support", "is not supported"):
		result.Code = "invalid_parameter"
		result.Message = fixedErrorMessages[result.Code]
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
			result.Message = invalidFieldPrefix + *providerParam
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

// contextLengthGenericMessage is used when the provider text carries no limit.
const contextLengthGenericMessage = "This model's maximum context length is exceeded by the request"

const (
	invalidFieldPrefix         = "Invalid "
	unsupportedParameterPrefix = "Unsupported parameter: "
	outputTokenRangePrefix     = ": requested "
	rangeMessageSeparator      = ", but the supported range is "
	exclusiveBoundSuffix       = " (exclusive)"
	pinnedValuePrefix          = ": only "
	pinnedValueSuffix          = " is allowed for this model"
)

var contextLengthSignals = []string{"context length", "context window", "context limit", "too many tokens", "input too long", "prompt too long", "prompt is too long", "token limit"}

var (
	contextMaxTokensRe    = regexp.MustCompile(`maximum context length is (\d+)`)
	contextOutputTokensRe = regexp.MustCompile(`requested (\d+) output tokens|(\d+) in the completion`)
	contextInputTokensRe  = regexp.MustCompile(`(\d+) input tokens|(\d+) in the messages`)

	// maxOutputTokensRequestedRe / maxOutputTokensMinRe / maxOutputTokensMaxRe
	// pull the three numbers out of a provider's "it has a maxOutputTokens
	// value of N but the supported range is from M to K" message.
	//
	// The numbers are matched as digit runs and re-validated as digits by
	// firstNumber, so what reaches the client is a number this package
	// parsed -- never a slice of the provider's own sentence. The wording
	// around them is this package's.
	//
	// Each pattern accepts the phrasings production actually uses:
	// "maxOutputTokens value of 200000" and "maxOutputTokens of 200000";
	// "the supported range is from 1 to 65537" and "the maximum allowed is
	// up to 65537".
	maxOutputTokensRequestedRe = regexp.MustCompile(`maxoutputtokens (?:value )?of (\d+)`)
	maxOutputTokensMinRe       = regexp.MustCompile(`supported range is from (\d+)(\s*\(exclusive\))?`)
	// The upper bound sits after the lower one, with provider-specific
	// filler in between: "from 1 (inclusive) to 65537 (exclusive)". The
	// filler is why the pattern cannot simply look for "to N" -- it has to
	// allow whatever the provider writes in between, and it must not stop
	// at the lower bound's own digits.
	maxOutputTokensMaxRe = regexp.MustCompile(`(?:supported range is from \d+[^0-9]*?to|up to|maximum allowed is) (\d+)(\s*\(exclusive\))?`)
)

// maxOutputTokensMessage rebuilds an out-of-range output-token message from
// the numbers in the provider text (digits only -- never the provider's own
// wording).
//
// The client's next action is a concrete number, so the message carries one:
// which value it asked for, and the range that would be accepted. "Invalid
// max_tokens" on its own leaves the client guessing what to put there
// instead.
//
// The rendering is CANONICAL: the message always names the numbers it was
// built from, so an operator reading it can tell what the client asked for
// and what it should have asked for.
//
// When the provider's text carries no parsable numbers the generic message
// is returned, so the client still learns WHICH parameter was rejected even
// though it does not learn the range.
func maxOutputTokensMessage(joined string) string {
	requested := firstNumber(maxOutputTokensRequestedRe, joined)
	lower := maxOutputTokensMinRe.FindStringSubmatch(joined)
	upper := maxOutputTokensMaxRe.FindStringSubmatch(joined)

	if requested == "" || upper == nil {
		return "Invalid max_output_tokens"
	}
	msg := invalidFieldPrefix + "max_output_tokens" + outputTokenRangePrefix + requested + rangeMessageSeparator
	if lower != nil {
		msg += "from " + lower[1]
		if lower[2] != "" {
			msg += exclusiveBoundSuffix
		}
		msg += " to "
	} else {
		msg += "up to "
	}
	msg += upper[1]
	if upper[2] != "" {
		msg += exclusiveBoundSuffix
	}
	return msg + "."
}

// contextLengthMessage rebuilds a context-overflow message from the numbers
// in the provider text (digits only — never the provider's own wording).
// The "This model's maximum context length is" prefix is the phrase
// OpenAI-compatible clients (LiteLLM's ContextWindowExceededError, which
// drives its context_window_fallbacks) match on, so it's kept even when
// the limit itself is unknown.
func contextLengthMessage(joined string) string {
	limit := firstNumber(contextMaxTokensRe, joined)
	output := firstNumber(contextOutputTokensRe, joined)
	input := firstNumber(contextInputTokensRe, joined)
	if limit == "" {
		return contextLengthGenericMessage
	}
	msg := "This model's maximum context length is " + limit + " tokens."
	switch {
	case input != "" && output != "":
		msg += " However, you requested " + output + " output tokens and your prompt contains at least " + input + " input tokens."
	case input != "":
		msg += " However, your prompt contains at least " + input + " input tokens."
	}
	return msg
}

func firstNumber(re *regexp.Regexp, text string) string {
	m := re.FindStringSubmatch(text)
	for _, g := range m[min(1, len(m)):] {
		if g != "" {
			return g
		}
	}
	return ""
}

const maxProviderErrorSignalBytes = 8 * 1024

// providerErrorInfoFromBody reads what a provider's error body carries: the
// provider's own words about the failure, the codes it states them by, and
// the parameter it rejected.
//
// It is the classifier's own reader of a body, and it reads the body the
// same way every other reader does (see ExtractProviderErrorInfo). The
// classifier used to have a reader of its own, and the two disagreed: the
// classifier's reader descended into a nested "detail" and read a
// machine-readable reason out of "error_type", while the readers on the
// status path read neither. The same body was therefore a content refusal
// on one path and an unclassified failure on another, and a rule that named
// it applied on one path and not on the other.
func providerErrorInfoFromBody(body []byte) ProviderErrorInfo {
	if len(body) > maxProviderErrorSignalBytes {
		body = body[:maxProviderErrorSignalBytes]
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return ProviderErrorInfo{}
	}

	var value any
	if err := json.Unmarshal(trimmed, &value); err != nil {
		return ProviderErrorInfo{Signals: []string{string(trimmed)}}
	}

	info := ProviderErrorInfo{}
	collectProviderErrorInfoValue(value, 0, &info)
	return info
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

func isOutputTokenLimitParam(param string) bool {
	switch param {
	case "max_tokens", "max_completion_tokens", "max_output_tokens":
		return true
	default:
		return false
	}
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

// pinnableParameters are the request parameters a provider may pin to a single
// value, and then complain about by name -- "invalid temperature: only 1 is
// allowed for this model". Only these are accepted as the offending
// parameter: the name comes out of the provider's own free text, so without
// a closed list a message like "invalid your-favourite-setting: only 1 is
// allowed" would put an arbitrary string into the client's "param" field,
// which is exactly the kind of provider-controlled text this package exists
// to keep out of the client-facing body.
var pinnableParameters = map[string]struct{}{
	"temperature":           {},
	"top_p":                 {},
	"top_k":                 {},
	"frequency_penalty":     {},
	"presence_penalty":      {},
	"repetition_penalty":    {},
	"max_tokens":            {},
	"max_completion_tokens": {},
	"max_output_tokens":     {},
	"n":                     {},
	"seed":                  {},
	"stop":                  {},
	"reasoning_effort":      {},
	"thinking":              {},
}

// requestParameterNames are the request fields a provider may report as
// unrecognized. Only these are accepted as the offending parameter: the
// name comes out of the provider's own free text, so without a closed list
// a message like "Unrecognized request argument supplied: your-secret-key"
// would put an arbitrary string into the client's "param" field.
var requestParameterNames = map[string]struct{}{
	"temperature":           {},
	"top_p":                 {},
	"top_k":                 {},
	"frequency_penalty":     {},
	"presence_penalty":      {},
	"repetition_penalty":    {},
	"max_tokens":            {},
	"max_completion_tokens": {},
	"max_output_tokens":     {},
	"max_prompt_tokens":     {},
	"n":                     {},
	"seed":                  {},
	"stop":                  {},
	"reasoning_effort":      {},
	"thinking":              {},
	"reasoning":             {},
	"enable_thinking":       {},
	"skip_special_tokens":   {},
	"response_format":       {},
	"tools":                 {},
	"tool_choice":           {},
	"parallel_tool_calls":   {},
	"stream_options":        {},
	"logprobs":              {},
	"top_logprobs":          {},
	"service_tier":          {},
	"user":                  {},
	"metadata":              {},
	"modalities":            {},
	"prediction":            {},
	"store":                 {},
}

// extractUnrecognizedArgumentName reads the parameter name out of a provider
// message of the shape "Unrecognized request argument supplied: <name>" (and
// its plural variant, which lists several names).
//
// Returns nil when the message names no KNOWN parameter, so an unrecognized
// wording falls back to the generic "Invalid request parameter" instead of
// guessing -- and a name that is not a request parameter is never passed
// through to the client.
func extractUnrecognizedArgumentName(signals []string) *string {
	const marker = "unrecognized request argument supplied:"
	const pluralMarker = "unrecognized request arguments supplied:"
	for _, signal := range signals {
		// The marker is located in a LOWERCASED copy, but the names are cut
		// out of the ORIGINAL. The two are reconciled by RUNE count, not by
		// byte offset: Unicode lowering can change a string's byte length
		// (U+212A "K" lowercases to one byte, U+0130 "İ" to two), so a
		// byte offset found in the lowered copy would point at the wrong
		// place in the original and the slice would start mid-name.
		//
		// This is the same fix extractPinnedParameterName applies, for the
		// same reason.
		lower := strings.ToLower(signal)
		original := []rune(signal)
		for _, m := range []string{marker, pluralMarker} {
			idx := strings.Index(lower, m)
			if idx < 0 {
				continue
			}
			markerRunes := utf8.RuneCountInString(m)
			startRunes := utf8.RuneCountInString(lower[:idx])
			if startRunes+markerRunes > len(original) {
				continue
			}
			rest := strings.TrimSpace(string(original[startRunes+markerRunes:]))
			// The list may hold several names ("enable_thinking, reasoning,
			// reasoning_effort, thinking"). The FIRST known one is reported:
			// the client needs one field to drop, and the provider lists them
			// in its own order.
			for _, name := range strings.Split(rest, ",") {
				name = strings.TrimSpace(name)
				name = strings.Trim(name, ".'\"")
				if name == "" {
					continue
				}
				if _, known := requestParameterNames[name]; known {
					field := name
					return &field
				}
			}
		}
	}
	return nil
}

// extractPinnedParameterName reads the parameter name out of a provider
// message of the shape "invalid <param>: only <value> is allowed for this
// model" (and its "not allowed" variant). Returns nil when the message does
// not name a known parameter, so an unrecognized wording falls back to the
// generic "Invalid request parameter" instead of guessing.
// pinnedParameterValueRe reads the value a provider says a parameter is
// pinned to: "invalid temperature: only 1 is allowed for this model" names
// the value 1.
//
// Only a NUMBER is taken from the provider's text, and it is re-validated as
// digits, so what reaches the client is a value this package parsed -- never
// a slice of the provider's own sentence. The wording around it is this
// package's.
var pinnedParameterValueRe = regexp.MustCompile(`only ([0-9]+(?:\.[0-9]+)?) is allowed`)

// pinnedParameterMessage builds the client-facing message for a parameter
// the provider pins to a single value.
//
// The value goes in, because it is the client's next action: a client that
// sent temperature=0 and is told "Invalid temperature" still does not know
// what to send instead, while "only 1 is allowed" says it exactly. This is
// the same reasoning as the output-token range message (see
// maxOutputTokensMessage), which reports the accepted range for the same
// reason.
//
// Falls back to naming the parameter alone when the provider states no
// parsable value, so the client still learns WHAT to fix.
func pinnedParameterMessage(param string, signals []string) string {
	base := invalidFieldPrefix + param
	for _, signal := range signals {
		if value := firstNumber(pinnedParameterValueRe, strings.ToLower(signal)); value != "" {
			return base + pinnedValuePrefix + value + pinnedValueSuffix
		}
	}
	return base
}

func extractPinnedParameterName(signals []string) *string {
	for _, signal := range signals {
		// The marker is located in a LOWERCASED copy, but the parameter
		// name is cut out of the ORIGINAL. The two are reconciled by RUNE
		// count, not by byte offset: Unicode lowering can change a
		// string's byte length (U+0130 "İ" lowercases to one byte), so a
		// byte offset found in the lowered copy would point at the wrong
		// place in the original and the slice would start mid-word.
		//
		// Counting runes instead means the slice starts at the same
		// CHARACTER the marker was found at, whatever its byte width.
		lower := strings.ToLower(signal)
		const marker = "invalid "
		idx := strings.Index(lower, marker)
		if idx < 0 {
			continue
		}
		// idx is a BYTE offset into the lowered copy; the slice below is
		// taken from the ORIGINAL by RUNE position. Converting through the
		// rune count keeps the two aligned even when the lowered and the
		// original spell the same character span with a different number
		// of bytes.
		markerRunes := utf8.RuneCountInString(marker)
		startRunes := utf8.RuneCountInString(lower[:idx])
		original := []rune(signal)
		if startRunes+markerRunes > len(original) {
			continue
		}
		rest := string(original[startRunes+markerRunes:])
		// The name runs up to the first delimiter the provider could put
		// there -- a colon ("invalid temperature: only 1 ...") or a space
		// ("invalid temperature value ...").
		end := 0
		for end < len(rest) && (isIdentifierByte(rest[end]) || rest[end] == '_') {
			end++
		}
		if end == 0 {
			continue
		}
		// The name is matched case-insensitively: providers capitalize it
		// freely ("Invalid Temperature: only 1 is allowed"), and a
		// capitalized name is the same parameter.
		name := asciiLower(rest[:end])
		if _, known := pinnableParameters[name]; !known {
			continue
		}
		field := name
		return &field
	}
	return nil
}

// asciiLower lowercases A-Z only, leaving every other byte -- including
// multi-byte UTF-8 sequences -- untouched.
//
// Unlike strings.ToLower it never changes a string's byte length, so an
// offset found in the lowered copy always points at the same place in the
// original. That matters wherever a lowercased copy is used to LOCATE
// something in the original string.
func asciiLower(s string) string {
	hasUpper := false
	for i := 0; i < len(s); i++ {
		if s[i] >= 'A' && s[i] <= 'Z' {
			hasUpper = true
			break
		}
	}
	if !hasUpper {
		return s
	}
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
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
		// The marker is located in a LOWERCASED copy, but the field is cut
		// out of the ORIGINAL. The two are reconciled by RUNE count, not by
		// byte offset: Unicode lowering can change a string's byte length
		// (U+212A "K" lowercases to one byte, U+0130 "İ" is two bytes that
		// lowercase to one), so a byte offset found in the lowered copy
		// would point at the wrong place in the original and the field
		// would be cut mid-name or missed entirely.
		//
		// This is the same fix extractUnrecognizedArgumentName applies, for
		// the same reason.
		lower := strings.ToLower(s)
		idx := strings.Index(lower, marker)
		if idx == -1 {
			continue
		}
		markerRunes := utf8.RuneCountInString(marker)
		startRunes := utf8.RuneCountInString(lower[:idx])
		original := []rune(s)
		if startRunes+markerRunes > len(original) {
			continue
		}
		rest := string(original[startRunes+markerRunes:])
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
	case IsContentPolicySignal(joined):
		// A provider-side content refusal is a fault of the request, not of the
		// credential: surface it as a clear 400 (the retry policy's content-policy
		// marker also makes it non-retryable) instead of a generic 502.
		// Checked before the generic validation branches so the real cause wins.
		//
		// The markers come from ContentPolicyMarkers -- the single list shared
		// with the proxy's status classification, the LiteLLM compatibility
		// layer and the retry policy. It used to be a fourth copy of its own,
		// and the copies drifted: a wording added to the other three left this
		// one behind, so a refusal phrased that way was correctly classified as
		// a non-retryable 400 yet reported to the client as the generic
		// "Invalid request" instead of the content-policy message.
		//
		// The markers stay narrowly content-policy-specific. A bare "safety" or
		// "blocklist" would also match unrelated validation messages (e.g. a
		// safety_identifier parameter error); a bare "blocked by the provider"
		// would swallow billing/geo/IP blocks ("Your account has been blocked by
		// the provider for suspected fraud"), replacing the real cause with a
		// misleading content-policy message.
		result.Code = "content_policy_violation"
		result.Message = fixedErrorMessages[result.Code]
		return result, true
	case hasSignal(joined, "invalid_image_size", "invalid image size"):
		result.Code = "invalid_image_size"
		result.Message = fixedErrorMessages[result.Code]
		if result.Param == nil {
			field := "size"
			result.Param = &field
		}
	case hasSignal(joined, "invalid_image", "invalid image", "could not decode image", "unable to decode image", "failed to decode image", "image is not valid", "image is invalid", "image_parse_error", "image file could not be processed", "unable to process input image", "unable to process the input image"):
		result.Code = "invalid_image"
		result.Message = fixedErrorMessages[result.Code]
		if result.Param == nil {
			field := "image"
			result.Param = &field
		}
	case hasSignal(joined, "missing_required_parameter", "missing required parameter", "missing required argument"):
		result.Code = "missing_required_parameter"
		result.Message = fixedErrorMessages[result.Code]
	case hasSignal(joined, "invalid_type", "invalid type", "invalid parameter type"):
		result.Code = "invalid_type"
		result.Message = fixedErrorMessages[result.Code]
	case hasSignal(joined, "invalid_json", "invalid json"):
		result.Code = "invalid_json"
		result.Message = fixedErrorMessages[result.Code]
	case hasSignal(joined, "invalid_multipart", "invalid multipart"):
		result.Code = "invalid_multipart"
		result.Message = fixedErrorMessages[result.Code]
	case hasSignal(joined, "invalid_value", "invalid value", "invalid parameter value"):
		result.Code = "invalid_value"
		result.Message = fixedErrorMessages[result.Code]
	case hasSignal(joined, "reasoning is mandatory", "thinking is mandatory"):
		// A provider that serves only reasoning models refuses a request
		// that turns reasoning off: "Reasoning is mandatory for this
		// endpoint and cannot be disabled." The client had sent
		// reasoning_effort=none (all 468 production requests did exactly
		// that), so the refusal is about the parameter the client set --
		// not about the request in general.
		//
		// The parameter is named here rather than read out of the message:
		// the provider's wording says "reasoning", while the parameter the
		// client actually sends is "reasoning_effort" (OpenAI's spelling)
		// or "thinking" (Anthropic's). Reporting the client's own spelling
		// is what makes the error actionable -- the client can go change
		// the field it set.
		//
		// The message says what to do rather than only what went wrong:
		// the fix is to enable reasoning, not to "fix" a parameter.
		result.Code = "reasoning_required"
		result.Message = fixedErrorMessages[result.Code]
		param := "reasoning_effort"
		result.Param = &param
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
