package upstreamerror

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

// ProviderErrorInfo is what the router reads out of a provider's error
// value: the provider's own words about the failure, the codes it states
// them by, and the parameter it rejected.
//
// The words and the codes are kept apart on purpose. The words are matched
// as text -- a rule may look for a wording anywhere in them -- while a code
// is matched only against what the provider wrote in a code-shaped field.
// Letting a rule's code be compared against the message text let a rule
// fire on a message that merely happened to read like the code: a provider
// that echoes its reason string as its message ("RESOURCE_EXHAUSTED") made
// a rule for that code answer for a failure it was never written for.
type ProviderErrorInfo struct {
	// Signals are the provider's own words: its message, its type, its
	// code, its status, its reason, and the same fields of the objects
	// nested in it.
	Signals []string
	// Codes are the provider's own error codes: what it wrote in its "code"
	// and "type" fields.
	Codes []string
	// Param is the parameter the provider rejected, when it names one.
	Param *string
	// ProviderMessage is the provider's own message, WHOLE -- the one it
	// wrote, before the signals above were trimmed to fit the bound they
	// carry.
	//
	// It is kept apart because a rule may relay the provider's message as it
	// is, and the check that it may be relayed (it must be short enough,
	// and carry nothing but printable text) has to see it WHOLE: a message
	// trimmed to the bound would pass a length check that a dump fails.
	ProviderMessage string
}

// ExtractErrorSignals extracts the text signals a provider's error value
// carries: its code, type, message, status and reason. The second return
// value reports whether the value held any usable signal at all.
//
// raw may be a bare string ({"error":"boom"}), a bare number
// ({"error":400} -- several providers state the status that way), or an
// object. The decode goes through map[string]any rather than a struct with
// string fields on purpose: a provider may put a NUMBER in "code" --
// OpenRouter, for one, forwards the upstream provider's own numeric code
// verbatim ({"error":{"code":502,"message":"..."}}) -- and a struct decode
// then fails the WHOLE object with "cannot unmarshal number into Go struct
// field", so the message and type are lost along with it. With the map
// decode a non-string value is simply not a signal, and the message beside
// it survives: an error whose only readable content is its message must
// still be classified by that message.
//
// The object is descended into, not read one level deep. A provider nests
// its error one level down as readily as it puts it at the top --
// {"error":{"detail":{"code":"InvalidParameter","message":"..."}}} is a
// shape production carried -- and a signal set that stopped at the first
// level read NOTHING out of such a body: the classifier found no code, no
// message and no status, so the failure fell through to the generic 502
// and every rule that would have named it went unused. The descent is
// bounded (see maxSignalDepth) and the fields it reads are the same ones
// at every level.
//
// Shared between the proxy's own classification (statusCodeFromProviderBodyError
// / statusCodeFromProviderStreamError) and the LiteLLM compatibility layer's
// (errorStatus) -- the two used to re-implement this independently, and the
// compatibility layer's copy had no numeric-code fallback and a narrower
// keyword set, so the same provider body could classify differently
// depending on which path it took.
func ExtractErrorSignals(raw json.RawMessage) ([]string, bool) {
	info, ok := ExtractProviderErrorInfo(raw)
	if !ok {
		return nil, false
	}
	return info.Signals, true
}

// ExtractProviderErrorInfo extracts the signals, the codes and the parameter
// from a provider's error value.
//
// It is the single reader of a provider's error body: the proxy's own
// classification, the LiteLLM compatibility layer's and the classifier's
// own all read a body through it, so they cannot disagree about what the
// body says. They used to: one of them read a nested "detail" and the
// others did not, and one of them read a machine-readable reason out of
// "metadata" and the others did not -- so the same body was a content
// refusal on one path and an unclassified failure on another.
func ExtractProviderErrorInfo(raw json.RawMessage) (ProviderErrorInfo, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ProviderErrorInfo{}, false
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		text = strings.TrimSpace(text)
		if text == "" {
			return ProviderErrorInfo{}, false
		}
		return ProviderErrorInfo{Signals: []string{text}}, true
	}

	// The error value itself can be a bare number: {"error":400} is how
	// several providers state the status when they answer with an HTTP 200.
	var number json.Number
	if err := json.Unmarshal(trimmed, &number); err == nil {
		if signal := number.String(); signal != "" {
			return ProviderErrorInfo{Signals: []string{signal}}, true
		}
		return ProviderErrorInfo{}, false
	}

	var fields map[string]any
	if err := json.Unmarshal(trimmed, &fields); err != nil {
		return ProviderErrorInfo{}, false
	}

	info := ProviderErrorInfo{}
	collectProviderErrorInfo(fields, 0, &info)

	if len(info.Signals) == 0 {
		return ProviderErrorInfo{}, false
	}
	return info, true
}

// maxSignalDepth bounds how deep the descent into a provider's error object
// goes. It is not about the shape a provider uses -- it is about a body
// that nests arbitrarily deep (or is a list of lists) not turning the
// classification into a walk.
const maxSignalDepth = 5

// maxProviderErrorSignals bounds how many signals are collected from one
// body.
const maxProviderErrorSignals = 32

// collectProviderErrorInfo gathers the signals, the codes and the parameter
// of one level of a provider's error object, and of the objects nested in it.
//
// The fields read at each level are the same: the ones a provider states its
// failure in ("message", "type", "code", "status", "reason"), the one it
// states the machine-readable reason in ("error_type", and the same under
// "metadata"), the one it names the rejected parameter in ("param",
// "parameter", "field"), and the ones it nests its error in ("error",
// "response", "details", "detail", "violations").
func collectProviderErrorInfo(fields map[string]any, depth int, info *ProviderErrorInfo) {
	if depth > maxSignalDepth || len(info.Signals) >= maxProviderErrorSignals {
		return
	}

	for _, key := range []string{"message", "type", "code", "status", "reason", "error_type"} {
		raw, exists := mapValueFold(fields, key)
		if !exists {
			continue
		}
		text, ok := raw.(string)
		if !ok {
			continue
		}
		if key == "message" && info.ProviderMessage == "" {
			info.ProviderMessage = text
		}
		if s := cleanProviderErrorSignal(text); s != "" {
			info.AddSignal(s)
		}
		// The provider's own codes are what it wrote in a code-shaped
		// field. They are collected apart from the free text, so that a
		// rule naming a code is matched against the CODE and never
		// against a message that happens to read like one.
		if key == "code" || key == "type" {
			info.AddCode(text)
		}
	}

	// A numeric "code" is kept, but as a FALLBACK signal only: it is the
	// provider's own code for the failure, and ClassifyProviderErrorSignals
	// uses it only when nothing else says what went wrong.
	//
	// It cannot simply become the response status: an aggregator forwards
	// the upstream provider's code verbatim ({"error":{"code":502,...}}), and
	// that describes the UPSTREAM's failure, not the status this router
	// should report to its client -- an upstream 502 may be a content
	// refusal (a 400) or a provider outage (a 502), and only the message
	// beside it tells the two apart.
	//
	// When it IS all there is -- the provider states a bare status and
	// nothing else ("code":400, no message) -- it is the only thing left to
	// report, and the provider's own 4xx is the right answer for the client.
	//
	// code != 0 excludes the one numeric value that is never a real signal:
	// {"code":0} (no code set, or a placeholder) must not count as "this
	// object has a signal" any more than an absent "code" field would.
	if code, ok := fields["code"].(float64); ok && code != 0 {
		info.AddSignal(strconv.Itoa(int(code)))
	}

	// "metadata": {"error_type": "..."} -- the machine-readable reason an
	// aggregator reports alongside the message. It is read at every level,
	// and it is read as a SIGNAL: a refusal that names itself only there
	// ("error_type": "content_filter") is a refusal, and the paths that
	// did not read it could not tell.
	if meta, ok := fields["metadata"].(map[string]any); ok {
		for _, key := range []string{"error_type", "type", "reason"} {
			if s, ok := meta[key].(string); ok {
				if s = strings.TrimSpace(s); s != "" {
					info.AddSignal(s)
				}
			}
		}
	}

	if info.Param == nil {
		for _, key := range []string{"param", "parameter", "field"} {
			raw, exists := mapValueFold(fields, key)
			if !exists {
				continue
			}
			text, ok := raw.(string)
			if !ok {
				continue
			}
			if p := cleanProviderErrorParam(text); p != "" {
				info.Param = &p
				break
			}
		}
	}

	for _, key := range []string{"error", "response", "details", "detail", "violations"} {
		raw, exists := mapValueFold(fields, key)
		if !exists {
			continue
		}
		collectProviderErrorInfoValue(raw, depth+1, info)
	}
}

// collectProviderErrorInfoValue descends into a nested value: another
// object, a list of them, or a bare string.
func collectProviderErrorInfoValue(value any, depth int, info *ProviderErrorInfo) {
	switch typed := value.(type) {
	case map[string]any:
		collectProviderErrorInfo(typed, depth, info)
	case []any:
		for _, item := range typed {
			collectProviderErrorInfoValue(item, depth, info)
		}
	case string:
		if s := cleanProviderErrorSignal(typed); s != "" {
			info.AddSignal(s)
		}
	}
}

// AddSignal appends a signal, keeping the list within its bound.
func (i *ProviderErrorInfo) AddSignal(signal string) {
	if len(i.Signals) >= maxProviderErrorSignals {
		return
	}
	i.Signals = append(i.Signals, signal)
}

// AddCode appends a provider code, keeping the list within its bound.
//
// The code is trimmed and compared case-insensitively by the rules, but it
// is kept as the provider wrote it: the same code may be spelled
// "InvalidParameter" by one provider and "invalidparameter" by another, and
// a rule that names one of them should match both.
func (i *ProviderErrorInfo) AddCode(code string) {
	trimmed := strings.TrimSpace(code)
	if trimmed == "" {
		return
	}
	for _, existing := range i.Codes {
		if existing == trimmed {
			return
		}
	}
	if len(i.Codes) >= maxProviderErrorSignals {
		return
	}
	i.Codes = append(i.Codes, trimmed)
}

// Merge folds another info into this one: its signals and codes are added,
// and its parameter is kept when this one has none.
func (i *ProviderErrorInfo) Merge(other ProviderErrorInfo) {
	for _, signal := range other.Signals {
		i.AddSignal(signal)
	}
	for _, code := range other.Codes {
		i.AddCode(code)
	}
	if i.Param == nil {
		i.Param = other.Param
	}
}

// ClassifyProviderErrorSignals maps a set of text signals (from
// ExtractErrorSignals, or assembled by a caller that already has its own
// envelope-decoded fields) onto the HTTP status the client should see.
//
// Shared between the proxy's own classification and the LiteLLM
// compatibility layer's, so the two cannot classify the same signals
// differently -- they used to: the compatibility layer's own copy knew only
// content-policy and a numeric code, with no rate_limit/timeout/overloaded
// keyword handling and a 422 default instead of a 502.
//
// Each keyword check runs on the signal with "_" folded to " "
// (normalizeKeywordSignal), so "rate_limit_exceeded" and the prose an
// aggregator relays verbatim ("Request timed out", "rate limit exceeded")
// match the same check instead of only the snake_case form.
//
// A caller that has the provider's own CODES -- the code field beside the
// message -- must use ClassifyProviderErrorInfo instead: a rule that names
// a code is matched against that code, and this form has no codes to match
// it against.
func ClassifyProviderErrorSignals(signals []string) int {
	return ClassifyProviderErrorInfo(ProviderErrorInfo{Signals: signals})
}

// ClassifyProviderErrorInfo maps a provider's error value onto the HTTP
// status the client should see.
//
// It is the one classification of a provider's failure: the status, the
// router's code and the router's message all come from ONE lookup of the
// rules, so they cannot disagree about which rule answered.
func ClassifyProviderErrorInfo(info ProviderErrorInfo) int {
	// A provider message the router has a rule for is answered with the
	// status the rule asks for: the router knows what the failure is worth
	// (a request's own fault, a 400), and the provider's wording alone would
	// not say so.
	//
	// It is checked FIRST, before the content-policy pass below: a rule
	// names one failure in one wording, and the operator wrote it for that
	// failure. A rule whose wording happens to contain a content-policy
	// marker ("content moderation") would otherwise be unreachable -- the
	// marker would claim every body carrying that wording.
	//
	// It is also checked before the provider's own status prefix below,
	// because the rule is the more precise of the two: it names the
	// failure, while the prefix only says what the provider considers the
	// failure to be.
	if status, ok := providerMessageStatus(info); ok {
		return status
	}

	// A content-policy refusal is checked in its own pass, before the
	// provider-side signals below, because an aggregator that wraps a refusal
	// reports it with the upstream's own code and error type (OpenRouter
	// forwards the upstream provider's numeric code verbatim) -- the refusal is
	// still a fault of the request, and classifying it as 503/502 would both
	// mislabel the cause and make the request look worth retrying on another
	// credential, which every credential refuses identically.
	//
	// The pass is deliberately narrow: it wins only over the PROVIDER-side
	// signals (overload, unavailability, internal error), never over the
	// request-side ones. A rate limit or a timeout is a property of the
	// account's state at that moment, not of the content -- reporting those
	// as 400 would tell the client "your prompt was refused" when the truth
	// is "slow down", and would stop a retry that would have succeeded.
	if HasContentPolicySignal(info.Signals) && !hasRequestSideSignal(info.Signals) {
		return http.StatusBadRequest
	}

	// A provider that states the failure's status in its own message is
	// taken at its word: Alibaba's DashScope writes "<400>
	// InternalError.Algo.X: <explanation>", and the status in that prefix
	// is the one the client should see. Without it the request was
	// answered 502 or 500 -- "Request failed", which reads like the
	// provider being down and invites a retry on another credential when
	// every credential would refuse it identically.
	//
	// The prefix is looked for before the keyword pass below, because a
	// provider's own status is more precise than anything its wording
	// suggests: the same message may contain "internal" (which the keyword
	// pass reads as a 500) while stating "<400>".
	if status, ok := providerPrefixedStatus(info.Signals); ok {
		return status
	}

	for _, signal := range info.Signals {
		signal = normalizeKeywordSignal(signal)
		switch {
		case signal == "":
			continue
		case strings.Contains(signal, "rate limit") || strings.Contains(signal, "too many requests"):
			return http.StatusTooManyRequests
		case strings.Contains(signal, "timeout") || strings.Contains(signal, "timed out"):
			return http.StatusRequestTimeout
		case strings.Contains(signal, "overloaded") || strings.Contains(signal, "service unavailable"):
			return http.StatusServiceUnavailable
		case strings.Contains(signal, "server error") || strings.Contains(signal, "internal"):
			return http.StatusInternalServerError
		}
	}

	// Nothing in the signals says what went wrong, but a provider stated a
	// status as a bare number ("code":400) -- that is the only thing it said,
	// and it is the provider's own status for the request, so it is what the
	// client should see. Only a real HTTP status is taken: an arbitrary
	// number ("code":42) says nothing about the failure and stays a 502.
	if status, ok := httpStatusFromNumericSignal(info.Signals); ok {
		return status
	}
	return http.StatusBadGateway
}

// httpStatusFromNumericSignal returns the HTTP status a signal list states as
// a bare number, if any. It looks at every signal, because the number can
// sit in any of them ("code":400, "status":429), and reports the first one
// that is a status the router can relay (400-599).
func httpStatusFromNumericSignal(signals []string) (int, bool) {
	for _, signal := range signals {
		signal = strings.TrimSpace(signal)
		if signal == "" || len(signal) > 3 {
			continue
		}
		status, err := strconv.Atoi(signal)
		if err != nil {
			continue
		}
		if status >= http.StatusBadRequest && status <= 599 {
			return status, true
		}
	}
	return 0, false
}

// normalizeKeywordSignal lowercases and trims a signal, and folds "_" to " "
// so "rate_limit_exceeded" and "Rate limit exceeded" match the same keyword
// check. Providers phrase the same failure both ways: an OpenAI-shaped error
// type is snake_case ("rate_limit_exceeded"), but the free-text message an
// aggregator relays verbatim is prose ("Request timed out", "Service
// unavailable") -- without folding, only the snake_case form matched, so a
// provider that wrote "Request timed out" classified as a generic 502
// instead of a 408, and "rate limit exceeded" (prose) lost to a
// content-policy signal sitting beside it that a snake_case
// "rate_limit_exceeded" would have outranked (see hasRequestSideSignal).
func normalizeKeywordSignal(signal string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(signal)), "_", " ")
}

// hasRequestSideSignal reports whether any signal describes a request-side
// condition -- a rate limit or a timeout -- which is a property of the
// account's state rather than of the content.
func hasRequestSideSignal(signals []string) bool {
	for _, signal := range signals {
		signal = normalizeKeywordSignal(signal)
		if strings.Contains(signal, "rate limit") || strings.Contains(signal, "too many requests") ||
			strings.Contains(signal, "timeout") || strings.Contains(signal, "timed out") {
			return true
		}
	}
	return false
}
