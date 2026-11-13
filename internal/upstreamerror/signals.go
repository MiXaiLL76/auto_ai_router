package upstreamerror

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

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
// The signals are collected from the object and its "metadata" sub-object:
// aggregators put the machine-readable reason ("provider_unavailable",
// "content_filter") there, next to the human-readable message.
//
// Shared between the proxy's own classification (statusCodeFromProviderBodyError
// / statusCodeFromProviderStreamError) and the LiteLLM compatibility layer's
// (errorStatus) -- the two used to re-implement this independently, and the
// compatibility layer's copy had no numeric-code fallback and a narrower
// keyword set, so the same provider body could classify differently
// depending on which path it took.
func ExtractErrorSignals(raw json.RawMessage) ([]string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, false
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		text = strings.TrimSpace(text)
		if text == "" {
			return nil, false
		}
		return []string{text}, true
	}

	// The error value itself can be a bare number: {"error":400} is how
	// several providers state the status when they answer with an HTTP 200.
	var number json.Number
	if err := json.Unmarshal(trimmed, &number); err == nil {
		if signal := number.String(); signal != "" {
			return []string{signal}, true
		}
		return nil, false
	}

	var fields map[string]any
	if err := json.Unmarshal(trimmed, &fields); err != nil {
		return nil, false
	}

	signals := make([]string, 0, 8)
	for _, key := range []string{"code", "type", "message", "status", "reason"} {
		if s, ok := fields[key].(string); ok {
			if s = strings.TrimSpace(s); s != "" {
				signals = append(signals, s)
			}
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
		signals = append(signals, strconv.Itoa(int(code)))
	}
	// "metadata": {"error_type": "..."} -- the machine-readable reason an
	// aggregator reports alongside the message.
	if meta, ok := fields["metadata"].(map[string]any); ok {
		for _, key := range []string{"error_type", "type", "reason"} {
			if s, ok := meta[key].(string); ok {
				if s = strings.TrimSpace(s); s != "" {
					signals = append(signals, s)
				}
			}
		}
	}

	if len(signals) == 0 {
		return nil, false
	}
	return signals, true
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
func ClassifyProviderErrorSignals(signals []string) int {
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
	if HasContentPolicySignal(signals) && !hasRequestSideSignal(signals) {
		return http.StatusBadRequest
	}

	for _, signal := range signals {
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
	if status, ok := httpStatusFromNumericSignal(signals); ok {
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
