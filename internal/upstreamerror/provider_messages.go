package upstreamerror

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// A provider that packs the status of the failure, its own error class and
// code, and the explanation meant for the client into ONE message.
//
// Alibaba's DashScope writes its errors this way:
//
//	{"error":{"code":"InvalidParameter",
//	          "message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}
//
// Three things are packed into that one message: the status the provider
// considers the failure to be (<400>), the provider's own error class and code
// (InternalError.Algo.InvalidParameter), and the explanation the client is
// meant to read (the text after the colon).
//
// Two things were wrong with how the router read it.
//
// The STATUS was lost. The classifier knew neither "<400>" nor a code it
// recognized, so the request was answered 500 or 502 -- "Request failed",
// which reads like the provider being down and invites a retry on another
// credential when every credential refuses it identically. The status in the
// prefix is the provider's own statement of what the failure is, and it is
// taken at its word (see ClassifyProviderErrorSignals).
//
// The EXPLANATION was lost too, even though it is the one part of the message
// that was written FOR the client: the request asked for something the model
// cannot do, and the provider said exactly what that was.
//
// The explanation is nevertheless the provider's free text, and this package
// does not hand a provider's words over as they are -- it hands out a small
// fixed set of messages instead. That rule is kept here rather than waived,
// because the providers that write an explanation this way also write
// internal identifiers into it: production carries "The parameter `size`
// specified in the request are not valid: ... Request id:
// 021795058711799a81c77d48bc6ef475723297888da3", and a request id is the
// upstream's own bookkeeping, not the client's.
//
// So the explanations that may be relayed are listed in
// providerRequestMessages -- each one read, checked and vetted by hand -- and
// the relay is an EXACT match against that list. A provider explanation that
// is not on the list is not relayed at all: the router's own generic wording
// stands, and the status the provider stated still reaches the client. A new
// provider message is therefore added by adding one line to the list, and
// nothing else in the router has to change.

// providerStatusPrefixRe matches a status a provider states at the start of
// its own message: "<400> ...".
//
// It is deliberately looser than providerPrefixedMessage below, which
// additionally requires the provider's own code and the explanation after a
// colon. The STATUS is worth taking even when the rest of the shape is not
// there: a provider that writes "<400> Request rejected" states the same
// thing as one that writes "<400> InternalError.Algo.InvalidParameter:
// ...". Requiring the full shape would throw the diagnosis away for a
// wording the provider changed.
var providerStatusPrefixRe = regexp.MustCompile(`^<(\d{3})>`)

// providerPrefixedMessage matches the full shape: the status the provider
// states, its error class and code, and the explanation after the colon.
//
// The code is a plain dotted path ("InternalError.Algo.InvalidParameter"), so
// a message that merely happens to contain a colon -- prose, a URL, a
// timestamp -- is not mistaken for one. The explanation is a single line: a
// message that carries a newline is not a prefixed message either.
var providerPrefixedMessage = regexp.MustCompile(`^<(\d{3})>\s*[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*\s*:\s*(.+)$`)

// providerRelayedStatuses are the statuses a provider's prefix may state.
//
// A status outside this list is not taken as one: the provider's own
// numbering may differ from HTTP's, and the router relays only the statuses
// it has a client-facing answer for.
var providerRelayedStatuses = map[int]struct{}{
	http.StatusBadRequest:            {},
	http.StatusUnauthorized:          {},
	http.StatusPaymentRequired:       {},
	http.StatusForbidden:             {},
	http.StatusNotFound:              {},
	http.StatusMethodNotAllowed:      {},
	http.StatusRequestTimeout:        {},
	http.StatusConflict:              {},
	http.StatusRequestEntityTooLarge: {},
	http.StatusUnprocessableEntity:   {},
	http.StatusTooManyRequests:       {},
	http.StatusInternalServerError:   {},
	http.StatusBadGateway:            {},
	http.StatusServiceUnavailable:    {},
	http.StatusGatewayTimeout:        {},
}

// providerPrefixedError is what a provider's own prefixed message says.
type providerPrefixedError struct {
	// Status is the status the provider states the failure carries (400).
	Status int
	// Explanation is the text after the colon, as the provider wrote it.
	Explanation string
}

// parseProviderPrefixedMessage reads a provider's prefixed message: the
// status it states and the explanation after the colon.
//
// It returns nil for a message that is not in that shape, for one whose
// status the router does not relay, and for one with no explanation.
func parseProviderPrefixedMessage(message string) *providerPrefixedError {
	message = strings.TrimSpace(message)
	if message == "" {
		return nil
	}

	match := providerPrefixedMessage.FindStringSubmatch(message)
	if match == nil {
		return nil
	}

	status, ok := relayedProviderStatus(match[1])
	if !ok {
		return nil
	}

	explanation := strings.TrimSpace(match[2])
	if explanation == "" {
		return nil
	}

	return &providerPrefixedError{Status: status, Explanation: explanation}
}

// relayedProviderStatus reads a status a provider states and reports whether
// the router relays it.
//
// A status outside the relayed set is not taken as one: the provider's own
// numbering may differ from HTTP's.
func relayedProviderStatus(statusText string) (int, bool) {
	status, err := strconv.Atoi(statusText)
	if err != nil {
		return 0, false
	}
	if _, ok := providerRelayedStatuses[status]; !ok {
		return 0, false
	}
	return status, true
}

// providerPrefixedStatus returns the status a provider's own message states,
// and whether it states one.
//
// The signals are the provider's own words about the failure (its code, its
// message, ...), so they are looked at one by one: a body may carry the
// prefix in its message and something else in its code.
func providerPrefixedStatus(signals []string) (int, bool) {
	for _, signal := range signals {
		signal = strings.TrimSpace(signal)
		match := providerStatusPrefixRe.FindStringSubmatch(signal)
		if match == nil {
			continue
		}
		if status, ok := relayedProviderStatus(match[1]); ok {
			return status, true
		}
	}
	return 0, false
}

// providerRequestMessage is the router's own answer to one provider
// explanation.
type providerRequestMessage struct {
	// Code is the router's code for the failure.
	Code string
	// Message is what the client is told.
	Message string
}

// providerRequestMessages lists the provider explanations that may be
// relayed to a client, keyed by the explanation normalized for comparison.
//
// Each entry was read by hand and checked to carry nothing but the
// diagnosis: no request id, no account name, no URL, no host, no path. An
// explanation that is not on this list is not relayed -- the router's own
// generic wording stands -- so a provider that starts writing an internal
// identifier into its explanations cannot leak it into a client's response
// just by being new.
//
// The key is the explanation lowercased with runs of whitespace collapsed to
// one space, so a provider that changes only the spacing or the
// capitalization still matches.
//
// The entries below are the ones production produced, in the shape
// "<400> InternalError.Algo.InvalidParameter: <explanation>".
//
// A limit that belongs to a model (the thinking budget below) is written out
// as it was observed. Should the provider change the limit, the entry stops
// matching and the client gets the generic wording again -- the status
// stays the 400 the provider stated, so the request is still reported as
// the provider's own refusal of it.
var providerRequestMessages = map[string]providerRequestMessage{
	"agent capabilities are not enabled for the current model.": {
		Code:    "invalid_request",
		Message: "Agent capabilities are not enabled for the current model.",
	},
	"the current model does not support pdf file input.": {
		Code:    "invalid_request",
		Message: "The current model does not support PDF file input.",
	},
	"the thinking_budget parameter must be a positive integer and not greater than 131072": {
		Code:    "invalid_request",
		Message: "The thinking_budget parameter must be a positive integer and not greater than 131072",
	},
}

// normalizeProviderExplanation reduces a provider's explanation to the form
// providerRequestMessages is keyed by: lowercased, with runs of whitespace
// collapsed to a single space.
func normalizeProviderExplanation(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

// lookupProviderRequestMessage returns the router's answer to a provider's
// explanation, and whether the explanation is one that may be relayed.
func lookupProviderRequestMessage(explanation string) (providerRequestMessage, bool) {
	entry, ok := providerRequestMessages[normalizeProviderExplanation(explanation)]
	return entry, ok
}

// classifyProviderPrefixedMessage classifies a provider's prefixed message
// into the router's own answer, and reports whether it could.
//
// Only a request fault is answered here, and only with an explanation that is
// on the relay list: another status has its own message (a rate limit, a
// timeout), and those are the router's to state, not the provider's.
//
// A refusal is not relayed either. The content-policy branch already names a
// refusal, and its name is what the retry policy and the client's own
// handling key off -- a second name for the same refusal would split them.
func classifyProviderPrefixedMessage(signals []string) (BadRequest, bool) {
	for _, signal := range signals {
		parsed := parseProviderPrefixedMessage(signal)
		if parsed == nil {
			continue
		}
		if parsed.Status != http.StatusBadRequest {
			return BadRequest{}, false
		}
		if IsContentPolicySignal(parsed.Explanation) {
			return BadRequest{}, false
		}
		entry, ok := lookupProviderRequestMessage(parsed.Explanation)
		if !ok {
			return BadRequest{}, false
		}
		return BadRequest{Message: entry.Message, Code: entry.Code}, true
	}
	return BadRequest{}, false
}

// isRouterProviderMessage reports whether message is the router's own answer
// to a provider explanation.
//
// It is the mirror of lookupProviderRequestMessage: the router writes the
// explanation as its own message, so a body carrying it is the router's own
// body -- and has to be recognized as one. Without that, a second
// classification pass (the LiteLLM compatibility mode re-classifies the body
// the router wrote, an AIR-to-AIR proxy chain hands one router the
// classification another already made) would replace the explanation with
// the generic "Invalid request".
func isRouterProviderMessage(code, message string) bool {
	for _, entry := range providerRequestMessages {
		if entry.Code == code && entry.Message == message {
			return true
		}
	}
	return false
}
