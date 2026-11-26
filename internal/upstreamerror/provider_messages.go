package upstreamerror

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// A provider may pack the status of the failure, its own error class and
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
// So the explanations that may be relayed are the ones a rule names, and a
// rule is a MATCH against the provider's words, never a pass-through: a
// provider explanation that no rule names is not relayed at all, and the
// router's own generic wording stands. The status the provider stated still
// reaches the client either way.
//
// The rules come from two places: the ones the router ships with
// (builtinProviderMessages) and the ones a deployment configures (the
// router's provider_messages config section, see SetProviderMessageRules).
// Adding a provider message is therefore a configuration change, not a change
// to the router.

// providerStatusPrefixRe matches a status a provider states at the start of
// its own message: "<400> ...".
//
// It is deliberately loose. The STATUS is worth taking even when the rest of
// the provider's shape is not there: a provider that writes "<400> Request
// rejected" states the same thing as one that writes "<400>
// InternalError.Algo.InvalidParameter: ...". Requiring the full shape would
// throw the diagnosis away for a wording the provider changed.
var providerStatusPrefixRe = regexp.MustCompile(`^<(\d{3})>`)

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

// ProviderMessageRule is one provider message the router may relay to a
// client.
//
// A rule names the provider's own words and the router's own answer to them.
// The provider's words are matched; the router's answer is what reaches the
// client.
type ProviderMessageRule struct {
	// ProviderCode is the provider's own error code ("code" in the error
	// object). When set, the failure must carry exactly this code.
	//
	// It is what makes one rule cover a whole family of failures: every
	// "InvalidParameter" from any provider, whatever the model and the
	// request, is the same kind of refusal.
	ProviderCode string
	// Match is text the provider's message must contain. When set, the
	// failure's message must contain it.
	//
	// It is what makes one rule cover the failures that share a wording
	// whatever the provider calls them ("Unsupported model: 'kimi-k2.7-code'"
	// and "Unsupported model: 'other-model'" are one failure).
	Match string
	// Code is the router's error code for the failure (the "code" field of
	// the client's error object).
	Code string
	// Message is what the client is told. It is the router's own wording,
	// not the provider's: this is the text that goes on the wire.
	Message string
	// Status is the HTTP status the client is answered with. Zero means the
	// router decides (the status the provider stated, or its own default).
	Status int
}

// builtinProviderMessages are the provider messages the router relays out of
// the box -- the ones production produced, in the shape "<400>
// InternalError.Algo.InvalidParameter: <explanation>".
//
// They match on the explanation only, and not on the provider's code: the
// same refusal is worth reporting whether the provider calls it
// "InvalidParameter" or something else.
//
// A limit that belongs to a model (the thinking budget below) is written out
// as it was observed. Should the provider change the limit, the rule stops
// matching and the client gets the generic wording again -- the status stays
// the 400 the provider stated, so the request is still reported as the
// provider's own refusal of it.
var builtinProviderMessages = []ProviderMessageRule{
	{
		Match:   "agent capabilities are not enabled for the current model.",
		Code:    "invalid_request",
		Message: "Agent capabilities are not enabled for the current model.",
		Status:  http.StatusBadRequest,
	},
	{
		Match:   "the current model does not support pdf file input.",
		Code:    "invalid_request",
		Message: "The current model does not support PDF file input.",
		Status:  http.StatusBadRequest,
	},
	{
		Match:   "the thinking_budget parameter must be a positive integer and not greater than 131072",
		Code:    "invalid_request",
		Message: "The thinking_budget parameter must be a positive integer and not greater than 131072",
		Status:  http.StatusBadRequest,
	},
}

// providerMessageRules holds the rules in force: the built-in ones and the
// deployment's own.
//
// The rules are read at startup and never change while the router runs, so
// the request path only takes a read lock; the lock is here for the write
// (SetProviderMessageRules), which happens once.
var (
	providerMessageRulesMu sync.RWMutex
	providerMessageRules   = builtinProviderMessages
)

// SetProviderMessageRules adds the deployment's own rules to the built-in
// ones.
//
// It is called once, at startup, with the router's provider_messages config
// section: from then on the rules are read on the request path, and a config
// change needs a restart (the deployment rolls the pods when the config
// changes).
//
// A configured rule that names the same text as a built-in one REPLACES it,
// in place: the deployment may reword what the router says about a failure
// the router already knows. A rule that names something else is added after
// the built-in ones, so it cannot shadow one of them by being broader.
func SetProviderMessageRules(rules []ProviderMessageRule) {
	merged := make([]ProviderMessageRule, 0, len(builtinProviderMessages)+len(rules))
	merged = append(merged, builtinProviderMessages...)

	for _, rule := range rules {
		replaced := false
		for i := range merged {
			if merged[i].Match == "" || rule.Match == "" {
				continue
			}
			if normalizeProviderMessageText(merged[i].Match) == normalizeProviderMessageText(rule.Match) {
				merged[i] = rule
				replaced = true
				break
			}
		}
		if !replaced {
			merged = append(merged, rule)
		}
	}

	providerMessageRulesMu.Lock()
	providerMessageRules = merged
	providerMessageRulesMu.Unlock()
}

// activeProviderMessageRules returns the rules in force.
func activeProviderMessageRules() []ProviderMessageRule {
	providerMessageRulesMu.RLock()
	defer providerMessageRulesMu.RUnlock()
	return providerMessageRules
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

// normalizeProviderMessageText reduces a text to the form the rules are
// matched against: lowercased, with runs of whitespace collapsed to a single
// space.
//
// The comparison ignores case and spacing on purpose: the provider is free
// to capitalize its message differently or to break a line in it, and a rule
// that stopped matching over that would be a rule the operator has to
// rewrite every time the provider edits a sentence.
func normalizeProviderMessageText(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

// providerMessageMatches reports whether a rule matches a provider failure.
//
// The provider's own code and the provider's own message are both compared.
// A rule that names a code must see that code; a rule that names text must
// see that text in the message. A rule that names both must see both.
//
// The code is compared as one of the signals (the provider states it in a
// field of its own, and the router does not know which field that was), and
// the text is compared as a substring of a signal -- so a provider that
// appends its own request id to the message still matches, and the request
// id is not relayed with it.
func providerMessageMatches(rule ProviderMessageRule, signals []string) bool {
	if rule.ProviderCode == "" && rule.Match == "" {
		return false
	}
	if rule.ProviderCode != "" {
		found := false
		for _, signal := range signals {
			if strings.EqualFold(strings.TrimSpace(signal), rule.ProviderCode) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if rule.Match != "" {
		needle := normalizeProviderMessageText(rule.Match)
		found := false
		for _, signal := range signals {
			if strings.Contains(normalizeProviderMessageText(signal), needle) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// lookupProviderMessageRule returns the rule that matches a provider
// failure, and whether one matched.
//
// The rules are tried in order, and the first match wins.
func lookupProviderMessageRule(signals []string) (ProviderMessageRule, bool) {
	for _, rule := range activeProviderMessageRules() {
		if providerMessageMatches(rule, signals) {
			return rule, true
		}
	}
	return ProviderMessageRule{}, false
}

// classifyProviderMessage classifies a provider's failure into the router's
// own answer, and reports whether it could.
//
// A refusal is not answered here. The content-policy branch already names a
// refusal, and its name is what the retry policy and the client's own
// handling key off -- a second name for the same refusal would split them.
func classifyProviderMessage(signals []string) (BadRequest, bool) {
	if HasContentPolicySignal(signals) {
		return BadRequest{}, false
	}
	rule, ok := lookupProviderMessageRule(signals)
	if !ok {
		return BadRequest{}, false
	}
	return BadRequest{Message: rule.Message, Code: rule.Code}, true
}

// providerMessageStatus returns the status a matched rule asks for, and
// whether a rule matched at all.
//
// A rule without a status leaves the decision to the router: the status the
// provider stated, or the router's own default.
func providerMessageStatus(signals []string) (int, bool) {
	if HasContentPolicySignal(signals) {
		return 0, false
	}
	rule, ok := lookupProviderMessageRule(signals)
	if !ok || rule.Status == 0 {
		return 0, false
	}
	return rule.Status, true
}

// isRouterProviderMessage reports whether message is the router's own answer
// to a provider message.
//
// It is the mirror of the rule lookup: the router writes the answer as its
// own message, so a body carrying it is the router's own body -- and has to
// be recognized as one. Without that, a second classification pass (the
// LiteLLM compatibility mode re-classifies the body the router wrote, an
// AIR-to-AIR proxy chain hands one router the classification another already
// made) would replace the answer with the generic "Invalid request".
func isRouterProviderMessage(code, message string) bool {
	for _, rule := range activeProviderMessageRules() {
		if rule.Code == code && rule.Message == message {
			return true
		}
	}
	return false
}
