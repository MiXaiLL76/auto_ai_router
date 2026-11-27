package upstreamerror

import (
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
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
// taken at its word (see ClassifyProviderErrorInfo).
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

// ReservedProviderMessageCodes are the router's own codes that a
// provider_messages rule may not reuse.
//
// These codes carry a meaning of their own -- "the account is blocked", "the
// session is blocked by policy", "the upstream could not fetch the content" --
// and the router's own bodies are recognized by them (see
// extractRouterClassification). A rule that reused one would produce a body
// the router cannot recognize as its own: the recognition expects a
// particular "type" beside the code, and the rule's path never produces
// that one. The client's answer would then be lost on the second
// classification pass -- the LiteLLM compatibility mode re-classifying the
// body the router wrote, an AIR-to-AIR chain handing one router what
// another already classified -- and replaced with the generic "Invalid
// request".
//
// A rule with one of these codes is refused at startup, with the name of the
// code in the error: a config that would lose the message on the second pass
// must not load.
var ReservedProviderMessageCodes = []string{
	accountBlockedCode,
	sessionBlockedCode,
	upstreamFetchFailedCode,
}

// IsReservedProviderMessageCode reports whether code is one of the router's
// own codes that a provider_messages rule may not reuse.
func IsReservedProviderMessageCode(code string) bool {
	for _, reserved := range ReservedProviderMessageCodes {
		if code == reserved {
			return true
		}
	}
	return false
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
	//
	// The {param} placeholder is replaced with the parameter the provider
	// named, so a rule can say WHICH parameter the request carried. The
	// replacement is the sanitized parameter name and nothing else (see
	// cleanProviderErrorParam): a provider that put free text in its
	// "param" field gets nothing substituted.
	Message string
	// Status is the HTTP status the client is answered with.
	Status int
	// Retryable says whether a failure that matched this rule may be
	// replayed on another credential.
	//
	// It is nil when the rule does not say: the router's own policy decides
	// then, and the status alone decides -- a 400 is in the router's
	// retryable set (a provider account may refuse a model another account
	// serves), and a 5xx is in it too.
	//
	// False is the answer for a rule about the REQUEST's own fault: the
	// model cannot do what was asked of it, every credential refuses it
	// identically, and replaying it only multiplies the attempts. Production
	// measured it: 441 requests, 1 236 attempts across the pool, not one
	// attempt succeeded.
	//
	// True is the answer for a rule about something a CREDENTIAL owns: the
	// model is missing from this account but another account serves it.
	Retryable *bool
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
//
// All three are about the REQUEST's own fault -- the model cannot do what
// was asked of it -- so they are not retryable: every credential refuses
// them identically, and replaying them only multiplies the attempts.
func builtinProviderMessages() []ProviderMessageRule {
	notRetryable := false
	return []ProviderMessageRule{
		{
			Match:     "agent capabilities are not enabled for the current model.",
			Code:      "invalid_request",
			Message:   "Agent capabilities are not enabled for the current model.",
			Status:    http.StatusBadRequest,
			Retryable: &notRetryable,
		},
		{
			Match:     "the current model does not support pdf file input.",
			Code:      "invalid_request",
			Message:   "The current model does not support PDF file input.",
			Status:    http.StatusBadRequest,
			Retryable: &notRetryable,
		},
		{
			Match:     "the thinking_budget parameter must be a positive integer and not greater than 131072",
			Code:      "invalid_request",
			Message:   "The thinking_budget parameter must be a positive integer and not greater than 131072",
			Status:    http.StatusBadRequest,
			Retryable: &notRetryable,
		},
	}
}

// compiledProviderMessageRule is a rule with its match text normalized once,
// at startup, so that the request path does not normalize the same static
// text for every error.
type compiledProviderMessageRule struct {
	rule ProviderMessageRule
	// normalizedMatch is the rule's Match folded for comparison, computed
	// once when the rules are set.
	normalizedMatch string
	// position is the rule's place in the set, which is the order the rules
	// are tried in. The lookup keeps the order while still skipping the
	// rules a failure cannot match (see lookupProviderMessageRule).
	position int
}

// providerMessageRuleSet is the rules in force, indexed for lookup.
//
// The index is what keeps the lookup cheap: a rule that names a provider
// code can only match a failure carrying that code, so the rules are
// bucketed by code and a lookup only walks the bucket of the codes the
// failure actually carries, plus the rules that name no code at all. Without
// it, every classified error walked every rule -- with the configured
// ceiling of a thousand rules, that was the bulk of the classification's
// cost on the errors that match nothing.
type providerMessageRuleSet struct {
	// all is every rule, in the order they are tried.
	all []compiledProviderMessageRule
	// byCode holds the rules that name a provider code, keyed by the
	// lowercased code.
	byCode map[string][]compiledProviderMessageRule
	// withoutCode holds the rules that name no provider code.
	withoutCode []compiledProviderMessageRule
}

// providerMessageRules holds the rules in force: the built-in ones and the
// deployment's own.
//
// The rules are read at startup and never change while the router runs, so
// the request path only takes a read lock; the lock is here for the write
// (SetProviderMessageRules), which happens once.
var (
	providerMessageRulesMu sync.RWMutex
	providerMessageRules   = buildProviderMessageRuleSet(builtinProviderMessages())
)

// buildProviderMessageRuleSet compiles and indexes a rule list.
func buildProviderMessageRuleSet(rules []ProviderMessageRule) providerMessageRuleSet {
	set := providerMessageRuleSet{
		all:         make([]compiledProviderMessageRule, 0, len(rules)),
		byCode:      make(map[string][]compiledProviderMessageRule),
		withoutCode: make([]compiledProviderMessageRule, 0, len(rules)),
	}
	for position, rule := range rules {
		compiled := compiledProviderMessageRule{
			rule:            rule,
			normalizedMatch: normalizeProviderMessageText(rule.Match),
			position:        position,
		}
		set.all = append(set.all, compiled)
		if rule.ProviderCode == "" {
			set.withoutCode = append(set.withoutCode, compiled)
			continue
		}
		key := strings.ToLower(rule.ProviderCode)
		set.byCode[key] = append(set.byCode[key], compiled)
	}
	return set
}

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
//
// A rule that names no text matches every failure carrying its code, so
// another such rule with the same code REPLACES it rather than shadowing it:
// two of them are the same rule twice, and the second would never fire. The
// config validation refuses that outright, so this is the belt to that
// braces.
func SetProviderMessageRules(rules []ProviderMessageRule) {
	// The deployment's own rules come FIRST.
	//
	// The order is the order the rules are tried in, and a rule that comes
	// first wins. A deployment's rule has to be able to say something
	// DIFFERENT about a failure the router already knows -- reword it, or
	// (the case this exists for) say that it must not be replayed on another
	// credential. A rule behind the built-in one could never do either: the
	// built-in rule matched first, and the operator's wording and its
	// retryable answer were both lost -- the operator saw their rule in the
	// config and a different answer on the wire.
	//
	// A rule that names the same text as a built-in one still REPLACES it
	// rather than sitting beside it: the same text is the same rule, and
	// two rules for one failure would be one of them dead.
	merged := make([]ProviderMessageRule, 0, len(rules)+len(builtinProviderMessages()))
	merged = append(merged, rules...)

	for _, builtin := range builtinProviderMessages() {
		// A built-in rule whose text a deployment rule already names is NOT
		// added: the deployment's rule is the same rule, and the
		// deployment's wording (and its retryable answer) is the one that
		// must stand.
		alreadyNamed := false
		for i := range merged {
			if merged[i].Match == "" || builtin.Match == "" {
				continue
			}
			if normalizeProviderMessageText(merged[i].Match) == normalizeProviderMessageText(builtin.Match) {
				alreadyNamed = true
				break
			}
		}
		if !alreadyNamed {
			merged = append(merged, builtin)
		}
	}

	// A rule without a text matches every failure carrying its code:
	// another such rule with the same code is the same rule twice.
	for i := 0; i < len(merged); i++ {
		if merged[i].Match != "" || merged[i].ProviderCode == "" {
			continue
		}
		for j := i + 1; j < len(merged); j++ {
			if merged[j].Match == "" && merged[j].ProviderCode == merged[i].ProviderCode {
				merged = append(merged[:j], merged[j+1:]...)
				j--
			}
		}
	}

	providerMessageRulesMu.Lock()
	providerMessageRules = buildProviderMessageRuleSet(merged)
	providerMessageRulesMu.Unlock()
}

// activeProviderMessageRules returns the rules in force.
func activeProviderMessageRules() providerMessageRuleSet {
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
// space and the trailing punctuation dropped.
//
// The comparison ignores case and spacing on purpose: the provider is free
// to capitalize its message differently or to break a line in it, and a rule
// that stopped matching over that would be a rule the operator has to
// rewrite every time the provider edits a sentence.
//
// The trailing punctuation is dropped for the same reason, and it matters
// for the REPLACEMENT of a built-in rule: the built-in texts end in a period
// and an operator's usually do not, so an exact comparison left the
// operator's rule shadowed by the built-in one -- the built-in rule matched
// first and the reworded message never reached the client. The operator
// would see their rule in the config and a different message on the wire.
//
// Only the punctuation at the END is dropped: a period inside the text is
// part of it ("e.g. ...", a version number, a file name).
func normalizeProviderMessageText(text string) string {
	return strings.TrimRight(strings.ToLower(strings.Join(strings.Fields(text), " ")), ".!?;:, ")
}

// providerMessageMatch is what a rule matched, and where.
//
// The signals that carried the match are reported apart from the rest,
// because the content-policy guard must judge the failure by the signals
// that did NOT produce the match: a rule whose own text contains a
// content-policy marker ("content moderation") is a rule about a parameter,
// and the marker inside its text says nothing about the failure.
type providerMessageMatch struct {
	rule ProviderMessageRule
	// matchedByText are the indexes of the signals that carried the rule's
	// match text.
	matchedByText []int
	// providerMessage is the provider's own text that the rule matched on --
	// the one signal that carried it. It is what the {verbatim} placeholder
	// relays: the wording that says exactly what is wrong, not a paraphrase
	// of it.
	providerMessage string
}

// lookupProviderMessageRule returns the rule that matches a provider
// failure, and whether one matched.
//
// The rules are tried in order, and the first match wins.
//
// The rules that name a provider code are looked up by the codes the failure
// carries, so a failure that carries no known code never walks them.
func lookupProviderMessageRule(info ProviderErrorInfo) (providerMessageMatch, bool) {
	rules := activeProviderMessageRules()

	// The signals are normalized ONCE per lookup, not once per rule: the
	// text of a signal does not depend on the rule it is compared against,
	// and redoing it for every rule made an unmatched error cost one
	// normalization per rule per signal.
	normalized := make([]string, len(info.Signals))
	for i, signal := range info.Signals {
		normalized[i] = normalizeProviderMessageText(signal)
	}

	// The candidates are the rules a failure could match: the ones that name
	// no code, plus the ones that name a code the failure carries. They are
	// then put back in the set's order, which is the order the rules are
	// tried in -- a deployment's rule comes before a built-in one, so a
	// rule that matches the same failure in a shorter wording still wins.
	candidates := make([]compiledProviderMessageRule, 0, len(rules.withoutCode)+4)
	candidates = append(candidates, rules.withoutCode...)
	for _, code := range info.Codes {
		candidates = append(candidates, rules.byCode[strings.ToLower(code)]...)
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].position < candidates[j].position
	})

	for _, candidate := range candidates {
		matched, ok := providerMessageMatches(candidate, info, normalized)
		if !ok {
			continue
		}
		return providerMessageMatch{
			rule:            candidate.rule,
			matchedByText:   matched,
			providerMessage: matchedProviderMessage(info, matched),
		}, true
	}
	return providerMessageMatch{}, false
}

// matchedProviderMessage returns the provider's own text that carried the
// rule's match -- the signal the rule matched on.
//
// It is the FIRST such signal: a rule matches on one wording, and that
// wording is what the {verbatim} placeholder relays. The others are the
// provider's other words about the same failure (its type, its code), and
// relaying those would put more of the provider's body in front of the
// client than the rule asked for.
func matchedProviderMessage(info ProviderErrorInfo, matched []int) string {
	for _, index := range matched {
		if index >= 0 && index < len(info.Signals) {
			return info.Signals[index]
		}
	}
	return ""
}

// providerMessageMatches reports whether a rule matches a provider failure,
// and which signals carried the rule's match text.
//
// The provider's own code and the provider's own message are both compared.
// A rule that names a code must see that code among the failure's codes; a
// rule that names text must see that text in the message. A rule that names
// both must see both.
//
// The text is compared as a substring, so a provider that appends its own
// request id to the message still matches -- and the request id is not
// relayed with it.
func providerMessageMatches(rule compiledProviderMessageRule, info ProviderErrorInfo, normalized []string) ([]int, bool) {
	if rule.rule.ProviderCode == "" && rule.rule.Match == "" {
		return nil, false
	}
	if rule.rule.ProviderCode != "" {
		found := false
		for _, code := range info.Codes {
			if strings.EqualFold(code, rule.rule.ProviderCode) {
				found = true
				break
			}
		}
		if !found {
			return nil, false
		}
	}
	if rule.rule.Match == "" {
		return nil, true
	}

	matched := make([]int, 0, 2)
	for i, text := range normalized {
		if strings.Contains(text, rule.normalizedMatch) {
			matched = append(matched, i)
		}
	}
	return matched, len(matched) > 0
}

// ProviderMessageClassification is what the router knows about a provider's
// failure: the status the client is answered with, and the router's own
// message and code for it.
//
// It is ONE value because the two halves come from ONE rule: computing the
// status and the message in two independent lookups let them disagree -- the
// status was taken from a rule while the message was not (or the other way
// round), and the client was answered with a status it did not get a
// message for, or with a message that belonged to a different rule.
type ProviderMessageClassification struct {
	// Status is the HTTP status the client is answered with.
	Status int
	// Code is the router's error code for the failure.
	Code string
	// Message is what the client is told.
	Message string
	// Param is the parameter the provider rejected, when it named one. It is
	// the provider's own name, sanitized (see cleanProviderErrorParam), and
	// it is what the client's "param" field carries -- so a client can act on
	// WHICH field the request carried, not just that some field was wrong.
	Param *string
	// Retryable says whether the failure may be replayed on another
	// credential. Nil when no rule said.
	Retryable *bool
}

// matchProviderMessageRule finds the rule that answers a provider's failure,
// and reports whether one did.
//
// It is the ONE place a rule is matched and judged. The status, the code, the
// message and the retryable answer all come from the rule it returns, so
// they cannot disagree about which rule answered -- and the content-policy
// guard is applied HERE, once, rather than in each of the callers.
//
// The guard has to be here and not in the callers: a caller that looked the
// rule up for itself would skip it, and a content-policy refusal would then
// take the rule's retryable answer -- "yes, try another credential" -- and
// the request would be replayed across the whole credential pool on a
// prompt the provider refused on policy grounds. Every credential refuses
// that identically, which is exactly why a refusal is not retryable.
//
// The refusal is judged by the signals that did NOT produce the match. A
// rule whose own wording contains a content-policy marker ("content
// moderation") is a rule about a parameter, and the marker inside its
// wording says nothing about the failure -- without this, such a rule would
// be permanently unreachable, with nothing in the config to say so. A
// refusal named anywhere else in the body (the machine-readable reason in
// "metadata", the type) still wins: the operator's rule was written for one
// wording, and a body that names a refusal in a different field is a
// refusal.
func matchProviderMessageRule(info ProviderErrorInfo) (providerMessageMatch, bool) {
	match, ok := lookupProviderMessageRule(info)
	if !ok {
		return providerMessageMatch{}, false
	}
	if isContentPolicyRefusal(info, match.matchedByText) {
		return providerMessageMatch{}, false
	}
	return match, true
}

// ClassifyProviderMessage classifies a provider's failure into the router's
// own answer: the status, the code and the message, all from ONE rule.
//
// It reports false when no rule matched, or when the failure is a
// content-policy refusal.
//
// A refusal is not answered here. The content-policy branch already names a
// refusal, and its name is what the retry policy and the client's own
// handling key off -- a second name for the same refusal would split them.
func ClassifyProviderMessage(info ProviderErrorInfo) (ProviderMessageClassification, bool) {
	match, ok := matchProviderMessageRule(info)
	if !ok {
		return ProviderMessageClassification{}, false
	}

	if match.rule.Status == 0 {
		return ProviderMessageClassification{}, false
	}

	// The parameter reaches the client ONLY when the rule's message asks for
	// it by name.
	//
	// The provider's "param" field says what the provider was complaining
	// about, and the rule matched on its own wording -- not necessarily on
	// the provider's parameter. A rule that names no parameter did not ask
	// for it to be repeated, so it is not repeated: the client's "param"
	// field would otherwise carry a field the rule never mentioned.
	//
	// And when the rule DOES name it, the parameter is repeated only if it
	// is one of the router's own request fields (see
	// providerParamNameAccepted): a provider that put free text there must
	// not get it into the client's body.
	param := (*string)(nil)
	if _, names := providerRuleParam(match.rule); names {
		if info.Param != nil && providerParamNameAccepted(*info.Param) {
			param = info.Param
		}
	}

	return ProviderMessageClassification{
		Status:    match.rule.Status,
		Code:      match.rule.Code,
		Message:   renderProviderMessageWithVerbatim(match.rule.Message, param, match.providerMessage),
		Retryable: match.rule.Retryable,
		Param:     param,
	}, true
}

// ProviderParamPlaceholder is the placeholder a rule's message uses to name
// the parameter the provider rejected.
//
// The router replaces it with the sanitized parameter name, so a rule can
// tell the client WHICH parameter the request carried. It is exported so the
// config package can validate a message that carries it.
const ProviderParamPlaceholder = "{param}"

// ProviderVerbatimPlaceholder is the placeholder a rule's message uses to
// relay the provider's own message as it is.
//
// It is for the failures whose whole point is the provider's wording: "Invalid
// schema for response_format 'X': In context=(), 'additionalProperties' is
// required to be supplied and to be false." says exactly what is wrong with
// the schema, and no paraphrase of it does.
//
// It is exported so the config package can validate a message that carries it.
const ProviderVerbatimPlaceholder = "{verbatim}"

// unnamedProviderParam is what a rule's message says when the provider named
// no parameter (or named one the router would not repeat).
const unnamedProviderParam = "the parameter"

// maxVerbatimMessageLen bounds the provider's message that a rule may relay
// as it is.
//
// A provider's message about a request is a sentence or two; anything much
// longer is a dump of something the client has no business seeing.
const maxVerbatimMessageLen = 512

// providerParamNames are the request fields a rule's message may name.
//
// It is a CLOSED list, and it has to be: the name comes out of the
// provider's own body, so without a list a provider could put an arbitrary
// string into the client's "param" field -- "exactly the channel for
// provider text that this package exists to close" (the same reason
// requestParameterNames exists for the router's own verdict).
//
// The list is the router's own set of request fields (requestParameterNames)
// plus the nested paths a provider may name for a field inside a body
// ("messages[0].content[1]"). A name outside it is not substituted: the rule's
// message says the parameter was not named, and the client's "param" field
// stays empty.
//
// A name is matched against the list in its NORMALIZED form (the case and
// the whitespace folded), so a provider that spells a field differently --
// "maxTokens" for "max_tokens" -- is still recognized as naming it.
var providerParamNames = buildProviderParamNames()

// buildProviderParamNames assembles the list of names a rule may repeat.
//
// It is a list of its own rather than requestParameterNames itself, so that
// adding a field here does not change what the router's own verdict accepts
// as an offending parameter.
func buildProviderParamNames() map[string]struct{} {
	names := make(map[string]struct{}, len(requestParameterNames)+8)
	for name := range requestParameterNames {
		names[name] = struct{}{}
	}
	// The body-shaped fields a provider names a path into: the offending
	// item inside the body ("messages[0].content[1].image_url.url").
	//
	// The names of the STRUCTURE are here too, not only the names of the
	// fields: a path names both ("content" is where the item sits, and
	// "image_url" is what is wrong with it), and a provider that names a
	// path names the structure on the way.
	for _, name := range []string{
		"messages", "input", "model", "stream", "prompt", "image", "file",
		"url", "contents", "system", "content", "image_url", "file_url",
		"text", "role", "part",
	} {
		names[name] = struct{}{}
	}
	return names
}

// providerParamNameAccepted reports whether a parameter name may be
// repeated to the client.
//
// The name must be one of the router's own request fields. The comparison is
// on the normalized form, so a provider that spells a field in another case
// is still recognized -- and a provider that puts anything else there is
// not.
func providerParamNameAccepted(name string) bool {
	if name == "" {
		return false
	}
	normalized := normalizeProviderMessageText(name)
	if _, known := providerParamNames[normalized]; known {
		return true
	}
	// A nested path: "messages[0].content[1]" -- the router's own
	// verdict names such paths too (extractQuotedInvalidField).
	if !strings.ContainsAny(name, "[]") {
		return false
	}
	// The path must start with a known field, and every step after it must
	// be an index or a known field.
	head := normalized
	if idx := strings.IndexAny(head, "["); idx >= 0 {
		head = head[:idx]
	}
	if _, known := providerParamNames[head]; !known {
		return false
	}
	rest := normalized[len(head):]
	for rest != "" {
		// A path step is an index, a field, or both: "messages[0].content[1]".
		if strings.HasPrefix(rest, "[") {
			end := strings.Index(rest, "]")
			if end < 0 {
				return false
			}
			index := rest[1:end]
			if index == "" {
				return false
			}
			for _, r := range index {
				if r < '0' || r > '9' {
					return false
				}
			}
			rest = rest[end+1:]
			continue
		}
		if strings.HasPrefix(rest, ".") {
			rest = rest[1:]
			end := strings.IndexAny(rest, ".[")
			if end < 0 {
				end = len(rest)
			}
			if _, known := providerParamNames[rest[:end]]; !known {
				return false
			}
			rest = rest[end:]
			continue
		}
		return false
	}
	return true
}

// providerRuleParam reports the parameter a rule's message names, and
// whether the rule names one at all.
//
// A rule that names no parameter says nothing about the parameter, so the
// client's "param" field stays empty: the provider's own "param" field is
// about whatever the provider was complaining about, and a rule that did not
// ask for it did not mean to repeat it.
func providerRuleParam(rule ProviderMessageRule) (*string, bool) {
	if !strings.Contains(rule.Message, ProviderParamPlaceholder) {
		return nil, false
	}
	return nil, true
}

// renderProviderMessage puts the parameter the provider named into a rule's
// message.
//
// The parameter is the provider's own, so it is checked against the router's
// list of request fields first (see providerParamNameAccepted): a provider
// that put free text in its "param" field gets a wording that names no
// parameter -- not its text.
//
// A rule whose message has no placeholder is returned as it is, so a rule
// that does not care about the parameter is unaffected.
func renderProviderMessage(message string, param *string) string {
	return renderProviderMessageWithVerbatim(message, param, "")
}

// renderProviderMessageWithVerbatim is renderProviderMessage with the
// provider's own message available for the {verbatim} placeholder.
//
// The provider's message is relayed as it is ONLY when it is safe to: short
// enough, and carrying nothing but printable text. A provider that answered
// with a dump, or with a control character, gets the router's own wording
// instead -- the placeholder is left for the operator to see, rather than
// the client receiving something unprintable.
func renderProviderMessageWithVerbatim(message string, param *string, providerMessage string) string {
	if strings.Contains(message, ProviderVerbatimPlaceholder) {
		if verbatim, ok := verbatimProviderMessage(providerMessage); ok {
			message = strings.ReplaceAll(message, ProviderVerbatimPlaceholder, verbatim)
		} else {
			message = strings.ReplaceAll(message, ProviderVerbatimPlaceholder, unnamedProviderMessage)
		}
	}
	if !strings.Contains(message, ProviderParamPlaceholder) {
		return message
	}
	if param != nil && providerParamNameAccepted(*param) {
		return strings.ReplaceAll(message, ProviderParamPlaceholder, *param)
	}
	return strings.ReplaceAll(message, ProviderParamPlaceholder, unnamedProviderParam)
}

// unnamedProviderMessage is what a rule's message says when the provider's own
// message may not be relayed as it is.
const unnamedProviderMessage = "the provider's message"

// verbatimProviderMessage reports whether the provider's message may be
// relayed to the client as it is, and returns it when it may.
//
// The checks are the ones the rest of this package applies to provider text:
// it must be short enough to be a sentence about the request, and it must
// carry nothing but printable text -- a control character would go into the
// client's JSON body as it is.
func verbatimProviderMessage(message string) (string, bool) {
	trimmed := strings.TrimSpace(message)
	if trimmed == "" || len(trimmed) > maxVerbatimMessageLen {
		return "", false
	}
	for _, r := range trimmed {
		if r == '\n' || r == '\r' || r == '\t' {
			continue
		}
		if unicode.IsControl(r) {
			return "", false
		}
	}
	return trimmed, true
}

// ProviderMessageRetryable reports whether a failure that matched a rule
// may be replayed on another credential, and whether a rule said at all.
//
// It is the rule's own answer, not the router's policy: a rule about the
// request's own fault says "no", and a rule about something a credential
// owns says "yes". A failure that matched no rule leaves the decision to
// the router's own retry policy.
//
// The rule is found through the SAME path the client's message is found
// through (matchProviderMessageRule), so the two cannot disagree: a
// content-policy refusal is answered by neither. It used to look the rule up
// for itself and skipped that guard -- so a refusal took the rule's
// "yes, try another credential" and was replayed across the whole
// credential pool, on a prompt every credential refuses identically.
func ProviderMessageRetryable(info ProviderErrorInfo) (retryable bool, said bool) {
	match, ok := matchProviderMessageRule(info)
	if !ok || match.rule.Retryable == nil {
		return false, false
	}
	return *match.rule.Retryable, true
}

// isContentPolicyRefusal reports whether the failure is a content-policy
// refusal, judged by the signals that did not produce the match.
func isContentPolicyRefusal(info ProviderErrorInfo, matchedByText []int) bool {
	skip := make(map[int]struct{}, len(matchedByText))
	for _, index := range matchedByText {
		skip[index] = struct{}{}
	}

	rest := make([]string, 0, len(info.Signals))
	for i, signal := range info.Signals {
		if _, excluded := skip[i]; excluded {
			continue
		}
		rest = append(rest, signal)
	}
	return HasContentPolicySignal(rest)
}

// providerMessageStatus returns the status a matched rule asks for, and
// whether a rule matched at all.
//
// It is a thin wrapper over ClassifyProviderMessage, for the callers that
// need only the status.
func providerMessageStatus(info ProviderErrorInfo) (int, bool) {
	classified, ok := ClassifyProviderMessage(info)
	if !ok {
		return 0, false
	}
	return classified.Status, true
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
//
// The answer is recognized by its CODE, its MESSAGE and the TYPE beside it --
// the type the router writes for the rule's status. The type is what keeps
// the recognition honest: a provider body that happens to carry the same
// code and message by coincidence is only taken for the router's own when it
// also carries the type that goes with the rule's status.
//
// A coincidence that does pass is harmless: the strings that matched ARE the
// router's own strings, so the client is answered with the same words and
// the same code either way.
// isRouterProviderMessageWithParam reports whether a body is the router's
// own answer for a provider message.
//
// The parameter the body carries and the type it carries are both checked
// here, in ONE pass over the rules: the body must carry the parameter the
// rule's message names (and only when the rule names one), and its type
// must be the one that goes with the rule's status.
//
// The type is what identifies a body as the router's own: the router writes
// the type that goes with the status, so a provider body that happens to
// carry the same code and message by coincidence is not taken for the
// router's own unless it also carries that type.
//
// An empty typ accepts any type.
func isRouterProviderMessageWithParam(code, message, typ string, param *string) bool {
	rules := activeProviderMessageRules()
	for _, rule := range rules.all {
		if rule.rule.Code != code {
			continue
		}
		if typ != "" && rule.rule.Status != 0 && routerErrorTypeForStatus(rule.rule.Status) != typ {
			continue
		}
		if _, names := providerRuleParam(rule.rule); names {
			// The rule names the parameter, so the body's message is
			// compared against the rule's template RENDERED with the
			// parameter the body carries: the same rendering the router did
			// when it wrote the body. So the two cannot disagree, and the
			// reverse-parsing (reconstructing the template and the
			// substitution from the message) is not needed.
			//
			// A body that carries NO parameter is compared against the
			// rendering with none either: the router wrote it that way
			// when the provider named no parameter (or named one the
			// router would not repeat), and it is still the router's own.
			rendered := renderProviderMessage(rule.rule.Message, param)
			if param == nil {
				rendered = renderProviderMessage(rule.rule.Message, nil)
			}
			if rendered != message {
				continue
			}
			return true
		}
		// The rule names no parameter, so the body must carry none.
		if param != nil {
			continue
		}
		if rule.rule.Message != message {
			continue
		}
		return true
	}
	return false
}

// routerErrorTypeForStatus is the "type" the router writes beside a status
// in the client's error object.
//
// It is the same mapping the proxy writes (proxy.errorTypeForStatus), kept
// here so that the router can recognize a body it wrote without depending
// on the proxy package. The two must agree: a body the router wrote is
// recognized by the type beside its code and message.
func routerErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusPaymentRequired:
		return "insufficient_quota"
	case http.StatusForbidden:
		return "permission_denied"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return "timeout_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusBadGateway:
		return "api_error"
	default:
		if status >= http.StatusInternalServerError {
			return "server_error"
		}
		return "invalid_request_error"
	}
}
