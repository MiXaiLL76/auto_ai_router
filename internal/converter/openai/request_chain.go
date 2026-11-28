package openai

import (
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

// The request rules of this package are applied in runs: a destination that
// speaks the OpenAI wire protocol gets up to five of them in a row, and the
// client-parameter clean-up before it another three. Each rule, called on its
// own, first looks for its field in the bytes of the whole body and then
// scans the body to edit it; a run of them reads a multi-megabyte
// conversation a dozen times to change a few small fields.
//
// RequestRules and ClientParamRules apply a run in one go: the body is
// scanned once, every rule works on the same set of top-level members, and
// the body is written back once -- or not at all when no rule had anything to
// do. The result is the same JSON the rules produce one after another, which
// remains the definition (applyOneByOne) and what the tests compare against.

// RequestRules is the run of rules ProviderConverter.RequestFrom applies to
// a request bound for an OpenAI-wire-protocol destination. Each field turns
// one rule on; the order of application is fixed and is the field order.
type RequestRules struct {
	// ChatTools applies ConvertWebSearchTools, then ForceWebSearchResults.
	// For Chat Completions bodies only.
	ChatTools bool
	// StripCacheSalt applies StripCacheSalt.
	StripCacheSalt bool
	// StreamOptionsIncludeUsageOnly applies RebuildStreamOptionsIncludeUsageOnly.
	StreamOptionsIncludeUsageOnly bool
	// StripOpenRouterOnlyFields applies StripOpenRouterOnlyFields.
	StripOpenRouterOnlyFields bool
	// StripVLLMOnlySamplingParams applies StripVLLMOnlySamplingParams.
	StripVLLMOnlySamplingParams bool
	// StripStreamOptions applies StripStreamOptions (the field is removed
	// altogether, for destinations that have no such concept).
	StripStreamOptions bool
}

// Apply applies the enabled rules to body.
func (r RequestRules) Apply(body []byte) []byte {
	if r == (RequestRules{}) {
		return body
	}
	obj, ok := jsonscan.ParseObject(body)
	if !ok || obj.Top().EscapedASCII() || obj.Top().HasInvalidNames() {
		// Not a JSON object: every rule leaves such a body alone, each in its
		// own way. Or some string in it is written with an escaped ASCII
		// character: the rules' own "is my field in these bytes" checks then
		// decide differently from a look at the decoded names, and they are
		// the definition. Or a name is not valid UTF-8: whether it gets
		// repaired depends on which rules re-encode the body, one by one.
		return r.applyOneByOne(body)
	}
	// The field-removing rules, applied on their own, decode a body into
	// interface{} and encode it again as soon as the name of their field
	// occurs anywhere in it, which replaces invalid UTF-8 in every string
	// (see updateJSONFieldDecode). A body that has any takes that path.
	if (r.StripCacheSalt || r.StripOpenRouterOnlyFields || r.StripVLLMOnlySamplingParams || r.StripStreamOptions) && !utf8.Valid(body) {
		return r.applyOneByOne(body)
	}
	fields := obj.Fields
	if r.ChatTools {
		convertWebSearchToolsFields(fields)
		forceWebSearchResultsFields(fields, true)
	}
	if r.StripCacheSalt {
		delete(fields, "cache_salt")
	}
	// reencoded: a rule that, on its own, writes the body out again even when
	// it changed nothing in it. That is observable only on a body with
	// repeated names (writing it out keeps one of each), and is reproduced.
	reencoded := false
	if r.StreamOptionsIncludeUsageOnly {
		_, reencoded = fields["stream_options"]
		rebuildStreamOptionsFields(fields)
	}
	if r.StripOpenRouterOnlyFields {
		delete(fields, "plugins")
		delete(fields, "provider")
	}
	if r.StripVLLMOnlySamplingParams {
		delete(fields, "chat_template_kwargs")
		delete(fields, "repetition_penalty")
		delete(fields, "length_penalty")
	}
	if r.StripStreamOptions {
		delete(fields, "stream_options")
	}
	// ConvertWebSearchTools does the same when it changes a body.
	if r.ChatTools && obj.Changed() && !utf8.Valid(body) {
		return r.applyOneByOne(body)
	}
	if !reencoded && !obj.Changed() {
		return body
	}
	return obj.Encode()
}

// applyOneByOne applies the same rules through their own entry points, one
// after another.
func (r RequestRules) applyOneByOne(body []byte) []byte {
	if r.ChatTools {
		body = ConvertWebSearchTools(body)
		body = ForceWebSearchResults(body)
	}
	if r.StripCacheSalt {
		body = StripCacheSalt(body)
	}
	if r.StreamOptionsIncludeUsageOnly {
		body = RebuildStreamOptionsIncludeUsageOnly(body)
	}
	if r.StripOpenRouterOnlyFields {
		body = StripOpenRouterOnlyFields(body)
	}
	if r.StripVLLMOnlySamplingParams {
		body = StripVLLMOnlySamplingParams(body)
	}
	if r.StripStreamOptions {
		body = StripStreamOptions(body)
	}
	return body
}

// ClientParamRules is the run of client-parameter rules the proxy applies
// before a request is converted for its destination: DropEmptyTools, then
// MapReasoningEffort, then the reasoning-model parameter rules.
type ClientParamRules struct {
	// ModelID is the provider's model name, for the reasoning-model rules.
	ModelID string
	// MapEffort, when set, is MapReasoningEffort's resolver.
	MapEffort func(string) (string, bool)
	// Chat selects ReplaceBodyParam (a Chat Completions body); otherwise
	// ReplaceResponsesBodyParam (a Responses API body).
	Chat bool
}

// Apply applies the rules to body.
func (r ClientParamRules) Apply(body []byte) []byte {
	obj, ok := jsonscan.ParseObject(body)
	if !ok || obj.Top().EscapedASCII() || obj.Top().HasInvalidNames() {
		return r.applyOneByOne(body) // see RequestRules.Apply
	}
	fields := obj.Fields
	dropEmptyToolsFields(fields)
	mapReasoningEffortFields(fields, r.MapEffort)
	if r.Chat {
		replaceBodyParamFields(r.ModelID, fields)
	} else {
		replaceResponsesBodyParamFields(r.ModelID, fields)
	}
	// ReplaceBodyParam writes a reasoning model's body out again whether or
	// not it changed anything (see RequestRules.Apply on why that matters);
	// the other rules return an unchanged body as it is.
	if (!r.Chat || !IsReasoningModel(r.ModelID)) && !obj.Changed() {
		return body
	}
	return obj.Encode()
}

// applyOneByOne applies the same rules through their own entry points, one
// after another.
func (r ClientParamRules) applyOneByOne(body []byte) []byte {
	body = DropEmptyTools(body)
	if r.MapEffort != nil {
		body = MapReasoningEffort(body, r.MapEffort)
	}
	if r.Chat {
		return ReplaceBodyParam(r.ModelID, body)
	}
	return ReplaceResponsesBodyParam(r.ModelID, body)
}
