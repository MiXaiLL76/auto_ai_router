package proxy

import (
	"encoding/json"

	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// applyDeepSeekToolUsageTo folds the router-run tool loop's usage into the
// request's own usage, so the billing path prices the whole loop.
//
// The loop's token counts are the sum of its steps', and the provider's own
// usage (of the last step) is what the body carries -- so the loop's counts
// replace the last step's rather than adding to them. The tool counters are
// the router's own and are added to the provider's, because they count
// something the provider never reported: the searches and fetches the router
// ran for the model.
//
// A nil loop usage (no loop ran) changes nothing.
func applyDeepSeekToolUsageTo(usage *converter.TokenUsage, loop *converter.TokenUsage) {
	if usage == nil || loop == nil {
		return
	}
	// The loop's counts already include the last step's (they are the sum of
	// every step), so the last step's own figures must not be counted twice.
	usage.PromptTokens = loop.PromptTokens
	usage.CompletionTokens = loop.CompletionTokens
	usage.CachedInputTokens = loop.CachedInputTokens
	usage.ReasoningTokens = loop.ReasoningTokens
	usage.WebSearchRequests += loop.WebSearchRequests
	usage.WebFetchRequests += loop.WebFetchRequests
	if usage.WebSearchContextSize == "" {
		usage.WebSearchContextSize = loop.WebSearchContextSize
	}
}

// setJSONInt sets an integer field in a raw JSON object, leaving the object
// unchanged when the value is not positive.
func setJSONInt(fields map[string]json.RawMessage, name string, value int) {
	if value <= 0 {
		return
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return
	}
	fields[name] = encoded
}

// withDeepSeekToolUsage writes the loop's summed usage into the last chunk of
// a streamed response.
//
// The streaming log reads its figures from the last chunk, so a chunk that
// carries only the last step's usage would bill the loop's last step alone.
// Putting the loop's totals there bills every step, as the client was
// charged for all of them.
//
// A chunk that cannot be parsed is returned as it is: the log then falls back
// to the request's own usage, which the caller has already filled.
func withDeepSeekToolUsage(chunk []byte, usage *converter.TokenUsage) []byte {
	if len(chunk) == 0 || usage == nil {
		return chunk
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(chunk, &fields); err != nil {
		return chunk
	}
	rawUsage, ok := fields["usage"]
	if !ok {
		return chunk
	}
	var usageFields map[string]json.RawMessage
	if err := json.Unmarshal(rawUsage, &usageFields); err != nil {
		return chunk
	}
	setJSONInt(usageFields, "prompt_tokens", usage.PromptTokens)
	setJSONInt(usageFields, "completion_tokens", usage.CompletionTokens)
	setJSONInt(usageFields, "total_tokens", usage.PromptTokens+usage.CompletionTokens)
	encodedUsage, err := json.Marshal(usageFields)
	if err != nil {
		return chunk
	}
	fields["usage"] = encodedUsage
	encoded, err := json.Marshal(fields)
	if err != nil {
		return chunk
	}
	return encoded
}
