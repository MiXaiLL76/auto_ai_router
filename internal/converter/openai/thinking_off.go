package openai

import (
	"bytes"
	"encoding/json"
)

// SetThinkingDisabled turns extended thinking off in an OpenAI-wire request
// body by putting Anthropic's own words into it, and removes the
// reasoning_effort field that asked for it.
//
// Why both: a provider that speaks the OpenAI wire protocol but fronts a
// Claude model does not read "reasoning_effort": "none" as "no reasoning".
// Measured against Requesty (router.requesty.ai) with
// anthropic/claude-haiku-5-5, a model that thinks by default:
//
//	reasoning_effort: "none"        -> reasoning STILL RUNS (and it also
//	                                    makes the model reason where the
//	                                    client asked for none)
//	thinking: {"type": "disabled"}   -> reasoning off
//
// Leaving reasoning_effort in the body is not harmless: with
// thinking disabled AND reasoning_effort "none" present, Requesty turns
// reasoning back ON -- the effort is the stronger signal there. So the
// effort goes, and the explicit disable takes its place.
//
// The body is returned unchanged when it already carries an explicit
// thinking config: the client asked for something specific, and that is
// not the router's to override.
func SetThinkingDisabled(body []byte) []byte {
	hasChatEffort := bytes.Contains(body, []byte(`"reasoning_effort"`))
	hasResponsesEffort := bytes.Contains(body, []byte(`"effort"`))
	if !hasChatEffort && !hasResponsesEffort {
		return body
	}
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	// The client said how it wants thinking handled; leave it alone.
	if _, exists := data["thinking"]; exists {
		return body
	}

	// The effective effort is the NESTED one when it is there: the Anthropic
	// converter resolves the same pair the same way (reasoning.effort wins
	// over reasoning_effort -- see openAIReasoningEffort), and a request
	// that carries both is a client's own contradiction, not something to
	// resolve differently on this path.
	effort, ok := effectiveReasoningEffort(data)
	if !ok || !isReasoningEffortOff(effort) {
		return body
	}
	// Both forms go: the one that asked for off AND the other one, whatever
	// it said. Leaving the other in place is what breaks the request -- a
	// body carrying both "reasoning_effort": "high" and reasoning.effort
	// "none" would keep the "high", and the provider reads the effort as
	// the stronger signal and turns reasoning back ON (measured against
	// Requesty).
	delete(data, "reasoning_effort")
	delete(data, "reasoning")
	data["thinking"] = map[string]any{"type": "disabled"}
	updated, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return updated
}

// effectiveReasoningEffort returns the reasoning effort a request asks for,
// and whether it asks for one at all.
//
// The Responses API carries it nested (reasoning: {effort: "none"}), the
// Chat API flat (reasoning_effort: "none"); a request may carry both, and
// then the nested one wins -- the same precedence the Anthropic converter
// applies (see openAIReasoningEffort).
func effectiveReasoningEffort(data map[string]any) (string, bool) {
	if reasoning, ok := data["reasoning"].(map[string]any); ok {
		if effort, ok := reasoning["effort"].(string); ok && effort != "" {
			return effort, true
		}
	}
	if effort, ok := data["reasoning_effort"].(string); ok && effort != "" {
		return effort, true
	}
	return "", false
}

// isReasoningEffortOff reports whether an OpenAI reasoning effort asks for
// no reasoning.
func isReasoningEffortOff(effort string) bool {
	switch effort {
	case "none", "disable", "disabled", "off":
		return true
	}
	return false
}
