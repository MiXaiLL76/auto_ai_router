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

	off := false
	if effort, ok := data["reasoning_effort"].(string); ok && isReasoningEffortOff(effort) {
		off = true
		delete(data, "reasoning_effort")
	}
	// The Responses API carries the effort nested: reasoning: {effort: "none"}.
	if reasoning, ok := data["reasoning"].(map[string]any); ok {
		if effort, ok := reasoning["effort"].(string); ok && isReasoningEffortOff(effort) {
			off = true
			delete(data, "reasoning")
		}
	}
	if !off {
		return body
	}
	data["thinking"] = map[string]any{"type": "disabled"}
	updated, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return updated
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
