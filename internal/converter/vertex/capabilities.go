package vertex

import "strings"

// This file collects per-model capability checks for fields the router copies into
// Vertex/Gemini generationConfig. Google validates generationConfig strictly: a knob
// the target model does not implement is rejected with 400 INVALID_ARGUMENT rather
// than ignored, and because a 400 is retryable the same doomed request is then
// replayed across every credential in the rotation. Deciding here — once, by model —
// keeps that class of failure out of the upstream entirely.
//
// The thinking-level floor is the same kind of check but lives next to the rest of
// the thinking logic; see lowestThinkingLevel in thinking.go.

// supportsPenalty reports whether the model accepts frequency_penalty /
// presence_penalty in generationConfig.
//
// Google dropped both knobs after Gemini 2.0. Gemini 2.5 and 3.x reject a request
// carrying either one with 400 INVALID_ARGUMENT ("Penalty is not enabled for this
// model"), on both the AI Studio and the Vertex endpoint. Both parameters are
// optional for the caller, so an unsupported one is dropped rather than turned into
// a client-visible error.
//
// Deliberately an allow-list: a deny-list would need extending on every Gemini
// release to keep new models from failing, while an unknown model here merely loses
// an optional knob. Non-Gemini models routed through this converter are left alone —
// the restriction is specific to Gemini's generationConfig.
func supportsPenalty(model string) bool {
	lower := strings.ToLower(model)
	if !strings.Contains(lower, "gemini") {
		return true
	}
	return strings.Contains(lower, "gemini-1.") || strings.Contains(lower, "gemini-2.0")
}
