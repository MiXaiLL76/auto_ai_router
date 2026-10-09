package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// This file collects per-model capability checks for fields the router copies into
// Vertex/Gemini generationConfig. Google validates generationConfig strictly: a knob
// the target model does not implement is rejected with 400 INVALID_ARGUMENT rather
// than ignored, and because a 400 is retryable the same doomed request is then
// replayed across every credential in the rotation. Deciding here — once, by model —
// keeps that class of failure out of the upstream entirely.
//
// Most checks go by the model name; a model whose name does not reveal its
// capabilities is pinned by ID in model_profile.go, and the checks here consult that
// profile first. The thinking-level floor is the same kind of check but lives next
// to the rest of the thinking logic; see lowestThinkingLevel in thinking.go.

// isImageModel reports whether the model generates images: its profile says so, or
// its name contains "image" (Gemini image models, Imagen).
func isImageModel(model string) bool {
	if profile := lookupGeminiModelProfile(model); profile != nil && profile.imageGeneration {
		return true
	}
	return strings.Contains(strings.ToLower(model), "image")
}

// isThinkingCapableModel returns true for models that support dynamic thinking
// (Gemini 2.5+, Gemini 3+, profiled models with thinking levels). These models think
// autonomously when ThinkingConfig is not set, causing unpredictable latency.
func isThinkingCapableModel(model string) bool {
	if profile := lookupGeminiModelProfile(model); profile != nil {
		return profile.hasThinkingLevels()
	}
	if isImageModel(model) {
		return false
	}
	lower := strings.ToLower(model)
	return strings.Contains(lower, "gemini-2.5") || strings.Contains(lower, "gemini-3")
}

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

// ApplyGenerationConstraints removes the generationConfig fields the model rejects.
// It runs after every request source is merged, so none can put one back.
// Exported for use by sub-packages (e.g. vertex/responses).
func ApplyGenerationConstraints(cfg *genai.GenerationConfig, model string) {
	if profile := lookupGeminiModelProfile(model); cfg != nil && profile != nil && profile.rejectsSamplingParams {
		cfg.Temperature, cfg.TopP, cfg.TopK, cfg.Seed = nil, nil, nil, nil
		cfg.ResponseLogprobs, cfg.Logprobs = false, nil
	}
}
