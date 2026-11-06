package vertex

import (
	"strings"

	"google.golang.org/genai"
)

// geminiModelProfile pins the capabilities of a Gemini model its ID does not reveal.
// The package otherwise decides by substring ("image", "gemini-3"), and
// gemini-nano-banana-2.1 contains neither. A profile wins over those heuristics.
type geminiModelProfile struct {
	id              string
	imageGeneration bool
	// image is the grid size mapping picks from; explicit values outside it are rejected.
	image geminiImageProfile
	// thinkingLevels are the accepted levels, lowest first; empty: no configurable thinking.
	thinkingLevels []genai.ThinkingLevel
	// defaultThinkingLevel is the model's own default, which the router never lowers.
	defaultThinkingLevel genai.ThinkingLevel
	// rejectsSamplingParams: the model answers 400 to temperature, topP, topK, seed, logprobs.
	rejectsSamplingParams bool
	// maxInputImages caps the image parts of one request (0: no cap).
	maxInputImages int
	// unbilledSearchContext: Google Search context is not charged as input tokens.
	unbilledSearchContext bool
	// thinkingLevelsOnly: the model takes nothing but its thinking levels. A native
	// thinking_budget or thinking_level MINIMAL is refused with 400 (Google answers
	// both with 400 rather than mapping them), and a compatibility request to turn
	// thinking off gets the lowest level: thinking cannot be disabled.
	thinkingLevelsOnly bool
	// ignoresSamplingParams: temperature, topP and topK have no effect and are dropped.
	ignoresSamplingParams bool
	// singleCandidate: candidateCount is rejected upstream, so n=1 is not sent and
	// n>1 is refused with 400.
	singleCandidate bool
	// noModelPrefill: the conversation may not end with a model turn.
	noModelPrefill bool
	// tokenLimit is one limit for input and output together (output counts thinking,
	// function calls and the answer); 0 when the model has separate limits.
	tokenLimit int
	// longDecode: generation past the single-request output cap stops with finish
	// reason CONTINUATION and goes on in requests carrying its continuation token.
	longDecode bool
	// functionCallIDs: a function response must carry its call's ID, and the call
	// its own thought signature. The chat route keeps the IDs Gemini gives calls and
	// sends them back on calls and responses; the Responses route carries the
	// signature in call_id (see vertex/responses/call_id.go). Without it the chat
	// route makes up tool call IDs and sends none, and a call_id stays the bare ID
	// (with a signature it runs past the 64 characters OpenAI takes).
	functionCallIDs bool
	// billsRepeatedSearches: every Google Search query the model runs is billed, a
	// second run with the same text too; without it a response's queries count once
	// per text.
	billsRepeatedSearches bool
}

var geminiModelProfiles = []geminiModelProfile{
	{
		// The 3.1 Flash Image grid without 512px. 9:21 is listed by Vertex only, not
		// by the Gemini API, so it stays out until both routes accept it.
		id:              "gemini-nano-banana-2.1",
		imageGeneration: true,
		image:           geminiImageProfile{ratios: gemini31FlashImageRatios, resolutions: gemini3ImageResolutions},
		thinkingLevels: []genai.ThinkingLevel{
			genai.ThinkingLevelMinimal,
			genai.ThinkingLevelMedium,
			genai.ThinkingLevelHigh,
		},
		defaultThinkingLevel:  genai.ThinkingLevelMedium,
		rejectsSamplingParams: true,
		maxInputImages:        14,
		unbilledSearchContext: true,
	},
	{
		// Gemini 4 Argon. Its name passes the Gemini 2.5 budget checks and a Gemini 3
		// rule would turn medium into HIGH, so its rules live here.
		id: "gemini-4-argon",
		thinkingLevels: []genai.ThinkingLevel{
			genai.ThinkingLevelLow,
			genai.ThinkingLevelMedium,
			genai.ThinkingLevelHigh,
		},
		defaultThinkingLevel:  genai.ThinkingLevelMedium,
		thinkingLevelsOnly:    true,
		ignoresSamplingParams: true,
		singleCandidate:       true,
		noModelPrefill:        true,
		tokenLimit:            1048576,
		longDecode:            true,
		functionCallIDs:       true,
		billsRepeatedSearches: true,
	},
}

// lookupGeminiModelProfile returns the profile for model, or nil. Provider prefixes
// ("google/", "publishers/google/models/") are ignored.
func lookupGeminiModelProfile(model string) *geminiModelProfile {
	name := strings.ToLower(strings.TrimSpace(model))
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	if name == "" {
		return nil
	}
	for i := range geminiModelProfiles {
		if matchesGeminiModelID(name, geminiModelProfiles[i].id) {
			return &geminiModelProfiles[i]
		}
	}
	return nil
}

// matchesGeminiModelID reports whether name is id or a version of it ("-preview…",
// "-001", "@001"); another variant such as "-lite" is a different model.
func matchesGeminiModelID(name, id string) bool {
	rest, ok := strings.CutPrefix(name, id)
	if !ok {
		return false
	}
	switch {
	case rest == "", strings.HasPrefix(rest, "@"), strings.HasPrefix(rest, "-preview"):
		return true
	default:
		return len(rest) > 1 && rest[0] == '-' && rest[1] >= '0' && rest[1] <= '9'
	}
}

// hasThinkingLevels reports whether the profile pins the model's thinking levels.
func (p *geminiModelProfile) hasThinkingLevels() bool {
	return p != nil && len(p.thinkingLevels) > 0
}

// thinkingLevel resolves a level or effort name to the highest supported level not
// above it ("low" → MINIMAL without LOW), the lowest one for "none". ok is false for
// a name that is not a level.
func (p *geminiModelProfile) thinkingLevel(requested string) (genai.ThinkingLevel, bool) {
	name := thinkingName(requested)
	if name == "none" {
		return p.thinkingLevels[0], true
	}
	rank, ok := thinkingLevelRank[name]
	if !ok {
		return "", false
	}
	resolved := p.thinkingLevels[0]
	for _, level := range p.thinkingLevels {
		if thinkingLevelRank[strings.ToLower(string(level))] <= rank {
			resolved = level
		}
	}
	return resolved, true
}

// thinkingConfig is thinkingLevel with the model default for an empty or unknown name.
func (p *geminiModelProfile) thinkingConfig(requested string, includeThoughts bool) *genai.ThinkingConfig {
	level, ok := p.thinkingLevel(requested)
	if !ok {
		level = p.defaultThinkingLevel
	}
	return &genai.ThinkingConfig{IncludeThoughts: includeThoughts, ThinkingLevel: level}
}
