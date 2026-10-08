package vertex

import (
	"fmt"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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

// thinkingLevelRank orders the normalized level names (see thinkingName).
var thinkingLevelRank = map[string]int{
	"minimal": 0,
	"low":     1,
	"medium":  2,
	"high":    3,
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

// ValidateInputImages rejects a request with more images than the model accepts,
// counting every image part of every turn: the upstream limit is per request.
func ValidateInputImages(model string, contents []*genai.Content) error {
	profile := lookupGeminiModelProfile(model)
	if profile == nil || profile.maxInputImages <= 0 {
		return nil
	}
	images := 0
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if isImagePart(part) {
				images++
			}
		}
	}
	if images <= profile.maxInputImages {
		return nil
	}
	return &converterutil.RequestValidationError{
		Param: "image",
		Code:  "too_many_images",
		Message: fmt.Sprintf("Too many input images: %d were sent, this model accepts at most %d per request",
			images, profile.maxInputImages),
	}
}

func isImagePart(part *genai.Part) bool {
	if part == nil {
		return false
	}
	mimeType := ""
	switch {
	case part.InlineData != nil:
		mimeType = part.InlineData.MIMEType
	case part.FileData != nil:
		mimeType = part.FileData.MIMEType
	}
	return strings.HasPrefix(strings.ToLower(mimeType), "image/")
}
