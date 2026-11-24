package vertex

import (
	"encoding/json"

	"google.golang.org/genai"
)

// VertexStreamingChunk wraps genai types for streaming response
type VertexStreamingChunk struct {
	Candidates    []*genai.Candidate                          `json:"candidates,omitempty"`
	UsageMetadata *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string                                      `json:"modelVersion,omitempty"`
	// PromptFeedback carries the prompt-level content block. Vertex/Gemini send
	// it only in the first stream chunk and only when no candidates were
	// generated because the prompt itself was refused (see the genai SDK's own
	// doc on GenerateContentResponsePromptFeedback). Dropping it here made such
	// a refusal look like a provider-side empty stream.
	PromptFeedback *genai.GenerateContentResponsePromptFeedback `json:"promptFeedback,omitempty"`
}

type VertexGenerationConfig struct {
	*genai.GenerationConfig
	ImageConfig *genai.ImageConfig `json:"imageConfig,omitempty"`
}

// VertexRequest represents the Vertex AI API request format
type VertexRequest struct {
	Contents          []*genai.Content        `json:"contents"`
	GenerationConfig  *VertexGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *genai.Content          `json:"systemInstruction,omitempty"`
	Tools             []*genai.Tool           `json:"tools,omitempty"`
	ToolConfig        *genai.ToolConfig       `json:"toolConfig,omitempty"`
}

// MarshalForModel applies the signature rules to the final model turns shared
// by Chat and Responses. Continuation requests reuse these serialized bytes.
func (r *VertexRequest) MarshalForModel(model string) ([]byte, error) {
	if !UsesPrefixBoundSignatures(model) {
		return json.Marshal(r)
	}
	for _, turn := range r.Contents {
		if turn.Role != "model" {
			continue
		}
		firstCall := true
		for _, part := range turn.Parts {
			if part.FunctionCall == nil {
				continue
			}
			if !firstCall {
				part.ThoughtSignature = nil
			}
			firstCall = false
		}
	}

	// Overlay the missing SDK field without re-encoding large media parts
	// or discarding the other generation and thinking settings.
	type thinking struct {
		*genai.ThinkingConfig
		PrefixMismatchBehavior string `json:"prefixMismatchBehavior"`
	}
	type generation struct {
		*VertexGenerationConfig
		Thinking thinking `json:"thinkingConfig"`
	}
	config := generation{
		VertexGenerationConfig: r.GenerationConfig,
		Thinking:               thinking{ThinkingConfig: r.GenerationConfig.ThinkingConfig, PrefixMismatchBehavior: "DROP"},
	}
	return json.Marshal(struct {
		*VertexRequest
		Generation generation `json:"generationConfig"`
	}{r, config})
}
