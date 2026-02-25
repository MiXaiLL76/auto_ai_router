package vertex

import "google.golang.org/genai"

// VertexStreamingChunk wraps genai types for streaming response
type VertexStreamingChunk struct {
	Candidates    []*genai.Candidate                          `json:"candidates,omitempty"`
	UsageMetadata *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata,omitempty"`
}

// VertexRequest represents the Vertex AI API request format
type VertexRequest struct {
	Contents          []*genai.Content        `json:"contents"`
	GenerationConfig  *genai.GenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *genai.Content          `json:"systemInstruction,omitempty"`
	Tools             []*genai.Tool           `json:"tools,omitempty"`
	ToolConfig        *genai.ToolConfig       `json:"toolConfig,omitempty"`
}
