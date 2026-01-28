package vertex_transform

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

// OpenAIRequest represents the OpenAI API request format
type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Temperature *float64        `json:"temperature,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        interface{}     `json:"stop,omitempty"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// VertexRequest represents the Vertex AI API request format
type VertexRequest struct {
	Contents          []VertexContent         `json:"contents"`
	GenerationConfig  *VertexGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *VertexContent          `json:"systemInstruction,omitempty"`
}

type VertexContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []VertexPart `json:"parts"`
}

type VertexPart struct {
	Text string `json:"text"`
}

type VertexGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

// OpenAIToVertex converts OpenAI format request to Vertex AI format
func OpenAIToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq OpenAIRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	vertexReq := VertexRequest{
		Contents: make([]VertexContent, 0),
	}

	// Handle generation config
	if openAIReq.Temperature != nil || openAIReq.MaxTokens != nil || openAIReq.TopP != nil {
		vertexReq.GenerationConfig = &VertexGenerationConfig{
			Temperature:     openAIReq.Temperature,
			MaxOutputTokens: openAIReq.MaxTokens,
			TopP:            openAIReq.TopP,
		}

		// Handle stop sequences
		if openAIReq.Stop != nil {
			switch stop := openAIReq.Stop.(type) {
			case string:
				vertexReq.GenerationConfig.StopSequences = []string{stop}
			case []interface{}:
				stopSeqs := make([]string, 0, len(stop))
				for _, s := range stop {
					if str, ok := s.(string); ok {
						stopSeqs = append(stopSeqs, str)
					}
				}
				vertexReq.GenerationConfig.StopSequences = stopSeqs
			}
		}
	}

	// Convert messages
	for _, msg := range openAIReq.Messages {
		if msg.Role == "system" {
			// System messages become systemInstruction
			vertexReq.SystemInstruction = &VertexContent{
				Parts: []VertexPart{{Text: msg.Content}},
			}
		} else {
			// Convert role mapping
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}

			vertexReq.Contents = append(vertexReq.Contents, VertexContent{
				Role:  role,
				Parts: []VertexPart{{Text: msg.Content}},
			})
		}
	}

	return json.Marshal(vertexReq)
}

// VertexToOpenAI converts Vertex AI response to OpenAI format
func VertexToOpenAI(vertexBody []byte, model string) ([]byte, error) {
	var vertexResp VertexResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex response: %w", err)
	}

	openAIResp := OpenAIResponse{
		ID:      generateID(),
		Object:  "chat.completion",
		Created: getCurrentTimestamp(),
		Model:   model,
		Choices: make([]OpenAIChoice, 0),
	}

	// Convert candidates to choices
	for i, candidate := range vertexResp.Candidates {
		var content string
		if len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		} else {
			// Handle case where parts is empty but we have a finish reason
			if candidate.FinishReason == "MAX_TOKENS" {
				content = "[Response truncated due to max tokens limit]"
			} else {
				content = "[No content generated]"
			}
		}

		choice := OpenAIChoice{
			Index: i,
			Message: OpenAIMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: mapFinishReason(candidate.FinishReason),
		}
		openAIResp.Choices = append(openAIResp.Choices, choice)
	}

	// Convert usage metadata
	if vertexResp.UsageMetadata != nil {
		openAIResp.Usage = &OpenAIUsage{
			PromptTokens:     vertexResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: vertexResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      vertexResp.UsageMetadata.TotalTokenCount,
		}
	}

	return json.Marshal(openAIResp)
}

// VertexResponse represents Vertex AI response format
type VertexResponse struct {
	Candidates    []VertexCandidate    `json:"candidates"`
	UsageMetadata *VertexUsageMetadata `json:"usageMetadata,omitempty"`
}

type VertexCandidate struct {
	Content      VertexContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"`
}

type VertexUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// OpenAIResponse represents OpenAI response format
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Helper functions
func generateID() string {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	return "chatcmpl-" + hex.EncodeToString(bytes)[:20]
}

func getCurrentTimestamp() int64 {
	return time.Now().Unix()
}

func mapFinishReason(vertexReason string) string {
	switch vertexReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY":
		return "content_filter"
	case "RECITATION":
		return "content_filter"
	default:
		return "stop"
	}
}
