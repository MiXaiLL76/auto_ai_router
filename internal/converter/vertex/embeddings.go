package vertex

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// Vertex AI embedding types (models/{model}:predict)

type VertexEmbeddingRequest struct {
	Instances  []VertexEmbeddingInstance  `json:"instances"`
	Parameters *VertexEmbeddingParameters `json:"parameters,omitempty"`
}

type VertexEmbeddingInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
}

type VertexEmbeddingParameters struct {
	OutputDimensionality *int  `json:"outputDimensionality,omitempty"`
	AutoTruncate         *bool `json:"autoTruncate,omitempty"`
}

type VertexEmbeddingResponse struct {
	Predictions []VertexEmbeddingPrediction `json:"predictions"`
	Metadata    *VertexEmbeddingMetadata    `json:"metadata,omitempty"`
}

type VertexEmbeddingPrediction struct {
	Embeddings VertexEmbeddingValues `json:"embeddings"`
}

type VertexEmbeddingValues struct {
	Values     []float64                  `json:"values"`
	Statistics *VertexEmbeddingStatistics `json:"statistics,omitempty"`
}

type VertexEmbeddingStatistics struct {
	TokenCount float64 `json:"token_count"`
	Truncated  bool    `json:"truncated"`
}

type VertexEmbeddingMetadata struct {
	BillableCharacterCount int `json:"billableCharacterCount,omitempty"`
}

// Gemini API embedding types (models/{model}:batchEmbedContents)

type GeminiEmbeddingRequest struct {
	Requests []GeminiEmbedRequest `json:"requests"`
}

type GeminiEmbedRequest struct {
	Model                string         `json:"model"`
	Content              *GeminiContent `json:"content"`
	TaskType             string         `json:"taskType,omitempty"`
	OutputDimensionality *int32         `json:"outputDimensionality,omitempty"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text string `json:"text"`
}

type GeminiEmbeddingResponse struct {
	Embeddings    []GeminiContentEmbedding `json:"embeddings"`
	UsageMetadata *GeminiEmbeddingUsage    `json:"usageMetadata,omitempty"`
}

type GeminiContentEmbedding struct {
	Values []float64 `json:"values"`
}

type GeminiEmbeddingUsage struct {
	PromptTokenCount int `json:"promptTokenCount"`
	TotalTokenCount  int `json:"totalTokenCount"`
}

// extractInputTexts parses the OpenAI input field into a slice of strings.
// Handles string, []string, and []interface{} (JSON arrays decode as []interface{}).
func extractInputTexts(input interface{}) ([]string, error) {
	switch v := input.(type) {
	case string:
		return []string{v}, nil
	case []interface{}:
		texts := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("unsupported input array element type: %T", item)
			}
			texts = append(texts, s)
		}
		return texts, nil
	case []string:
		return v, nil
	default:
		return nil, fmt.Errorf("unsupported input type: %T", input)
	}
}

// OpenAIEmbeddingToVertex converts an OpenAI embedding request to Vertex AI format.
func OpenAIEmbeddingToVertex(body []byte) ([]byte, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse embedding request: %w", err)
	}

	texts, err := extractInputTexts(req.Input)
	if err != nil {
		return nil, err
	}

	instances := make([]VertexEmbeddingInstance, len(texts))
	for i, text := range texts {
		instances[i] = VertexEmbeddingInstance{Content: text}
	}

	vertexReq := VertexEmbeddingRequest{
		Instances: instances,
	}

	if req.Dimensions != nil {
		vertexReq.Parameters = &VertexEmbeddingParameters{
			OutputDimensionality: req.Dimensions,
		}
	}

	return json.Marshal(vertexReq)
}

// OpenAIEmbeddingToGemini converts an OpenAI embedding request to Gemini API format.
func OpenAIEmbeddingToGemini(body []byte, model string) ([]byte, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse embedding request: %w", err)
	}

	texts, err := extractInputTexts(req.Input)
	if err != nil {
		return nil, err
	}

	modelRef := "models/" + model

	requests := make([]GeminiEmbedRequest, len(texts))
	for i, text := range texts {
		gr := GeminiEmbedRequest{
			Model: modelRef,
			Content: &GeminiContent{
				Parts: []GeminiPart{{Text: text}},
			},
		}
		if req.Dimensions != nil {
			dim := int32(*req.Dimensions)
			gr.OutputDimensionality = &dim
		}
		requests[i] = gr
	}

	geminiReq := GeminiEmbeddingRequest{
		Requests: requests,
	}

	return json.Marshal(geminiReq)
}

// VertexEmbeddingToOpenAI converts a Vertex AI embedding response to OpenAI format.
func VertexEmbeddingToOpenAI(body []byte, model string) ([]byte, error) {
	var resp VertexEmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse vertex embedding response: %w", err)
	}

	data := make([]openai.OpenAIEmbeddingData, len(resp.Predictions))
	totalTokens := 0

	for i, pred := range resp.Predictions {
		data[i] = openai.OpenAIEmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: pred.Embeddings.Values,
		}
		if pred.Embeddings.Statistics != nil {
			totalTokens += int(math.Round(pred.Embeddings.Statistics.TokenCount))
		}
	}

	openaiResp := openai.OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: openai.OpenAIEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	return json.Marshal(openaiResp)
}

// GeminiEmbeddingToOpenAI converts a Gemini API embedding response to OpenAI format.
func GeminiEmbeddingToOpenAI(body []byte, model string) ([]byte, error) {
	var resp GeminiEmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse gemini embedding response: %w", err)
	}

	data := make([]openai.OpenAIEmbeddingData, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		data[i] = openai.OpenAIEmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: emb.Values,
		}
	}

	var promptTokens, totalTokens int
	if resp.UsageMetadata != nil {
		promptTokens = resp.UsageMetadata.PromptTokenCount
		totalTokens = resp.UsageMetadata.TotalTokenCount
	}

	openaiResp := openai.OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: openai.OpenAIEmbeddingUsage{
			PromptTokens: promptTokens,
			TotalTokens:  totalTokens,
		},
	}

	return json.Marshal(openaiResp)
}

// BuildVertexEmbeddingURL constructs the Vertex AI URL for embeddings.
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexEmbeddingURL(cred *config.CredentialConfig, modelID string) string {
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// BuildGeminiEmbeddingURL constructs the Gemini API URL for embeddings.
// Format: {base_url}/v1beta/models/{model}:batchEmbedContents
func BuildGeminiEmbeddingURL(cred *config.CredentialConfig, modelID string) string {
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	return fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents", baseURL, modelID)
}
