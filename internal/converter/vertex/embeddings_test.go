package vertex

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

// TestOpenAIEmbeddingToVertex_DimensionsWrongTypeReportsParam covers a client sending
// "dimensions" as a string instead of a number. Before this fix, the raw
// json.Unmarshal error reached the proxy layer as a plain error, which doesn't match
// *converterutil.RequestValidationError and falls through to a generic 500.
func TestOpenAIEmbeddingToVertex_DimensionsWrongTypeReportsParam(t *testing.T) {
	body := []byte(`{"model":"text-embedding-3-small","input":"hello","dimensions":"x"}`)

	_, err := OpenAIEmbeddingToVertex(body)
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "dimensions", validationErr.Param)
	assert.Equal(t, "invalid_type", validationErr.Code)
}

// TestOpenAIEmbeddingToGemini_DimensionsWrongTypeReportsParam is the Gemini-route
// equivalent of the Vertex test above; same request struct, same converter bug.
func TestOpenAIEmbeddingToGemini_DimensionsWrongTypeReportsParam(t *testing.T) {
	body := []byte(`{"model":"gemini-embedding-001","input":"hello","dimensions":"x"}`)

	_, err := OpenAIEmbeddingToGemini(body, "gemini-embedding-001")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "dimensions", validationErr.Param)
	assert.Equal(t, "invalid_type", validationErr.Code)
}

func TestOpenAIEmbeddingToVertex_SingleString(t *testing.T) {
	req := openai.OpenAIEmbeddingRequest{
		Input: "Hello, world!",
		Model: "text-embedding-004",
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	result, err := OpenAIEmbeddingToVertex(body)
	require.NoError(t, err)

	var vertexReq VertexEmbeddingRequest
	require.NoError(t, json.Unmarshal(result, &vertexReq))

	assert.Len(t, vertexReq.Instances, 1)
	assert.Equal(t, "Hello, world!", vertexReq.Instances[0].Content)
	assert.Nil(t, vertexReq.Parameters)
}

func TestOpenAIEmbeddingToVertex_MultipleStrings(t *testing.T) {
	req := openai.OpenAIEmbeddingRequest{
		Input: []interface{}{"text1", "text2", "text3"},
		Model: "text-embedding-004",
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	result, err := OpenAIEmbeddingToVertex(body)
	require.NoError(t, err)

	var vertexReq VertexEmbeddingRequest
	require.NoError(t, json.Unmarshal(result, &vertexReq))

	assert.Len(t, vertexReq.Instances, 3)
	assert.Equal(t, "text1", vertexReq.Instances[0].Content)
	assert.Equal(t, "text2", vertexReq.Instances[1].Content)
	assert.Equal(t, "text3", vertexReq.Instances[2].Content)
}

func TestOpenAIEmbeddingToVertex_WithDimensions(t *testing.T) {
	dims := 512
	req := openai.OpenAIEmbeddingRequest{
		Input:      "test text",
		Model:      "text-embedding-004",
		Dimensions: &dims,
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	result, err := OpenAIEmbeddingToVertex(body)
	require.NoError(t, err)

	var vertexReq VertexEmbeddingRequest
	require.NoError(t, json.Unmarshal(result, &vertexReq))

	require.NotNil(t, vertexReq.Parameters)
	require.NotNil(t, vertexReq.Parameters.OutputDimensionality)
	assert.Equal(t, 512, *vertexReq.Parameters.OutputDimensionality)
}

func TestOpenAIEmbeddingToGemini_SingleString(t *testing.T) {
	req := openai.OpenAIEmbeddingRequest{
		Input: "Hello, world!",
		Model: "text-embedding-004",
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	result, err := OpenAIEmbeddingToGemini(body, "text-embedding-004")
	require.NoError(t, err)

	var geminiReq GeminiEmbeddingRequest
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	assert.Len(t, geminiReq.Requests, 1)
	assert.Equal(t, "models/text-embedding-004", geminiReq.Requests[0].Model)
	require.NotNil(t, geminiReq.Requests[0].Content)
	assert.Len(t, geminiReq.Requests[0].Content.Parts, 1)
	assert.Equal(t, "Hello, world!", geminiReq.Requests[0].Content.Parts[0].Text)
}

func TestOpenAIEmbeddingToGemini_WithDimensions(t *testing.T) {
	dims := 256
	req := openai.OpenAIEmbeddingRequest{
		Input:      []interface{}{"text1", "text2"},
		Model:      "text-embedding-004",
		Dimensions: &dims,
	}
	body, err := json.Marshal(req)
	require.NoError(t, err)

	result, err := OpenAIEmbeddingToGemini(body, "text-embedding-004")
	require.NoError(t, err)

	var geminiReq GeminiEmbeddingRequest
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	assert.Len(t, geminiReq.Requests, 2)
	for _, r := range geminiReq.Requests {
		require.NotNil(t, r.OutputDimensionality)
		assert.Equal(t, int32(256), *r.OutputDimensionality)
	}
}

func TestVertexEmbeddingToOpenAI(t *testing.T) {
	resp := VertexEmbeddingResponse{
		Predictions: []VertexEmbeddingPrediction{
			{
				Embeddings: VertexEmbeddingValues{
					Values: []float64{0.1, 0.2, 0.3},
					Statistics: &VertexEmbeddingStatistics{
						TokenCount: 5,
					},
				},
			},
			{
				Embeddings: VertexEmbeddingValues{
					Values: []float64{0.4, 0.5, 0.6},
					Statistics: &VertexEmbeddingStatistics{
						TokenCount: 3,
					},
				},
			},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	result, err := VertexEmbeddingToOpenAI(body, "text-embedding-004")
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, "list", openaiResp.Object)
	assert.Equal(t, "text-embedding-004", openaiResp.Model)
	assert.Len(t, openaiResp.Data, 2)
	assert.Equal(t, "embedding", openaiResp.Data[0].Object)
	assert.Equal(t, 0, openaiResp.Data[0].Index)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, openaiResp.Data[0].Embedding)
	assert.Equal(t, 1, openaiResp.Data[1].Index)
	assert.Equal(t, []float64{0.4, 0.5, 0.6}, openaiResp.Data[1].Embedding)
	assert.Equal(t, 8, openaiResp.Usage.PromptTokens)
	assert.Equal(t, 8, openaiResp.Usage.TotalTokens)
}

func TestGeminiEmbeddingToOpenAI_NoInputTexts(t *testing.T) {
	// Simulates Gemini API batchEmbedContents response (no statistics).
	// Without inputTexts, token counts fall back to 0.
	resp := genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{
			{Values: []float32{0.1, 0.2, 0.3}},
			{Values: []float32{0.4, 0.5, 0.6}},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	result, err := GeminiEmbeddingToOpenAI(body, "text-embedding-004", nil)
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, "list", openaiResp.Object)
	assert.Equal(t, "text-embedding-004", openaiResp.Model)
	assert.Len(t, openaiResp.Data, 2)
	assert.Equal(t, "embedding", openaiResp.Data[0].Object)
	// float32→float64 conversion; values are approximate
	assert.InDelta(t, 0.1, openaiResp.Data[0].Embedding[0], 1e-5)
	assert.Equal(t, 0, openaiResp.Usage.PromptTokens)
}

func TestGeminiEmbeddingToOpenAI_TokenEstimation(t *testing.T) {
	// Gemini API omits statistics; tokens must be estimated from inputTexts.
	resp := genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{
			{Values: []float32{0.1, 0.2, 0.3}},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	inputTexts := []string{"Hello world"}
	result, err := GeminiEmbeddingToOpenAI(body, "gemini-embedding-001", inputTexts)
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Greater(t, openaiResp.Usage.PromptTokens, 0, "expected estimated token count > 0")
	assert.Equal(t, openaiResp.Usage.PromptTokens, openaiResp.Usage.TotalTokens)
}

func TestGeminiEmbeddingToOpenAI_WithStatistics(t *testing.T) {
	// Simulates Vertex embedContent response that does include statistics.
	tokenCount := float32(5)
	resp := genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{
			{
				Values: []float32{0.1, 0.2, 0.3},
				Statistics: &genai.ContentEmbeddingStatistics{
					TokenCount: tokenCount,
				},
			},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	result, err := GeminiEmbeddingToOpenAI(body, "gemini-embedding-001", nil)
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, 5, openaiResp.Usage.PromptTokens)
	assert.Equal(t, 5, openaiResp.Usage.TotalTokens)
}

func TestGeminiEmbeddingToOpenAI_BatchTokenEstimation(t *testing.T) {
	// Batch estimation: tokens for each text are summed.
	resp := genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{
			{Values: []float32{0.1}},
			{Values: []float32{0.2}},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	inputTexts := []string{"Hello world", "Another sentence here"}
	result, err := GeminiEmbeddingToOpenAI(body, "gemini-embedding-001", inputTexts)
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	singleResp := genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{{Values: []float32{0.1}}},
	}
	singleBody, _ := json.Marshal(singleResp)
	singleResult, _ := GeminiEmbeddingToOpenAI(singleBody, "gemini-embedding-001", []string{"Hello world"})
	var singleOpenAI openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(singleResult, &singleOpenAI))

	assert.Greater(t, openaiResp.Usage.PromptTokens, singleOpenAI.Usage.PromptTokens,
		"batch tokens should be greater than single-text tokens")
}

func TestBuildVertexEmbeddingURL(t *testing.T) {
	cred := &config.CredentialConfig{
		ProjectID: "my-project",
		Location:  "us-central1",
	}
	url := BuildVertexEmbeddingURL(cred, "text-embedding-004")
	assert.Equal(t, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/publishers/google/models/text-embedding-004:predict", url)
}

func TestBuildVertexEmbeddingURL_Global(t *testing.T) {
	cred := &config.CredentialConfig{
		ProjectID: "my-project",
		Location:  "global",
	}
	url := BuildVertexEmbeddingURL(cred, "text-embedding-004")
	assert.Equal(t, "https://aiplatform.googleapis.com/v1beta1/projects/my-project/locations/global/publishers/google/models/text-embedding-004:predict", url)
}

func TestBuildGeminiEmbeddingURL(t *testing.T) {
	cred := &config.CredentialConfig{
		BaseURL: "https://generativelanguage.googleapis.com",
	}
	url := BuildGeminiEmbeddingURL(cred, "text-embedding-004")
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents", url)
}

func TestExtractInputTexts(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    []string
		wantErr bool
	}{
		{
			name:  "single string",
			input: "hello",
			want:  []string{"hello"},
		},
		{
			name:  "string slice via interface",
			input: []interface{}{"a", "b", "c"},
			want:  []string{"a", "b", "c"},
		},
		{
			name:    "non-string array element",
			input:   []interface{}{123},
			wantErr: true,
		},
		{
			name:    "unsupported type",
			input:   42,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractInputTexts(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}
