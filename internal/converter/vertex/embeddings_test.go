package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestGeminiEmbeddingToOpenAI(t *testing.T) {
	resp := GeminiEmbeddingResponse{
		Embeddings: []GeminiContentEmbedding{
			{Values: []float64{0.1, 0.2, 0.3}},
			{Values: []float64{0.4, 0.5, 0.6}},
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	result, err := GeminiEmbeddingToOpenAI(body, "text-embedding-004")
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, "list", openaiResp.Object)
	assert.Equal(t, "text-embedding-004", openaiResp.Model)
	assert.Len(t, openaiResp.Data, 2)
	assert.Equal(t, "embedding", openaiResp.Data[0].Object)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, openaiResp.Data[0].Embedding)
	assert.Equal(t, 0, openaiResp.Usage.PromptTokens) // Gemini doesn't return token counts by default
}

func TestGeminiEmbeddingToOpenAI_WithUsageMetadata(t *testing.T) {
	resp := GeminiEmbeddingResponse{
		Embeddings: []GeminiContentEmbedding{
			{Values: []float64{0.1, 0.2, 0.3}},
		},
		UsageMetadata: &GeminiEmbeddingUsage{
			PromptTokenCount: 5,
			TotalTokenCount:  5,
		},
	}
	body, err := json.Marshal(resp)
	require.NoError(t, err)

	result, err := GeminiEmbeddingToOpenAI(body, "text-embedding-004")
	require.NoError(t, err)

	var openaiResp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, 5, openaiResp.Usage.PromptTokens)
	assert.Equal(t, 5, openaiResp.Usage.TotalTokens)
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
