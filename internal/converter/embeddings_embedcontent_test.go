package converter

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func embeddingsConverter(providerType config.ProviderType, model string) *ProviderConverter {
	return New(providerType, RequestMode{IsEmbeddings: true, ModelID: model, DisplayModelID: "google/" + model})
}

func TestProviderConverter_VertexGeminiEmbedding2FansOutOneCallPerInput(t *testing.T) {
	conv := embeddingsConverter(config.ProviderTypeVertexAI, "gemini-embedding-2")
	body, err := conv.RequestFrom([]byte(`{"model":"google/gemini-embedding-2","input":["first",["second","third"]],"dimensions":768}`))
	require.NoError(t, err)

	fanOut := conv.EmbeddingFanOutBodies()
	require.Len(t, fanOut, 2)
	assert.JSONEq(t, `{"content":{"parts":[{"text":"first"}]},"embedContentConfig":{"outputDimensionality":768}}`, string(fanOut[0]))
	assert.JSONEq(t, `{"content":{"parts":[{"text":"second"},{"text":"third"}]},"embedContentConfig":{"outputDimensionality":768}}`, string(fanOut[1]))
	assert.JSONEq(t, `{"requests":[`+string(fanOut[0])+`,`+string(fanOut[1])+`]}`, string(body), "logged as one envelope")

	cred := &config.CredentialConfig{ProjectID: "grant", Location: "global"}
	assert.Equal(t, "https://aiplatform.googleapis.com/v1beta1/projects/grant/locations/global/publishers/google/models/gemini-embedding-2:embedContent", conv.BuildURL(cred))

	merged, err := vertex.MergeEmbedContentResponses([][]byte{
		[]byte(`{"embedding":{"values":[0.1]},"usageMetadata":{"promptTokenCount":1,"promptTokensDetails":[{"modality":"TEXT","tokenCount":1}]}}`),
		[]byte(`{"embedding":{"values":[0.2]},"usageMetadata":{"promptTokenCount":2,"promptTokensDetails":[{"modality":"TEXT","tokenCount":2}]}}`),
	})
	require.NoError(t, err)
	openAIBody, err := conv.ResponseTo(merged)
	require.NoError(t, err)
	var resp struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(openAIBody, &resp))
	require.Len(t, resp.Data, 2)
	assert.Equal(t, []float64{0.2}, resp.Data[1].Embedding)
	assert.Equal(t, 1, resp.Data[1].Index)

	usage := conv.UsageFromResponse(openAIBody)
	require.NotNil(t, usage)
	assert.Equal(t, 3, usage.PromptTokens)
	assert.Equal(t, 0, usage.CompletionTokens)
}

func TestProviderConverter_VertexGeminiEmbedding2SingleInputIsOneCall(t *testing.T) {
	conv := embeddingsConverter(config.ProviderTypeVertexAI, "gemini-embedding-2")
	body, err := conv.RequestFrom([]byte(`{"input":"hello"}`))
	require.NoError(t, err)
	assert.Nil(t, conv.EmbeddingFanOutBodies())
	assert.JSONEq(t, `{"content":{"parts":[{"text":"hello"}]}}`, string(body))
}

func TestProviderConverter_GeminiAPIGeminiEmbedding2PicksMethodByInputCount(t *testing.T) {
	cred := &config.CredentialConfig{BaseURL: "https://generativelanguage.googleapis.com"}

	single := embeddingsConverter(config.ProviderTypeGemini, "gemini-embedding-2")
	body, err := single.RequestFrom([]byte(`{"input":"hello"}`))
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"hello"}]}}`, string(body))
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent", single.BuildURL(cred))
	assert.Nil(t, single.EmbeddingFanOutBodies(), "the Gemini API batches natively")

	batch := embeddingsConverter(config.ProviderTypeGemini, "gemini-embedding-2")
	_, err = batch.RequestFrom([]byte(`{"input":["a","b"]}`))
	require.NoError(t, err)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents", batch.BuildURL(cred))

	openAIBody, err := single.ResponseTo([]byte(`{"embedding":{"values":[0.5]},"usageMetadata":{"promptTokenCount":4}}`))
	require.NoError(t, err)
	usage := single.UsageFromResponse(openAIBody)
	require.NotNil(t, usage)
	assert.Equal(t, 4, usage.PromptTokens)
	assert.False(t, single.EmbeddingUsageEstimated())
}

func TestProviderConverter_GeminiEmbedding2ReplyWithoutUsageIsEstimatedAndFlagged(t *testing.T) {
	conv := embeddingsConverter(config.ProviderTypeVertexAI, "gemini-embedding-2")
	_, err := conv.RequestFrom([]byte(`{"input":["What is the meaning of life?","abcd"]}`))
	require.NoError(t, err)

	openAIBody, err := conv.ResponseTo([]byte(`{"embeddings":[{"values":[1]},{"values":[2]}]}`))
	require.NoError(t, err)
	assert.True(t, conv.EmbeddingUsageEstimated())
	usage := conv.UsageFromResponse(openAIBody)
	require.NotNil(t, usage, "a reply without usage must not bill $0")
	assert.Equal(t, 8, usage.PromptTokens)
}

func TestProviderConverter_GeminiEmbedding001KeepsLegacyPaths(t *testing.T) {
	vertexConv := embeddingsConverter(config.ProviderTypeVertexAI, "gemini-embedding-001")
	body, err := vertexConv.RequestFrom([]byte(`{"input":["a","b"]}`))
	require.NoError(t, err)
	assert.JSONEq(t, `{"instances":[{"content":"a"},{"content":"b"}]}`, string(body))
	assert.Nil(t, vertexConv.EmbeddingFanOutBodies())
	assert.Equal(t, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/grant/locations/us-central1/publishers/google/models/gemini-embedding-001:predict",
		vertexConv.BuildURL(&config.CredentialConfig{ProjectID: "grant", Location: "us-central1"}))

	geminiConv := embeddingsConverter(config.ProviderTypeGemini, "gemini-embedding-001")
	_, err = geminiConv.RequestFrom([]byte(`{"input":"one"}`))
	require.NoError(t, err)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents",
		geminiConv.BuildURL(&config.CredentialConfig{BaseURL: "https://generativelanguage.googleapis.com"}))

	// A text-only model still refuses multimodal input, now as a 400.
	_, err = geminiConv.RequestFrom([]byte(`{"input":[{"type":"text","text":"x"}]}`))
	require.Error(t, err)
}

func TestExtractTokenUsage_EmbeddingModalityBreakdown(t *testing.T) {
	usage := ExtractTokenUsageWithOptions([]byte(`{
		"object":"list",
		"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
		"usage":{"prompt_tokens":1300,"total_tokens":1300,
			"prompt_tokens_details":{"text_tokens":132,"image_tokens":258,"audio_tokens":250,"video_tokens":660}}
	}`), TokenUsageExtractionOptions{})
	require.NotNil(t, usage)
	assert.Equal(t, 1300, usage.PromptTokens)
	assert.Equal(t, 0, usage.CompletionTokens)
	assert.Equal(t, 258, usage.ImageTokens)
	assert.Equal(t, 250, usage.AudioInputTokens)
	assert.Equal(t, 660, usage.VideoInputTokens)
}

func TestExtractTokenUsage_ResponsesInputVideoTokens(t *testing.T) {
	usage := ExtractTokenUsage([]byte(`{"usage":{"input_tokens":100,"output_tokens":5,"input_tokens_details":{"image_tokens":20,"video_tokens":30}}}`))
	require.NotNil(t, usage)
	assert.Equal(t, 20, usage.ImageTokens)
	assert.Equal(t, 30, usage.VideoInputTokens)
}

func TestTokenUsage_MergeNonZeroKeepsVideoTokens(t *testing.T) {
	usage := &TokenUsage{PromptTokens: 10, VideoInputTokens: 7}
	usage.MergeNonZero(&TokenUsage{PromptTokens: 12})
	assert.Equal(t, 7, usage.VideoInputTokens)
	usage.MergeNonZero(&TokenUsage{VideoInputTokens: 9})
	assert.Equal(t, 9, usage.VideoInputTokens)
}
