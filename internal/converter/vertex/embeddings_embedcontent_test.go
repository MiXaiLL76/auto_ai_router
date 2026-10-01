package vertex

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	testPNG = base64.StdEncoding.EncodeToString([]byte("\x89PNG fake image bytes"))
	testMP3 = base64.StdEncoding.EncodeToString([]byte("ID3 fake audio bytes"))
	testPDF = base64.StdEncoding.EncodeToString([]byte("%PDF-1.7 fake document"))
	testMP4 = base64.StdEncoding.EncodeToString([]byte("fake mp4 video bytes"))
)

// decodeVertexEmbedContent decodes one Vertex embedContent body into plain
// maps, i.e. the wire shape Google receives.
func decodeVertexEmbedContent(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var decoded map[string]any
	require.NoError(t, json.Unmarshal(body, &decoded))
	return decoded
}

func wireParts(t *testing.T, request map[string]any) []map[string]any {
	t.Helper()
	content, ok := request["content"].(map[string]any)
	require.True(t, ok, "content missing: %v", request)
	rawParts, ok := content["parts"].([]any)
	require.True(t, ok, "parts missing: %v", content)
	parts := make([]map[string]any, len(rawParts))
	for i, raw := range rawParts {
		parts[i] = raw.(map[string]any)
	}
	return parts
}

func requireValidationError(t *testing.T, err error, param string) {
	t.Helper()
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr), "want a 400 validation error, got %T: %v", err, err)
	if param != "" {
		assert.Equal(t, param, validationErr.Param)
	}
}

func TestIsEmbedContentModel(t *testing.T) {
	for model, want := range map[string]bool{
		"gemini-embedding-2":                true,
		"gemini-embedding-2-preview":        true,
		"google/gemini-embedding-2":         true,
		"GEMINI-EMBEDDING-2":                true,
		"gemini-embedding-001":              false,
		"google/gemini-embedding-001":       false,
		"gemini-embedding-exp-03-07":        false,
		"text-embedding-004":                false,
		"text-multilingual-embedding-002":   false,
		"gemini-3.6-flash":                  false,
		"multimodalembedding@001":           false,
		"vertex_ai/gemini-embedding-2":      true,
		" gemini-embedding-2 ":              true,
		"gemini-embedding-2-something-next": true,
	} {
		assert.Equal(t, want, IsEmbedContentModel(model), model)
	}
}

func TestOpenAIEmbeddingToVertexEmbedContent_StringKeepsOneCall(t *testing.T) {
	bodies, err := OpenAIEmbeddingToVertexEmbedContent([]byte(`{"model":"gemini-embedding-2","input":"What is the meaning of life?"}`))
	require.NoError(t, err)
	require.Len(t, bodies, 1)

	request := decodeVertexEmbedContent(t, bodies[0])
	assert.NotContains(t, request, "embedContentConfig", "no dimensions requested")
	assert.NotContains(t, request, "instances", "embedContent, not :predict")
	parts := wireParts(t, request)
	require.Len(t, parts, 1)
	assert.Equal(t, "What is the meaning of life?", parts[0]["text"])
}

func TestOpenAIEmbeddingToVertexEmbedContent_ArrayOfStringsOneCallPerInputInOrder(t *testing.T) {
	bodies, err := OpenAIEmbeddingToVertexEmbedContent([]byte(`{"input":["first","second","third"],"dimensions":768}`))
	require.NoError(t, err)
	require.Len(t, bodies, 3)
	for i, want := range []string{"first", "second", "third"} {
		request := decodeVertexEmbedContent(t, bodies[i])
		assert.Equal(t, want, wireParts(t, request)[0]["text"])
		assert.Equal(t, map[string]any{"outputDimensionality": float64(768)}, request["embedContentConfig"])
	}
}

func TestOpenAIEmbeddingToVertexEmbedContent_MixedInputIsOneAggregatedContent(t *testing.T) {
	body := fmt.Sprintf(`{"input":[[
		"An image of a dog",
		{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},
		{"type":"input_audio","input_audio":{"data":"%s","format":"mp3"}},
		{"type":"file","file":{"file_data":"data:application/pdf;base64,%s","filename":"doc.pdf"}},
		{"type":"video_url","video_url":{"url":"gs://bucket/clip.mp4"}}
	]]}`, testPNG, testMP3, testPDF)

	bodies, err := OpenAIEmbeddingToVertexEmbedContent([]byte(body))
	require.NoError(t, err)
	require.Len(t, bodies, 1, "an inner array is one vector")

	parts := wireParts(t, decodeVertexEmbedContent(t, bodies[0]))
	require.Len(t, parts, 5)
	assert.Equal(t, "An image of a dog", parts[0]["text"])
	assert.Equal(t, map[string]any{"mimeType": "image/png", "data": testPNG}, parts[1]["inlineData"])
	assert.Equal(t, map[string]any{"mimeType": "audio/mpeg", "data": testMP3}, parts[2]["inlineData"])
	assert.Equal(t, map[string]any{"mimeType": "application/pdf", "data": testPDF}, parts[3]["inlineData"])
	assert.Equal(t, map[string]any{"mimeType": "video/mp4", "fileUri": "gs://bucket/clip.mp4"}, parts[4]["fileData"])
}

func TestOpenAIEmbeddingToVertexEmbedContent_ItemShapes(t *testing.T) {
	body := fmt.Sprintf(`{"input":[
		"plain text",
		{"type":"image_url","image_url":"https://example.com/cat.jpg"},
		{"content":[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]},
		{"type":"input_file","file_url":"gs://bucket/report","mime_type":"application/pdf"},
		{"type":"video_url","video_url":{"url":"data:video/mp4;base64,%s"}}
	]}`, testPNG, testMP4)

	bodies, err := OpenAIEmbeddingToVertexEmbedContent([]byte(body))
	require.NoError(t, err)
	require.Len(t, bodies, 5, "one vector per top-level item")

	assert.Equal(t, "plain text", wireParts(t, decodeVertexEmbedContent(t, bodies[0]))[0]["text"])
	assert.Equal(t, map[string]any{"mimeType": "image/jpeg", "fileUri": "https://example.com/cat.jpg"},
		wireParts(t, decodeVertexEmbedContent(t, bodies[1]))[0]["fileData"])
	mixed := wireParts(t, decodeVertexEmbedContent(t, bodies[2]))
	require.Len(t, mixed, 2)
	assert.Equal(t, "caption", mixed[0]["text"])
	assert.Equal(t, "image/png", mixed[1]["inlineData"].(map[string]any)["mimeType"])
	assert.Equal(t, map[string]any{"mimeType": "application/pdf", "fileUri": "gs://bucket/report"},
		wireParts(t, decodeVertexEmbedContent(t, bodies[3]))[0]["fileData"])
	assert.Equal(t, map[string]any{"mimeType": "video/mp4", "data": testMP4},
		wireParts(t, decodeVertexEmbedContent(t, bodies[4]))[0]["inlineData"])
}

func TestOpenAIEmbeddingToVertexEmbedContent_BareBase64FileNeedsMIMEType(t *testing.T) {
	bodies, err := OpenAIEmbeddingToVertexEmbedContent([]byte(fmt.Sprintf(
		`{"input":{"type":"file","file":{"file_data":"%s","filename":"scan.pdf"}}}`, testPDF)))
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"mimeType": "application/pdf", "data": testPDF},
		wireParts(t, decodeVertexEmbedContent(t, bodies[0]))[0]["inlineData"])

	_, err = OpenAIEmbeddingToVertexEmbedContent([]byte(fmt.Sprintf(
		`{"input":{"type":"file","file":{"file_data":"%s"}}}`, testPDF)))
	requireValidationError(t, err, "input[0].file.file_data")
}

func TestOpenAIEmbeddingToVertexEmbedContent_RefusesWhatItCannotEmbed(t *testing.T) {
	tooMany := make([]string, MaxEmbedContentInputs+1)
	for i := range tooMany {
		tooMany[i] = `"x"`
	}
	for name, tc := range map[string]struct {
		body  string
		param string
	}{
		"missing input":           {`{"model":"gemini-embedding-2"}`, "input"},
		"empty array":             {`{"input":[]}`, "input"},
		"token ids":               {`{"input":[1,2,3]}`, "input[0]"},
		"nested token ids":        {`{"input":[[1,2,3]]}`, "input[0][0]"},
		"empty part list":         {`{"input":[[]]}`, "input[0]"},
		"unknown part type":       {`{"input":[{"type":"hologram"}]}`, "input[0].type"},
		"object without type":     {`{"input":[{"text":"hi"}]}`, "input[0]"},
		"text part without text":  {`{"input":[{"type":"text"}]}`, "input[0].text"},
		"image without mime":      {`{"input":[{"type":"image_url","image_url":{"url":"https://example.com/image"}}]}`, "input[0].image_url"},
		"private url":             {`{"input":[{"type":"image_url","image_url":{"url":"http://127.0.0.1/a.png"}}]}`, "input[0].image_url"},
		"file scheme":             {`{"input":[{"type":"image_url","image_url":{"url":"file:///etc/a.png"}}]}`, "input[0].image_url"},
		"empty data url":          {`{"input":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}`, "input[0].image_url"},
		"bad audio base64":        {`{"input":[{"type":"input_audio","input_audio":{"data":"not base64!","format":"wav"}}]}`, "input[0].input_audio.data"},
		"file without reference":  {`{"input":[{"type":"file","file":{"filename":"a.pdf"}}]}`, "input[0].file"},
		"zero dimensions":         {`{"input":"hi","dimensions":0}`, "dimensions"},
		"too many inputs":         {`{"input":[` + strings.Join(tooMany, ",") + `]}`, "input"},
		"mixed item bad element":  {`{"input":[["ok", true]]}`, "input[0][1]"},
		"content of wrong type":   {`{"input":[{"content":42}]}`, "input[0].content"},
		"top-level unknown value": {`{"input":true}`, "input[0]"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := OpenAIEmbeddingToVertexEmbedContent([]byte(tc.body))
			requireValidationError(t, err, tc.param)
		})
	}
}

func TestOpenAIEmbeddingToVertexEmbedContent_OversizedInlineDataIs413(t *testing.T) {
	withMaxBase64Size(t, 8)
	_, err := OpenAIEmbeddingToVertexEmbedContent([]byte(fmt.Sprintf(
		`{"input":{"type":"input_audio","input_audio":{"data":"%s","format":"mp3"}}}`, testMP3)))
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr), "got %v", err)
	assert.Equal(t, 413, validationErr.StatusCode)
}

// TestEmbedContent_DimensionsWrongTypeReportsParam is the embedContent-family
// counterpart of TestOpenAIEmbeddingToVertex_DimensionsWrongTypeReportsParam:
// a mistyped field is a 400 naming the param, not a generic 500.
func TestEmbedContent_DimensionsWrongTypeReportsParam(t *testing.T) {
	body := []byte(`{"model":"gemini-embedding-2","input":"hello","dimensions":"x"}`)

	_, err := OpenAIEmbeddingToVertexEmbedContent(body)
	testhelpers.RequireValidationError(t, err, "dimensions", "invalid_type")

	_, _, err = OpenAIEmbeddingToGeminiEmbedContent(body, "gemini-embedding-2")
	testhelpers.RequireValidationError(t, err, "dimensions", "invalid_type")
}

func TestEmbedContentFanOutEnvelope(t *testing.T) {
	envelope := EmbedContentFanOutEnvelope([][]byte{[]byte(`{"a":1}`), []byte(`{"b":2}`)})
	assert.JSONEq(t, `{"requests":[{"a":1},{"b":2}]}`, string(envelope))
}

func TestOpenAIEmbeddingToGeminiEmbedContent_SingleInputUsesEmbedContent(t *testing.T) {
	body := fmt.Sprintf(`{"input":[["An image of a dog",{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]],"dimensions":256}`, testPNG)
	converted, inputs, err := OpenAIEmbeddingToGeminiEmbedContent([]byte(body), "gemini-embedding-2")
	require.NoError(t, err)
	assert.Equal(t, 1, inputs)

	var request map[string]any
	require.NoError(t, json.Unmarshal(converted, &request))
	assert.NotContains(t, request, "requests", "one vector goes to :embedContent")
	assert.Equal(t, "models/gemini-embedding-2", request["model"])
	assert.Equal(t, float64(256), request["outputDimensionality"])
	parts := wireParts(t, request)
	require.Len(t, parts, 2)
	assert.Equal(t, "An image of a dog", parts[0]["text"])
	assert.Equal(t, map[string]any{"mimeType": "image/png", "data": testPNG}, parts[1]["inlineData"])
}

func TestOpenAIEmbeddingToGeminiEmbedContent_SeveralInputsUseBatch(t *testing.T) {
	body := fmt.Sprintf(`{"input":["task: classification | query: a dog",{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}`, testPNG)
	converted, inputs, err := OpenAIEmbeddingToGeminiEmbedContent([]byte(body), "gemini-embedding-2")
	require.NoError(t, err)
	assert.Equal(t, 2, inputs)

	var batch GeminiEmbeddingRequest
	require.NoError(t, json.Unmarshal(converted, &batch))
	require.Len(t, batch.Requests, 2)
	for _, request := range batch.Requests {
		assert.Equal(t, "models/gemini-embedding-2", request.Model)
		assert.Nil(t, request.OutputDimensionality)
	}
	assert.Equal(t, "task: classification | query: a dog", batch.Requests[0].Content.Parts[0].Text)
	require.NotNil(t, batch.Requests[1].Content.Parts[0].InlineData)
	assert.Equal(t, "image/png", batch.Requests[1].Content.Parts[0].InlineData.MIMEType)
}

func TestEmbedContentToOpenAI_SingleEmbedContentWithModalityUsage(t *testing.T) {
	// Vertex AI embedContent: text + one image, 42 + 258 tokens.
	body := `{
		"embedding": {"values": [0.25, -0.5, 1]},
		"usageMetadata": {
			"promptTokenCount": 300,
			"totalTokenCount": 300,
			"promptTokensDetails": [{"modality": "TEXT", "tokenCount": 42}, {"modality": "IMAGE", "tokenCount": 258}]
		},
		"truncated": false
	}`
	result, estimated, err := EmbedContentToOpenAI([]byte(body), "gemini-embedding-2", nil)
	require.NoError(t, err)
	assert.False(t, estimated)

	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	assert.Equal(t, "list", resp.Object)
	assert.Equal(t, "gemini-embedding-2", resp.Model)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, 0, resp.Data[0].Index)
	assert.Equal(t, []float64{0.25, -0.5, 1}, resp.Data[0].Embedding)
	assert.Equal(t, 300, resp.Usage.PromptTokens)
	assert.Equal(t, 300, resp.Usage.TotalTokens)
	assert.Equal(t, &openai.OpenAIEmbeddingPromptDetails{TextTokens: 42, ImageTokens: 258}, resp.Usage.PromptTokensDetails)
	assert.NotContains(t, string(result), "completion_tokens", "vectors are not output tokens")
}

func TestEmbedContentToOpenAI_GeminiBatchWithAudioVideoAndDocument(t *testing.T) {
	// Gemini API batchEmbedContents spells the breakdown promptTokenDetails.
	body := `{
		"embeddings": [{"values": [0.1]}, {"values": [0.2]}, {"values": [0.3]}],
		"usageMetadata": {
			"promptTokenCount": 1300,
			"promptTokenDetails": [
				{"modality": "AUDIO", "tokenCount": 250},
				{"modality": "VIDEO", "tokenCount": 660},
				{"modality": "DOCUMENT", "tokenCount": 258},
				{"modality": "TEXT", "tokenCount": 100}
			]
		}
	}`
	result, _, err := EmbedContentToOpenAI([]byte(body), "gemini-embedding-2", nil)
	require.NoError(t, err)

	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	require.Len(t, resp.Data, 3)
	for i := range resp.Data {
		assert.Equal(t, i, resp.Data[i].Index)
	}
	assert.Equal(t, []float64{0.3}, resp.Data[2].Embedding)
	assert.Equal(t, 1300, resp.Usage.PromptTokens)
	// 32 prompt tokens have no modality: they stay text, priced at the text rate.
	assert.Equal(t, &openai.OpenAIEmbeddingPromptDetails{TextTokens: 132, ImageTokens: 258, AudioTokens: 250, VideoTokens: 660},
		resp.Usage.PromptTokensDetails)
}

func TestEmbedContentToOpenAI_NoUsageFallsBackToTextEstimate(t *testing.T) {
	// A reply without usageMetadata must not bill $0: the text parts are
	// estimated at ~4 characters per token (28 + 4 characters -> 7 + 1), the
	// image cannot be sized and is left out, and the caller is told.
	request := `{"model":"gemini-embedding-2","input":["What is the meaning of life?",
		[{"type":"text","text":"abcd"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]]}`
	result, estimated, err := EmbedContentToOpenAI([]byte(`{"embeddings":[{"values":[0.1]},{"values":[0.2]}]}`), "gemini-embedding-2", []byte(request))
	require.NoError(t, err)
	assert.True(t, estimated)
	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	assert.Equal(t, 8, resp.Usage.PromptTokens)
	assert.Equal(t, 8, resp.Usage.TotalTokens)
	assert.Nil(t, resp.Usage.PromptTokensDetails)
}

func TestEmbedContentToOpenAI_NoUsageAndNoRequestIsZeroButFlagged(t *testing.T) {
	result, estimated, err := EmbedContentToOpenAI([]byte(`{"embedding":{"values":[0.1,0.2]}}`), "gemini-embedding-2", nil)
	require.NoError(t, err)
	assert.True(t, estimated, "the caller still learns the reply had no usage")
	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	assert.Equal(t, 0, resp.Usage.PromptTokens)
}

func TestEmbedContentToOpenAI_DetailsAboveCountRaisePromptTokens(t *testing.T) {
	result, _, err := EmbedContentToOpenAI([]byte(`{"embedding":{"values":[1]},"usageMetadata":{"promptTokensDetails":[{"modality":"IMAGE","tokenCount":258}]}}`), "gemini-embedding-2", nil)
	require.NoError(t, err)
	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	assert.Equal(t, 258, resp.Usage.PromptTokens)
	assert.Equal(t, &openai.OpenAIEmbeddingPromptDetails{ImageTokens: 258}, resp.Usage.PromptTokensDetails)
}

func TestEmbedContentToOpenAI_RejectsResponseWithoutEmbeddings(t *testing.T) {
	_, _, err := EmbedContentToOpenAI([]byte(`{"usageMetadata":{"promptTokenCount":3}}`), "gemini-embedding-2", nil)
	require.Error(t, err)
}

func TestMergeEmbedContentResponses_KeepsOrderAndSumsUsage(t *testing.T) {
	merged, err := MergeEmbedContentResponses([][]byte{
		[]byte(`{"embedding":{"values":[1,1]},"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5,"promptTokensDetails":[{"modality":"TEXT","tokenCount":5}]}}`),
		[]byte(`{"embedding":{"values":[2,2]},"usageMetadata":{"promptTokenCount":263,"totalTokenCount":263,"promptTokensDetails":[{"modality":"TEXT","tokenCount":5},{"modality":"IMAGE","tokenCount":258}]}}`),
		[]byte(`{"embedding":{"values":[3,3]},"usageMetadata":{"promptTokenCount":75,"totalTokenCount":75,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":75}]}}`),
	})
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"embeddings": [{"values":[1,1]},{"values":[2,2]},{"values":[3,3]}],
		"usageMetadata": {
			"promptTokenCount": 343,
			"totalTokenCount": 343,
			"promptTokensDetails": [
				{"modality":"TEXT","tokenCount":10},
				{"modality":"IMAGE","tokenCount":258},
				{"modality":"AUDIO","tokenCount":75}
			]
		}
	}`, string(merged))

	result, _, err := EmbedContentToOpenAI(merged, "gemini-embedding-2", nil)
	require.NoError(t, err)
	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	require.Len(t, resp.Data, 3)
	assert.Equal(t, []float64{2, 2}, resp.Data[1].Embedding)
	assert.Equal(t, 2, resp.Data[2].Index)
	assert.Equal(t, 343, resp.Usage.PromptTokens)
	assert.Equal(t, &openai.OpenAIEmbeddingPromptDetails{TextTokens: 10, ImageTokens: 258, AudioTokens: 75}, resp.Usage.PromptTokensDetails)
}

func TestMergeEmbedContentResponses_RejectsReplyWithoutEmbedding(t *testing.T) {
	_, err := MergeEmbedContentResponses([][]byte{[]byte(`{"embedding":{"values":[1]}}`), []byte(`{"usageMetadata":{}}`)})
	require.Error(t, err)
}

func TestGeminiEmbeddingToOpenAI_PrefersUsageMetadataOverEstimate(t *testing.T) {
	body := `{"embeddings":[{"values":[0.1]},{"values":[0.2]}],"usageMetadata":{"promptTokenCount":9}}`
	result, err := GeminiEmbeddingToOpenAI([]byte(body), "gemini-embedding-001", []string{"a much longer text than nine tokens would suggest at all", "b"})
	require.NoError(t, err)
	var resp openai.OpenAIEmbeddingResponse
	require.NoError(t, json.Unmarshal(result, &resp))
	assert.Equal(t, 9, resp.Usage.PromptTokens)
	assert.Equal(t, 9, resp.Usage.TotalTokens)
}

func TestBuildVertexEmbedContentURL(t *testing.T) {
	assert.Equal(t,
		"https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/gemini-embedding-2:embedContent",
		BuildVertexEmbedContentURL(&config.CredentialConfig{ProjectID: "p", Location: "global"}, "gemini-embedding-2"))
	assert.Equal(t,
		"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/publishers/google/models/gemini-embedding-2:embedContent",
		BuildVertexEmbedContentURL(&config.CredentialConfig{ProjectID: "p", Location: "us-central1"}, "gemini-embedding-2"))
}

func TestBuildGeminiEmbedContentURL(t *testing.T) {
	cred := &config.CredentialConfig{BaseURL: "https://generativelanguage.googleapis.com/"}
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent",
		BuildGeminiEmbedContentURL(cred, "gemini-embedding-2", 1))
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents",
		BuildGeminiEmbedContentURL(cred, "gemini-embedding-2", 3))
}
