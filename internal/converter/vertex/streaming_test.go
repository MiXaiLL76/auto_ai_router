package vertex

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestTransformVertexStreamToOpenAI_PreservesGroundedSearchUsage(t *testing.T) {
	chunk := map[string]interface{}{
		"candidates": []map[string]interface{}{{
			"content": map[string]interface{}{
				"role":  "model",
				"parts": []map[string]interface{}{{"text": "grounded"}},
			},
			"groundingMetadata": map[string]interface{}{
				"webSearchQueries": []string{"query one", "query two"},
			},
			"finishReason": "STOP",
		}},
		"usageMetadata": map[string]interface{}{
			"promptTokenCount":     5,
			"candidatesTokenCount": 2,
			"totalTokenCount":      7,
		},
	}
	data, err := json.Marshal(chunk)
	require.NoError(t, err)
	var out bytes.Buffer
	require.NoError(t, TransformVertexStreamToOpenAI(
		strings.NewReader("data: "+string(data)+"\n\ndata: [DONE]\n\n"),
		"gemini-test",
		&out,
	))

	assert.Contains(t, out.String(), `"server_tool_use":{"web_search_requests":2}`)
}

func TestTransformVertexStreamToOpenAI_AccumulatesDistinctSearchesAcrossChunks(t *testing.T) {
	makeChunk := func(query string, includeUsage bool) string {
		chunk := map[string]interface{}{
			"candidates": []map[string]interface{}{{
				"content": map[string]interface{}{
					"role":  "model",
					"parts": []map[string]interface{}{{"text": query}},
				},
				"groundingMetadata": map[string]interface{}{
					"webSearchQueries": []string{query},
				},
			}},
		}
		if includeUsage {
			chunk["usageMetadata"] = map[string]interface{}{
				"promptTokenCount":     5,
				"candidatesTokenCount": 2,
				"totalTokenCount":      7,
			}
		}
		data, err := json.Marshal(chunk)
		require.NoError(t, err)
		return "data: " + string(data) + "\n\n"
	}

	stream := makeChunk("query one", false) +
		makeChunk("query two", false) +
		makeChunk("query one", true) +
		"data: [DONE]\n\n"

	var out bytes.Buffer
	require.NoError(t, TransformVertexStreamToOpenAI(
		strings.NewReader(stream),
		"gemini-test",
		&out,
	))

	assert.Contains(t, out.String(), `"server_tool_use":{"web_search_requests":2}`)
}

func TestTransformVertexStreamToOpenAI_PreservesLargeInlineImage(t *testing.T) {
	imageBytes := bytes.Repeat([]byte{0xab}, 900*1024) // base64 makes the SSE line exceed 1 MiB
	chunk := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				InlineData: &genai.Blob{MIMEType: "image/png", Data: imageBytes},
			}}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 1120,
			CandidatesTokensDetails: []*genai.ModalityTokenCount{{
				Modality: genai.MediaModalityImage, TokenCount: 1120,
			}},
		},
	}
	data, err := json.Marshal(chunk)
	require.NoError(t, err)
	require.Greater(t, len(data), 1024*1024)

	var out bytes.Buffer
	require.NoError(t, TransformVertexStreamToOpenAI(
		strings.NewReader("data: "+string(data)+"\n\ndata: [DONE]\n\n"),
		"gemini-image",
		&out,
	))

	line := strings.Split(out.String(), "\n")[0]
	var converted openai.OpenAIStreamingChunk
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &converted))
	require.Len(t, converted.Choices, 1)
	require.Len(t, converted.Choices[0].Delta.Images, 1)
	require.NotNil(t, converted.Choices[0].Delta.Images[0].ImageURL)
	assert.Equal(t,
		"data:image/png;base64,"+base64.StdEncoding.EncodeToString(imageBytes),
		converted.Choices[0].Delta.Images[0].ImageURL.URL,
	)
	require.NotNil(t, converted.Usage)
	require.NotNil(t, converted.Usage.CompletionTokensDetails)
	assert.Equal(t, 1120, converted.Usage.CompletionTokensDetails.ImageTokens)
}

func TestConvertVertexFunctionCallToStreamingOpenAI(t *testing.T) {
	t.Run("valid function call with name and args", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "get_weather",
			Args: map[string]interface{}{
				"city": "Tokyo",
			},
		}
		thoughtSig := []byte("test-signature")

		result := convertVertexFunctionCallToStreamingOpenAI(fc, thoughtSig, 0)

		assert.Equal(t, 0, result.Index)
		assert.Equal(t, "function", result.Type)
		assert.NotEmpty(t, result.ID)
		require.NotNil(t, result.Function)
		assert.Equal(t, "get_weather", result.Function.Name)
		assert.Contains(t, result.Function.Arguments, "Tokyo")

		// Check thoughtSignature is preserved in provider_specific_fields
		require.NotNil(t, result.ProviderSpecificFields)
		assert.NotEmpty(t, result.ProviderSpecificFields["thought_signature"])
	})

	t.Run("function call with nil args produces empty JSON object", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "no_args_fn",
			Args: nil,
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 1)

		assert.Equal(t, 1, result.Index)
		require.NotNil(t, result.Function)
		assert.Equal(t, "no_args_fn", result.Function.Name)
		assert.Equal(t, "{}", result.Function.Arguments)
	})

	t.Run("nil thought signature sets skip flag", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "test_fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 0)

		require.NotNil(t, result.ProviderSpecificFields)
		assert.Equal(t, true, result.ProviderSpecificFields["skip_thought_signature_validator"])
	})

	t.Run("empty thought signature sets skip flag", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "test_fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, []byte{}, 0)

		require.NotNil(t, result.ProviderSpecificFields)
		assert.Equal(t, true, result.ProviderSpecificFields["skip_thought_signature_validator"])
	})

	t.Run("index is preserved", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 5)
		assert.Equal(t, 5, result.Index)
	})
}

// TestTransformVertexStreamToOpenAI_ToolCallsFinishInLaterChunk: Gemini 3 may
// stream a function call in one chunk and its STOP in a later chunk that carries
// no call (empty text, a thought signature). The choice must still finish with
// "tool_calls", or an agent loop stops instead of running the tool.
func TestTransformVertexStreamToOpenAI_ToolCallsFinishInLaterChunk(t *testing.T) {
	finishReasons := func(stream string) []string {
		var out bytes.Buffer
		require.NoError(t, TransformVertexStreamToOpenAI(strings.NewReader(stream), "gemini-3-flash-preview", &out))
		var reasons []string
		for _, line := range strings.Split(out.String(), "\n") {
			payload, ok := strings.CutPrefix(line, "data: ")
			if !ok || payload == "[DONE]" {
				continue
			}
			var chunk openai.OpenAIStreamingChunk
			require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
			for _, choice := range chunk.Choices {
				if choice.FinishReason != nil {
					reasons = append(reasons, *choice.FinishReason)
				}
			}
		}
		return reasons
	}
	call := `data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]}}]}` + "\n\n"
	stop := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"c2ln"}]},"finishReason":"STOP"}]}` + "\n\n"
	text := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Sunny"}]}}]}` + "\n\n"

	assert.Equal(t, []string{"tool_calls"}, finishReasons(call+stop), "call and STOP in different chunks")
	assert.Equal(t, []string{"stop"}, finishReasons(text+stop), "no call: STOP stays stop")

	// Candidates are tracked apart: only the one that called a tool finishes with tool_calls.
	twoCandidates := `data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Sunny"}]}},` +
		`{"index":1,"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{}}}]}}]}` + "\n\n" +
		`data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"},` +
		`{"index":1,"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}]}` + "\n\n"
	assert.Equal(t, []string{"stop", "tool_calls"}, finishReasons(twoCandidates))
}
