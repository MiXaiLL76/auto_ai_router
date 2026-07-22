package vertexresponses

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestCandidatesToOutputItems_InlineImageBecomesImageGenerationCall(t *testing.T) {
	imageBytes := []byte("generated-png")
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Here is the image."},
						{InlineData: &genai.Blob{MIMEType: "image/png", Data: imageBytes}},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	output := candidatesToOutputItems(vertexResp)
	require.Len(t, output, 2)
	assert.Equal(t, "message", output[0].Type)
	assert.Equal(t, "Here is the image.", output[0].Content[0].Text)

	imageCall := output[1]
	assert.Equal(t, "image_generation_call", imageCall.Type)
	assert.Equal(t, "completed", imageCall.Status)
	assert.NotEmpty(t, imageCall.ID)
	assert.Equal(t, base64.StdEncoding.EncodeToString(imageBytes), imageCall.Result)
	assert.Equal(t, "png", imageCall.OutputFormat)
}

func TestUsageMetadataToUsage_PreservesImageOutputTokens(t *testing.T) {
	meta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     22,
		CandidatesTokenCount: 1127,
		TotalTokenCount:      1149,
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: 7},
			{Modality: genai.MediaModalityImage, TokenCount: 1120},
		},
	}

	usage := usageMetadataToUsage(meta)
	require.NotNil(t, usage)
	assert.Equal(t, 22, usage.InputTokens)
	assert.Equal(t, 1127, usage.OutputTokens)
	assert.Equal(t, 1120, usage.OutputTokensDetails.ImageTokens)

	raw, err := json.Marshal(usage)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"input_tokens": 22,
		"output_tokens": 1127,
		"total_tokens": 1149,
		"input_tokens_details": {"cached_tokens": 0},
		"output_tokens_details": {"reasoning_tokens": 0, "image_tokens": 1120}
	}`, string(raw))
}

func TestVertexToResponsesResponse_ImageGenerationRoundTrip(t *testing.T) {
	providerBody := `{
		"candidates": [{
			"content": {"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {
			"promptTokenCount": 22,
			"candidatesTokenCount": 1120,
			"totalTokenCount": 1142,
			"candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}]
		}
	}`

	resp, err := VertexToResponsesResponse([]byte(providerBody), "gemini-3.1-flash-image-preview", "resp_test", 123)
	require.NoError(t, err)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "image_generation_call", resp.Output[0].Type)
	assert.Equal(t, "aW1hZ2U=", resp.Output[0].Result)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 1120, resp.Usage.OutputTokensDetails.ImageTokens)
}

func TestCandidatesToOutputItems_CodeInterpreter_CodeAndResult(t *testing.T) {
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{ExecutableCode: &genai.ExecutableCode{Code: "print(1+1)", Language: "PYTHON"}},
						{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: "OUTCOME_OK", Output: "2\n"}},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	output := candidatesToOutputItems(vertexResp)
	require.Len(t, output, 1)

	ci := output[0]
	assert.Equal(t, "code_interpreter_call", ci.Type)
	assert.Equal(t, "completed", ci.Status)
	assert.Equal(t, "print(1+1)", ci.Code)

	outputs, ok := ci.Outputs.([]map[string]interface{})
	require.True(t, ok, "Outputs should be []map[string]interface{}")
	require.Len(t, outputs, 1)
	assert.Equal(t, "text", outputs[0]["type"])
	assert.Equal(t, "2\n", outputs[0]["text"])
}

func TestCandidatesToOutputItems_CodeInterpreter_CodeOnly(t *testing.T) {
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{ExecutableCode: &genai.ExecutableCode{Code: "x = 42"}},
					},
				},
			},
		},
	}

	output := candidatesToOutputItems(vertexResp)
	require.Len(t, output, 1)
	assert.Equal(t, "code_interpreter_call", output[0].Type)
	assert.Equal(t, "x = 42", output[0].Code)
	assert.Nil(t, output[0].Outputs, "no result yet — Outputs should be nil")
}

func TestCandidatesToOutputItems_CodeInterpreter_InterleavedWithText(t *testing.T) {
	// Gemini may return text, then code, then result, then more text.
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Let me calculate:"},
						{ExecutableCode: &genai.ExecutableCode{Code: "2+2"}},
						{CodeExecutionResult: &genai.CodeExecutionResult{Output: "4"}},
						{Text: "The answer is 4."},
					},
				},
			},
		},
	}

	output := candidatesToOutputItems(vertexResp)
	// message("Let me calculate:"), code_interpreter_call, message("The answer is 4.")
	require.Len(t, output, 3)

	assert.Equal(t, "message", output[0].Type)
	assert.Equal(t, "Let me calculate:", output[0].Content[0].Text)

	assert.Equal(t, "code_interpreter_call", output[1].Type)
	assert.Equal(t, "2+2", output[1].Code)
	outputs := output[1].Outputs.([]map[string]interface{})
	assert.Equal(t, "4", outputs[0]["text"])

	assert.Equal(t, "message", output[2].Type)
	assert.Equal(t, "The answer is 4.", output[2].Content[0].Text)
}

func TestCandidatesToOutputItems_GroundingMetadataAddsAnnotationsAndWebSearchCall(t *testing.T) {
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Paris is the capital of France."},
					},
				},
				GroundingMetadata: &genai.GroundingMetadata{
					WebSearchQueries: []string{"capital of france"},
					GroundingChunks: []*genai.GroundingChunk{
						{
							Web: &genai.GroundingChunkWeb{
								URI:   "https://example.com/paris",
								Title: "Paris reference",
							},
						},
					},
					GroundingSupports: []*genai.GroundingSupport{
						{
							GroundingChunkIndices: []int32{0},
							Segment: &genai.Segment{
								PartIndex:  0,
								StartIndex: 0,
								EndIndex:   5,
							},
						},
					},
				},
			},
		},
	}

	output := candidatesToOutputItems(vertexResp)
	require.Len(t, output, 2)

	require.Equal(t, "message", output[0].Type)
	require.Len(t, output[0].Content, 1)
	assert.Equal(t, "output_text", output[0].Content[0].Type)
	assert.Equal(t, "Paris is the capital of France.", output[0].Content[0].Text)
	require.Len(t, output[0].Content[0].Annotations, 1)
	assert.Equal(t, "url_citation", output[0].Content[0].Annotations[0].Type)
	assert.Equal(t, "https://example.com/paris", output[0].Content[0].Annotations[0].URL)
	assert.Equal(t, "Paris reference", output[0].Content[0].Annotations[0].Title)
	assert.Equal(t, 0, output[0].Content[0].Annotations[0].StartIndex)
	assert.Equal(t, 5, output[0].Content[0].Annotations[0].EndIndex)

	assert.Equal(t, "web_search_call", output[1].Type)
	assert.Equal(t, []string{"capital of france"}, output[1].Queries)
}

func TestVertexToResponsesResponse_RequiredSchemaFields(t *testing.T) {
	vertexResp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "call_1",
								Name: "get_weather",
								Args: map[string]any{"location": "Paris"},
							},
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	resp := buildResponsesResponse(vertexResp, "gemini-test", "resp_test", 123)

	assert.Equal(t, "auto", resp.ToolChoice)
	assert.Equal(t, "disabled", resp.Truncation)
	assert.Equal(t, "default", resp.ServiceTier)
	require.NotNil(t, resp.Temperature)
	assert.Equal(t, 1.0, *resp.Temperature)
	require.NotNil(t, resp.TopP)
	assert.Equal(t, 1.0, *resp.TopP)
	require.NotNil(t, resp.Text)

	raw, err := json.Marshal(resp)
	require.NoError(t, err)
	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &parsed))
	assert.Equal(t, "auto", parsed["tool_choice"])
	assert.Equal(t, "disabled", parsed["truncation"])
	assert.Equal(t, "default", parsed["service_tier"])
	assert.Equal(t, float64(1), parsed["temperature"])
	assert.Equal(t, float64(1), parsed["top_p"])
	_, hasText := parsed["text"]
	assert.True(t, hasText)
}
