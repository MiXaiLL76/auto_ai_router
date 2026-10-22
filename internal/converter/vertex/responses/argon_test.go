package vertexresponses

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

const argon = "gemini-4-argon"

func argonVertexRequest(t *testing.T, body string) *vertex.VertexRequest {
	t.Helper()
	out, err := ResponsesRequestToVertex([]byte(body), argon)
	require.NoError(t, err)
	var req vertex.VertexRequest
	require.NoError(t, json.Unmarshal(out, &req))
	return &req
}

func TestArgonResponsesThinkingLevels(t *testing.T) {
	for effort, want := range map[string]genai.ThinkingLevel{
		"":        genai.ThinkingLevelMedium,
		"minimal": genai.ThinkingLevelLow,
		"none":    genai.ThinkingLevelLow,
		"low":     genai.ThinkingLevelLow,
		"medium":  genai.ThinkingLevelMedium,
		"high":    genai.ThinkingLevelHigh,
		"xhigh":   genai.ThinkingLevelHigh,
	} {
		t.Run("effort "+effort, func(t *testing.T) {
			reasoning := ""
			if effort != "" {
				reasoning = `"reasoning": {"effort": "` + effort + `"}, `
			}
			req := argonVertexRequest(t, `{"model": "gemini-4-argon", `+reasoning+`"temperature": 0.3, "input": "hi"}`)
			tc := req.GenerationConfig.ThinkingConfig
			require.NotNil(t, tc)
			assert.Equal(t, want, tc.ThinkingLevel)
			assert.Nil(t, tc.ThinkingBudget)
			assert.Nil(t, req.GenerationConfig.Temperature, "temperature has no effect on this model")
		})
	}
}

func TestArgonResponsesRejectsPrefillAndOversizedOutput(t *testing.T) {
	_, err := ResponsesRequestToVertex([]byte(`{"model": "gemini-4-argon", "input": [
		{"role": "user", "content": "Write a poem"},
		{"role": "assistant", "content": "Roses are"}]}`), argon)
	testhelpers.RequireValidationError(t, err, "input", "invalid_value")

	_, err = ResponsesRequestToVertex([]byte(`{"model": "gemini-4-argon", "max_output_tokens": 1048577, "input": "hi"}`), argon)
	testhelpers.RequireValidationError(t, err, "max_output_tokens", "invalid_value")

	req := argonVertexRequest(t, `{"model": "gemini-4-argon", "max_output_tokens": 262144, "input": "hi"}`)
	assert.Equal(t, int32(262144), req.GenerationConfig.MaxOutputTokens)
}

// A full Responses tool loop with two functions: the call IDs and the thought
// signature come back through call_id, the calls of the turn make one model
// content, each result gets its function's real name and the results go in the
// order of the calls.
func TestArgonResponsesToolLoop(t *testing.T) {
	turn1, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-weather", Name: "get_weather", Args: map[string]any{"city": "Moscow"}},
					ThoughtSignature: []byte("signature/1+")},
				{FunctionCall: &genai.FunctionCall{ID: "fc-time", Name: "get_time", Args: map[string]any{}}},
			}},
			FinishReason: genai.FinishReasonStop,
		}},
	})
	require.NoError(t, err)
	resp, err := VertexToResponsesResponse(turn1, argon, "", 0)
	require.NoError(t, err)
	var calls []responses.OutputItem
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			calls = append(calls, item)
		}
	}
	require.Len(t, calls, 2)
	assert.Equal(t, "fc-weather__thought__"+base64.RawURLEncoding.EncodeToString([]byte("signature/1+")), calls[0].CallID)
	assert.Equal(t, "fc-time", calls[1].CallID)

	// Turn 2: the calls go back as returned, the results without names and the
	// second one first, with a reasoning item in between.
	input, err := json.Marshal([]map[string]any{
		{"role": "user", "content": "Weather and time in Moscow?"},
		{"type": "reasoning", "summary": []any{}},
		{"type": "function_call", "call_id": calls[0].CallID, "name": calls[0].Name, "arguments": calls[0].Arguments},
		{"type": "function_call", "call_id": calls[1].CallID, "name": calls[1].Name, "arguments": calls[1].Arguments},
		{"type": "function_call_output", "call_id": calls[1].CallID, "output": "15:00"},
		{"type": "function_call_output", "call_id": calls[0].CallID, "output": "20C"},
	})
	require.NoError(t, err)
	req := argonVertexRequest(t, `{"model": "gemini-4-argon", "input": `+string(input)+`}`)
	require.Len(t, req.Contents, 3)

	model := req.Contents[1]
	assert.Equal(t, "model", model.Role)
	require.Len(t, model.Parts, 2, "one model turn with both calls")
	assert.Equal(t, "fc-weather", model.Parts[0].FunctionCall.ID)
	assert.Equal(t, []byte("signature/1+"), model.Parts[0].ThoughtSignature)
	assert.Equal(t, "fc-time", model.Parts[1].FunctionCall.ID)
	assert.Equal(t, []byte("skip_thought_signature_validator"), model.Parts[1].ThoughtSignature)

	results := req.Contents[2]
	require.Len(t, results.Parts, 2)
	assert.Equal(t, "fc-weather", results.Parts[0].FunctionResponse.ID)
	assert.Equal(t, "get_weather", results.Parts[0].FunctionResponse.Name)
	assert.Equal(t, "fc-time", results.Parts[1].FunctionResponse.ID)
	assert.Equal(t, "get_time", results.Parts[1].FunctionResponse.Name, "the name comes from the call, not from call_id")
}

func TestDecodeCallID(t *testing.T) {
	id, signature := decodeCallID("call_abc")
	assert.Equal(t, "call_abc", id)
	assert.Nil(t, signature)

	id, signature = decodeCallID("fc-1__thought__" + base64.StdEncoding.EncodeToString([]byte("sig?>")))
	assert.Equal(t, "fc-1", id)
	assert.Equal(t, []byte("sig?>"), signature, "standard base64 is read too")
}

func TestArgonResponsesMergedContinuation(t *testing.T) {
	first, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:           &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "Part one. "}}},
			FinishReason:      genai.FinishReasonContinuation,
			ContinuationToken: []byte("tok"),
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 800, CandidatesTokenCount: 30000, ThoughtsTokenCount: 2768},
	})
	require.NoError(t, err)
	second, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "Part two."}}},
			FinishReason: genai.FinishReasonStop,
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 33568, CachedContentTokenCount: 33000, CandidatesTokenCount: 100},
	})
	require.NoError(t, err)
	merged, err := vertex.MergeContinuationResponses([][]byte{first, second})
	require.NoError(t, err)

	resp, err := VertexToResponsesResponse(merged, argon, "", 0)
	require.NoError(t, err)
	assert.Equal(t, "completed", resp.Status)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 800+33568, resp.Usage.InputTokens)
	assert.Equal(t, 30000+2768+100, resp.Usage.OutputTokens)
	assert.Equal(t, 2768, resp.Usage.OutputTokensDetails.ReasoningTokens)
	assert.Equal(t, 33000, resp.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, 800, resp.Usage.InputTokensDetails.InitialPromptTokens)

	// The first response alone, not continued: cut short at the output cap.
	resp, err = VertexToResponsesResponse(first, argon, "", 0)
	require.NoError(t, err)
	assert.Equal(t, "incomplete", resp.Status)
	require.NotNil(t, resp.IncompleteDetails)
	assert.Equal(t, "max_output_tokens", resp.IncompleteDetails.Reason)
}

const continuedResponsesStream = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"plan","thought":true}]}}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":10}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"CONTINUATION","continuationToken":"dG9r","groundingMetadata":{"webSearchQueries":["q"]}}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":65536}}

`

const finalResponsesSegment = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Answer."}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["q"]}}],"usageMetadata":{"promptTokenCount":66536,"cachedContentTokenCount":66000,"candidatesTokenCount":2}}

`

func streamToResponses(t *testing.T, stream string) (*responses.Response, string) {
	t.Helper()
	var out strings.Builder
	var completed *responses.Response
	require.NoError(t, TransformVertexStreamToResponses(strings.NewReader(stream), &out, argon, "", nil,
		func(r *responses.Response) { completed = r }))
	require.NotNil(t, completed)
	return completed, out.String()
}

func TestArgonResponsesContinuedStream(t *testing.T) {
	resp, sse := streamToResponses(t, continuedResponsesStream+finalResponsesSegment)
	assert.Equal(t, "completed", resp.Status)
	assert.Equal(t, 1, strings.Count(sse, "event: response.completed"))
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 1000+66536, resp.Usage.InputTokens)
	assert.Equal(t, 65536+2, resp.Usage.OutputTokens)
	assert.Equal(t, 66000, resp.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, 1000, resp.Usage.InputTokensDetails.InitialPromptTokens)
	require.NotNil(t, resp.Usage.ServerToolUse)
	assert.Equal(t, 2, resp.Usage.ServerToolUse.WebSearchRequests, "one query in each request")
}

func TestArgonResponsesStreamLeftAtContinuationIsIncomplete(t *testing.T) {
	resp, _ := streamToResponses(t, continuedResponsesStream)
	assert.Equal(t, "incomplete", resp.Status)
	require.NotNil(t, resp.IncompleteDetails)
	assert.Equal(t, "max_output_tokens", resp.IncompleteDetails.Reason)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 65536, resp.Usage.OutputTokens, "the finished request's usage is kept")
}
