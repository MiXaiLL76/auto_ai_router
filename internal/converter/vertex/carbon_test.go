package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

const carbon = "gemini-4-carbon"

// carbonRequest converts an OpenAI chat body for gemini-4-carbon and returns the
// Vertex request as generic JSON, to see exactly which fields are sent.
func carbonRequest(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := OpenAIToVertex([]byte(body), false, false, carbon, "application/json")
	require.NoError(t, err)
	var req map[string]any
	require.NoError(t, json.Unmarshal(out, &req))
	return req
}

func carbonError(t *testing.T, body string) error {
	t.Helper()
	_, err := OpenAIToVertex([]byte(body), false, false, carbon, "application/json")
	return err
}

func TestCarbonThinkingLevels(t *testing.T) {
	tests := []struct {
		name   string
		params string
		want   string
	}{
		{"default is MEDIUM", ``, "MEDIUM"},
		{"low", `"reasoning_effort": "low",`, "LOW"},
		{"medium stays MEDIUM", `"reasoning_effort": "medium",`, "MEDIUM"},
		{"high", `"reasoning_effort": "high",`, "HIGH"},
		{"xhigh", `"reasoning_effort": "xhigh",`, "HIGH"},
		// Thinking cannot be turned off: the compatible ways to ask get the lowest level.
		{"minimal", `"reasoning_effort": "minimal",`, "LOW"},
		{"none", `"reasoning_effort": "none",`, "LOW"},
		{"disable", `"reasoning_effort": "disable",`, "LOW"},
		{"anthropic disabled", `"thinking": {"type": "disabled"},`, "LOW"},
		{"anthropic adaptive", `"thinking": {"type": "adaptive"},`, "MEDIUM"},
		{"anthropic budget", `"thinking": {"type": "enabled", "budget_tokens": 20000},`, "HIGH"},
		{"native level", `"thinking_config": {"thinking_level": "high"},`, "HIGH"},
		{"native level camelCase", `"extra_body": {"generation_config": {"thinkingConfig": {"thinkingLevel": "LOW"}}},`, "LOW"},
		{"thinking_level shorthand", `"thinking_level": "medium",`, "MEDIUM"},
		{"native none", `"thinking_config": {"thinking_level": "none"},`, "LOW"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := carbonRequest(t, `{"model": "gemini-4-carbon", `+tt.params+userTurn+`}`)
			assert.Equal(t, tt.want, thinkingLevel(t, req))
		})
	}
}

func TestCarbonIncludeThoughts(t *testing.T) {
	req := carbonRequest(t, `{"model": "gemini-4-carbon", "thinking_config": {"include_thoughts": true}, `+userTurn+`}`)
	tc := generationConfig(t, req)["thinkingConfig"].(map[string]any)
	assert.Equal(t, true, tc["includeThoughts"])
	assert.Equal(t, "MEDIUM", tc["thinkingLevel"])
}

func TestCarbonRejectsNativeBudgetAndMinimal(t *testing.T) {
	tests := []struct {
		name   string
		params string
		param  string
		code   string
	}{
		{"thinking_config budget", `"thinking_config": {"thinking_budget": 1024},`, "thinking_budget", "unsupported_parameter"},
		{"thinking_config budget zero", `"thinkingConfig": {"thinkingBudget": 0},`, "thinking_budget", "unsupported_parameter"},
		{"dynamic budget", `"thinking_budget": -1,`, "thinking_budget", "unsupported_parameter"},
		{"extra_body budget", `"extra_body": {"thinking_budget": 8192},`, "thinking_budget", "unsupported_parameter"},
		{"generation_config budget", `"extra_body": {"generation_config": {"thinking_config": {"thinking_budget": 512, "thinking_level": "low"}}},`, "thinking_budget", "unsupported_parameter"},
		{"native MINIMAL", `"thinking_config": {"thinking_level": "MINIMAL"},`, "thinking_level", "unsupported_value"},
		{"MINIMAL shorthand", `"thinkingLevel": "THINKING_LEVEL_MINIMAL",`, "thinking_level", "unsupported_value"},
		{"extra_body MINIMAL", `"extra_body": {"thinking_level": "minimal"},`, "thinking_level", "unsupported_value"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testhelpers.RequireValidationError(t, carbonError(t, `{"model": "gemini-4-carbon", `+tt.params+userTurn+`}`), tt.param, tt.code)
		})
	}
}

// The Gemini 4 rules are the profile's: other models keep their behavior.
func TestCarbonRulesDoNotReachOtherModels(t *testing.T) {
	out, err := OpenAIToVertex([]byte(`{"model": "g", "n": 2, "thinking_budget": 1024, `+
		`"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure,"}]}`),
		false, false, "gemini-3.1-pro-preview", "application/json")
	require.NoError(t, err)
	var req VertexRequest
	require.NoError(t, json.Unmarshal(out, &req))
	assert.Equal(t, int32(2), req.GenerationConfig.CandidateCount)
	assert.Equal(t, genai.ThinkingLevelLow, req.GenerationConfig.ThinkingConfig.ThinkingLevel)
}

func TestCarbonCandidateCount(t *testing.T) {
	req := carbonRequest(t, `{"model": "gemini-4-carbon", "n": 1, `+userTurn+`}`)
	assert.NotContains(t, generationConfig(t, req), "candidateCount", "n=1 is not sent")

	testhelpers.RequireValidationError(t, carbonError(t, `{"model": "gemini-4-carbon", "n": 2, `+userTurn+`}`), "n", "unsupported_value")
	testhelpers.RequireValidationError(t,
		carbonError(t, `{"model": "gemini-4-carbon", "extra_body": {"generation_config": {"candidate_count": 3}}, `+userTurn+`}`),
		"candidate_count", "unsupported_value")
}

func TestCarbonDropsIgnoredAndRejectedSamplingParams(t *testing.T) {
	req := carbonRequest(t, `{"model": "gemini-4-carbon", "temperature": 0.2, "top_p": 0.5, `+
		`"frequency_penalty": 0.3, "presence_penalty": 0.1, "seed": 7, "max_tokens": 4096, `+
		`"extra_body": {"generation_config": {"top_k": 5}}, `+userTurn+`}`)
	cfg := generationConfig(t, req)
	for _, field := range []string{"temperature", "topP", "topK", "frequencyPenalty", "presencePenalty"} {
		assert.NotContains(t, cfg, field)
	}
	assert.Equal(t, float64(7), cfg["seed"])
	assert.Equal(t, float64(4096), cfg["maxOutputTokens"])
}

func TestCarbonOutputLimitIsTheSharedTokenLimit(t *testing.T) {
	req := carbonRequest(t, `{"model": "gemini-4-carbon", "max_completion_tokens": 1048576, `+userTurn+`}`)
	assert.Equal(t, float64(1048576), generationConfig(t, req)["maxOutputTokens"], "long decodes may ask for the whole limit")

	testhelpers.RequireValidationError(t, carbonError(t, `{"model": "gemini-4-carbon", "max_tokens": 1048577, `+userTurn+`}`),
		"max_tokens", "invalid_value")
	testhelpers.RequireValidationError(t,
		carbonError(t, `{"model": "gemini-4-carbon", "max_tokens": 10, "max_completion_tokens": 2000000, `+userTurn+`}`),
		"max_completion_tokens", "invalid_value")
}

func TestCarbonRejectsAssistantPrefill(t *testing.T) {
	err := carbonError(t, `{"model": "gemini-4-carbon", "messages": [{"role": "user", "content": "Write a poem"}, `+
		`{"role": "assistant", "content": "Roses are"}]}`)
	testhelpers.RequireValidationError(t, err, "messages", "invalid_value")
	assert.Contains(t, err.Error(), "prefill")
}

// A full tool loop with two functions: Gemini's call IDs reach the client and come
// back on the calls and their results, the results go in the order of the calls
// whatever order the client sends them in, and the thought signature goes back on
// the call that carried it.
func TestCarbonToolLoopKeepsCallIDsOrderAndSignature(t *testing.T) {
	// Turn 1: the model calls two functions.
	turn1, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-weather", Name: "get_weather", Args: map[string]any{"city": "Moscow"}},
					ThoughtSignature: []byte("signature-1")},
				{FunctionCall: &genai.FunctionCall{ID: "fc-time", Name: "get_time", Args: map[string]any{"tz": "MSK"}}},
			}},
			FinishReason: genai.FinishReasonStop,
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 50, CandidatesTokenCount: 20, ThoughtsTokenCount: 30},
	})
	require.NoError(t, err)
	converted, err := VertexToOpenAI(turn1, carbon)
	require.NoError(t, err)
	var resp1 openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &resp1))
	require.Len(t, resp1.Choices, 1)
	assert.Equal(t, "tool_calls", resp1.Choices[0].FinishReason)
	calls := resp1.Choices[0].Message.ToolCalls
	require.Len(t, calls, 2)
	assert.Equal(t, "fc-weather", calls[0].ID)
	assert.Equal(t, "fc-time", calls[1].ID)

	// Turn 2: the client sends the calls back with their results, the second first.
	assistant, err := json.Marshal(resp1.Choices[0].Message)
	require.NoError(t, err)
	body := `{"model": "gemini-4-carbon", "tools": [` +
		`{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},` +
		`{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}], ` +
		`"messages": [{"role": "user", "content": "Weather and time in Moscow?"}, ` + string(assistant) + `, ` +
		`{"role": "tool", "tool_call_id": "fc-time", "content": "{\"time\": \"15:00\"}"}, ` +
		`{"role": "tool", "tool_call_id": "fc-weather", "content": "{\"temp\": 20}"}]}`
	out, err := OpenAIToVertex([]byte(body), false, false, carbon, "application/json")
	require.NoError(t, err)
	var req2 VertexRequest
	require.NoError(t, json.Unmarshal(out, &req2))
	require.Len(t, req2.Contents, 3)

	model := req2.Contents[1]
	assert.Equal(t, "model", model.Role)
	require.Len(t, model.Parts, 2)
	assert.Equal(t, "fc-weather", model.Parts[0].FunctionCall.ID)
	assert.Equal(t, []byte("signature-1"), model.Parts[0].ThoughtSignature)
	assert.Equal(t, "fc-time", model.Parts[1].FunctionCall.ID)

	results := req2.Contents[2]
	assert.Equal(t, "user", results.Role)
	require.Len(t, results.Parts, 2)
	assert.Equal(t, "fc-weather", results.Parts[0].FunctionResponse.ID)
	assert.Equal(t, "get_weather", results.Parts[0].FunctionResponse.Name)
	assert.Equal(t, map[string]any{"temp": float64(20)}, results.Parts[0].FunctionResponse.Response)
	assert.Equal(t, "fc-time", results.Parts[1].FunctionResponse.ID)
	assert.Equal(t, "get_time", results.Parts[1].FunctionResponse.Name)
}

// Streamed function calls keep Gemini's IDs, and a call coming in a later chunk
// does not restart the OpenAI tool call index.
func TestCarbonStreamedCallsKeepIDsAndIndexes(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-1","name":"a","args":{}}}]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-2","name":"b","args":{}}}]},"finishReason":"STOP"}]}`)
	chunks := carbonTransformStream(t, stream)
	var calls []openai.OpenAIStreamingToolCall
	for _, chunk := range chunks {
		for _, choice := range chunk.Choices {
			calls = append(calls, choice.Delta.ToolCalls...)
		}
	}
	require.Len(t, calls, 2)
	assert.Equal(t, "fc-1", calls[0].ID)
	assert.Equal(t, 0, calls[0].Index)
	assert.Equal(t, "fc-2", calls[1].ID)
	assert.Equal(t, 1, calls[1].Index)
}

// Gemini's call IDs reach the client and go back only on carbon: another model's
// tool calls get router-made IDs, and its calls and responses go without IDs.
func TestFunctionCallIDsOnlyOnCarbon(t *testing.T) {
	const other = "gemini-3.1-pro-preview"
	turn1, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "get_weather", Args: map[string]any{}}},
			}},
			FinishReason: genai.FinishReasonStop,
		}},
	})
	require.NoError(t, err)
	converted, err := VertexToOpenAI(turn1, other)
	require.NoError(t, err)
	var resp openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &resp))
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1)
	id := resp.Choices[0].Message.ToolCalls[0].ID
	assert.NotEmpty(t, id)
	assert.NotEqual(t, "fc-1", id)

	stream := sseChunk(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-1","name":"get_weather","args":{}}}]},"finishReason":"STOP"}]}`)
	var streamed []openai.OpenAIStreamingToolCall
	for _, chunk := range transformStreamFor(t, stream, other) {
		for _, choice := range chunk.Choices {
			streamed = append(streamed, choice.Delta.ToolCalls...)
		}
	}
	require.Len(t, streamed, 1)
	assert.NotEmpty(t, streamed[0].ID)
	assert.NotEqual(t, "fc-1", streamed[0].ID)

	body := `{"model": "g", "messages": [{"role": "user", "content": "Weather?"}, ` +
		`{"role": "assistant", "tool_calls": [{"id": "fc-1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}, ` +
		`{"role": "tool", "tool_call_id": "fc-1", "content": "{\"temp\": 20}"}]}`
	for model, wantID := range map[string]string{other: "", carbon: "fc-1"} {
		out, err := OpenAIToVertex([]byte(body), false, false, model, "application/json")
		require.NoError(t, err)
		var req VertexRequest
		require.NoError(t, json.Unmarshal(out, &req))
		require.Len(t, req.Contents, 3, model)
		assert.Equal(t, wantID, req.Contents[1].Parts[0].FunctionCall.ID, model)
		assert.Equal(t, wantID, req.Contents[2].Parts[0].FunctionResponse.ID, model)
		assert.Equal(t, "get_weather", req.Contents[2].Parts[0].FunctionResponse.Name, model)
	}
}

// Behind an alias, carbon is known by Gemini's modelVersion: its call IDs are kept
// and every search run is billed.
func TestCarbonResponseRulesFollowModelVersion(t *testing.T) {
	body, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "lookup", Args: map[string]any{}}},
			}},
			FinishReason:      genai.FinishReasonStop,
			GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q", "q"}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5, CandidatesTokenCount: 2},
		ModelVersion:  carbon,
	})
	require.NoError(t, err)
	converted, err := VertexToOpenAI(body, "my-alias")
	require.NoError(t, err)
	var resp openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &resp))
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "fc-1", resp.Choices[0].Message.ToolCalls[0].ID)
	require.NotNil(t, resp.Usage.ServerToolUse)
	assert.Equal(t, 2, resp.Usage.ServerToolUse.WebSearchRequests)
}

// A query text a response lists twice is two billed queries on carbon and one on
// any other model; the responses of a continued generation add up either way.
func TestRepeatedSearchQueriesBilledOnlyOnCarbon(t *testing.T) {
	const other = "gemini-3.1-pro-preview"
	grounded := []*genai.Candidate{{GroundingMetadata: &genai.GroundingMetadata{
		WebSearchQueries: []string{"x", "x", "y"}, ImageSearchQueries: []string{"x", "x"}}}}
	assert.Equal(t, 5, CountWebSearchRequests(grounded, carbon))
	assert.Equal(t, 3, CountWebSearchRequests(grounded, other))
	assert.Equal(t, 3, CountWebSearchRequests(grounded, ""))

	var queries WebSearchQueries
	queries.Add(grounded)
	queries.CloseResponse()
	queries.Add(grounded)
	assert.Equal(t, 10, queries.Count(carbon))
	assert.Equal(t, 6, queries.Count(other))

	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"a"}]},"groundingMetadata":{"webSearchQueries":["x","x"]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["x","x","y"]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(transformStreamFor(t, stream, other))
	require.NotNil(t, usage)
	require.NotNil(t, usage.ServerToolUse)
	assert.Equal(t, 2, usage.ServerToolUse.WebSearchRequests)
}

func carbonTransformStream(t *testing.T, stream string) []openai.OpenAIStreamingChunk {
	t.Helper()
	return transformStreamFor(t, stream, carbon)
}

func TestCarbonContinuedStreamIsOneGeneration(t *testing.T) {
	chunks := carbonTransformStream(t, continuedStream+finalSegment)

	assert.Equal(t, []string{"stop"}, finishReasons(chunks), "the CONTINUATION finish is not passed on")
	var content, reasoning string
	for _, chunk := range chunks {
		for _, choice := range chunk.Choices {
			content += choice.Delta.Content
			reasoning += choice.Delta.ReasoningContent
		}
	}
	assert.Equal(t, "The answer is 42.", content)
	assert.Equal(t, "plan more", reasoning)

	usage := lastUsage(chunks)
	require.NotNil(t, usage)
	assert.Equal(t, 1000+66536, usage.PromptTokens, "each request's final prompt, once")
	assert.Equal(t, 65536+6+20, usage.CompletionTokens)
	assert.Equal(t, 65536+20, usage.CompletionTokensDetails.ReasoningTokens)
	assert.Equal(t, 66000, usage.PromptTokensDetails.CachedTokens, "the re-read context is billed as its cache hits")
	assert.Equal(t, 1000, usage.PromptTokensDetails.InitialPromptTokens)
	// The same query text, run once in each request: two billed queries.
	assert.Equal(t, 2, usage.ServerToolUse.WebSearchRequests)
}

func TestCarbonStreamLeftAtContinuationEndsAsLength(t *testing.T) {
	chunks := carbonTransformStream(t, continuedStream)
	assert.Equal(t, []string{"length"}, finishReasons(chunks))
	usage := lastUsage(chunks)
	require.NotNil(t, usage)
	assert.Equal(t, 1000, usage.PromptTokens)
	assert.Equal(t, 65536, usage.CompletionTokens, "the request's usage is kept")
}

// On carbon the SSE metadata a stream repeats is not billed twice; queries a chunk
// lists twice are two runs.
func TestCarbonStreamWebSearchQueriesDeduplicateRepeatsNotRuns(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"a"}]},"groundingMetadata":{"webSearchQueries":["x","x"]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["x","x","y"]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(carbonTransformStream(t, stream))
	require.NotNil(t, usage)
	assert.Equal(t, 3, usage.ServerToolUse.WebSearchRequests)
}

func TestCarbonEnabledSearchWithoutQueriesIsNotBilled(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"no search needed"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(carbonTransformStream(t, stream))
	require.NotNil(t, usage)
	assert.Nil(t, usage.ServerToolUse)
}

func carbonContinuationResponse(t *testing.T, parts []*genai.Part, finish genai.FinishReason, token string,
	usage *genai.GenerateContentResponseUsageMetadata, grounding *genai.GroundingMetadata) []byte {
	t.Helper()
	body, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:           &genai.Content{Role: "model", Parts: parts},
			FinishReason:      finish,
			ContinuationToken: []byte(token),
			GroundingMetadata: grounding,
		}},
		UsageMetadata: usage,
		ModelVersion:  carbon,
	})
	require.NoError(t, err)
	return body
}

func TestCarbonUncontinuedResponseEndsAsLength(t *testing.T) {
	body := carbonContinuationResponse(t, []*genai.Part{{Text: "cut"}}, genai.FinishReasonContinuation, "tok",
		&genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 32768}, nil)
	converted, err := VertexToOpenAI(body, carbon)
	require.NoError(t, err)
	var resp openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &resp))
	assert.Equal(t, "length", resp.Choices[0].FinishReason)
	assert.Equal(t, 32768, resp.Usage.CompletionTokens)
	assert.Zero(t, InitialPromptTokens(body), "a single response carries no merged field")
}

func TestCarbonSupportsLongDecode(t *testing.T) {
	assert.True(t, SupportsLongDecode("gemini-4-carbon"))
	assert.True(t, SupportsLongDecode("publishers/google/models/gemini-4-carbon"))
}
