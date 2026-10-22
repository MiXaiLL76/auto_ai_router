package vertex

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

const argon = "gemini-4-argon"

// argonRequest converts an OpenAI chat body for gemini-4-argon and returns the
// Vertex request as generic JSON, to see exactly which fields are sent.
func argonRequest(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := OpenAIToVertex([]byte(body), false, false, argon, "application/json")
	require.NoError(t, err)
	var req map[string]any
	require.NoError(t, json.Unmarshal(out, &req))
	return req
}

func argonError(t *testing.T, body string) error {
	t.Helper()
	_, err := OpenAIToVertex([]byte(body), false, false, argon, "application/json")
	return err
}

func generationConfig(t *testing.T, req map[string]any) map[string]any {
	t.Helper()
	cfg, ok := req["generationConfig"].(map[string]any)
	require.True(t, ok, "generationConfig: %v", req)
	return cfg
}

func thinkingLevel(t *testing.T, req map[string]any) string {
	t.Helper()
	tc, _ := generationConfig(t, req)["thinkingConfig"].(map[string]any)
	level, _ := tc["thinkingLevel"].(string)
	_, hasBudget := tc["thinkingBudget"]
	assert.False(t, hasBudget, "Gemini 4 takes no thinkingBudget")
	return level
}

const userTurn = `"messages": [{"role": "user", "content": "hi"}]`

func TestArgonThinkingLevels(t *testing.T) {
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
			req := argonRequest(t, `{"model": "gemini-4-argon", `+tt.params+userTurn+`}`)
			assert.Equal(t, tt.want, thinkingLevel(t, req))
		})
	}
}

func TestArgonIncludeThoughts(t *testing.T) {
	req := argonRequest(t, `{"model": "gemini-4-argon", "thinking_config": {"include_thoughts": true}, `+userTurn+`}`)
	tc := generationConfig(t, req)["thinkingConfig"].(map[string]any)
	assert.Equal(t, true, tc["includeThoughts"])
	assert.Equal(t, "MEDIUM", tc["thinkingLevel"])
}

func TestArgonRejectsNativeBudgetAndMinimal(t *testing.T) {
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
			testhelpers.RequireValidationError(t, argonError(t, `{"model": "gemini-4-argon", `+tt.params+userTurn+`}`), tt.param, tt.code)
		})
	}
}

// The Gemini 4 rules are the profile's: other models keep their behavior.
func TestArgonRulesDoNotReachOtherModels(t *testing.T) {
	out, err := OpenAIToVertex([]byte(`{"model": "g", "n": 2, "thinking_budget": 1024, `+
		`"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure,"}]}`),
		false, false, "gemini-3.1-pro-preview", "application/json")
	require.NoError(t, err)
	var req VertexRequest
	require.NoError(t, json.Unmarshal(out, &req))
	assert.Equal(t, int32(2), req.GenerationConfig.CandidateCount)
	assert.Equal(t, genai.ThinkingLevelLow, req.GenerationConfig.ThinkingConfig.ThinkingLevel)
}

func TestArgonCandidateCount(t *testing.T) {
	req := argonRequest(t, `{"model": "gemini-4-argon", "n": 1, `+userTurn+`}`)
	assert.NotContains(t, generationConfig(t, req), "candidateCount", "n=1 is not sent")

	testhelpers.RequireValidationError(t, argonError(t, `{"model": "gemini-4-argon", "n": 2, `+userTurn+`}`), "n", "unsupported_value")
	testhelpers.RequireValidationError(t,
		argonError(t, `{"model": "gemini-4-argon", "extra_body": {"generation_config": {"candidate_count": 3}}, `+userTurn+`}`),
		"candidate_count", "unsupported_value")
}

func TestArgonDropsIgnoredAndRejectedSamplingParams(t *testing.T) {
	req := argonRequest(t, `{"model": "gemini-4-argon", "temperature": 0.2, "top_p": 0.5, `+
		`"frequency_penalty": 0.3, "presence_penalty": 0.1, "seed": 7, "max_tokens": 4096, `+
		`"extra_body": {"generation_config": {"top_k": 5}}, `+userTurn+`}`)
	cfg := generationConfig(t, req)
	for _, field := range []string{"temperature", "topP", "topK", "frequencyPenalty", "presencePenalty"} {
		assert.NotContains(t, cfg, field)
	}
	assert.Equal(t, float64(7), cfg["seed"])
	assert.Equal(t, float64(4096), cfg["maxOutputTokens"])
}

func TestArgonOutputLimitIsTheSharedTokenLimit(t *testing.T) {
	req := argonRequest(t, `{"model": "gemini-4-argon", "max_completion_tokens": 1048576, `+userTurn+`}`)
	assert.Equal(t, float64(1048576), generationConfig(t, req)["maxOutputTokens"], "long decodes may ask for the whole limit")

	testhelpers.RequireValidationError(t, argonError(t, `{"model": "gemini-4-argon", "max_tokens": 1048577, `+userTurn+`}`),
		"max_tokens", "invalid_value")
	testhelpers.RequireValidationError(t,
		argonError(t, `{"model": "gemini-4-argon", "max_tokens": 10, "max_completion_tokens": 2000000, `+userTurn+`}`),
		"max_completion_tokens", "invalid_value")
}

func TestArgonRejectsAssistantPrefill(t *testing.T) {
	err := argonError(t, `{"model": "gemini-4-argon", "messages": [{"role": "user", "content": "Write a poem"}, `+
		`{"role": "assistant", "content": "Roses are"}]}`)
	testhelpers.RequireValidationError(t, err, "messages", "invalid_value")
	assert.Contains(t, err.Error(), "prefill")
}

// A full tool loop with two functions: Gemini's call IDs reach the client and come
// back on the calls and their results, the results go in the order of the calls
// whatever order the client sends them in, and the thought signature goes back on
// the call that carried it.
func TestArgonToolLoopKeepsCallIDsOrderAndSignature(t *testing.T) {
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
	converted, err := VertexToOpenAI(turn1, argon)
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
	body := `{"model": "gemini-4-argon", "tools": [` +
		`{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},` +
		`{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}], ` +
		`"messages": [{"role": "user", "content": "Weather and time in Moscow?"}, ` + string(assistant) + `, ` +
		`{"role": "tool", "tool_call_id": "fc-time", "content": "{\"time\": \"15:00\"}"}, ` +
		`{"role": "tool", "tool_call_id": "fc-weather", "content": "{\"temp\": 20}"}]}`
	out, err := OpenAIToVertex([]byte(body), false, false, argon, "application/json")
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
func TestArgonStreamedCallsKeepIDsAndIndexes(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-1","name":"a","args":{}}}]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-2","name":"b","args":{}}}]},"finishReason":"STOP"}]}`)
	chunks := transformStream(t, stream)
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

// Gemini's call IDs reach the client and go back only on argon: another model's
// tool calls get router-made IDs, and its calls and responses go without IDs.
func TestFunctionCallIDsOnlyOnArgon(t *testing.T) {
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
	for model, wantID := range map[string]string{other: "", argon: "fc-1"} {
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

// Behind an alias, argon is known by Gemini's modelVersion: its call IDs are kept
// and every search run is billed.
func TestArgonResponseRulesFollowModelVersion(t *testing.T) {
	body, err := json.Marshal(genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "lookup", Args: map[string]any{}}},
			}},
			FinishReason:      genai.FinishReasonStop,
			GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q", "q"}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5, CandidatesTokenCount: 2},
		ModelVersion:  argon,
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

// A query text a response lists twice is two billed queries on argon and one on
// any other model; the responses of a continued generation add up either way.
func TestRepeatedSearchQueriesBilledOnlyOnArgon(t *testing.T) {
	const other = "gemini-3.1-pro-preview"
	grounded := []*genai.Candidate{{GroundingMetadata: &genai.GroundingMetadata{
		WebSearchQueries: []string{"x", "x", "y"}, ImageSearchQueries: []string{"x", "x"}}}}
	assert.Equal(t, 5, CountWebSearchRequests(grounded, argon))
	assert.Equal(t, 3, CountWebSearchRequests(grounded, other))
	assert.Equal(t, 3, CountWebSearchRequests(grounded, ""))

	var queries WebSearchQueries
	queries.Add(grounded)
	queries.CloseResponse()
	queries.Add(grounded)
	assert.Equal(t, 10, queries.Count(argon))
	assert.Equal(t, 6, queries.Count(other))

	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"a"}]},"groundingMetadata":{"webSearchQueries":["x","x"]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["x","x","y"]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(transformStreamFor(t, stream, other))
	require.NotNil(t, usage)
	require.NotNil(t, usage.ServerToolUse)
	assert.Equal(t, 2, usage.ServerToolUse.WebSearchRequests)
}

func sseChunk(t *testing.T, payload string) string {
	t.Helper()
	require.True(t, json.Valid([]byte(payload)), payload)
	return "data: " + payload + "\r\n\r\n"
}

func transformStream(t *testing.T, stream string) []openai.OpenAIStreamingChunk {
	t.Helper()
	return transformStreamFor(t, stream, argon)
}

func transformStreamFor(t *testing.T, stream, model string) []openai.OpenAIStreamingChunk {
	t.Helper()
	var out strings.Builder
	require.NoError(t, TransformVertexStreamToOpenAI(strings.NewReader(stream), model, &out))
	var chunks []openai.OpenAIStreamingChunk
	for _, frame := range strings.Split(out.String(), "\n\n") {
		payload := strings.TrimPrefix(strings.TrimSpace(frame), "data: ")
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk openai.OpenAIStreamingChunk
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk), payload)
		chunks = append(chunks, chunk)
	}
	require.True(t, strings.HasSuffix(out.String(), "data: [DONE]\n\n"))
	return chunks
}

func finishReasons(chunks []openai.OpenAIStreamingChunk) []string {
	var reasons []string
	for _, chunk := range chunks {
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				reasons = append(reasons, *choice.FinishReason)
			}
		}
	}
	return reasons
}

func lastUsage(chunks []openai.OpenAIStreamingChunk) *openai.OpenAIUsage {
	var usage *openai.OpenAIUsage
	for _, chunk := range chunks {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	return usage
}

// Two requests of one long decode: the first spent all its tokens thinking (a
// checkpoint token mid-stream, then CONTINUATION), the second answers. Each
// request's usage is cumulative within it; the generation's is their sum.
const continuedStream = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"plan","thought":true}]}}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":10,"totalTokenCount":1010}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":" more","thought":true}]},"continuationToken":"Y2hlY2twb2ludA=="}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":40000,"totalTokenCount":41000}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"CONTINUATION","continuationToken":"dG9rLTE=","groundingMetadata":{"webSearchQueries":["argon pricing"]}}],"usageMetadata":{"promptTokenCount":1000,"thoughtsTokenCount":65536,"totalTokenCount":66536}}

`

const finalSegment = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"The answer"}]}}],"usageMetadata":{"promptTokenCount":66536,"cachedContentTokenCount":66000,"candidatesTokenCount":3,"totalTokenCount":66539}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":" is 42."}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["argon pricing"]}}],"usageMetadata":{"promptTokenCount":66536,"cachedContentTokenCount":66000,"candidatesTokenCount":6,"thoughtsTokenCount":20,"totalTokenCount":66562}}

`

func TestArgonContinuedStreamIsOneGeneration(t *testing.T) {
	chunks := transformStream(t, continuedStream+finalSegment)

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

func TestArgonStreamLeftAtContinuationEndsAsLength(t *testing.T) {
	chunks := transformStream(t, continuedStream)
	assert.Equal(t, []string{"length"}, finishReasons(chunks))
	usage := lastUsage(chunks)
	require.NotNil(t, usage)
	assert.Equal(t, 1000, usage.PromptTokens)
	assert.Equal(t, 65536, usage.CompletionTokens, "the request's usage is kept")
}

// On argon the SSE metadata a stream repeats is not billed twice; queries a chunk
// lists twice are two runs.
func TestArgonStreamWebSearchQueriesDeduplicateRepeatsNotRuns(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"a"}]},"groundingMetadata":{"webSearchQueries":["x","x"]}}]}`) +
		sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["x","x","y"]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(transformStream(t, stream))
	require.NotNil(t, usage)
	assert.Equal(t, 3, usage.ServerToolUse.WebSearchRequests)
}

func TestArgonEnabledSearchWithoutQueriesIsNotBilled(t *testing.T) {
	stream := sseChunk(t, `{"candidates":[{"content":{"parts":[{"text":"no search needed"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	usage := lastUsage(transformStream(t, stream))
	require.NotNil(t, usage)
	assert.Nil(t, usage.ServerToolUse)
}

func continuationResponse(t *testing.T, parts []*genai.Part, finish genai.FinishReason, token string,
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
		ModelVersion:  argon,
	})
	require.NoError(t, err)
	return body
}

func TestMergeContinuationResponses(t *testing.T) {
	first := continuationResponse(t,
		[]*genai.Part{{Text: "thinking", Thought: true}, {Text: "Part one. "}},
		genai.FinishReasonContinuation, "tok",
		&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 1200, ToolUsePromptTokenCount: 300, CandidatesTokenCount: 20000, ThoughtsTokenCount: 12768,
			TotalTokenCount: 34268, PromptTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 900}, {Modality: genai.MediaModalityImage, TokenCount: 300}},
		},
		&genai.GroundingMetadata{
			WebSearchQueries: []string{"q"},
			GroundingChunks:  []*genai.GroundingChunk{{Web: &genai.GroundingChunkWeb{URI: "https://a.example"}}},
		})
	second := continuationResponse(t,
		[]*genai.Part{{Text: "Part two."}},
		genai.FinishReasonStop, "",
		&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 34000, CachedContentTokenCount: 33000, CandidatesTokenCount: 500, TotalTokenCount: 34500,
			PromptTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 33700}, {Modality: genai.MediaModalityImage, TokenCount: 300}},
		},
		&genai.GroundingMetadata{
			WebSearchQueries: []string{"q"},
			GroundingChunks:  []*genai.GroundingChunk{{Web: &genai.GroundingChunkWeb{URI: "https://b.example"}}},
			GroundingSupports: []*genai.GroundingSupport{{
				Segment:               &genai.Segment{PartIndex: 0, StartIndex: 0, EndIndex: 9},
				GroundingChunkIndices: []int32{0},
			}},
		})

	merged, err := MergeContinuationResponses([][]byte{first, second})
	require.NoError(t, err)
	assert.Equal(t, int32(1200), InitialPromptTokens(merged))

	var resp genai.GenerateContentResponse
	require.NoError(t, json.Unmarshal(merged, &resp))
	candidate := resp.Candidates[0]
	require.Len(t, candidate.Content.Parts, 3)
	assert.Equal(t, "Part two.", candidate.Content.Parts[2].Text)
	assert.Equal(t, genai.FinishReasonStop, candidate.FinishReason)
	assert.Empty(t, candidate.ContinuationToken)
	support := candidate.GroundingMetadata.GroundingSupports[0]
	assert.Equal(t, int32(2), support.Segment.PartIndex, "points at the second request's part")
	assert.Equal(t, []int32{1}, support.GroundingChunkIndices, "and at its chunk")
	assert.Equal(t, 2, CountWebSearchRequests(resp.Candidates, argon))

	usage := resp.UsageMetadata
	assert.Equal(t, int32(35200), usage.PromptTokenCount)
	assert.Equal(t, int32(33000), usage.CachedContentTokenCount)
	assert.Equal(t, int32(20500), usage.CandidatesTokenCount)
	assert.Equal(t, int32(12768), usage.ThoughtsTokenCount)
	assert.Equal(t, int32(300), usage.ToolUsePromptTokenCount)
	assert.Equal(t, []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 34600}, {Modality: genai.MediaModalityImage, TokenCount: 600}},
		usage.PromptTokensDetails)

	converted, err := VertexToOpenAI(merged, argon)
	require.NoError(t, err)
	var openAIResp openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &openAIResp))
	choice := openAIResp.Choices[0]
	assert.Equal(t, "Part one. Part two.", choice.Message.Content)
	assert.Equal(t, "thinking", choice.Message.ReasoningContent)
	assert.Equal(t, "stop", choice.FinishReason)
	assert.Equal(t, 35200+300, openAIResp.Usage.PromptTokens, "tool-use prompt tokens are billed as input")
	assert.Equal(t, 20500+12768, openAIResp.Usage.CompletionTokens)
	assert.Equal(t, 33000, openAIResp.Usage.PromptTokensDetails.CachedTokens)
	assert.Equal(t, 600, openAIResp.Usage.PromptTokensDetails.ImageTokens)
	assert.Equal(t, 1200, openAIResp.Usage.PromptTokensDetails.InitialPromptTokens)
	assert.Equal(t, 2, openAIResp.Usage.ServerToolUse.WebSearchRequests)
}

func TestArgonUncontinuedResponseEndsAsLength(t *testing.T) {
	body := continuationResponse(t, []*genai.Part{{Text: "cut"}}, genai.FinishReasonContinuation, "tok",
		&genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 32768}, nil)
	converted, err := VertexToOpenAI(body, argon)
	require.NoError(t, err)
	var resp openai.OpenAIResponse
	require.NoError(t, json.Unmarshal(converted, &resp))
	assert.Equal(t, "length", resp.Choices[0].FinishReason)
	assert.Equal(t, 32768, resp.Usage.CompletionTokens)
	assert.Zero(t, InitialPromptTokens(body), "a single response carries no merged field")
}

func TestContinuationRequestResendsTheRequestWithTheToken(t *testing.T) {
	original := []byte(`{"contents":[{"role":"user","parts":[{"text":"prove it"}]}],"systemInstruction":{"parts":[{"text":"be formal"}]},` +
		`"tools":[{"googleSearch":{}}],"generationConfig":{"maxOutputTokens":131072,"thinkingConfig":{"thinkingLevel":"HIGH"}}}`)
	next, err := ContinuationRequest(original, "dG9rLTE=")
	require.NoError(t, err)
	var got, want map[string]any
	require.NoError(t, json.Unmarshal(next, &got))
	require.NoError(t, json.Unmarshal(original, &want))
	assert.Equal(t, "dG9rLTE=", got["continuationToken"])
	delete(got, "continuationToken")
	assert.Equal(t, want, got, "contents, systemInstruction, tools and maxOutputTokens go again unchanged")
}

func TestReadContinuation(t *testing.T) {
	finish, token, ok := ReadContinuation([]byte(`{"candidates":[{"finishReason":"CONTINUATION","continuationToken":"dA=="}]}`))
	assert.True(t, ok)
	assert.True(t, IsContinuation(finish))
	assert.Equal(t, "dA==", token)

	finish, token, ok = ReadContinuation([]byte(`{"candidates":[{"continuationToken":"Y3A="}]}`))
	assert.True(t, ok)
	assert.False(t, IsContinuation(finish), "a checkpoint token alone asks for nothing")
	assert.Equal(t, "Y3A=", token)

	_, _, ok = ReadContinuation([]byte(`not json`))
	assert.False(t, ok)
	assert.True(t, SupportsLongDecode("gemini-4-argon"))
	assert.True(t, SupportsLongDecode("publishers/google/models/gemini-4-argon"))
	assert.False(t, SupportsLongDecode("gemini-3.1-pro-preview"))
}
