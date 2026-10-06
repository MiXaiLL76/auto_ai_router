package converter

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// xAI chat usage as documented: reasoning on top of completion_tokens
// (32 + 9 + 94 = 135), the purchase cost in cost_in_usd_ticks.
const xaiChatUsageBody = `{"id":"c1","object":"chat.completion","model":"grok-4.7","choices":[],
	"usage":{"prompt_tokens":32,"completion_tokens":9,"total_tokens":135,
	"prompt_tokens_details":{"text_tokens":32,"audio_tokens":0,"image_tokens":0,"cached_tokens":6},
	"completion_tokens_details":{"reasoning_tokens":94,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},
	"num_sources_used":0,"cost_in_usd_ticks":37756000}}`

// The same model through Requesty: reasoning inside completion_tokens
// (250 + 323 = 573), cost in USD.
const requestyChatUsageBody = `{"id":"c2","object":"chat.completion","model":"grok-4.6","choices":[],
	"usage":{"completion_tokens":323,"completion_tokens_details":{"reasoning_tokens":309},
	"prompt_tokens":250,"prompt_tokens_details":{"cached_tokens":128},"total_tokens":573,"cost":0.002246}}`

func TestExtractTokenUsage_XAIChatReasoningIsAdditive(t *testing.T) {
	usage := ExtractTokenUsage([]byte(xaiChatUsageBody))

	require.NotNil(t, usage)
	assert.Equal(t, 32, usage.PromptTokens)
	assert.Equal(t, 9, usage.CompletionTokens)
	assert.Equal(t, 94, usage.ReasoningTokens)
	assert.Equal(t, 6, usage.CachedInputTokens)
	assert.Equal(t, ReasoningAccountingAdditive, usage.ReasoningAccounting)
	assert.InDelta(t, 0.0037756, usage.ProviderCostUSD, 1e-12)
	assert.False(t, usage.ServerToolUsageReported, "no server-side tool object in this response")
}

func TestExtractTokenUsage_RequestyChatReasoningIsIncluded(t *testing.T) {
	usage := ExtractTokenUsage([]byte(requestyChatUsageBody))

	require.NotNil(t, usage)
	assert.Equal(t, 323, usage.CompletionTokens)
	assert.Equal(t, 309, usage.ReasoningTokens)
	assert.Equal(t, ReasoningAccountingIncluded, usage.ReasoningAccounting)
	assert.InDelta(t, 0.002246, usage.ProviderCostUSD, 1e-12)
}

func TestDetectReasoningAccounting(t *testing.T) {
	tests := []struct {
		name                                 string
		prompt, completion, reasoning, total int
		want                                 string
	}{
		{"additive", 32, 9, 94, 135, ReasoningAccountingAdditive},
		{"included", 250, 323, 309, 573, ReasoningAccountingIncluded},
		{"reasoning only while thinking", 100, 0, 50, 150, ReasoningAccountingAdditive},
		{"no reasoning", 10, 5, 0, 15, ""},
		{"no total", 10, 5, 3, 0, ""},
		{"total matches neither sum", 10, 5, 3, 99, ""},
		{"included sum but reasoning exceeds completion", 10, 2, 5, 12, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, detectReasoningAccounting(tt.prompt, tt.completion, tt.reasoning, tt.total))
		})
	}
}

func TestExtractTokenUsage_XAIResponsesServerSideTools(t *testing.T) {
	// Three web_search_call items but only two successful searches: xAI bills
	// usage.server_side_tool_usage_details, not the attempts.
	body := []byte(`{"id":"resp_1","object":"response","status":"completed","model":"grok-4.7",
		"output":[
			{"type":"web_search_call","id":"ws_1","status":"completed"},
			{"type":"web_search_call","id":"ws_2","status":"completed"},
			{"type":"web_search_call","id":"ws_3","status":"completed"},
			{"type":"x_search_call","id":"xs_1","status":"completed"},
			{"type":"code_interpreter_call","id":"ci_1","status":"completed"},
			{"type":"image_generation_call","id":"ig_1","status":"completed","result":"AAAA"},
			{"type":"image_generation_call","id":"ie_2","status":"completed","result":"AAAA"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi","annotations":[{"type":"url_citation","url":"https://x.com/a/status/1"}]}]}],
		"usage":{"input_tokens":5000,"input_tokens_details":{"cached_tokens":1000},"output_tokens":200,
			"output_tokens_details":{"reasoning_tokens":800},"total_tokens":6000,"num_sources_used":0,"num_server_side_tools_used":4,
			"cost_in_usd_ticks":123456789,"cost_in_nano_usd":12345678,
			"server_side_tool_usage_details":{"web_search_calls":2,"x_search_calls":1,"x_posts_fetched":44,"x_users_fetched":3,
				"code_interpreter_calls":1,"file_search_calls":3,"mcp_calls":1,"document_search_calls":2,"image_generation_calls":2}}}`)

	usage := ExtractTokenUsage(body)

	require.NotNil(t, usage)
	assert.Equal(t, 5000, usage.PromptTokens)
	assert.Equal(t, 200, usage.CompletionTokens)
	assert.Equal(t, 800, usage.ReasoningTokens)
	assert.Equal(t, ReasoningAccountingAdditive, usage.ReasoningAccounting)
	assert.True(t, usage.ServerToolUsageReported)
	assert.Equal(t, 2, usage.WebSearchRequests, "the usage counter wins over the three output items")
	assert.Equal(t, 1, usage.XSearchCalls)
	assert.Equal(t, 44, usage.XSearchPosts)
	assert.Equal(t, 3, usage.XSearchProfiles)
	assert.Equal(t, 1, usage.CodeExecutionCalls)
	assert.Equal(t, 2, usage.AttachmentSearchCalls)
	assert.Equal(t, 3, usage.CollectionsSearchCalls)
	assert.Equal(t, 1, usage.MCPCalls)
	assert.Equal(t, 1, usage.ImageToolGenerations)
	assert.Equal(t, 1, usage.ImageToolEdits)
	assert.InDelta(t, 0.0123456789, usage.ProviderCostUSD, 1e-15, "ticks win over nano USD")
}

func TestExtractTokenUsage_ReportedZeroWebSearchIsAuthoritative(t *testing.T) {
	withCounter := []byte(`{"object":"response","status":"completed",
		"output":[{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"web_search_call","id":"ws_2","status":"completed"},
			{"type":"message","role":"assistant","content":[]}],
		"usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110,"server_side_tool_usage_details":{"web_search_calls":0}}}`)
	withoutCounter := []byte(`{"object":"response","status":"completed",
		"output":[{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"web_search_call","id":"ws_2","status":"completed"}],
		"usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110}}`)

	reported := ExtractTokenUsage(withCounter)
	require.NotNil(t, reported)
	assert.True(t, reported.ServerToolUsageReported)
	assert.Zero(t, reported.WebSearchRequests, "a reported zero must not be replaced by output items")

	missing := ExtractTokenUsage(withoutCounter)
	require.NotNil(t, missing)
	assert.False(t, missing.ServerToolUsageReported)
	assert.Equal(t, 2, missing.WebSearchRequests, "without a counter the completed items are still counted")
}

func TestExtractTokenUsage_XSearchOnlyIsNotBilledAsWebSearch(t *testing.T) {
	// X Search returns X post citations; they are no proof of a web search.
	body := []byte(`{"object":"response","status":"completed",
		"output":[{"type":"x_search_call","id":"xs_1","status":"completed"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi","annotations":[{"type":"url_citation","url":"https://x.com/a/status/1"}]}]}],
		"usage":{"input_tokens":5000,"output_tokens":200,"output_tokens_details":{"reasoning_tokens":800},"total_tokens":6000,
			"server_side_tool_usage_details":{"web_search_calls":0,"x_search_calls":3,"x_posts_fetched":120,"x_users_fetched":0}}}`)

	usage := ExtractTokenUsage(body)

	require.NotNil(t, usage)
	assert.Zero(t, usage.WebSearchRequests)
	assert.Equal(t, 120, usage.XSearchPosts)
	assert.Equal(t, 3, usage.XSearchCalls)
}

func TestExtractTokenUsage_StreamTerminalEvents(t *testing.T) {
	for _, eventType := range []string{"response.completed", "response.incomplete"} {
		t.Run(eventType, func(t *testing.T) {
			body := []byte(`{"type":"` + eventType + `","response":{"id":"resp_1","status":"completed","output":[
				{"type":"web_search_call","id":"ws_1","status":"completed"},
				{"type":"image_generation_call","id":"ie_1","status":"completed"}],
				"usage":{"input_tokens":300,"output_tokens":20,"output_tokens_details":{"reasoning_tokens":30},"total_tokens":350,
				"cost_in_usd_ticks":5000000000,
				"server_side_tool_usage_details":{"web_search_calls":1,"x_posts_fetched":7,"image_generation_calls":1}}}}`)

			usage := ExtractTokenUsage(body)

			require.NotNil(t, usage)
			assert.Equal(t, 300, usage.PromptTokens)
			assert.Equal(t, 20, usage.CompletionTokens)
			assert.Equal(t, ReasoningAccountingAdditive, usage.ReasoningAccounting)
			assert.True(t, usage.ServerToolUsageReported)
			assert.Equal(t, 1, usage.WebSearchRequests)
			assert.Equal(t, 7, usage.XSearchPosts)
			assert.Equal(t, 0, usage.ImageToolGenerations)
			assert.Equal(t, 1, usage.ImageToolEdits)
			assert.InDelta(t, 0.5, usage.ProviderCostUSD, 1e-12)
		})
	}
}

func imageCalls(n int) converterutil.ServerSideToolUsage {
	return converterutil.ServerSideToolUsage{ImageGenerationCalls: n}
}

func TestImageGenerationToolImages(t *testing.T) {
	items := []extractedOutputItem{
		{Type: "image_generation_call", ID: "ig_1", Status: "completed"},
		{Type: "image_generation_call", ID: "ie_2", Status: "completed"},
		{Type: "image_generation_call", ID: "ig_3", Status: "failed"},
		{Type: "message", ID: "msg_1"},
	}

	gens, edits := imageGenerationToolImages(items, nil, imageCalls(0), false)
	assert.Equal(t, [2]int{1, 1}, [2]int{gens, edits}, "without a counter the completed items are counted")

	gens, edits = imageGenerationToolImages(items, nil, imageCalls(3), true)
	assert.Equal(t, [2]int{2, 1}, [2]int{gens, edits}, "the counter is the total, items only tell the edits")

	gens, edits = imageGenerationToolImages(items, nil, imageCalls(0), true)
	assert.Equal(t, [2]int{0, 0}, [2]int{gens, edits}, "a reported zero is authoritative")

	gens, edits = imageGenerationToolImages(nil, items, imageCalls(0), false)
	assert.Equal(t, [2]int{1, 1}, [2]int{gens, edits}, "items of the streamed terminal event count too")
}

func TestExtractTokenUsage_ProviderCostShapes(t *testing.T) {
	tests := []struct {
		name string
		cost string
		want float64
	}{
		{"ticks", `"cost_in_usd_ticks":158500`, 0.00001585},
		{"nano", `"cost_in_nano_usd":2500`, 0.0000025},
		{"aggregator cost", `"cost":0.25`, 0.25},
		{"object is ignored", `"cost":{"total":1}`, 0},
		{"string is ignored", `"cost_in_usd_ticks":"10"`, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := ExtractTokenUsage([]byte(`{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,` + tt.cost + `}}`))
			require.NotNil(t, usage, "an unexpected cost shape must not drop the token counters")
			assert.Equal(t, 10, usage.PromptTokens)
			assert.InDelta(t, tt.want, usage.ProviderCostUSD, 1e-15)
		})
	}
}

func TestExtractTokenUsage_AnthropicFlatCacheSkipsReasoningDetection(t *testing.T) {
	usage := ExtractTokenUsage([]byte(`{"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":20,"total_tokens":35,
		"output_tokens_details":{"reasoning_tokens":5}}}`))
	require.NotNil(t, usage)
	assert.Equal(t, 30, usage.PromptTokens)
	assert.Empty(t, usage.ReasoningAccounting)
}

func TestMergeNonZero_ServerToolUsageIsCumulative(t *testing.T) {
	merged := &TokenUsage{}

	// An item-derived web search count from an earlier chunk...
	merged.MergeNonZero(&TokenUsage{WebSearchRequests: 3})
	assert.Equal(t, 3, merged.WebSearchRequests)

	// ...is replaced by the provider's object, zeros included.
	merged.MergeNonZero(&TokenUsage{PromptTokens: 10, ServerToolUsageReported: true, XSearchPosts: 5})
	assert.Equal(t, 0, merged.WebSearchRequests)
	assert.Equal(t, 5, merged.XSearchPosts)

	// The same cumulative object repeated by later chunks is not summed.
	merged.MergeNonZero(&TokenUsage{PromptTokens: 10, ServerToolUsageReported: true, XSearchPosts: 5})
	assert.Equal(t, 5, merged.XSearchPosts)

	// The total of the terminal event replaces the running figure.
	merged.MergeNonZero(&TokenUsage{PromptTokens: 12, ServerToolUsageReported: true, XSearchPosts: 9, WebSearchRequests: 1})
	assert.Equal(t, 9, merged.XSearchPosts)
	assert.Equal(t, 1, merged.WebSearchRequests)

	// A later chunk without the object cannot override it.
	merged.MergeNonZero(&TokenUsage{WebSearchRequests: 4, CodeExecutionCalls: 2, ReasoningAccounting: ReasoningAccountingAdditive, ProviderCostUSD: 0.5})
	assert.Equal(t, 1, merged.WebSearchRequests)
	assert.Equal(t, 0, merged.CodeExecutionCalls)
	assert.Equal(t, ReasoningAccountingAdditive, merged.ReasoningAccounting)
	assert.InDelta(t, 0.5, merged.ProviderCostUSD, 1e-12)
	assert.Equal(t, 12, merged.PromptTokens)
}

func TestTokenUsage_NormalizeClampsNewFields(t *testing.T) {
	usage := (&TokenUsage{
		XSearchPosts:        -1,
		ImageToolEdits:      -2,
		ReasoningAccounting: "bogus",
		ProviderCostUSD:     -3,
	}).Normalize()

	assert.Zero(t, usage.XSearchPosts)
	assert.Zero(t, usage.ImageToolEdits)
	assert.Empty(t, usage.ReasoningAccounting)
	assert.Zero(t, usage.ProviderCostUSD)
}
