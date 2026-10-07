package converterutil

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServerSideToolUsage_DecodesXAIObject(t *testing.T) {
	// Shape of xAI's usage.server_side_tool_usage_details (Responses API docs).
	usage := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(`{
		"web_search_calls": 2,
		"x_search_calls": 2,
		"x_posts_fetched": 44,
		"x_users_fetched": 3,
		"code_interpreter_calls": 1,
		"file_search_calls": 4,
		"mcp_calls": 5,
		"document_search_calls": 6,
		"image_generation_calls": 7
	}`)}

	got, ok := usage.ServerSideToolUsage()

	require.True(t, ok)
	assert.Equal(t, ServerSideToolUsage{
		WebSearchCalls:         2,
		XSearchCalls:           2,
		XPostsFetched:          44,
		XUsersFetched:          3,
		CodeExecutionCalls:     1,
		AttachmentSearchCalls:  6,
		CollectionsSearchCalls: 4,
		MCPCalls:               5,
		ImageGenerationCalls:   7,

		WebSearchCallsReported:       true,
		ImageGenerationCallsReported: true,
	}, got)
}

func TestServerSideToolUsage_AbsentIsNotZero(t *testing.T) {
	for _, raw := range []string{``, `null`, `[]`, `"x"`, `{"web_search_calls":`} {
		t.Run(raw, func(t *testing.T) {
			_, ok := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(raw)}.ServerSideToolUsage()
			assert.False(t, ok, "no usable object means no data, not zero usage")
		})
	}

	got, ok := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(`{"web_search_calls":0}`)}.ServerSideToolUsage()
	require.True(t, ok, "an object reporting zero is reported usage")
	assert.Equal(t, ServerSideToolUsage{WebSearchCallsReported: true}, got)
}

func TestServerSideToolUsage_CounterPresence(t *testing.T) {
	tests := []struct {
		name               string
		raw                string
		web, image         int
		webSeen, imageSeen bool
	}{
		{"counters missing", `{"x_search_calls":1}`, 0, 0, false, false},
		{"counters zero", `{"web_search_calls":0,"image_generation_calls":0}`, 0, 0, true, true},
		{"counters set", `{"web_search_calls":2,"image_generation_calls":3}`, 2, 3, true, true},
		{"counters null", `{"web_search_calls":null,"image_generation_calls":null}`, 0, 0, false, false},
		{"counters malformed", `{"web_search_calls":"2","image_generation_calls":-1}`, 0, 0, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(tt.raw)}.ServerSideToolUsage()
			require.True(t, ok)
			assert.Equal(t, tt.web, got.WebSearchCalls)
			assert.Equal(t, tt.webSeen, got.WebSearchCallsReported)
			assert.Equal(t, tt.image, got.ImageGenerationCalls)
			assert.Equal(t, tt.imageSeen, got.ImageGenerationCallsReported)
		})
	}
}

func TestServerSideToolUsage_AliasesAreNotSummed(t *testing.T) {
	got, ok := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(`{
		"code_interpreter_calls": 2, "code_execution_calls": 2,
		"file_search_calls": 1, "collections_search_calls": 3,
		"document_search_calls": 4, "attachment_search_calls": 1
	}`)}.ServerSideToolUsage()

	require.True(t, ok)
	assert.Equal(t, 2, got.CodeExecutionCalls)
	assert.Equal(t, 3, got.CollectionsSearchCalls)
	assert.Equal(t, 4, got.AttachmentSearchCalls)
}

func TestServerSideToolUsage_MalformedCounterDoesNotHideOthers(t *testing.T) {
	got, ok := ToolUsageExtensions{ServerSideToolUsageDetails: json.RawMessage(`{
		"web_search_calls": "two", "x_posts_fetched": -5, "x_users_fetched": null,
		"code_interpreter_calls": 1.9, "mcp_calls": {"n": 1}, "file_search_calls": 3
	}`)}.ServerSideToolUsage()

	require.True(t, ok)
	assert.Equal(t, ServerSideToolUsage{CodeExecutionCalls: 1, CollectionsSearchCalls: 3}, got)
}

func TestToolUsageExtensions_RoundTripKeepsServerSideToolUsage(t *testing.T) {
	var usage struct {
		InputTokens int `json:"input_tokens"`
		ToolUsageExtensions
	}
	require.NoError(t, json.Unmarshal([]byte(`{"input_tokens":5,"server_side_tool_usage_details":{"x_posts_fetched":9}}`), &usage))

	encoded, err := json.Marshal(usage)
	require.NoError(t, err)
	assert.JSONEq(t, `{"input_tokens":5,"server_side_tool_usage_details":{"x_posts_fetched":9}}`, string(encoded))
}
