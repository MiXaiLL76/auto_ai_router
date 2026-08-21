package anthropicresponses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropicToResponsesResponse_TextContent(t *testing.T) {
	body := `{
		"id": "msg_01",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "Hello, world!"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "response", resp.Object)
	assert.Equal(t, "claude-opus-4-5", resp.Model)
	assert.Equal(t, "completed", resp.Status)
	assert.Nil(t, resp.IncompleteDetails)

	require.Len(t, resp.Output, 1)
	msgItem := resp.Output[0]
	assert.Equal(t, "message", msgItem.Type)
	assert.Equal(t, "assistant", msgItem.Role)
	require.Len(t, msgItem.Content, 1)
	assert.Equal(t, "output_text", msgItem.Content[0].Type)
	assert.Equal(t, "Hello, world!", msgItem.Content[0].Text)

	require.NotNil(t, resp.Usage)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 5, resp.Usage.OutputTokens)
	assert.Equal(t, 15, resp.Usage.TotalTokens)
}

func TestAnthropicToResponsesResponse_MaxTokens(t *testing.T) {
	body := `{
		"id": "msg_02",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "Partial..."}],
		"stop_reason": "max_tokens",
		"usage": {"input_tokens": 100, "output_tokens": 200}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	assert.Equal(t, "incomplete", resp.Status)
	require.NotNil(t, resp.IncompleteDetails)
	assert.Equal(t, "max_output_tokens", resp.IncompleteDetails.Reason)
}

func TestAnthropicToResponsesResponse_ToolUse(t *testing.T) {
	body := `{
		"id": "msg_03",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "tool_use",
				"id": "tool_abc",
				"name": "get_weather",
				"input": {"city": "London"}
			}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 30}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	require.Len(t, resp.Output, 1)
	fc := resp.Output[0]
	assert.Equal(t, "function_call", fc.Type)
	assert.Equal(t, "tool_abc", fc.CallID)
	assert.Equal(t, "get_weather", fc.Name)

	var args map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(fc.Arguments), &args))
	assert.Equal(t, "London", args["city"])
}

// TestAnthropicToResponsesResponse_WebSearch verifies that a server_tool_use
// web_search block becomes a web_search_call output item (instead of being
// silently dropped, as it used to be before switch-case web_search support
// was added), and that citations on the following text block become
// url_citation annotations with offsets recovered from the cited_text.
func TestAnthropicToResponsesResponse_WebSearch(t *testing.T) {
	body := `{
		"id": "msg_05",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "server_tool_use",
				"id": "srvtoolu_01",
				"name": "web_search",
				"input": {"query": "claude shannon birth date"}
			},
			{
				"type": "web_search_tool_result",
				"tool_use_id": "srvtoolu_01",
				"content": [
					{"type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Claude_Shannon", "title": "Claude Shannon - Wikipedia"}
				]
			},
			{
				"type": "text",
				"text": "Claude Shannon was born on April 30, 1916.",
				"citations": [
					{
						"type": "web_search_result_location",
						"url": "https://en.wikipedia.org/wiki/Claude_Shannon",
						"title": "Claude Shannon - Wikipedia",
						"cited_text": "born on April 30, 1916"
					}
				]
			}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 20, "output_tokens": 30}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	require.Len(t, resp.Output, 2)

	wsCall := resp.Output[0]
	assert.Equal(t, "web_search_call", wsCall.Type)
	assert.Equal(t, "completed", wsCall.Status)
	require.Len(t, wsCall.Queries, 1)
	assert.Equal(t, "claude shannon birth date", wsCall.Queries[0])

	msg := resp.Output[1]
	assert.Equal(t, "message", msg.Type)
	require.Len(t, msg.Content, 1)
	text := msg.Content[0]
	assert.Equal(t, "Claude Shannon was born on April 30, 1916.", text.Text)
	require.Len(t, text.Annotations, 1)
	ann := text.Annotations[0]
	assert.Equal(t, "url_citation", ann.Type)
	assert.Equal(t, "https://en.wikipedia.org/wiki/Claude_Shannon", ann.URL)
	assert.Equal(t, "Claude Shannon - Wikipedia", ann.Title)
	wantStart := strings.Index(text.Text, "born on April 30, 1916")
	assert.Equal(t, wantStart, ann.StartIndex)
	assert.Equal(t, wantStart+len("born on April 30, 1916"), ann.EndIndex)
}

// TestAnthropicToResponsesResponse_WebSearchCitationNotFoundSkipped verifies
// that a citation whose cited_text doesn't appear verbatim in the block's
// text is skipped rather than emitted with a bogus zero offset.
func TestAnthropicToResponsesResponse_WebSearchCitationNotFoundSkipped(t *testing.T) {
	body := `{
		"id": "msg_06",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "text",
				"text": "Some answer text.",
				"citations": [
					{"type": "web_search_result_location", "url": "https://example.com", "cited_text": "not present anywhere"}
				]
			}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 5, "output_tokens": 5}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	require.Len(t, resp.Output, 1)
	require.Len(t, resp.Output[0].Content, 1)
	assert.Empty(t, resp.Output[0].Content[0].Annotations)
}

func TestAnthropicToResponsesResponse_Thinking(t *testing.T) {
	body := `{
		"id": "msg_04",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "thinking",
				"thinking": "I am reasoning about this...",
				"signature": "enc_sig_xyz"
			},
			{"type": "text", "text": "Here is my answer."}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 50, "output_tokens": 80}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	// Output: reasoning item first, then message item
	require.Len(t, resp.Output, 2)

	reasoning := resp.Output[0]
	assert.Equal(t, "reasoning", reasoning.Type)
	require.Len(t, reasoning.Summary, 1)
	assert.Equal(t, "summary_text", reasoning.Summary[0].Type)
	assert.Equal(t, "I am reasoning about this...", reasoning.Summary[0].Text)
	assert.Equal(t, "enc_sig_xyz", reasoning.EncryptedContent)

	msg := resp.Output[1]
	assert.Equal(t, "message", msg.Type)
	require.Len(t, msg.Content, 1)
	assert.Equal(t, "Here is my answer.", msg.Content[0].Text)
}

func TestAnthropicToResponsesResponse_EmptyContent(t *testing.T) {
	body := `{
		"id": "msg_05",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 5, "output_tokens": 0}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	// Should return at least one message item (even if empty)
	require.NotEmpty(t, resp.Output)
	assert.Equal(t, "message", resp.Output[0].Type)
}

func TestAnthropicToResponsesResponse_CachedTokens(t *testing.T) {
	body := `{
		"id": "msg_06",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "ok"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50,
			"cache_read_input_tokens": 80
		}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	assert.Equal(t, 80, resp.Usage.InputTokensDetails.CachedTokens)
}

func TestAnthropicToResponsesResponse_CacheUsageUsesInclusiveInputTotal(t *testing.T) {
	body := `{
		"id": "msg_cache_total",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "ok"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 10,
			"cache_read_input_tokens": 80,
			"cache_creation_input_tokens": 20,
			"cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 15}
		}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 200, resp.Usage.InputTokens)
	assert.Equal(t, 210, resp.Usage.TotalTokens)

	raw, err := json.Marshal(resp.Usage.InputTokensDetails)
	require.NoError(t, err)
	var details map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &details))
	assert.Equal(t, float64(80), details["cached_tokens"])
	assert.Equal(t, float64(20), details["cache_creation_tokens"])
	ttlDetails := details["cache_creation_token_details"].(map[string]interface{})
	assert.Equal(t, float64(5), ttlDetails["ephemeral_5m_input_tokens"])
	assert.Equal(t, float64(15), ttlDetails["ephemeral_1h_input_tokens"])
}

func TestAnthropicToResponsesResponse_CustomResponseID(t *testing.T) {
	body := `{
		"id": "msg_07",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "hi"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 1, "output_tokens": 1}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "resp_custom_id", 1234567890)
	require.NoError(t, err)

	assert.Equal(t, "resp_custom_id", resp.ID)
	assert.Equal(t, int64(1234567890), resp.CreatedAt)
}

func TestAnthropicToResponsesResponse_RequiredSchemaFields(t *testing.T) {
	body := `{
		"id": "msg_schema",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "tool_use", "id": "tool_abc", "name": "get_weather", "input": {"city": "London"}}],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 30}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

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

func TestAnthropicStopReasonToStatus(t *testing.T) {
	tests := []struct {
		reason   string
		status   string
		hasExtra bool
	}{
		{"end_turn", "completed", false},
		{"tool_use", "completed", false},
		{"stop_sequence", "completed", false},
		{"", "completed", false},
		{"max_tokens", "incomplete", true},
		{"unknown_reason", "completed", false},
	}

	for _, tc := range tests {
		t.Run(tc.reason, func(t *testing.T) {
			status, details := anthropicStopReasonToStatus(tc.reason)
			assert.Equal(t, tc.status, status)
			if tc.hasExtra {
				require.NotNil(t, details)
			} else {
				assert.Nil(t, details)
			}
		})
	}
}

func TestAnthropicToResponsesResponse_ComputerCall(t *testing.T) {
	body := `{
		"id": "msg_computer",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "tool_use",
				"id": "toolu_abc123",
				"name": "computer",
				"input": {"action": "screenshot"}
			}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	require.Len(t, resp.Output, 1)
	ci := resp.Output[0]
	assert.Equal(t, "computer_call", ci.Type)
	assert.Equal(t, "completed", ci.Status)
	assert.Equal(t, "toolu_abc123", ci.CallID)
	assert.Equal(t, "computer", ci.Name)
	action, ok := ci.Action.(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "screenshot", action["action"])
}

func TestAnthropicToResponsesResponse_ToolUseWithoutAction_IsFunctionCall(t *testing.T) {
	body := `{
		"id": "msg_fn",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [
			{
				"type": "tool_use",
				"id": "toolu_fn1",
				"name": "get_weather",
				"input": {"city": "Paris"}
			}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	require.Len(t, resp.Output, 1)
	fc := resp.Output[0]
	assert.Equal(t, "function_call", fc.Type)
	assert.Equal(t, "toolu_fn1", fc.CallID)
	assert.Equal(t, "get_weather", fc.Name)
	assert.Contains(t, fc.Arguments, "Paris")
}

func TestAnthropicUsageToUsage(t *testing.T) {
	body := `{
		"id": "msg_08",
		"type": "message",
		"role": "assistant",
		"model": "claude-opus-4-5",
		"content": [{"type": "text", "text": "x"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50,
			"cache_read_input_tokens": 30
		}
	}`

	resp, err := AnthropicToResponsesResponse([]byte(body), "claude-opus-4-5", "", 0)
	require.NoError(t, err)

	assert.Equal(t, 130, resp.Usage.InputTokens)
	assert.Equal(t, 50, resp.Usage.OutputTokens)
	assert.Equal(t, 180, resp.Usage.TotalTokens)
	assert.Equal(t, 30, resp.Usage.InputTokensDetails.CachedTokens)
}
