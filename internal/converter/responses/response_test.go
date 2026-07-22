package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatToResponse_BasicText(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "The capital of France is Paris."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 8,
			"total_tokens": 18
		}
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	assert.Equal(t, "response", resp.Object)
	assert.Equal(t, "gpt-4o", resp.Model)
	assert.Equal(t, "completed", resp.Status)
	assert.Equal(t, int64(1700000000), resp.CreatedAt)

	assert.Len(t, resp.Output, 1)
	assert.Equal(t, "message", resp.Output[0].Type)
	assert.Equal(t, "assistant", resp.Output[0].Role)
	assert.Equal(t, "completed", resp.Output[0].Status)
	assert.Len(t, resp.Output[0].Content, 1)
	assert.Equal(t, "output_text", resp.Output[0].Content[0].Type)
	assert.Equal(t, "The capital of France is Paris.", resp.Output[0].Content[0].Text)
	assert.NotNil(t, resp.Output[0].Content[0].Annotations)
}

func TestChatToResponse_ContentArray(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
			},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	assert.Len(t, resp.Output, 1)
	assert.Equal(t, "message", resp.Output[0].Type)
	assert.Equal(t, "hi", resp.Output[0].Content[0].Text)
}

func TestChatToResponse_Refusal(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"refusal": "I can not do that"
			},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	assert.Len(t, resp.Output, 1)
	assert.Equal(t, "output_refusal", resp.Output[0].Content[0].Type)
	assert.Equal(t, "I can not do that", resp.Output[0].Content[0].Refusal)
}

func TestChatToResponse_ContentArray_NoText(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
			},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	// Fallback empty message is created when no output items are generated
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "message", resp.Output[0].Type)
	assert.Equal(t, "", resp.Output[0].Content[0].Text)
}

func TestChatToResponse_ToolCalls(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_xyz",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"Paris\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 15,
			"completion_tokens": 10,
			"total_tokens": 25
		}
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	assert.Equal(t, "completed", resp.Status)

	// Should have function_call output item (no message since content is empty)
	var fcItems []OutputItem
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			fcItems = append(fcItems, item)
		}
	}
	assert.Len(t, fcItems, 1)
	assert.Equal(t, "call_xyz", fcItems[0].CallID)
	assert.Equal(t, "get_weather", fcItems[0].Name)
	assert.Equal(t, `{"city":"Paris"}`, fcItems[0].Arguments)
	assert.Equal(t, "completed", fcItems[0].Status)
}

func TestChatToResponse_RequiredSchemaFields(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "z-ai/glm-4.7-flash",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "I'll check the current weather in San Francisco for you.",
				"tool_calls": [{
					"id": "call_xyz",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"location\":\"San Francisco, CA\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 188,
			"completion_tokens": 26,
			"total_tokens": 214
		}
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Equal(t, "disabled", parsed["truncation"])
	assert.Equal(t, "default", parsed["service_tier"])
	assert.Equal(t, "auto", parsed["tool_choice"])
	assert.Equal(t, float64(1), parsed["temperature"])
	assert.Equal(t, float64(1), parsed["top_p"])
	_, hasText := parsed["text"]
	assert.True(t, hasText)
}

func TestChatToResponse_Usage(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hi"},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 100,
			"completion_tokens": 50,
			"total_tokens": 150,
			"prompt_tokens_details": {"cached_tokens": 20},
			"completion_tokens_details": {"reasoning_tokens": 10}
		}
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	require.NotNil(t, resp.Usage)
	assert.Equal(t, 100, resp.Usage.InputTokens)
	assert.Equal(t, 50, resp.Usage.OutputTokens)
	assert.Equal(t, 150, resp.Usage.TotalTokens)
	assert.NotNil(t, resp.Usage.InputTokensDetails)
	assert.Equal(t, 20, resp.Usage.InputTokensDetails.CachedTokens)
	assert.NotNil(t, resp.Usage.OutputTokensDetails)
	assert.Equal(t, 10, resp.Usage.OutputTokensDetails.ReasoningTokens)
}

func TestChatToResponse_GeminiImageAndImageTokenUsage(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-image",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gemini-3.1-flash-image-preview",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 22,
			"completion_tokens": 1120,
			"total_tokens": 1142,
			"completion_tokens_details": {"image_tokens": 1120}
		}
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "image_generation_call", resp.Output[0].Type)
	assert.Equal(t, "aW1hZ2U=", resp.Output[0].Result)
	assert.Equal(t, "png", resp.Output[0].OutputFormat)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 1120, resp.Usage.OutputTokensDetails.ImageTokens)
}

func TestChatToResponse_Status(t *testing.T) {
	tests := []struct {
		finishReason string
		status       string
	}{
		{"stop", "completed"},
		{"length", "incomplete"},
		{"content_filter", "incomplete"},
		{"tool_calls", "completed"},
	}

	for _, tt := range tests {
		t.Run(tt.finishReason, func(t *testing.T) {
			ccBody := `{
				"id": "chatcmpl-abc123",
				"object": "chat.completion",
				"created": 1700000000,
				"model": "gpt-4o",
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "hi"},
					"finish_reason": "` + tt.finishReason + `"
				}]
			}`

			result, err := ChatToResponse([]byte(ccBody))
			require.NoError(t, err)

			var resp Response
			require.NoError(t, json.Unmarshal(result, &resp))

			assert.Equal(t, tt.status, resp.Status)
		})
	}
}

func TestChatToResponse_EmptyContent(t *testing.T) {
	// Tool calls with no text content
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "fn1", "arguments": "{}"}
				}]
			},
			"finish_reason": "tool_calls"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	// Should have only function_call items, no empty message
	for _, item := range resp.Output {
		if item.Type == "message" {
			assert.Fail(t, "should not have message output when content is empty")
		}
	}
	assert.Len(t, resp.Output, 1)
	assert.Equal(t, "function_call", resp.Output[0].Type)
}

func TestChatToResponse_MultipleChoices(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hi"},
			"finish_reason": "stop"
		},{
			"index": 1,
			"message": {"role": "assistant", "content": "hello"},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	assert.Len(t, resp.Output, 2)
	assert.Equal(t, "hi", resp.Output[0].Content[0].Text)
	assert.Equal(t, "hello", resp.Output[1].Content[0].Text)
}

func TestChatToResponse_Structure(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hi"},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	// ID should start with "resp_"
	assert.True(t, strings.HasPrefix(resp.ID, "resp_"))
	assert.Equal(t, "response", resp.Object)

	// Output item IDs should have proper prefixes
	assert.True(t, strings.HasPrefix(resp.Output[0].ID, "msg_"))
}

// Phase 3: WithExtraOutputItems option

func TestChatToResponse_WithExtraOutputItems(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-abc",
		"object": "chat.completion",
		"created": 1700000001,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hello"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
	}`

	extra := []OutputItem{
		{
			Type:    "web_search_call",
			ID:      "ws_injected",
			Status:  "completed",
			Queries: []string{"grounding query"},
		},
	}

	result, err := ChatToResponse([]byte(ccBody), WithExtraOutputItems(extra))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	// Should have the regular message item + the injected web_search_call item
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "message", resp.Output[0].Type)
	assert.Equal(t, "web_search_call", resp.Output[1].Type)
	assert.Equal(t, "ws_injected", resp.Output[1].ID)
	assert.Equal(t, []string{"grounding query"}, resp.Output[1].Queries)
}

func TestChatToResponse_WithExtraOutputItems_Multiple(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-xyz",
		"object": "chat.completion",
		"created": 1700000002,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "found it"},
			"finish_reason": "stop"
		}]
	}`

	extra := []OutputItem{
		{Type: "web_search_call", ID: "ws_1", Status: "completed"},
		{Type: "web_search_call", ID: "ws_2", Status: "completed"},
	}

	result, err := ChatToResponse([]byte(ccBody), WithExtraOutputItems(extra))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	require.Len(t, resp.Output, 3)
	assert.Equal(t, "message", resp.Output[0].Type)
	assert.Equal(t, "ws_1", resp.Output[1].ID)
	assert.Equal(t, "ws_2", resp.Output[2].ID)
}

func TestOutputContentMarshalJSON_TextAlwaysPresent(t *testing.T) {
	// Regression: output_text content must always include "text" in JSON even when
	// the value is an empty string. When "text" is absent the Python OpenAI SDK parses
	// the attribute as None, causing TypeError in "".join([None, ...]).
	cases := []struct {
		name     string
		content  OutputContent
		wantText interface{} // expected value of "text" key in JSON
	}{
		{
			name:     "non-empty text",
			content:  OutputContent{Type: "output_text", Text: "Paris"},
			wantText: "Paris",
		},
		{
			name:     "empty text",
			content:  OutputContent{Type: "output_text", Text: ""},
			wantText: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.content)
			require.NoError(t, err)
			var m map[string]interface{}
			require.NoError(t, json.Unmarshal(b, &m))
			val, ok := m["text"]
			assert.True(t, ok, `"text" key must be present in output_text JSON`)
			assert.Equal(t, tc.wantText, val)
			// annotations must also be present (even empty)
			_, hasAnnotations := m["annotations"]
			assert.True(t, hasAnnotations, `"annotations" key must be present for output_text`)
		})
	}
}

func TestOutputContentMarshalJSON_NonOutputTextOmitsEmptyText(t *testing.T) {
	// For non-output_text types, empty text is still omitted (no change to existing behaviour).
	c := OutputContent{Type: "summary_text", Text: ""}
	b, err := json.Marshal(c)
	require.NoError(t, err)
	var m map[string]interface{}
	require.NoError(t, json.Unmarshal(b, &m))
	_, hasText := m["text"]
	assert.False(t, hasText, `"text" key must be absent for non-output_text when empty`)
}

func TestChatToResponse_WithoutOptions_Unchanged(t *testing.T) {
	ccBody := `{
		"id": "chatcmpl-base",
		"object": "chat.completion",
		"created": 1700000003,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "ok"},
			"finish_reason": "stop"
		}]
	}`

	result, err := ChatToResponse([]byte(ccBody))
	require.NoError(t, err)

	var resp Response
	require.NoError(t, json.Unmarshal(result, &resp))

	// No extra items — only the message
	require.Len(t, resp.Output, 1)
	assert.Equal(t, "message", resp.Output[0].Type)
}
