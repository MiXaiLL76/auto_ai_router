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
