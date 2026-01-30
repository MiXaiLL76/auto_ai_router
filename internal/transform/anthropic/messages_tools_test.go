package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
)

// TestAnthropicToolRequest tests that tools in OpenAI format are passed through to Anthropic
func TestAnthropicToolRequest(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		checkFn func(t *testing.T, req *anthropic.MessageNewParams)
	}{
		{
			name: "developer role as system instruction",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [
					{"role": "developer", "content": "You are a helpful assistant"},
					{"role": "user", "content": "Hello"}
				]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *anthropic.MessageNewParams) {
				assert.Len(t, req.System, 1)
				assert.Equal(t, req.System[0].Text, "You are a helpful assistant")
				assert.Len(t, req.Messages, 1)
				assert.Equal(t, anthropic.MessageParamRoleUser, req.Messages[0].Role)
			},
		},
		{
			name: "developer role combined with system",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [
					{"role": "system", "content": "Base system prompt"},
					{"role": "developer", "content": "Developer instructions"},
					{"role": "user", "content": "Hello"}
				]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *anthropic.MessageNewParams) {
				assert.Len(t, req.System, 2)
				assert.Contains(t, req.System[0].Text, "Base system prompt")
				assert.Contains(t, req.System[1].Text, "Developer instructions")
				assert.Len(t, req.Messages, 1)
			},
		},
		{
			name: "tool request conversion",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "What tools do I have?"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "get_weather",
						"description": "Get weather for a location",
						"parameters": {
							"type": "object",
							"properties": {
								"location": {"type": "string"}
							}
						}
					}
				}],
				"tool_choice": "auto"
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *anthropic.MessageNewParams) {
				assert.NotNil(t, req.Tools)
				assert.Len(t, req.Tools, 1)
				assert.NotNil(t, req.ToolChoice)

				// Verify tool converted to Anthropic format
				tool := req.Tools[0]
				assert.Equal(t, "get_weather", tool.OfTool.Name)
				// Description is a param.Opt[string]
			},
		},
		{
			name: "multiple tools",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "Help"}],
				"tools": [
					{
						"type": "function",
						"function": {
							"name": "tool1",
							"description": "Tool 1",
							"parameters": {"type": "object"}
						}
					},
					{
						"type": "function",
						"function": {
							"name": "tool2",
							"description": "Tool 2",
							"parameters": {"type": "object"}
						}
					}
				]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *anthropic.MessageNewParams) {
				assert.Len(t, req.Tools, 2)
				assert.Equal(t, "tool1", req.Tools[0].OfTool.Name)
				assert.Equal(t, "tool2", req.Tools[1].OfTool.Name)
			},
		},
		{
			name: "no tools in request",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "Hello"}]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *anthropic.MessageNewParams) {
				assert.Nil(t, req.Tools)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := OpenAIToAnthropic([]byte(tt.input))

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)

			var req anthropic.MessageNewParams
			err = json.Unmarshal(result, &req)
			assert.NoError(t, err)

			tt.checkFn(t, &req)
		})
	}
}

// TestAnthropicToolResponse tests that tool_use in stop_reason maps correctly
func TestAnthropicToolResponse(t *testing.T) {
	tests := []struct {
		name           string
		stopReason     string
		expectedReason string
	}{
		{
			name:           "tool_use stop reason",
			stopReason:     "tool_use",
			expectedReason: "tool_calls",
		},
		{
			name:           "end_turn stop reason",
			stopReason:     "end_turn",
			expectedReason: "stop",
		},
		{
			name:           "max_tokens stop reason",
			stopReason:     "max_tokens",
			expectedReason: "length",
		},
		{
			name:           "stop_sequence stop reason",
			stopReason:     "stop_sequence",
			expectedReason: "stop",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapAnthropicStopReason(tt.stopReason)
			assert.Equal(t, tt.expectedReason, result)
		})
	}
}

// TestAnthropicToOpenAIWithToolCall tests tool_use response conversion to tool_calls
func TestAnthropicToOpenAIWithToolCall(t *testing.T) {
	anthropicResp := `{
		"id": "msg_test123",
		"type": "message",
		"role": "assistant",
		"content": [
			{
				"type": "tool_use",
				"id": "tool_use_123",
				"name": "get_weather",
				"input": {"location": "London"}
			}
		],
		"model": "claude-3-opus-20240229",
		"stop_reason": "tool_use",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50
		}
	}`

	result, err := AnthropicToOpenAI([]byte(anthropicResp), "claude-3-opus-20240229")
	assert.NoError(t, err)

	var openAIResp interface{}
	err = json.Unmarshal(result, &openAIResp)
	assert.NoError(t, err)

	respMap := openAIResp.(map[string]interface{})
	choices := respMap["choices"].([]interface{})
	assert.Len(t, choices, 1)

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	// Verify tool_calls are present
	toolCalls := message["tool_calls"].([]interface{})
	assert.Len(t, toolCalls, 1)

	toolCall := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "tool_use_123", toolCall["id"])
	assert.Equal(t, "function", toolCall["type"])

	function := toolCall["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", function["name"])

	// Arguments should be a JSON string
	args := function["arguments"].(string)
	assert.JSONEq(t, `{"location":"London"}`, args)

	// Verify finish reason
	assert.Equal(t, "tool_calls", choice["finish_reason"])
}

// TestAnthropicToOpenAIWithMixedContent tests tool_use mixed with text
func TestAnthropicToOpenAIWithMixedContent(t *testing.T) {
	anthropicResp := `{
		"id": "msg_test456",
		"type": "message",
		"role": "assistant",
		"content": [
			{
				"type": "text",
				"text": "Let me check the weather for you."
			},
			{
				"type": "tool_use",
				"id": "tool_use_456",
				"name": "get_weather",
				"input": {"location": "Paris", "unit": "celsius"}
			}
		],
		"model": "claude-3-opus-20240229",
		"stop_reason": "tool_use",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50
		}
	}`

	result, err := AnthropicToOpenAI([]byte(anthropicResp), "claude-3-opus-20240229")
	assert.NoError(t, err)

	var openAIResp interface{}
	err = json.Unmarshal(result, &openAIResp)
	assert.NoError(t, err)

	respMap := openAIResp.(map[string]interface{})
	choices := respMap["choices"].([]interface{})

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	// Verify text content
	assert.Equal(t, "Let me check the weather for you.", message["content"])

	// Verify tool_calls
	toolCalls := message["tool_calls"].([]interface{})
	assert.Len(t, toolCalls, 1)

	toolCall := toolCalls[0].(map[string]interface{})
	function := toolCall["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", function["name"])
}
