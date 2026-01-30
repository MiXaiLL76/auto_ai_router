package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestAnthropicToolRequest tests that tools in OpenAI format are passed through to Anthropic
func TestAnthropicToolRequest(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		checkFn func(t *testing.T, req *AnthropicRequest)
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
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Equal(t, "You are a helpful assistant", req.System)
				assert.Len(t, req.Messages, 1)
				assert.Equal(t, "user", req.Messages[0].Role)
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
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				systemStr, ok := req.System.(string)
				assert.True(t, ok)
				assert.Contains(t, systemStr, "Base system prompt")
				assert.Contains(t, systemStr, "Developer instructions")
				assert.Len(t, req.Messages, 1)
			},
		},
		{
			name: "assistant with tool_calls",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [
					{"role": "user", "content": "Call a function"},
					{
						"role": "assistant",
						"content": "I'll call the function",
						"tool_calls": [
							{
								"id": "call_123",
								"type": "function",
								"function": {
									"name": "test_function",
									"arguments": "{\"param1\": 123, \"param2\": \"value\"}"
								}
							}
						]
					}
				]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Len(t, req.Messages, 2)

				// Check assistant message with tool_calls
				assistantMsg := req.Messages[1]
				assert.Equal(t, "assistant", assistantMsg.Role)

				// Content should be array with text and tool_use blocks
				contentBlocks, ok := assistantMsg.Content.([]interface{})
				assert.True(t, ok, "Content should be array for tool_calls")
				assert.Len(t, contentBlocks, 2)

				// First block should be text
				textBlock, ok := contentBlocks[0].(map[string]interface{})
				assert.True(t, ok)
				assert.Equal(t, "text", textBlock["type"])
				assert.Equal(t, "I'll call the function", textBlock["text"])

				// Second block should be tool_use
				toolBlock, ok := contentBlocks[1].(map[string]interface{})
				assert.True(t, ok)
				assert.Equal(t, "tool_use", toolBlock["type"])
				assert.Equal(t, "call_123", toolBlock["id"])
				assert.Equal(t, "test_function", toolBlock["name"])

				// Check input is parsed map, not string
				input, ok := toolBlock["input"].(map[string]interface{})
				assert.True(t, ok, "Input should be parsed map")
				assert.Equal(t, float64(123), input["param1"])
				assert.Equal(t, "value", input["param2"])
			},
		},
		{
			name: "assistant with null content and tool_calls",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [
					{"role": "user", "content": "Call function"},
					{
						"role": "assistant",
						"content": null,
						"tool_calls": [
							{
								"id": "call_456",
								"type": "function",
								"function": {
									"name": "get_weather",
									"arguments": "{\"location\": \"London\"}"
								}
							}
						]
					}
				]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Len(t, req.Messages, 2)

				assistantMsg := req.Messages[1]
				contentBlocks, ok := assistantMsg.Content.([]interface{})
				assert.True(t, ok)

				// Should have only tool_use block (no text)
				assert.Len(t, contentBlocks, 1)

				toolBlock, ok := contentBlocks[0].(map[string]interface{})
				assert.True(t, ok)
				assert.Equal(t, "tool_use", toolBlock["type"])
				assert.Equal(t, "get_weather", toolBlock["name"])
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
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.NotNil(t, req.Tools)
				assert.Len(t, req.Tools, 1)
				assert.NotNil(t, req.ToolChoice)

				// Verify tool converted to Anthropic format
				toolMap, ok := req.Tools[0].(map[string]interface{})
				assert.True(t, ok)
				assert.Equal(t, "get_weather", toolMap["name"])
				assert.Equal(t, "Get weather for a location", toolMap["description"])
				assert.NotNil(t, toolMap["input_schema"])
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
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Len(t, req.Tools, 2)
			},
		},
		{
			name: "no tools in request",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "Hello"}]
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Nil(t, req.Tools)
				assert.Nil(t, req.ToolChoice)
			},
		},
		{
			name: "tool_choice string value",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "Test"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "test",
						"description": "Test",
						"parameters": {"type": "object"}
					}
				}],
				"tool_choice": "required"
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.Equal(t, "required", req.ToolChoice)
			},
		},
		{
			name: "tool_choice object value",
			input: `{
				"model": "claude-3-opus-20240229",
				"messages": [{"role": "user", "content": "Test"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "test",
						"description": "Test",
						"parameters": {"type": "object"}
					}
				}],
				"tool_choice": {
					"type": "function",
					"function": {"name": "test"}
				}
			}`,
			wantErr: false,
			checkFn: func(t *testing.T, req *AnthropicRequest) {
				assert.NotNil(t, req.ToolChoice)
				toolChoice, ok := req.ToolChoice.(map[string]interface{})
				assert.True(t, ok)
				assert.Equal(t, "function", toolChoice["type"])
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

			var req AnthropicRequest
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

// TestAnthropicToolResponseContent documents the limitation with tool_use blocks
func TestAnthropicToolResponseLimitation(t *testing.T) {
	// Current limitation: AnthropicToOpenAI only handles "text" type blocks
	// It does not process "tool_use" blocks from Anthropic responses
	//
	// An Anthropic response with tool_use looks like:
	// {
	//   "content": [
	//     {
	//       "type": "tool_use",
	//       "id": "tool_use_123",
	//       "name": "get_weather",
	//       "input": {"location": "London"}
	//     }
	//   ],
	//   "stop_reason": "tool_use"
	// }
	//
	// This needs to be converted to OpenAI tool_calls format:
	// {
	//   "choices": [{
	//     "message": {
	//       "tool_calls": [{
	//         "id": "tool_use_123",
	//         "type": "function",
	//         "function": {
	//           "name": "get_weather",
	//           "arguments": "{\"location\": \"London\"}"
	//         }
	//       }]
	//     }
	//   }]
	// }

	input := `{
		"id": "msg_123",
		"type": "message",
		"role": "assistant",
		"content": [
			{
				"type": "tool_use",
				"id": "tool_use_456",
				"name": "get_weather",
				"input": {"location": "London"}
			}
		],
		"stop_reason": "tool_use",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50
		}
	}`

	result, err := AnthropicToOpenAI([]byte(input), "claude-3-opus")
	assert.NoError(t, err)

	var respMap map[string]interface{}
	err = json.Unmarshal(result, &respMap)
	assert.NoError(t, err)

	// Current behavior: tool_use block is ignored, no content extracted
	// Expected behavior: tool_use block should be converted to tool_calls
	choices, ok := respMap["choices"].([]interface{})
	assert.True(t, ok)

	choice, ok := choices[0].(map[string]interface{})
	assert.True(t, ok)

	message, ok := choice["message"].(map[string]interface{})
	assert.True(t, ok)

	// Currently content will be empty because tool_use blocks are not processed
	content := message["content"].(string)
	assert.Equal(t, "", content)

	// Finish reason is correctly mapped to "tool_calls"
	finishReason := choice["finish_reason"].(string)
	assert.Equal(t, "tool_calls", finishReason)

	t.Log("Note: tool_use response handling requires AnthropicToOpenAI enhancement")
}

// TestAnthropicToolRequestWithMessages tests tools work with various message formats
func TestAnthropicToolRequestWithMessages(t *testing.T) {
	input := `{
		"model": "claude-3-opus-20240229",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What's the weather in London?"}
				]
			}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": "get_weather",
				"description": "Get weather",
				"parameters": {
					"type": "object",
					"properties": {
						"location": {"type": "string"}
					},
					"required": ["location"]
				}
			}
		}],
		"temperature": 0.7,
		"max_tokens": 500
	}`

	result, err := OpenAIToAnthropic([]byte(input))
	assert.NoError(t, err)

	var req AnthropicRequest
	err = json.Unmarshal(result, &req)
	assert.NoError(t, err)

	// Verify all parameters preserved
	assert.Equal(t, "claude-3-opus-20240229", req.Model)
	assert.Len(t, req.Messages, 1)
	assert.NotNil(t, req.Tools)
	assert.Equal(t, 0.7, *req.Temperature)
	assert.Equal(t, 500, req.MaxTokens)
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
	assert.Equal(t, `{"location":"London"}`, args)

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

	// Verify complex input arguments
	args := function["arguments"].(string)
	var parsedArgs map[string]interface{}
	err = json.Unmarshal([]byte(args), &parsedArgs)
	assert.NoError(t, err)
	assert.Equal(t, "Paris", parsedArgs["location"])
	assert.Equal(t, "celsius", parsedArgs["unit"])
}

// TestAnthropicToOpenAIWithMultipleToolCalls tests multiple tool_use blocks
func TestAnthropicToOpenAIWithMultipleToolCalls(t *testing.T) {
	anthropicResp := `{
		"id": "msg_test789",
		"type": "message",
		"role": "assistant",
		"content": [
			{
				"type": "tool_use",
				"id": "tool_use_1",
				"name": "get_weather",
				"input": {"location": "London"}
			},
			{
				"type": "tool_use",
				"id": "tool_use_2",
				"name": "get_weather",
				"input": {"location": "Paris"}
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

	// Verify multiple tool_calls
	toolCalls := message["tool_calls"].([]interface{})
	assert.Len(t, toolCalls, 2)

	// Check first tool call
	toolCall1 := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "tool_use_1", toolCall1["id"])

	// Check second tool call
	toolCall2 := toolCalls[1].(map[string]interface{})
	assert.Equal(t, "tool_use_2", toolCall2["id"])
}
