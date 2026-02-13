package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/transform/common"
	"github.com/stretchr/testify/assert"
	"google.golang.org/genai"
)

func TestOpenAIToolsConversion(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "basic tool conversion",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "What's the weather?"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "get_weather",
						"description": "Get weather for a location",
						"parameters": {
							"type": "object",
							"properties": {
								"location": {
									"type": "string",
									"description": "City name"
								},
								"unit": {
									"type": "string",
									"enum": ["celsius", "fahrenheit"],
									"description": "Temperature unit"
								}
							},
							"required": ["location"]
						}
					}
				}]
			}`,
			expected: map[string]interface{}{
				"tools": []interface{}{
					map[string]interface{}{
						"functionDeclarations": []interface{}{
							map[string]interface{}{
								"name":        "get_weather",
								"description": "Get weather for a location",
								"parameters": map[string]interface{}{
									"type": "OBJECT",
									"properties": map[string]interface{}{
										"location": map[string]interface{}{
											"type":        "STRING",
											"description": "City name",
										},
										"unit": map[string]interface{}{
											"type":        "STRING",
											"description": "Temperature unit",
											"enum":        []interface{}{"celsius", "fahrenheit"},
										},
									},
									"required": []interface{}{"location"},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple tools",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Help me"}],
				"tools": [
					{
						"type": "function",
						"function": {
							"name": "get_weather",
							"description": "Get weather",
							"parameters": {
								"type": "object",
								"properties": {
									"location": {"type": "string"}
								}
							}
						}
					},
					{
						"type": "function",
						"function": {
							"name": "search",
							"description": "Search the web",
							"parameters": {
								"type": "object",
								"properties": {
									"query": {"type": "string"}
								},
								"required": ["query"]
							}
						}
					}
				]
			}`,
			expected: map[string]interface{}{
				"tools": []interface{}{
					map[string]interface{}{
						"functionDeclarations": []interface{}{
							map[string]interface{}{
								"name": "get_weather",
							},
							map[string]interface{}{
								"name": "search",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool with array parameter type",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "process_items",
						"description": "Process multiple items",
						"parameters": {
							"type": "object",
							"properties": {
								"items": {
									"type": "array",
									"items": {"type": "string"},
									"description": "List of items"
								}
							}
						}
					}
				}]
			}`,
			expected: map[string]interface{}{
				"tools": []interface{}{
					map[string]interface{}{
						"functionDeclarations": []interface{}{
							map[string]interface{}{
								"name": "process_items",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool with nested object parameters",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"tools": [{
					"type": "function",
					"function": {
						"name": "create_user",
						"description": "Create a new user",
						"parameters": {
							"type": "object",
							"properties": {
								"name": {"type": "string", "description": "User name"},
								"email": {"type": "string", "description": "User email"},
								"age": {"type": "integer", "description": "User age"}
							},
							"required": ["name", "email"]
						}
					}
				}]
			}`,
			expected: map[string]interface{}{
				"tools": []interface{}{
					map[string]interface{}{
						"functionDeclarations": []interface{}{
							map[string]interface{}{
								"name":        "create_user",
								"description": "Create a new user",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:     "no tools in request",
			input:    `{"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "Hello"}]}`,
			expected: map[string]interface{}{},
			wantErr:  false,
		},
		{
			name:    "invalid json",
			input:   `{"invalid": json}`,
			wantErr: true,
		},
		{
			name: "tool with no function field",
			input: `{
				"model": "gemini-2.5-pro",
				"messages": [{"role": "user", "content": "Test"}],
				"tools": [{"type": "function"}]
			}`,
			expected: map[string]interface{}{},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := OpenAIToVertex([]byte(tt.input), false, "")

			if tt.wantErr {
				assert.Error(t, err, "Expected error but got none")
				return
			}

			assert.NoError(t, err, "Unexpected error")

			var resultMap map[string]interface{}
			err = json.Unmarshal(result, &resultMap)
			assert.NoError(t, err, "Failed to unmarshal result")

			// Check if tools field exists when expected
			if len(tt.expected) > 0 && tt.expected["tools"] != nil {
				assert.NotNil(t, resultMap["tools"], "Expected tools field in result")

				// Check structure
				tools, ok := resultMap["tools"].([]interface{})
				assert.True(t, ok, "Expected tools to be array")
				assert.Greater(t, len(tools), 0, "Expected at least one tool")

				if len(tools) > 0 {
					toolObj, ok := tools[0].(map[string]interface{})
					assert.True(t, ok, "Expected tool to be object")

					_, hasFuncDecls := toolObj["functionDeclarations"]
					assert.True(t, hasFuncDecls, "Expected functionDeclarations field")
				}
			} else if len(tt.expected) == 0 || tt.expected["tools"] == nil {
				// Should not have tools field or it should be empty
				tools, exists := resultMap["tools"]
				if exists {
					toolsArr, ok := tools.([]interface{})
					assert.True(t, ok || len(toolsArr) == 0, "Expected empty tools or no tools field")
				}
			}
		})
	}
}

func TestConvertOpenAIToolsToVertex(t *testing.T) {
	tests := []struct {
		name          string
		input         []interface{}
		expectedLen   int
		expectedNames []string
	}{
		{
			name: "single tool",
			input: []interface{}{
				map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        "test_function",
						"description": "A test function",
						"parameters": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"arg1": map[string]interface{}{
									"type":        "string",
									"description": "Argument 1",
								},
							},
						},
					},
				},
			},
			expectedLen:   1,
			expectedNames: []string{"test_function"},
		},
		{
			name: "multiple tools",
			input: []interface{}{
				map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        "func1",
						"description": "First function",
						"parameters": map[string]interface{}{
							"type": "object",
						},
					},
				},
				map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        "func2",
						"description": "Second function",
						"parameters": map[string]interface{}{
							"type": "object",
						},
					},
				},
			},
			expectedLen:   2,
			expectedNames: []string{"func1", "func2"},
		},
		{
			name:        "empty input",
			input:       []interface{}{},
			expectedLen: 0,
		},
		{
			name: "tool without function field",
			input: []interface{}{
				map[string]interface{}{
					"type": "function",
				},
			},
			expectedLen: 0,
		},
		{
			name: "non-function type tool",
			input: []interface{}{
				map[string]interface{}{
					"type": "code_interpreter",
				},
			},
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertOpenAIToolsToVertex(tt.input)

			if tt.expectedLen == 0 {
				assert.Nil(t, result, "Expected nil result")
				return
			}

			assert.NotNil(t, result, "Expected non-nil result")
			assert.Len(t, result, 1, "Expected single tool wrapper")

			tool := result[0]
			assert.Len(t, tool.FunctionDeclarations, tt.expectedLen, "Function declaration count mismatch")

			for i, expectedName := range tt.expectedNames {
				assert.Equal(t, expectedName, tool.FunctionDeclarations[i].Name, "Function name mismatch at index %d", i)
			}
		})
	}
}

func TestConvertOpenAIParamsToVertex(t *testing.T) {
	tests := []struct {
		name               string
		input              map[string]interface{}
		expectedType       string
		expectedPropsCount int
		expectedRequired   []string
	}{
		{
			name: "basic parameters",
			input: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "City name",
					},
				},
				"required": []interface{}{"location"},
			},
			expectedType:       "OBJECT",
			expectedPropsCount: 1,
			expectedRequired:   []string{"location"},
		},
		{
			name: "multiple properties with different types",
			input: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type": "string",
					},
					"age": map[string]interface{}{
						"type": "integer",
					},
					"active": map[string]interface{}{
						"type": "boolean",
					},
				},
				"required": []interface{}{"name", "age"},
			},
			expectedType:       "OBJECT",
			expectedPropsCount: 3,
			expectedRequired:   []string{"name", "age"},
		},
		{
			name: "property with enum",
			input: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"unit": map[string]interface{}{
						"type": "string",
						"enum": []interface{}{"celsius", "fahrenheit"},
					},
				},
			},
			expectedType:       "OBJECT",
			expectedPropsCount: 1,
		},
		{
			name: "empty parameters",
			input: map[string]interface{}{
				"type": "object",
			},
			expectedType:       "OBJECT",
			expectedPropsCount: 0,
		},
		{
			name: "array parameter",
			input: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"items": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
			expectedType:       "OBJECT",
			expectedPropsCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertOpenAIParamsToGenaiSchema(tt.input)

			assert.NotNil(t, result, "Expected non-nil schema")
			assert.Equal(t, genai.Type(tt.expectedType), result.Type, "Type mismatch")
			assert.Equal(t, tt.expectedPropsCount, len(result.Properties), "Properties count mismatch")

			if len(tt.expectedRequired) > 0 {
				assert.Equal(t, tt.expectedRequired, result.Required, "Required fields mismatch")
			}
		})
	}
}

func TestGetString(t *testing.T) {
	tests := []struct {
		name     string
		input    map[string]interface{}
		key      string
		expected string
	}{
		{
			name: "existing string value",
			input: map[string]interface{}{
				"name": "test",
			},
			key:      "name",
			expected: "test",
		},
		{
			name: "missing key",
			input: map[string]interface{}{
				"name": "test",
			},
			key:      "missing",
			expected: "",
		},
		{
			name: "non-string value",
			input: map[string]interface{}{
				"count": 42,
			},
			key:      "count",
			expected: "",
		},
		{
			name:     "empty map",
			input:    map[string]interface{}{},
			key:      "any",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := common.GetString(tt.input, tt.key)
			assert.Equal(t, tt.expected, result, "Unexpected result")
		})
	}
}

func TestToolsWithOtherParameters(t *testing.T) {
	// Test that tools work alongside other parameters
	input := `{
		"model": "gemini-2.5-pro",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": "What's the weather?"}
		],
		"temperature": 0.7,
		"max_tokens": 200,
		"top_p": 0.9,
		"tools": [{
			"type": "function",
			"function": {
				"name": "get_weather",
				"description": "Get weather information",
				"parameters": {
					"type": "object",
					"properties": {
						"location": {"type": "string"}
					},
					"required": ["location"]
				}
			}
		}]
	}`

	result, err := OpenAIToVertex([]byte(input), false, "")
	assert.NoError(t, err)

	var resultMap map[string]interface{}
	err = json.Unmarshal(result, &resultMap)
	assert.NoError(t, err)

	// Check all fields are present
	assert.NotNil(t, resultMap["contents"])
	assert.NotNil(t, resultMap["systemInstruction"])
	assert.NotNil(t, resultMap["generationConfig"])
	assert.NotNil(t, resultMap["tools"])

	// Verify generation config
	genConfig, ok := resultMap["generationConfig"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, 0.7, genConfig["temperature"])
	assert.Equal(t, float64(200), genConfig["maxOutputTokens"])
	assert.Equal(t, 0.9, genConfig["topP"])

	// Verify tools
	tools, ok := resultMap["tools"].([]interface{})
	assert.True(t, ok)
	assert.Len(t, tools, 1)
}

func TestStreamingLimitations(t *testing.T) {
	// Document current limitations of streaming with tool calls
	// TransformVertexStreamToOpenAI in streaming.go:
	// - Only processes text content (line 77: if part.Text != "")
	// - Does not handle tool_use parts from Vertex responses
	// - Would need to extract function name and arguments from tool_use blocks
	// - And convert them to OpenAI's tool_calls format

	// This is a documentation test - actual streaming with tool_calls
	// requires enhancement to streaming.go TransformVertexStreamToOpenAI
	t.Log("Streaming with tool_calls requires TransformVertexStreamToOpenAI enhancement")
}

// TestVertexToOpenAIWithFunctionCall tests function call response conversion to tool_calls
func TestVertexToOpenAIWithFunctionCall(t *testing.T) {
	vertexResp := `{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"functionCall": {
								"name": "get_weather",
								"args": {
									"location": "London"
								}
							}
						}
					]
				},
				"finishReason": "TOOL_CALL"
			}
		],
		"usageMetadata": {
			"promptTokenCount": 100,
			"candidatesTokenCount": 50,
			"totalTokenCount": 150
		}
	}`

	result, err := VertexToOpenAI([]byte(vertexResp), "gemini-2.0-pro")
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
	assert.NotEmpty(t, toolCall["id"])
	assert.Equal(t, "function", toolCall["type"])

	function := toolCall["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", function["name"])

	// Arguments should be a JSON string
	args := function["arguments"].(string)
	var parsedArgs map[string]interface{}
	err = json.Unmarshal([]byte(args), &parsedArgs)
	assert.NoError(t, err)
	assert.Equal(t, "London", parsedArgs["location"])

	// Verify finish reason
	assert.Equal(t, "tool_calls", choice["finish_reason"])
}

// TestVertexToOpenAIWithMixedContent tests function call mixed with text
func TestVertexToOpenAIWithFunctionCallAndText(t *testing.T) {
	vertexResp := `{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"text": "I'll check the weather for you."
						},
						{
							"functionCall": {
								"name": "get_weather",
								"args": {
									"location": "Paris",
									"unit": "celsius"
								}
							}
						}
					]
				},
				"finishReason": "TOOL_CALL"
			}
		],
		"usageMetadata": {
			"promptTokenCount": 100,
			"candidatesTokenCount": 50,
			"totalTokenCount": 150
		}
	}`

	result, err := VertexToOpenAI([]byte(vertexResp), "gemini-2.0-pro")
	assert.NoError(t, err)

	var openAIResp interface{}
	err = json.Unmarshal(result, &openAIResp)
	assert.NoError(t, err)

	respMap := openAIResp.(map[string]interface{})
	choices := respMap["choices"].([]interface{})

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	// Verify text content
	assert.Equal(t, "I'll check the weather for you.", message["content"])

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

// TestVertexToOpenAIWithMultipleFunctionCalls tests multiple function calls
func TestVertexToOpenAIWithMultipleFunctionCalls(t *testing.T) {
	vertexResp := `{
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"functionCall": {
								"name": "get_weather",
								"args": {"location": "London"}
							}
						},
						{
							"functionCall": {
								"name": "get_weather",
								"args": {"location": "Paris"}
							}
						}
					]
				},
				"finishReason": "TOOL_CALL"
			}
		],
		"usageMetadata": {
			"promptTokenCount": 100,
			"candidatesTokenCount": 50,
			"totalTokenCount": 150
		}
	}`

	result, err := VertexToOpenAI([]byte(vertexResp), "gemini-2.0-pro")
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
	function1 := toolCall1["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", function1["name"])

	// Check second tool call
	toolCall2 := toolCalls[1].(map[string]interface{})
	function2 := toolCall2["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", function2["name"])
}

// TestMapFinishReasonToolCall tests TOOL_CALL finish reason mapping
func TestMapFinishReasonToolCall(t *testing.T) {
	tests := []struct {
		vertexReason string
		expected     string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"RECITATION", "content_filter"},
		{"TOOL_CALL", "tool_calls"},
		{"UNKNOWN_REASON", "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.vertexReason, func(t *testing.T) {
			result := mapFinishReason(tt.vertexReason)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestToolMessageConversion tests that tool messages are properly converted
func TestToolMessageConversion(t *testing.T) {
	input := `{
		"model": "gemini-2.5-pro",
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": "I'll check", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]},
			{"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "call_123"},
			{"role": "user", "content": "Thanks!"}
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
		}]
	}`

	result, err := OpenAIToVertex([]byte(input), false, "")
	assert.NoError(t, err)

	var resultMap map[string]interface{}
	err = json.Unmarshal(result, &resultMap)
	assert.NoError(t, err)

	// Check contents
	contents, ok := resultMap["contents"].([]interface{})
	assert.True(t, ok, "Expected contents to be array")

	// Should have at least 4 messages: user, assistant (model), tool->user, user
	assert.GreaterOrEqual(t, len(contents), 3)

	// Find the tool result message - should be converted to user role
	toolResultFound := false
	for _, c := range contents {
		content, ok := c.(map[string]interface{})
		assert.True(t, ok)

		if role, ok := content["role"].(string); ok && role == "user" {
			if parts, ok := content["parts"].([]interface{}); ok && len(parts) > 0 {
				part, ok := parts[0].(map[string]interface{})
				assert.True(t, ok)

				if text, ok := part["text"].(string); ok && text == "Sunny, 22°C" {
					toolResultFound = true
					break
				}
			}
		}
	}

	assert.True(t, toolResultFound, "Tool result should be converted to user message with role='user'")
}

// TestToolMessageWithoutToolCallId tests tool messages without explicit tool_call_id
func TestToolMessageWithoutToolCallId(t *testing.T) {
	input := `{
		"model": "gemini-2.5-pro",
		"messages": [
			{"role": "user", "content": "Process this"},
			{"role": "tool", "content": "Result from external tool"}
		]
	}`

	result, err := OpenAIToVertex([]byte(input), false, "")
	assert.NoError(t, err)

	var resultMap map[string]interface{}
	err = json.Unmarshal(result, &resultMap)
	assert.NoError(t, err)

	contents, ok := resultMap["contents"].([]interface{})
	assert.True(t, ok)

	// Tool message should be converted to user role
	toolMsgFound := false
	for _, c := range contents {
		content := c.(map[string]interface{})
		if role, ok := content["role"].(string); ok && role == "user" {
			if parts, ok := content["parts"].([]interface{}); ok && len(parts) > 0 {
				part := parts[0].(map[string]interface{})
				if text, ok := part["text"].(string); ok && text == "Result from external tool" {
					toolMsgFound = true
					break
				}
			}
		}
	}

	assert.True(t, toolMsgFound, "Tool message should be converted to user message")
}
