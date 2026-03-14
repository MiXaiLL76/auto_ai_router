package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsResponsesAPI(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected bool
	}{
		{
			name:     "string input without messages",
			body:     `{"model":"gpt-4o","input":"hello"}`,
			expected: true,
		},
		{
			name:     "array input without messages",
			body:     `{"model":"gpt-4o","input":[{"role":"user","content":"hello"}]}`,
			expected: true,
		},
		{
			name:     "messages without input",
			body:     `{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`,
			expected: false,
		},
		{
			name:     "both input and messages",
			body:     `{"model":"gpt-4o","input":"hello","messages":[]}`,
			expected: false,
		},
		{
			name:     "empty body",
			body:     ``,
			expected: false,
		},
		{
			name:     "invalid json",
			body:     `{broken`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsResponsesAPI([]byte(tt.body))
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRequestToChat_StringInput(t *testing.T) {
	body := `{"model":"gpt-4o","input":"What is 2+2?","temperature":0.5}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	// Should have messages, not input
	assert.Contains(t, parsed, "messages")
	assert.NotContains(t, parsed, "input")

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 1)

	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
	assert.Equal(t, "What is 2+2?", msg["content"])

	// temperature should be preserved
	assert.Equal(t, 0.5, parsed["temperature"])
}

func TestRequestToChat_MessageInput(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi!"},
			{"role": "user", "content": "How are you?"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 3)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[1].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[2].(map[string]interface{})["role"])
}

func TestRequestToChat_InputObject(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": {"role": "user", "content": "Hello object"}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 1)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "Hello object", messages[0].(map[string]interface{})["content"])
}

func TestRequestToChat_Instructions(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"instructions": "You are a pirate.",
		"input": "Hello"
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 2)

	// First message should be developer
	assert.Equal(t, "developer", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "You are a pirate.", messages[0].(map[string]interface{})["content"])

	// Second should be user
	assert.Equal(t, "user", messages[1].(map[string]interface{})["role"])

	// instructions should be removed
	assert.NotContains(t, parsed, "instructions")
}

func TestRequestToChat_InstructionsArray(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"instructions": [{"role": "system", "content": "System msg"}],
		"input": "Hello"
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 2)
	assert.Equal(t, "system", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "System msg", messages[0].(map[string]interface{})["content"])
	assert.Equal(t, "user", messages[1].(map[string]interface{})["role"])
}

func TestRequestToChat_MaxOutputTokens(t *testing.T) {
	body := `{"model":"gpt-4o","input":"hi","max_output_tokens":100}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Contains(t, parsed, "max_completion_tokens")
	assert.Equal(t, float64(100), parsed["max_completion_tokens"])
	assert.NotContains(t, parsed, "max_output_tokens")
}

func TestRequestToChat_Tools(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "What's the weather?",
		"tools": [
			{
				"type": "function",
				"name": "get_weather",
				"description": "Get weather info",
				"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	tools := parsed["tools"].([]interface{})
	assert.Len(t, tools, 1)

	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "function", tool["type"])

	// Should be nested format
	funcDef := tool["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", funcDef["name"])
	assert.Equal(t, "Get weather info", funcDef["description"])
	assert.NotNil(t, funcDef["parameters"])
}

func TestRequestToChat_ToolsNestedFormat(t *testing.T) {
	// Tools passed in Chat Completions nested format (function key wrapping fields)
	body := `{
		"model": "gpt-4o",
		"input": "What's the weather?",
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather info",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	tools := parsed["tools"].([]interface{})
	assert.Len(t, tools, 1)

	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "function", tool["type"])

	funcDef := tool["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", funcDef["name"])
	assert.Equal(t, "Get weather info", funcDef["description"])
	assert.NotNil(t, funcDef["parameters"])
}

func TestRequestToChat_InputImageURLAsObject(t *testing.T) {
	// image_url provided as object {url: "...", detail: "..."} (SDK may send this form)
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_image", "image_url": {"url": "data:image/png;base64,AAA", "detail": "auto"}}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "image_url", part["type"])
	img := part["image_url"].(map[string]interface{})
	assert.Equal(t, "data:image/png;base64,AAA", img["url"])
}

func TestRequestToChat_ToolChoice(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "hi",
		"tool_choice": {"type": "function", "name": "get_weather"}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	tc := parsed["tool_choice"].(map[string]interface{})
	assert.Equal(t, "function", tc["type"])

	funcMap := tc["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", funcMap["name"])
}

func TestRequestToChat_ToolChoiceString(t *testing.T) {
	body := `{"model":"gpt-4o","input":"hi","tool_choice":"auto"}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Equal(t, "auto", parsed["tool_choice"])
}

func TestRequestToChat_ToolChoiceUnsupported(t *testing.T) {
	body := `{"model":"gpt-4o","input":"hi","tool_choice":{"type":"file_search"}}`
	_, err := RequestToChat([]byte(body))
	require.Error(t, err)
}

func TestRequestToChat_FunctionCallOutput(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "What's the weather?"},
			{
				"type": "function_call",
				"call_id": "call_123",
				"name": "get_weather",
				"arguments": "{\"city\":\"Paris\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_123",
				"output": "Sunny, 25C"
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 3)

	// First: user message
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])

	// Second: assistant with tool_calls
	assistantMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", assistantMsg["role"])
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	assert.Len(t, toolCalls, 1)
	tc := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "call_123", tc["id"])
	assert.Equal(t, "function", tc["type"])
	funcInfo := tc["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", funcInfo["name"])
	assert.Equal(t, `{"city":"Paris"}`, funcInfo["arguments"])

	// Third: tool message
	toolMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "call_123", toolMsg["tool_call_id"])
	assert.Equal(t, "Sunny, 25C", toolMsg["content"])
}

func TestRequestToChat_FunctionCallOutput_Object(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"type": "function_call_output", "call_id": "call_123", "output": {"ok": true}}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 1)
	toolMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "call_123", toolMsg["tool_call_id"])
	assert.Equal(t, `{"ok":true}`, toolMsg["content"])
}

func TestRequestToChat_MultipleFunctionCallsMerged(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Do two things"},
			{
				"type": "function_call",
				"call_id": "call_1",
				"name": "get_weather",
				"arguments": "{\"city\":\"Paris\"}"
			},
			{
				"type": "function_call",
				"call_id": "call_2",
				"name": "get_time",
				"arguments": "{\"tz\":\"UTC\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_1",
				"output": "Sunny"
			},
			{
				"type": "function_call_output",
				"call_id": "call_2",
				"output": "12:00"
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// user + 1 assistant (merged) + 2 tool outputs = 4
	assert.Len(t, messages, 4)

	// First: user message
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])

	// Second: single assistant message with TWO tool_calls
	assistantMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", assistantMsg["role"])
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	assert.Len(t, toolCalls, 2)

	tc1 := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "call_1", tc1["id"])
	assert.Equal(t, "get_weather", tc1["function"].(map[string]interface{})["name"])

	tc2 := toolCalls[1].(map[string]interface{})
	assert.Equal(t, "call_2", tc2["id"])
	assert.Equal(t, "get_time", tc2["function"].(map[string]interface{})["name"])

	// Third and Fourth: tool messages
	assert.Equal(t, "tool", messages[2].(map[string]interface{})["role"])
	assert.Equal(t, "tool", messages[3].(map[string]interface{})["role"])
}

func TestRequestToChat_JsonSchemaFormat(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "Output JSON",
		"text": {
			"format": {
				"type": "json_schema",
				"name": "my_schema",
				"schema": {"type": "object", "properties": {"name": {"type": "string"}}},
				"strict": true
			}
		}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	rf := parsed["response_format"].(map[string]interface{})
	assert.Equal(t, "json_schema", rf["type"])

	// Should be wrapped in json_schema key for Chat Completions
	jsonSchema := rf["json_schema"].(map[string]interface{})
	assert.Equal(t, "my_schema", jsonSchema["name"])
	assert.NotNil(t, jsonSchema["schema"])
	assert.Equal(t, true, jsonSchema["strict"])

	// "type" should NOT be inside json_schema (it's at the top level only)
	_, hasType := jsonSchema["type"]
	assert.False(t, hasType)

	assert.NotContains(t, parsed, "text")
}

func TestRequestToChat_ContentParts(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Describe this image"},
					{"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 1)

	content := messages[0].(map[string]interface{})["content"].([]interface{})
	assert.Len(t, content, 2)

	// input_text -> text
	textPart := content[0].(map[string]interface{})
	assert.Equal(t, "text", textPart["type"])
	assert.Equal(t, "Describe this image", textPart["text"])

	// input_image -> image_url
	imagePart := content[1].(map[string]interface{})
	assert.Equal(t, "image_url", imagePart["type"])
	imgURL := imagePart["image_url"].(map[string]interface{})
	assert.Equal(t, "https://example.com/img.png", imgURL["url"])
	assert.Equal(t, "high", imgURL["detail"])
}

func TestRequestToChat_InputAudio(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_audio", "data": "BASE64", "format": "wav"}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "input_audio", part["type"])
	audio := part["input_audio"].(map[string]interface{})
	assert.Equal(t, "BASE64", audio["data"])
	assert.Equal(t, "wav", audio["format"])
}

func TestRequestToChat_InputImageDataURL(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_image", "image_url": "data:image/png;base64,AAA"}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "image_url", part["type"])
	img := part["image_url"].(map[string]interface{})
	assert.Equal(t, "data:image/png;base64,AAA", img["url"])
}

func TestRequestToChat_ResponsesAPIFieldsDropped(t *testing.T) {
	// type, phase, status are Responses-API-only message fields.
	// They must be stripped before forwarding to Chat Completions providers
	// which reject unknown parameters on message objects.
	body := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "assistant", "content": "draft", "phase": "commentary", "status": "completed"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "assistant", msg["role"])
	assert.Equal(t, "draft", msg["content"])
	assert.NotContains(t, msg, "type", "type must be stripped from Chat Completions messages")
	assert.NotContains(t, msg, "phase", "phase must be stripped from Chat Completions messages")
	assert.NotContains(t, msg, "status", "status must be stripped from Chat Completions messages")
}

func TestRequestToChat_InputOutputTextAsInput(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "assistant",
				"content": [
					{"type": "output_text", "text": "previous"}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "text", part["type"])
	assert.Equal(t, "previous", part["text"])
}

func TestRequestToChat_InputOutputRefusalAsInput(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "assistant",
				"content": [
					{"type": "output_refusal", "refusal": "nope"}
				]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "text", part["type"])
	assert.Equal(t, "nope", part["text"])
}

func TestRequestToChat_InputImageFileIDUnsupported(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_image", "file_id": "file_123"}
				]
			}
		]
	}`
	_, err := RequestToChat([]byte(body))
	require.Error(t, err)
}

func TestRequestToChat_InputFileUnsupported(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "input_file", "file_id": "file_123", "filename": "test.pdf"}
				]
			}
		]
	}`
	_, err := RequestToChat([]byte(body))
	require.Error(t, err)
}

func TestRequestToChat_Reasoning(t *testing.T) {
	body := `{
		"model": "o1",
		"input": "Think carefully about this.",
		"reasoning": {"effort": "high"}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Equal(t, "high", parsed["reasoning_effort"])
	assert.NotContains(t, parsed, "reasoning")
}

func TestRequestToChat_ReasoningNone(t *testing.T) {
	// effort: "none" should NOT be converted to reasoning_effort
	body := `{
		"model": "gpt-4o",
		"input": "Say pong",
		"reasoning": {"effort": "none"}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.NotContains(t, parsed, "reasoning_effort", "effort='none' should not be converted")
	assert.NotContains(t, parsed, "reasoning")
}

func TestRequestToChat_TextFormat(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "Output JSON",
		"text": {"format": {"type": "json_object"}}
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	rf := parsed["response_format"].(map[string]interface{})
	assert.Equal(t, "json_object", rf["type"])
	assert.NotContains(t, parsed, "text")
}

func TestRequestToChat_PreservesOtherFields(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "hello",
		"temperature": 0.7,
		"top_p": 0.9,
		"stream": true,
		"user": "test-user",
		"conversation": "conv_123",
		"include": ["message.output_text.logprobs"],
		"stream_options": {"include_obfuscation": true},
		"truncation": "auto",
		"safety_identifier": "user_abc",
		"service_tier": "flex"
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Equal(t, "gpt-4o", parsed["model"])
	assert.Equal(t, 0.7, parsed["temperature"])
	assert.Equal(t, 0.9, parsed["top_p"])
	assert.Equal(t, true, parsed["stream"])
	assert.Equal(t, "test-user", parsed["user"])
	assert.NotContains(t, parsed, "conversation")
	assert.NotContains(t, parsed, "include")
	assert.NotContains(t, parsed, "stream_options")
	assert.NotContains(t, parsed, "truncation")
	assert.NotContains(t, parsed, "safety_identifier")
	assert.NotContains(t, parsed, "service_tier")
}

func TestRequestToChat_NonFunctionToolsRemoved(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": "search the web",
		"tools": [
			{"type": "web_search", "name": "web_search"},
			{"type": "function", "name": "my_func", "description": "My function", "parameters": {}}
		]
	}`
	_, err := RequestToChat([]byte(body))
	require.Error(t, err)
}

func TestRequestToChat_MessageWithType(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	assert.Len(t, messages, 1)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "Hello", messages[0].(map[string]interface{})["content"])
}
