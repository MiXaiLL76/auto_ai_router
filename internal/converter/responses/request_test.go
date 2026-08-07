package responses

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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
	// max_output_tokens must be converted to max_tokens (universal Chat Completions
	// parameter). Renaming to max_completion_tokens for reasoning models is done
	// later by openai.ReplaceBodyParam after conversion.
	body := `{"model":"gpt-4o","input":"hi","max_output_tokens":100}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.Contains(t, parsed, "max_tokens")
	assert.Equal(t, float64(100), parsed["max_tokens"])
	assert.NotContains(t, parsed, "max_output_tokens")
	assert.NotContains(t, parsed, "max_completion_tokens")
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

func TestRequestToChat_ToolChoiceNonFunctionPassthrough(t *testing.T) {
	// Non-function tool_choice (e.g. web_search_preview, file_search) references
	// Responses-API built-in tools.  RequestToChat passes it through so that
	// provider-specific converters downstream can handle it.
	body := `{"model":"gpt-4o","input":"hi","tool_choice":{"type":"file_search"}}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	tc := parsed["tool_choice"].(map[string]interface{})
	assert.Equal(t, "file_search", tc["type"])
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
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_file.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
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

func TestPrepareCodexPassthrough_DropsReasoningNone(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o-mini",
		"input": "Say pong",
		"reasoning": {"effort": "none"}
	}`)

	result := PrepareCodexPassthrough(body, false)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	assert.NotContains(t, parsed, "reasoning", "reasoning.effort='none' should be dropped for native passthrough")
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

func TestRequestToChat_NonFunctionToolsFiltered(t *testing.T) {
	// Non-function tools (web_search, web_search_preview, etc.) are Responses-API
	// built-in constructs with no Chat Completions equivalent. RequestToChat filters
	// them out so generic Chat Completions providers don't reject the request.
	// Provider-specific native paths (Vertex, Anthropic) handle built-in tools
	// through their own converters, which bypass RequestToChat entirely.
	body := `{
		"model": "gpt-4o",
		"input": "search the web",
		"tools": [
			{"type": "web_search", "name": "web_search"},
			{"type": "web_search_preview"},
			{"type": "function", "name": "my_func", "description": "My function", "parameters": {}}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	tools := parsed["tools"].([]interface{})
	assert.Len(t, tools, 1, "only function tools must be kept; non-function tools are filtered out")

	// Only the function tool remains in nested format
	tool0 := tools[0].(map[string]interface{})
	assert.Equal(t, "function", tool0["type"])
	funcDef := tool0["function"].(map[string]interface{})
	assert.Equal(t, "my_func", funcDef["name"])
}

func TestRequestToChat_AllNonFunctionToolsRemoved(t *testing.T) {
	// When all tools are non-function, tools and tool_choice must both be removed
	// to avoid sending an empty tools array or orphaned tool_choice to providers.
	body := `{
		"model": "gpt-4o",
		"input": "search the web",
		"tools": [{"type": "web_search_preview"}, {"type": "file_search"}],
		"tool_choice": "auto"
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	_, hasTools := parsed["tools"]
	assert.False(t, hasTools, "tools key must be absent when no function tools remain")

	_, hasToolChoice := parsed["tool_choice"]
	assert.False(t, hasToolChoice, "tool_choice must be removed when no function tools remain")
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

// Phase 2: input item type tests

func TestRequestToChat_ReasoningItem(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Think hard"},
			{
				"type": "reasoning",
				"id": "rs_001",
				"summary": [{"type": "summary_text", "text": "The answer is 42."}],
				"encrypted_content": "enc_should_be_dropped"
			},
			{"role": "user", "content": "So?"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 3)

	// reasoning → synthetic assistant message with summary text
	reasoningMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", reasoningMsg["role"])
	content := reasoningMsg["content"].([]interface{})
	require.Len(t, content, 1)
	part := content[0].(map[string]interface{})
	assert.Equal(t, "text", part["type"])
	assert.Contains(t, part["text"].(string), "The answer is 42.")

	// encrypted_content must not appear
	assert.NotContains(t, reasoningMsg, "encrypted_content")
}

func TestRequestToChat_ReasoningItem_NoSummary(t *testing.T) {
	// A reasoning item with no summary should produce no message
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "hi"},
			{"type": "reasoning", "id": "rs_002"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// Only the user message — reasoning with no summary is skipped
	assert.Len(t, messages, 1)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
}

func TestRequestToChat_WebSearchCall(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Search for cats"},
			{
				"type": "web_search_call",
				"id": "ws_001",
				"name": "web_search",
				"results": [{"title": "Cat facts", "url": "https://example.com"}]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// user + assistant (function_call) + tool (results)
	require.Len(t, messages, 3)

	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])

	// assistant with tool_calls
	assistantMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", assistantMsg["role"])
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	require.Len(t, toolCalls, 1)
	tc := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "ws_001", tc["id"])

	// tool result
	toolMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "ws_001", toolMsg["tool_call_id"])
	assert.NotEmpty(t, toolMsg["content"])
}

func TestRequestToChat_ComputerCallOutput(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Click the button"},
			{
				"type": "computer_call_output",
				"id": "cc_001",
				"output": {"image_url": "data:image/png;base64,abc123"}
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 2)

	// computer_call_output → user message with image_url
	computerMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "user", computerMsg["role"])
	content := computerMsg["content"].([]interface{})
	require.Len(t, content, 1)
	part := content[0].(map[string]interface{})
	assert.Equal(t, "image_url", part["type"])
	imgURL := part["image_url"].(map[string]interface{})
	assert.Equal(t, "data:image/png;base64,abc123", imgURL["url"])
}

func TestRequestToChat_ComputerCallOutput_NoImage(t *testing.T) {
	// Without image_url, falls back to text placeholder
	body := `{
		"model": "gpt-4o",
		"input": [
			{"type": "computer_call_output", "id": "cc_002", "output": {}}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
	content := msg["content"].([]interface{})
	part := content[0].(map[string]interface{})
	assert.Equal(t, "text", part["type"])
}

func TestRequestToChat_CodeInterpreterCall(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Run the code"},
			{
				"type": "code_interpreter_call",
				"id": "ci_001",
				"code": "print('hello')",
				"outputs": [{"type": "text", "text": "hello"}]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 3)

	// assistant with code_interpreter function call
	assistantMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", assistantMsg["role"])
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	tc := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "ci_001", tc["id"])
	fn := tc["function"].(map[string]interface{})
	assert.Equal(t, "code_interpreter", fn["name"])
	assert.Contains(t, fn["arguments"].(string), "print")

	// tool result with outputs
	toolMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "ci_001", toolMsg["tool_call_id"])
	assert.Contains(t, toolMsg["content"].(string), "hello")
}

func TestRequestToChat_FileSearchCall(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Find the doc"},
			{
				"type": "file_search_call",
				"id": "fs_001",
				"results": [{"file_id": "file_abc", "filename": "doc.txt", "score": 0.9}]
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 3)

	// assistant with file_search function call
	assistantMsg := messages[1].(map[string]interface{})
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	tc := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "fs_001", tc["id"])
	fn := tc["function"].(map[string]interface{})
	assert.Equal(t, "file_search", fn["name"])

	// tool result
	toolMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Contains(t, toolMsg["content"].(string), "file_abc")
}

func TestRequestToChat_ImageGenerationCall(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Draw a cat"},
			{
				"type": "image_generation_call",
				"id": "ig_001",
				"result": "data:image/png;base64,iVBORw0KGgo="
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 2)

	// image_generation_call → assistant message with image_url
	imgMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", imgMsg["role"])
	content := imgMsg["content"].([]interface{})
	require.Len(t, content, 1)
	part := content[0].(map[string]interface{})
	assert.Equal(t, "image_url", part["type"])
	imgURL := part["image_url"].(map[string]interface{})
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgo=", imgURL["url"])
}

func TestRequestToChat_ImageGenerationCall_NoResult(t *testing.T) {
	// image_generation_call with no result should be skipped
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Draw a cat"},
			{"type": "image_generation_call", "id": "ig_002"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// Only the user message — empty image_generation_call is skipped
	assert.Len(t, messages, 1)
}

func TestRequestToChat_MCPToolCall(t *testing.T) {
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Use MCP"},
			{
				"type": "mcp_tool_call",
				"id": "mcp_001",
				"name": "get_weather",
				"server_label": "weather_svc",
				"input": {"city": "London"},
				"output": {"temp": 15, "unit": "C"}
			}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 3)

	// assistant with mcp function call (server_label + name combined)
	assistantMsg := messages[1].(map[string]interface{})
	toolCalls := assistantMsg["tool_calls"].([]interface{})
	tc := toolCalls[0].(map[string]interface{})
	assert.Equal(t, "mcp_001", tc["id"])
	fn := tc["function"].(map[string]interface{})
	assert.Equal(t, "weather_svc_get_weather", fn["name"])
	assert.Contains(t, fn["arguments"].(string), "London")

	// tool result with output
	toolMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "mcp_001", toolMsg["tool_call_id"])
	assert.Contains(t, toolMsg["content"].(string), "15")
}

func TestRequestToChat_MCPToolCall_NoID(t *testing.T) {
	// mcp_tool_call with no id should be skipped
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Use MCP"},
			{"type": "mcp_tool_call", "name": "get_weather"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// Only the user message
	assert.Len(t, messages, 1)
}

// Phase 10: Reasoning round-trip tests

func TestOutputToInputItems_Reasoning_WithSummary(t *testing.T) {
	output := []OutputItem{
		{
			Type: "reasoning",
			ID:   "rs_001",
			Summary: []OutputContent{
				{Type: "summary_text", Text: "I thought about it carefully."},
			},
		},
	}
	items := outputToInputItems(output)
	require.Len(t, items, 1)

	item := items[0].(map[string]interface{})
	assert.Equal(t, "reasoning", item["type"])
	assert.Equal(t, "rs_001", item["id"])

	summary := item["summary"].([]interface{})
	require.Len(t, summary, 1)
	s := summary[0].(map[string]interface{})
	assert.Equal(t, "summary_text", s["type"])
	assert.Equal(t, "I thought about it carefully.", s["text"])

	_, hasEncrypted := item["encrypted_content"]
	assert.False(t, hasEncrypted, "should not include encrypted_content when empty")
}

func TestOutputToInputItems_Reasoning_WithEncryptedContent(t *testing.T) {
	output := []OutputItem{
		{
			Type:             "reasoning",
			ID:               "rs_002",
			EncryptedContent: "enc_abc123",
			Summary: []OutputContent{
				{Type: "summary_text", Text: "summary text"},
			},
		},
	}
	items := outputToInputItems(output)
	require.Len(t, items, 1)

	item := items[0].(map[string]interface{})
	assert.Equal(t, "reasoning", item["type"])
	assert.Equal(t, "enc_abc123", item["encrypted_content"])
	assert.NotNil(t, item["summary"])
}

func TestOutputToInputItems_Reasoning_Empty_Skipped(t *testing.T) {
	// Reasoning items with no summary and no encrypted_content are dropped.
	output := []OutputItem{
		{Type: "reasoning", ID: "rs_empty"},
		{Type: "message", Role: "assistant", Content: []OutputContent{{Type: "output_text", Text: "hi"}}},
	}
	items := outputToInputItems(output)
	require.Len(t, items, 1, "empty reasoning item should be skipped")
	assert.Equal(t, "message", items[0].(map[string]interface{})["type"])
}

func TestOutputToInputItems_Reasoning_MixedWithMessage(t *testing.T) {
	// Reasoning items interleaved with message items are all preserved.
	output := []OutputItem{
		{
			Type: "reasoning",
			ID:   "rs_1",
			Summary: []OutputContent{
				{Type: "summary_text", Text: "Step 1"},
				{Type: "summary_text", Text: "Step 2"},
			},
		},
		{
			Type:    "message",
			Role:    "assistant",
			Content: []OutputContent{{Type: "output_text", Text: "Answer"}},
		},
	}
	items := outputToInputItems(output)
	require.Len(t, items, 2)

	reasoning := items[0].(map[string]interface{})
	assert.Equal(t, "reasoning", reasoning["type"])
	summary := reasoning["summary"].([]interface{})
	assert.Len(t, summary, 2)

	msg := items[1].(map[string]interface{})
	assert.Equal(t, "message", msg["type"])
}

func TestPrependHistoryToInput_WithReasoning(t *testing.T) {
	// Reasoning items from a previous turn are injected into the next request's input.
	currentBody := `{"model":"gpt-4o","input":[{"role":"user","content":"continue"}]}`

	prevOutput := []OutputItem{
		{
			Type: "reasoning",
			ID:   "rs_prev",
			Summary: []OutputContent{
				{Type: "summary_text", Text: "I considered the options."},
			},
		},
		{
			Type:    "message",
			Role:    "assistant",
			Content: []OutputContent{{Type: "output_text", Text: "Here is my answer."}},
		},
	}

	result, err := PrependHistoryToInput([]byte(currentBody), nil, prevOutput)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	input := parsed["input"].([]interface{})
	// Should have: reasoning, message, user
	require.Len(t, input, 3)

	rs := input[0].(map[string]interface{})
	assert.Equal(t, "reasoning", rs["type"])
	assert.Equal(t, "rs_prev", rs["id"])

	msg := input[1].(map[string]interface{})
	assert.Equal(t, "message", msg["type"])

	user := input[2].(map[string]interface{})
	assert.Equal(t, "user", user["role"])
}

func TestPrependHistoryToInput_ReasoningEncryptedContent(t *testing.T) {
	// encrypted_content is preserved through history injection (for Anthropic round-trips).
	currentBody := `{"model":"claude","input":[{"role":"user","content":"follow-up"}]}`

	prevOutput := []OutputItem{
		{
			Type:             "reasoning",
			ID:               "rs_enc",
			EncryptedContent: "sigma:encrypted_thinking_token",
		},
	}

	result, err := PrependHistoryToInput([]byte(currentBody), nil, prevOutput)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	input := parsed["input"].([]interface{})
	require.Len(t, input, 2)

	rs := input[0].(map[string]interface{})
	assert.Equal(t, "reasoning", rs["type"])
	assert.Equal(t, "sigma:encrypted_thinking_token", rs["encrypted_content"])
}

func TestOutputToInputItems_ComputerCall(t *testing.T) {
	output := []OutputItem{
		{
			Type:   "computer_call",
			ID:     "cc_01",
			CallID: "cc_01",
			Name:   "computer",
			Action: map[string]interface{}{"action": "screenshot"},
		},
	}

	items := outputToInputItems(output)
	require.Len(t, items, 1)

	ci := items[0].(map[string]interface{})
	assert.Equal(t, "computer_call", ci["type"])
	assert.Equal(t, "cc_01", ci["id"])
	assert.Equal(t, "cc_01", ci["call_id"])
	assert.Equal(t, "computer", ci["name"])
	action := ci["action"].(map[string]interface{})
	assert.Equal(t, "screenshot", action["action"])
}

func TestOutputToInputItems_ComputerCallOutput(t *testing.T) {
	output := []OutputItem{
		{
			Type:   "computer_call_output",
			ID:     "cco_01",
			CallID: "cc_01",
			Output: map[string]interface{}{"type": "computer_screenshot", "image_url": "https://example.com/shot.png"},
		},
	}

	items := outputToInputItems(output)
	require.Len(t, items, 1)

	cco := items[0].(map[string]interface{})
	assert.Equal(t, "computer_call_output", cco["type"])
	assert.Equal(t, "cc_01", cco["call_id"])
	out := cco["output"].(map[string]interface{})
	assert.Equal(t, "https://example.com/shot.png", out["image_url"])
}

func TestOutputToInputItems_WebSearchCall(t *testing.T) {
	output := []OutputItem{
		{
			Type:    "web_search_call",
			ID:      "ws_01",
			Queries: []string{"capital of France"},
		},
	}

	items := outputToInputItems(output)
	require.Len(t, items, 1)

	ws := items[0].(map[string]interface{})
	assert.Equal(t, "web_search_call", ws["type"])
	assert.Equal(t, "ws_01", ws["id"])
	assert.Equal(t, []string{"capital of France"}, ws["queries"])
}

func TestOutputToInputItems_CodeInterpreterCall(t *testing.T) {
	output := []OutputItem{
		{
			Type:    "code_interpreter_call",
			ID:      "ci_01",
			Code:    "print(1+1)",
			Outputs: []map[string]interface{}{{"type": "text", "text": "2"}},
		},
	}

	items := outputToInputItems(output)
	require.Len(t, items, 1)

	ci := items[0].(map[string]interface{})
	assert.Equal(t, "code_interpreter_call", ci["type"])
	assert.Equal(t, "ci_01", ci["id"])
	assert.Equal(t, "print(1+1)", ci["code"])
	require.NotNil(t, ci["outputs"])
}

func TestPrependHistoryToInput_WithComputerCallRoundTrip(t *testing.T) {
	currentBody := `{"model":"claude","input":[{"role":"user","content":"continue"}]}`

	prevOutput := []OutputItem{
		{
			Type:   "computer_call",
			ID:     "cc_hist",
			CallID: "cc_hist",
			Name:   "computer",
			Action: map[string]interface{}{"action": "click", "coordinate": []int{100, 200}},
		},
		{
			Type:   "computer_call_output",
			ID:     "cco_hist",
			CallID: "cc_hist",
			Output: map[string]interface{}{"type": "computer_screenshot", "image_url": "https://example.com/after.png"},
		},
	}

	result, err := PrependHistoryToInput([]byte(currentBody), nil, prevOutput)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	input := parsed["input"].([]interface{})
	// computer_call + computer_call_output + user message
	require.Len(t, input, 3)

	cc := input[0].(map[string]interface{})
	assert.Equal(t, "computer_call", cc["type"])
	assert.Equal(t, "cc_hist", cc["call_id"])

	cco := input[1].(map[string]interface{})
	assert.Equal(t, "computer_call_output", cco["type"])
	assert.Equal(t, "cc_hist", cco["call_id"])
}

func TestRequestToChat_MixedSpecialItems(t *testing.T) {
	// Multi-item conversation with function_call + reasoning + user reply
	body := `{
		"model": "gpt-4o",
		"input": [
			{"role": "user", "content": "Q1"},
			{
				"type": "function_call",
				"call_id": "call_1",
				"name": "search",
				"arguments": "{}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_1",
				"output": "result"
			},
			{
				"type": "reasoning",
				"id": "rs_1",
				"summary": [{"type": "summary_text", "text": "Found it."}]
			},
			{"role": "user", "content": "Q2"}
		]
	}`
	result, err := RequestToChat([]byte(body))
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	messages := parsed["messages"].([]interface{})
	// user, assistant(function_call), tool, assistant(reasoning), user
	require.Len(t, messages, 5)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[1].(map[string]interface{})["role"])
	assert.Equal(t, "tool", messages[2].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[3].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[4].(map[string]interface{})["role"])
}
