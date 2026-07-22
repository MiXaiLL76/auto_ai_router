package vertexresponses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestResponsesRequestToVertex_StringInput(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": "Hello, world!",
		"temperature": 0.7,
		"max_output_tokens": 512
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	contents := vr["contents"].([]interface{})
	require.Len(t, contents, 1)
	c := contents[0].(map[string]interface{})
	assert.Equal(t, "user", c["role"])
	parts := c["parts"].([]interface{})
	assert.Equal(t, "Hello, world!", parts[0].(map[string]interface{})["text"])

	cfg := vr["generationConfig"].(map[string]interface{})
	assert.Equal(t, float64(0.7), cfg["temperature"])
	assert.Equal(t, float64(512), cfg["maxOutputTokens"])
}

func TestResponsesRequestToVertex_Instructions(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"instructions": "You are a helpful assistant.",
		"input": "Hello"
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	sysInst := vr["systemInstruction"].(map[string]interface{})
	parts := sysInst["parts"].([]interface{})
	assert.Equal(t, "You are a helpful assistant.", parts[0].(map[string]interface{})["text"])
}

func TestResponsesRequestToVertex_FunctionTool(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": "Search for something",
		"tools": [
			{
				"type": "function",
				"name": "search",
				"description": "Search the web",
				"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
			}
		]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	tools := vr["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	funcDecls := tool["functionDeclarations"].([]interface{})
	require.Len(t, funcDecls, 1)
	fd := funcDecls[0].(map[string]interface{})
	assert.Equal(t, "search", fd["name"])
	assert.Equal(t, "Search the web", fd["description"])
}

func TestResponsesRequestToVertex_WebSearchTool(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": "What's the weather?",
		"tools": [{"type": "web_search_preview"}]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	tools := vr["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	// googleSearch tool should be present
	assert.Contains(t, tool, "googleSearch")
}

func TestResponsesRequestToVertex_CodeInterpreterTool(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": "Run some code",
		"tools": [{"type": "code_interpreter"}]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	tools := vr["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Contains(t, tool, "codeExecution")
}

func TestResponsesRequestToVertex_ImageGenerationToolRequestsImageModality(t *testing.T) {
	body := `{
		"model": "gemini-3.1-flash-image-preview",
		"input": "Generate a product image",
		"tools": [{"type": "image_generation"}]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-3.1-flash-image-preview")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))
	assert.NotContains(t, vr, "tools", "image_generation is a response modality, not a Vertex tool")

	cfg := vr["generationConfig"].(map[string]interface{})
	assert.Equal(t, []interface{}{"IMAGE"}, cfg["responseModalities"])
}

func TestResponsesRequestToVertex_ConversationHistory(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "user", "content": "How are you?"}
		]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	contents := vr["contents"].([]interface{})
	require.Len(t, contents, 3)
	assert.Equal(t, "user", contents[0].(map[string]interface{})["role"])
	assert.Equal(t, "model", contents[1].(map[string]interface{})["role"])
	assert.Equal(t, "user", contents[2].(map[string]interface{})["role"])
}

func TestResponsesRequestToVertex_FunctionCallHistory(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": [
			{"role": "user", "content": "Call the function"},
			{
				"type": "function_call",
				"call_id": "call_1",
				"name": "get_weather",
				"arguments": "{\"city\": \"London\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_1",
				"name": "get_weather",
				"output": "{\"temp\": 15}"
			}
		]
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	contents := vr["contents"].([]interface{})
	// user + model(functionCall) + user(functionResponse)
	require.Len(t, contents, 3)
	assert.Equal(t, "user", contents[0].(map[string]interface{})["role"])
	assert.Equal(t, "model", contents[1].(map[string]interface{})["role"])
	assert.Equal(t, "user", contents[2].(map[string]interface{})["role"])

	// Verify function call part
	modelParts := contents[1].(map[string]interface{})["parts"].([]interface{})
	fc := modelParts[0].(map[string]interface{})["functionCall"].(map[string]interface{})
	assert.Equal(t, "get_weather", fc["name"])

	// Verify function response part
	userParts := contents[2].(map[string]interface{})["parts"].([]interface{})
	fr := userParts[0].(map[string]interface{})["functionResponse"].(map[string]interface{})
	assert.Equal(t, "get_weather", fr["name"])
}

func TestResponsesRequestToVertex_Reasoning(t *testing.T) {
	body := `{
		"model": "gemini-2.5-flash",
		"input": "Think about this",
		"reasoning": {"effort": "high"}
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.5-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	cfg := vr["generationConfig"].(map[string]interface{})
	thinkingCfg := cfg["thinkingConfig"].(map[string]interface{})
	// Gemini 2.5 flash: high effort → budget=24576
	assert.Equal(t, float64(24576), thinkingCfg["thinkingBudget"])
}

func TestResponsesRequestToVertex_JSONSchemaFormat(t *testing.T) {
	body := `{
		"model": "gemini-2.0-flash",
		"input": "Return JSON",
		"text": {
			"format": {
				"type": "json_schema",
				"name": "result",
				"schema": {"type": "object", "properties": {"value": {"type": "string"}}}
			}
		}
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	cfg := vr["generationConfig"].(map[string]interface{})
	assert.Equal(t, "application/json", cfg["responseMimeType"])
	responseSchema := cfg["responseSchema"].(map[string]interface{})
	assert.Equal(t, "OBJECT", responseSchema["type"])
	properties := responseSchema["properties"].(map[string]interface{})
	valueSchema := properties["value"].(map[string]interface{})
	assert.Equal(t, "STRING", valueSchema["type"])
}

func TestContentPartToVertexParts_InputText(t *testing.T) {
	part := map[string]interface{}{"type": "input_text", "text": "hello"}
	parts, err := contentPartToVertexParts(part)
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, "hello", parts[0].Text)
}

func TestContentPartToVertexParts_InputImage_DataURL(t *testing.T) {
	// A tiny valid PNG in base64
	pngB64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	part := map[string]interface{}{
		"type":      "input_image",
		"image_url": "data:image/png;base64," + pngB64,
	}
	parts, err := contentPartToVertexParts(part)
	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].InlineData)
	assert.Equal(t, "image/png", parts[0].InlineData.MIMEType)
}

func TestContentPartToVertexParts_InputImage_URL(t *testing.T) {
	part := map[string]interface{}{
		"type":      "input_image",
		"image_url": "https://example.com/photo.jpg",
	}
	parts, err := contentPartToVertexParts(part)
	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].FileData)
	assert.Equal(t, "image/jpeg", parts[0].FileData.MIMEType)
	assert.Equal(t, "https://example.com/photo.jpg", parts[0].FileData.FileURI)
}

func TestContentPartToVertexParts_UnknownType(t *testing.T) {
	part := map[string]interface{}{"type": "unknown_type"}
	parts, err := contentPartToVertexParts(part)
	assert.NoError(t, err)
	assert.Nil(t, parts)
}

func TestResponsesToolsToVertex_Mixed(t *testing.T) {
	tools := []struct {
		Type string
		Name string
	}{
		{"function", "search"},
		{"web_search_preview", ""},
		{"code_interpreter", ""},
		{"file_search", ""}, // unsupported, should be skipped
	}

	var respTools []interface{}
	for _, t := range tools {
		m := map[string]interface{}{"type": t.Type}
		if t.Name != "" {
			m["name"] = t.Name
		}
		respTools = append(respTools, m)
	}

	b, _ := json.Marshal(map[string]interface{}{
		"model": "gemini-2.0-flash",
		"input": "test",
		"tools": respTools,
	})

	result, err := ResponsesRequestToVertex(b, "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	vertexTools := vr["tools"].([]interface{})
	// Built-in tools take priority; function declarations are dropped.
	// googleSearch + codeExecution = 2 tools (function "search" silently dropped).
	assert.Len(t, vertexTools, 2)
	toolMap0 := vertexTools[0].(map[string]interface{})
	toolMap1 := vertexTools[1].(map[string]interface{})
	assert.Contains(t, toolMap0, "googleSearch")
	assert.Contains(t, toolMap1, "codeExecution")
}

func TestResponsesRequestToVertex_WebSearchOnlyNoToolConfig(t *testing.T) {
	// Regression: when only non-function tools are present (e.g. web_search_preview),
	// ToolConfig must NOT be set. Vertex rejects FunctionCallingConfig when there are
	// no FunctionDeclarations: "Function calling config is set without function_declarations."
	body := `{
		"model": "gemini-2.0-flash",
		"input": "search the web",
		"tools": [{"type": "web_search_preview"}],
		"tool_choice": "auto"
	}`

	result, err := ResponsesRequestToVertex([]byte(body), "gemini-2.0-flash")
	require.NoError(t, err)

	var vr map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &vr))

	// ToolConfig must be absent (no function declarations to configure)
	_, hasToolConfig := vr["toolConfig"]
	assert.False(t, hasToolConfig, "toolConfig must not be set when only non-function tools are present")

	// The web_search_preview tool must still be converted to googleSearch
	tools, ok := vr["tools"].([]interface{})
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Contains(t, tools[0].(map[string]interface{}), "googleSearch")
}

// TestVertexSchemaConversion verifies that function parameters are converted to genai.Schema.
func TestVertexSchemaConversion(t *testing.T) {
	params := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"city": map[string]interface{}{"type": "string"},
		},
		"required": []interface{}{"city"},
	}

	schema := interfaceToSchema(params)
	require.NotNil(t, schema)
	assert.Equal(t, genai.TypeObject, schema.Type)
	assert.Contains(t, schema.Properties, "city")
}
