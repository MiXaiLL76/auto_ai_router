package anthropicresponses

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponsesRequestToAnthropic_StringInput(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "Hello, world!",
		"temperature": 0.7,
		"max_output_tokens": 512
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	assert.Equal(t, "claude-opus-4-5", ar["model"])
	assert.Equal(t, float64(512), ar["max_tokens"])
	assert.Equal(t, float64(0.7), ar["temperature"])

	messages := ar["messages"].([]interface{})
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
	assert.Equal(t, "Hello, world!", msg["content"])
}

func TestResponsesRequestToAnthropic_Instructions(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"instructions": "You are a helpful assistant.",
		"input": "Hello"
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	assert.Equal(t, "You are a helpful assistant.", ar["system"])
}

func TestResponsesRequestToAnthropic_MaxTokensDefault(t *testing.T) {
	body := `{"model": "claude-opus-4-5", "input": "hi"}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	// Default is 4096
	assert.Equal(t, float64(4096), ar["max_tokens"])
}

func TestResponsesRequestToAnthropic_InputFileFileData(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{
			"role": "user",
			"content": [
				{"type": "input_file", "filename": "test.pdf", "file_data": "data:application/pdf;base64,JVBERi0="},
				{"type": "input_text", "text": "read it"}
			]
		}]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.NoError(t, err)

	block := firstUserContentBlock(t, result)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "base64", source["type"])
	assert.Equal(t, "application/pdf", source["media_type"])
	assert.Equal(t, "JVBERi0=", source["data"])
}

func TestResponsesRequestToAnthropic_InputFileFileIDUnsupported(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file-abc"}]}]
	}`

	_, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_file.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
}

func TestResponsesRequestToAnthropic_InputImageFileIDUnsupported(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file-abc"}]}]
	}`

	_, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_image.file_id", validationErr.Param)
	assert.Equal(t, "file_id is not supported for this route", validationErr.Message)
}

func TestResponsesRequestToAnthropic_InputFileFileURL(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{"role": "user", "content": [{"type": "input_file", "file_url": "https://example.com/document.pdf"}]}]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.NoError(t, err)

	block := firstUserContentBlock(t, result)
	assert.Equal(t, "document", block["type"])
	source := block["source"].(map[string]interface{})
	assert.Equal(t, "url", source["type"])
	assert.Equal(t, "https://example.com/document.pdf", source["url"])
}

func TestResponsesRequestToAnthropic_InputFileMalformedBase64(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{"role": "user", "content": [{"type": "input_file", "file_data": "data:application/pdf;base64,###"}]}]
	}`

	_, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_file.file_data", validationErr.Param)
}

func TestResponsesRequestToAnthropic_InputFileMissingSource(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": [{"role": "user", "content": [{"type": "input_file"}]}]
	}`

	_, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.True(t, errors.As(err, &validationErr))
	assert.Equal(t, "input_file", validationErr.Param)
}

func TestResponsesRequestToAnthropic_FunctionTool(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "Search something",
		"tools": [
			{
				"type": "function",
				"name": "search",
				"description": "Search the web",
				"parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	tools := ar["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "search", tool["name"])
	assert.Equal(t, "Search the web", tool["description"])
}

func TestResponsesRequestToAnthropic_ComputerUseTool(t *testing.T) {
	w, h := 1280, 800
	body, _ := json.Marshal(map[string]interface{}{
		"model": "claude-opus-4-5",
		"input": "Click something",
		"tools": []map[string]interface{}{
			{
				"type":           "computer_use_preview",
				"display_width":  w,
				"display_height": h,
			},
		},
	})

	result, err := ResponsesRequestToAnthropic(body, "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	tools := ar["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "computer_20241022", tool["type"])
	assert.Equal(t, "computer", tool["name"])
	assert.Equal(t, float64(1280), tool["display_width_px"])
	assert.Equal(t, float64(800), tool["display_height_px"])

	// Beta header should be set
	betas := ar["anthropic_beta"].([]interface{})
	assert.Contains(t, betas, "computer-use-2024-10-22")
}

func TestResponsesRequestToAnthropic_UnsupportedToolsSkipped(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "test",
		"tools": [
			{"type": "web_search_preview"},
			{"type": "file_search"},
			{"type": "code_interpreter"}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	// All unsupported tools are dropped → no tools key or empty
	if toolsRaw, ok := ar["tools"]; ok {
		tools, _ := toolsRaw.([]interface{})
		assert.Len(t, tools, 0)
	}
}

func TestResponsesRequestToAnthropic_ToolChoiceNone(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "test",
		"tool_choice": "none"
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	tc := ar["tool_choice"].(map[string]interface{})
	assert.Equal(t, "none", tc["type"])
}

func TestResponsesRequestToAnthropic_ToolChoiceRequired(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "test",
		"tool_choice": "required"
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	tc := ar["tool_choice"].(map[string]interface{})
	assert.Equal(t, "any", tc["type"])
}

func TestResponsesRequestToAnthropic_ToolChoiceFunction(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "test",
		"tool_choice": {"type": "function", "name": "search"}
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	tc := ar["tool_choice"].(map[string]interface{})
	assert.Equal(t, "tool", tc["type"])
	assert.Equal(t, "search", tc["name"])
}

func TestResponsesRequestToAnthropic_ReasoningEffortHigh(t *testing.T) {
	body := `{
		"model": "claude-opus-4-8",
		"input": "Think about this",
		"reasoning": {"effort": "high"}
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-8")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	// claude-opus-4-8 uses adaptive thinking: type="adaptive", no budget_tokens
	thinking := ar["thinking"].(map[string]interface{})
	assert.Equal(t, "adaptive", thinking["type"])
	assert.Equal(t, "summarized", thinking["display"])
	assert.Nil(t, thinking["budget_tokens"])

	// effort goes in output_config, not in thinking
	outputConfig := ar["output_config"].(map[string]interface{})
	assert.Equal(t, "high", outputConfig["effort"])

	betas := ar["anthropic_beta"].([]interface{})
	assert.Contains(t, betas, "effort-2025-11-24")
}

func TestResponsesRequestToAnthropic_ReasoningEffortLow(t *testing.T) {
	body := `{
		"model": "claude-opus-4-8",
		"input": "test",
		"reasoning": {"effort": "low"}
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-8")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	thinking := ar["thinking"].(map[string]interface{})
	assert.Equal(t, "adaptive", thinking["type"])
	assert.Equal(t, "summarized", thinking["display"])
	assert.Nil(t, thinking["budget_tokens"])

	outputConfig := ar["output_config"].(map[string]interface{})
	assert.Equal(t, "low", outputConfig["effort"])
}

func TestResponsesRequestToAnthropic_ReasoningBudgetRaisesMaxTokens(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-5",
		"input": "test",
		"reasoning": {"effort": "low"},
		"max_output_tokens": 5000
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-sonnet-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	thinking := ar["thinking"].(map[string]interface{})
	assert.Equal(t, "enabled", thinking["type"])
	assert.Equal(t, float64(5000), thinking["budget_tokens"])
	assert.Equal(t, float64(6024), ar["max_tokens"])
}

func TestResponsesRequestToAnthropic_ReasoningEffortNone(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": "test",
		"reasoning": {"effort": "none"}
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	// effort=none should not enable thinking
	assert.Nil(t, ar["thinking"])
}

func TestResponsesRequestToAnthropic_ConversationHistory(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "user", "content": "How are you?"}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})
	require.Len(t, messages, 3)
	assert.Equal(t, "user", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[1].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[2].(map[string]interface{})["role"])
}

func TestResponsesRequestToAnthropic_FunctionCallHistory(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Call the function"},
			{
				"type": "function_call",
				"call_id": "call_abc",
				"name": "get_weather",
				"arguments": "{\"city\": \"Paris\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_abc",
				"name": "get_weather",
				"output": "{\"temp\": 20}"
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})
	// user + assistant(tool_use) + user(tool_result)
	require.Len(t, messages, 3)

	// Assistant message has tool_use block
	assistantMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "assistant", assistantMsg["role"])
	assistantContent := assistantMsg["content"].([]interface{})
	require.Len(t, assistantContent, 1)
	toolUse := assistantContent[0].(map[string]interface{})
	assert.Equal(t, "tool_use", toolUse["type"])
	assert.Equal(t, "call_abc", toolUse["id"])
	assert.Equal(t, "get_weather", toolUse["name"])

	// User message has tool_result block
	userMsg := messages[2].(map[string]interface{})
	assert.Equal(t, "user", userMsg["role"])
	userContent := userMsg["content"].([]interface{})
	require.Len(t, userContent, 1)
	toolResult := userContent[0].(map[string]interface{})
	assert.Equal(t, "tool_result", toolResult["type"])
	assert.Equal(t, "call_abc", toolResult["tool_use_id"])
}

func TestResponsesRequestToAnthropic_ReasoningItemInHistory(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Think about this"},
			{
				"type": "reasoning",
				"summary": [{"type": "summary_text", "text": "I'm thinking..."}]
			},
			{"role": "assistant", "content": "Here is my answer"}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})
	// user + assistant(thinking) + assistant(text)
	require.GreaterOrEqual(t, len(messages), 2)
}

func TestResponsesRequestToAnthropic_ComputerCallHistory(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Take a screenshot"},
			{
				"type": "computer_call",
				"call_id": "cc_abc123",
				"name": "computer",
				"action": {"action": "screenshot"}
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})
	require.GreaterOrEqual(t, len(messages), 2)

	var foundToolUse bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "assistant" {
			if blocks, ok := msg["content"].([]interface{}); ok {
				for _, b := range blocks {
					block := b.(map[string]interface{})
					if block["type"] == "tool_use" && block["id"] == "cc_abc123" {
						foundToolUse = true
						assert.Equal(t, "computer", block["name"])
						input := block["input"].(map[string]interface{})
						assert.Equal(t, "screenshot", input["action"])
					}
				}
			}
		}
	}
	assert.True(t, foundToolUse, "expected tool_use block for computer_call")
}

func TestResponsesRequestToAnthropic_ComputerCallOutputHistory_URL(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Take a screenshot"},
			{
				"type": "computer_call",
				"call_id": "cc_abc123",
				"name": "computer",
				"action": {"action": "screenshot"}
			},
			{
				"type": "computer_call_output",
				"call_id": "cc_abc123",
				"output": {
					"type": "computer_screenshot",
					"image_url": "https://example.com/shot.png"
				}
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})

	var foundToolResult bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "user" {
			if blocks, ok := msg["content"].([]interface{}); ok {
				for _, b := range blocks {
					block := b.(map[string]interface{})
					if block["type"] == "tool_result" && block["tool_use_id"] == "cc_abc123" {
						foundToolResult = true
						content := block["content"].([]interface{})
						require.Len(t, content, 1)
						imgBlock := content[0].(map[string]interface{})
						assert.Equal(t, "image", imgBlock["type"])
						src := imgBlock["source"].(map[string]interface{})
						assert.Equal(t, "url", src["type"])
						assert.Equal(t, "https://example.com/shot.png", src["url"])
					}
				}
			}
		}
	}
	assert.True(t, foundToolResult, "expected tool_result for computer_call_output")
}

func TestResponsesRequestToAnthropic_ComputerCallOutputHistory_Base64(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Take a screenshot"},
			{
				"type": "computer_call",
				"call_id": "cc_b64",
				"name": "computer",
				"action": {"action": "screenshot"}
			},
			{
				"type": "computer_call_output",
				"call_id": "cc_b64",
				"output": {
					"type": "computer_screenshot",
					"image_base64": "abc123base64data",
					"media_type": "image/jpeg"
				}
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})

	var foundToolResult bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "user" {
			if blocks, ok := msg["content"].([]interface{}); ok {
				for _, b := range blocks {
					block := b.(map[string]interface{})
					if block["type"] == "tool_result" && block["tool_use_id"] == "cc_b64" {
						foundToolResult = true
						content := block["content"].([]interface{})
						require.Len(t, content, 1)
						imgBlock := content[0].(map[string]interface{})
						assert.Equal(t, "image", imgBlock["type"])
						src := imgBlock["source"].(map[string]interface{})
						assert.Equal(t, "base64", src["type"])
						assert.Equal(t, "image/jpeg", src["media_type"])
						assert.Equal(t, "abc123base64data", src["data"])
					}
				}
			}
		}
	}
	assert.True(t, foundToolResult, "expected tool_result with base64 image for computer_call_output")
}

func firstUserContentBlock(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &ar))
	messages := ar["messages"].([]interface{})
	require.NotEmpty(t, messages)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.NotEmpty(t, content)
	return content[0].(map[string]interface{})
}

func TestResponsesRequestToAnthropic_ReasoningItemWithEncryptedContent(t *testing.T) {
	body := `{
		"model": "claude-opus-4-5",
		"input": [
			{"role": "user", "content": "Think"},
			{
				"type": "reasoning",
				"encrypted_content": "encrypted_sig_xyz"
			}
		]
	}`

	result, err := ResponsesRequestToAnthropic([]byte(body), "claude-opus-4-5")
	require.NoError(t, err)

	var ar map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &ar))

	messages := ar["messages"].([]interface{})
	// Should have user + assistant(thinking with signature)
	require.GreaterOrEqual(t, len(messages), 2)

	// Find the assistant thinking message
	var foundThinking bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "assistant" {
			if blocks, ok := msg["content"].([]interface{}); ok {
				for _, b := range blocks {
					block := b.(map[string]interface{})
					if block["type"] == "thinking" && block["signature"] == "encrypted_sig_xyz" {
						foundThinking = true
					}
				}
			}
		}
	}
	assert.True(t, foundThinking, "expected thinking block with signature")
}
