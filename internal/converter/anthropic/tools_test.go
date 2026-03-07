package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMapToolChoice(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice interface{}
		wantType   string
		wantNil    bool
	}{
		{"nil", nil, "", true},
		{"none", "none", "none", false},
		{"auto", "auto", "auto", false},
		{"required", "required", "any", false},
		{"unknown_string", "invalid", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapToolChoice(tt.toolChoice)
			if tt.wantNil {
				assert.Nil(t, result)
			} else {
				m := result.(map[string]interface{})
				assert.Equal(t, tt.wantType, m["type"])
			}
		})
	}

	t.Run("function_object", func(t *testing.T) {
		toolChoice := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "get_weather",
			},
		}
		result := mapToolChoice(toolChoice)
		require.NotNil(t, result)
		m := result.(map[string]interface{})
		assert.Equal(t, "tool", m["type"])
		assert.Equal(t, "get_weather", m["name"])
	})

	t.Run("function_object_empty_name", func(t *testing.T) {
		toolChoice := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "",
			},
		}
		result := mapToolChoice(toolChoice)
		assert.Nil(t, result)
	})
}

func TestConvertOpenAIToolsToAnthropic(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		assert.Nil(t, convertOpenAIToolsToAnthropic(nil))
		assert.Nil(t, convertOpenAIToolsToAnthropic([]interface{}{}))
	})

	t.Run("function_tool", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather info",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"city": map[string]interface{}{"type": "string"},
						},
					},
				},
			},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "get_weather", result[0].Name)
		assert.Equal(t, "Get weather info", result[0].Description)
		assert.NotNil(t, result[0].InputSchema)
	})

	t.Run("builtin_tools", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type":              "computer_use",
				"display_width_px":  float64(1920),
				"display_height_px": float64(1080),
			},
			map[string]interface{}{"type": "text_editor"},
			map[string]interface{}{"type": "bash"},
			map[string]interface{}{"type": "web_search"},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 4)

		assert.Equal(t, "computer_20241022", result[0].Type)
		assert.Equal(t, "computer", result[0].Name)
		assert.Equal(t, 1920, result[0].DisplayWidthPx)
		assert.Equal(t, 1080, result[0].DisplayHeightPx)

		assert.Equal(t, "text_editor_20241022", result[1].Type)
		assert.Equal(t, "str_replace_editor", result[1].Name)

		assert.Equal(t, "bash_20241022", result[2].Type)
		assert.Equal(t, "bash", result[2].Name)

		assert.Equal(t, "web_search_20250305", result[3].Type)
		assert.Equal(t, "web_search", result[3].Name)
	})

	t.Run("function_without_name_skipped", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"description": "No name function",
				},
			},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		assert.Nil(t, result)
	})
}

func TestConvertToolCallsToAnthropicContent(t *testing.T) {
	t.Run("valid_tool_call", func(t *testing.T) {
		toolCalls := []interface{}{
			map[string]interface{}{
				"id": "call_1",
				"function": map[string]interface{}{
					"name":      "get_weather",
					"arguments": `{"city":"NYC"}`,
				},
			},
		}
		blocks := convertToolCallsToAnthropicContent(toolCalls)
		require.Len(t, blocks, 1)
		assert.Equal(t, "tool_use", blocks[0].Type)
		assert.Equal(t, "call_1", blocks[0].ID)
		assert.Equal(t, "get_weather", blocks[0].Name)

		input := blocks[0].Input.(map[string]interface{})
		assert.Equal(t, "NYC", input["city"])
	})

	t.Run("invalid_json_arguments", func(t *testing.T) {
		toolCalls := []interface{}{
			map[string]interface{}{
				"id": "call_2",
				"function": map[string]interface{}{
					"name":      "func1",
					"arguments": "not-valid-json",
				},
			},
		}
		blocks := convertToolCallsToAnthropicContent(toolCalls)
		require.Len(t, blocks, 1)
		// Invalid JSON → empty map
		input := blocks[0].Input.(map[string]interface{})
		assert.Empty(t, input)
	})

	t.Run("empty_arguments", func(t *testing.T) {
		toolCalls := []interface{}{
			map[string]interface{}{
				"id": "call_3",
				"function": map[string]interface{}{
					"name":      "func2",
					"arguments": "",
				},
			},
		}
		blocks := convertToolCallsToAnthropicContent(toolCalls)
		require.Len(t, blocks, 1)
		input := blocks[0].Input.(map[string]interface{})
		assert.Empty(t, input)
	})
}
