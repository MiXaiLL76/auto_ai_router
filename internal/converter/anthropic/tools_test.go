package anthropic

import (
	"encoding/json"
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

	// allowed_tools is passed through by mapToolChoice so that OpenAIToAnthropic
	// can intercept and call expandAllowedTools afterwards.
	t.Run("allowed_tools_passthrough", func(t *testing.T) {
		toolChoice := map[string]interface{}{
			"type": "allowed_tools",
			"mode": "auto",
			"tools": []interface{}{
				map[string]interface{}{"type": "tool", "name": "get_weather"},
			},
		}
		result := mapToolChoice(toolChoice)
		require.NotNil(t, result)
		m := result.(map[string]interface{})
		// must come back unchanged so OpenAIToAnthropic can detect and expand it
		assert.Equal(t, "allowed_tools", m["type"])
	})

	t.Run("anthropic_auto_object_passthrough", func(t *testing.T) {
		toolChoice := map[string]interface{}{"type": "auto"}
		result := mapToolChoice(toolChoice)
		require.NotNil(t, result)
		assert.Equal(t, toolChoice, result)
	})

	t.Run("anthropic_any_object_passthrough", func(t *testing.T) {
		toolChoice := map[string]interface{}{"type": "any"}
		result := mapToolChoice(toolChoice)
		require.NotNil(t, result)
		assert.Equal(t, toolChoice, result)
	})

	t.Run("anthropic_tool_object_passthrough", func(t *testing.T) {
		toolChoice := map[string]interface{}{"type": "tool", "name": "my_func"}
		result := mapToolChoice(toolChoice)
		require.NotNil(t, result)
		assert.Equal(t, toolChoice, result)
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

		assert.Equal(t, "computer_20250124", result[0].Type)
		assert.Equal(t, "computer", result[0].Name)
		assert.Equal(t, 1920, result[0].DisplayWidthPx)
		assert.Equal(t, 1080, result[0].DisplayHeightPx)

		assert.Equal(t, "text_editor_20250124", result[1].Type)
		assert.Equal(t, "str_replace_editor", result[1].Name)

		assert.Equal(t, "bash_20250124", result[2].Type)
		assert.Equal(t, "bash", result[2].Name)

		assert.Equal(t, "web_search_20250305", result[3].Type)
		assert.Equal(t, "web_search", result[3].Name)
	})

	t.Run("native_versioned_web_search_preserved", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type":     "web_search_20250305",
				"name":     "web_search",
				"max_uses": float64(5),
			},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "web_search_20250305", result[0].Type)
		assert.Equal(t, "web_search", result[0].Name)
		assert.Equal(t, 5, result[0].MaxUses)
	})

	t.Run("native_versioned_web_search_future_version_preserved", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{"type": "web_search_20260318", "name": "web_search"},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "web_search_20260318", result[0].Type)
	})

	t.Run("web_search_options_copied", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type":            "web_search",
				"max_uses":        float64(3),
				"allowed_domains": []interface{}{"example.com"},
				"user_location":   map[string]interface{}{"type": "approximate", "city": "NYC"},
			},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "web_search_20250305", result[0].Type)
		assert.Equal(t, 3, result[0].MaxUses)
		assert.Equal(t, []interface{}{"example.com"}, result[0].AllowedDomains)
		assert.Equal(t, "NYC", result[0].UserLocation.(map[string]interface{})["city"])
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

// TestConvertOpenAIToolsToAnthropic_VersionedBuiltinTools covers built-in tools that already
// carry their versioned Anthropic type. An Anthropic Messages request declaring
// {"type":"web_search_20250305","name":"web_search"} keeps that exact shape through
// MessagesToChat, so the conversion back has to recognise the versioned identifier — matching
// only the unversioned aliases silently dropped the tool and the provider answered as if no
// tool had been offered.
func TestConvertOpenAIToolsToAnthropic_VersionedBuiltinTools(t *testing.T) {
	t.Run("web_search_passes_through", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{"type": "web_search_20250305", "name": "web_search"},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "web_search_20250305", result[0].Type)
		assert.Equal(t, "web_search", result[0].Name)
		assert.Nil(t, result[0].InputSchema, "server tools carry no input_schema")
	})

	t.Run("name_defaults_per_tool_family", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{"type": "web_search_20250305"},
			map[string]interface{}{"type": "text_editor_20250124"},
			map[string]interface{}{"type": "bash_20250124"},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 3)
		assert.Equal(t, "web_search", result[0].Name)
		assert.Equal(t, "str_replace_editor", result[1].Name)
		assert.Equal(t, "bash", result[2].Name)
	})

	t.Run("computer_keeps_display_dimensions", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type":              "computer_20250124",
				"name":              "computer",
				"display_width_px":  float64(1280),
				"display_height_px": float64(800),
			},
		}
		result := convertOpenAIToolsToAnthropic(tools)
		require.Len(t, result, 1)
		assert.Equal(t, "computer_20250124", result[0].Type)
		assert.Equal(t, 1280, result[0].DisplayWidthPx)
		assert.Equal(t, 800, result[0].DisplayHeightPx)
	})

	t.Run("unknown_type_still_dropped", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{"type": "file_search", "name": "file_search"},
		}
		assert.Nil(t, convertOpenAIToolsToAnthropic(tools))
	})
}

func TestExpandAllowedTools(t *testing.T) {
	allTools := []AnthropicTool{
		{Name: "get_weather"},
		{Name: "search_docs"},
		{Name: "calculate"},
	}

	t.Run("filters_to_allowed_subset_auto", func(t *testing.T) {
		tc := map[string]interface{}{
			"type": "allowed_tools",
			"mode": "auto",
			"tools": []interface{}{
				map[string]interface{}{"type": "tool", "name": "get_weather"},
			},
		}
		choice, tools := expandAllowedTools(tc, allTools)
		m := choice.(map[string]interface{})
		assert.Equal(t, "auto", m["type"])
		require.Len(t, tools, 1)
		assert.Equal(t, "get_weather", tools[0].Name)
	})

	t.Run("mode_any_maps_to_any", func(t *testing.T) {
		tc := map[string]interface{}{
			"type": "allowed_tools",
			"mode": "any",
			"tools": []interface{}{
				map[string]interface{}{"name": "get_weather"},
				map[string]interface{}{"name": "calculate"},
			},
		}
		choice, tools := expandAllowedTools(tc, allTools)
		m := choice.(map[string]interface{})
		assert.Equal(t, "any", m["type"])
		assert.Len(t, tools, 2)
	})

	t.Run("empty_allowed_list_keeps_all_tools", func(t *testing.T) {
		tc := map[string]interface{}{
			"type":  "allowed_tools",
			"mode":  "auto",
			"tools": []interface{}{},
		}
		choice, tools := expandAllowedTools(tc, allTools)
		m := choice.(map[string]interface{})
		assert.Equal(t, "auto", m["type"])
		assert.Len(t, tools, 3) // unchanged
	})
}

func TestOpenAIToAnthropic_AllowedToolsExpanded(t *testing.T) {
	// allowed_tools must be expanded: only the allowed tool should remain in tools[],
	// and tool_choice must become {"type":"auto"} (Bedrock-compatible).
	body := `{
		"model": "claude-sonnet-4-5",
		"messages": [{"role": "user", "content": "test"}],
		"tools": [
			{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
			{"type": "function", "function": {"name": "search_docs",  "parameters": {"type": "object", "properties": {}}}}
		],
		"tool_choice": {
			"type": "allowed_tools",
			"mode": "auto",
			"tools": [{"type": "tool", "name": "get_weather"}]
		},
		"max_tokens": 200
	}`

	out, err := OpenAIToAnthropic([]byte(body), "")
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &req))

	// tool_choice must be plain auto, not allowed_tools
	tc, ok := req["tool_choice"].(map[string]interface{})
	require.True(t, ok, "tool_choice should be present")
	assert.Equal(t, "auto", tc["type"])
	assert.Nil(t, tc["tools"], "tools key must not be present in converted tool_choice")

	// only get_weather must remain in the tools array
	tools, ok := req["tools"].([]interface{})
	require.True(t, ok)
	require.Len(t, tools, 1)
	toolName := tools[0].(map[string]interface{})["name"]
	assert.Equal(t, "get_weather", toolName)
}

func TestOpenAIToAnthropic_AllowedToolsViaExtraBody(t *testing.T) {
	// extra_body["tool_choice"] path: same expansion must apply.
	body := `{
		"model": "claude-sonnet-4-5",
		"messages": [{"role": "user", "content": "test"}],
		"tools": [
			{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
			{"type": "function", "function": {"name": "search_docs",  "parameters": {"type": "object", "properties": {}}}}
		],
		"extra_body": {
			"tool_choice": {
				"type": "allowed_tools",
				"mode": "auto",
				"tools": [{"type": "tool", "name": "get_weather"}]
			}
		},
		"max_tokens": 200
	}`

	out, err := OpenAIToAnthropic([]byte(body), "")
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &req))

	tc, ok := req["tool_choice"].(map[string]interface{})
	require.True(t, ok, "tool_choice should be present")
	assert.Equal(t, "auto", tc["type"])

	tools, ok := req["tools"].([]interface{})
	require.True(t, ok)
	require.Len(t, tools, 1)
	assert.Equal(t, "get_weather", tools[0].(map[string]interface{})["name"])
}

func TestOpenAIToAnthropic_ExtraBodyToolChoiceOverridesRegular(t *testing.T) {
	// When both tool_choice and extra_body["tool_choice"] are set,
	// extra_body takes precedence.
	body := `{
		"model": "claude-sonnet-4-5",
		"messages": [{"role": "user", "content": "test"}],
		"tool_choice": "required",
		"extra_body": {
			"tool_choice": {"type": "none"}
		},
		"max_tokens": 100
	}`

	out, err := OpenAIToAnthropic([]byte(body), "")
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &req))

	tc, ok := req["tool_choice"].(map[string]interface{})
	require.True(t, ok, "tool_choice should be present")
	assert.Equal(t, "none", tc["type"], "extra_body tool_choice should override regular tool_choice")
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
