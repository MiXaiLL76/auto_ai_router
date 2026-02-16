package anthropic

import (
	"encoding/json"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// mapToolChoice maps an OpenAI tool_choice value to the Anthropic tool_choice format.
//
//	"none"      → {"type": "none"}
//	"auto"      → {"type": "auto"}
//	"required"  → {"type": "any"}
//	{function}  → {"type": "tool", "name": "<name>"}
func mapToolChoice(toolChoice interface{}) interface{} {
	if toolChoice == nil {
		return nil
	}
	switch choice := toolChoice.(type) {
	case string:
		switch choice {
		case "none":
			return map[string]interface{}{"type": "none"}
		case "auto":
			return map[string]interface{}{"type": "auto"}
		case "required":
			return map[string]interface{}{"type": "any"}
		}
	case map[string]interface{}:
		// {"type": "function", "function": {"name": "func_name"}}
		if funcObj, ok := choice["function"].(map[string]interface{}); ok {
			if name, ok := funcObj["name"].(string); ok && name != "" {
				return map[string]interface{}{
					"type": "tool",
					"name": name,
				}
			}
		}
	}
	return nil
}

// convertOpenAIToolsToAnthropic converts an OpenAI tools array to Anthropic tool definitions.
//
// Standard function tools become AnthropicTool with Name/Description/InputSchema.
// Anthropic built-in tools (computer_use, text_editor, bash, web_search) are mapped to
// their versioned type identifiers.
func convertOpenAIToolsToAnthropic(openAITools []interface{}) []AnthropicTool {
	if len(openAITools) == 0 {
		return nil
	}
	var tools []AnthropicTool
	for _, toolInterface := range openAITools {
		toolMap, ok := toolInterface.(map[string]interface{})
		if !ok {
			continue
		}
		toolType, _ := toolMap["type"].(string)
		switch toolType {
		case "function":
			if funcObj, ok := toolMap["function"].(map[string]interface{}); ok {
				tool := AnthropicTool{
					Name:        converterutil.GetString(funcObj, "name"),
					Description: converterutil.GetString(funcObj, "description"),
				}
				if tool.Name == "" {
					continue
				}
				if params, ok := funcObj["parameters"].(map[string]interface{}); ok {
					tool.InputSchema = convertOpenAISchemaToAnthropic(params)
				} else {
					tool.InputSchema = map[string]interface{}{"type": "object"}
				}
				tools = append(tools, tool)
			}
		case "computer_use":
			tool := AnthropicTool{
				Type: "computer_20241022",
				Name: "computer",
			}
			if w, ok := toolMap["display_width_px"].(float64); ok {
				tool.DisplayWidthPx = int(w)
			}
			if h, ok := toolMap["display_height_px"].(float64); ok {
				tool.DisplayHeightPx = int(h)
			}
			tools = append(tools, tool)
		case "text_editor":
			tools = append(tools, AnthropicTool{
				Type: "text_editor_20241022",
				Name: "str_replace_editor",
			})
		case "bash":
			tools = append(tools, AnthropicTool{
				Type: "bash_20241022",
				Name: "bash",
			})
		case "web_search", "web_search_preview":
			tools = append(tools, AnthropicTool{
				Type: "web_search_20250305",
				Name: "web_search",
			})
		}
	}
	if len(tools) == 0 {
		return nil
	}
	return tools
}

// convertToolCallsToAnthropicContent converts OpenAI tool_calls (from an assistant message)
// into Anthropic tool_use content blocks.
func convertToolCallsToAnthropicContent(toolCalls []interface{}) []ContentBlock {
	var blocks []ContentBlock
	for _, tc := range toolCalls {
		tcMap, ok := tc.(map[string]interface{})
		if !ok {
			continue
		}
		id := converterutil.GetString(tcMap, "id")
		var name, argsStr string
		if funcObj, ok := tcMap["function"].(map[string]interface{}); ok {
			name = converterutil.GetString(funcObj, "name")
			argsStr = converterutil.GetString(funcObj, "arguments")
		}
		var input interface{}
		if argsStr != "" {
			var parsed map[string]interface{}
			if err := json.Unmarshal([]byte(argsStr), &parsed); err == nil {
				input = parsed
			} else {
				input = map[string]interface{}{}
			}
		} else {
			input = map[string]interface{}{}
		}
		blocks = append(blocks, ContentBlock{
			Type:  "tool_use",
			ID:    id,
			Name:  name,
			Input: input,
		})
	}
	return blocks
}
