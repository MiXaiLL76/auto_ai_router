package anthropic

import (
	"encoding/json"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
)

// mapToolChoice maps an OpenAI tool_choice value to the Anthropic tool_choice format.
//
//	"none"            → {"type": "none"}
//	"auto"            → {"type": "auto"}
//	"required"        → {"type": "any"}
//	{type:function}   → {"type": "tool", "name": "<name>"}
//	{type:allowed_tools/auto/any/none/tool} → passed through as-is (Anthropic-native)
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
		// OpenAI format: {"type": "function", "function": {"name": "func_name"}}
		if funcObj, ok := choice["function"].(map[string]interface{}); ok {
			if name, ok := funcObj["name"].(string); ok && name != "" {
				return map[string]interface{}{
					"type": "tool",
					"name": name,
				}
			}
		}
		// Anthropic-native format (e.g. {"type":"allowed_tools",...},
		// {"type":"auto"}, {"type":"any"}, {"type":"none"}, {"type":"tool",...}):
		// pass through unchanged — the Python OpenAI SDK merges extra_body into
		// the top-level request, so these arrive as req.ToolChoice directly.
		if tcType, ok := choice["type"].(string); ok && tcType != "function" {
			return choice
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
				tool.CacheControl = toolMap["cache_control"]
				tools = append(tools, tool)
			}
		case "computer_use":
			tool := AnthropicTool{
				Type: "computer_20250124", // — updated to latest version
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
				Type: "text_editor_20250124", // — updated to latest version
				Name: "str_replace_editor",
			})
		case "bash":
			tools = append(tools, AnthropicTool{
				Type: "bash_20250124", // — updated to latest version
				Name: "bash",
			})
		case "web_search", "web_search_preview":
			tool := AnthropicTool{
				Type: "web_search_20250305",
				Name: "web_search",
			}
			applyWebSearchOptions(&tool, toolMap)
			tools = append(tools, tool)
		default:
			// Built-in tools that already carry their versioned Anthropic type identifier
			// (e.g. "web_search_20250305") reach this switch whenever the request started
			// life as an Anthropic Messages call: MessagesToChat keeps such tools verbatim
			// in the intermediate chat body, so on the way back they match none of the
			// unversioned aliases above. Dropping them leaves the provider with no tool at
			// all and the model answers as if it had never been offered one.
			if builtin, ok := anthropicBuiltinToolFromVersionedType(toolMap, toolType); ok {
				if strings.HasPrefix(toolType, "web_search_") {
					applyWebSearchOptions(&builtin, toolMap)
				}
				tools = append(tools, builtin)
			}
		}
	}
	if len(tools) == 0 {
		return nil
	}
	return tools
}

// applyWebSearchOptions copies the web_search tool's optional fields (per the
// Anthropic Tool definition: max_uses, allowed_domains, blocked_domains,
// user_location, cache_control) from the client-supplied tool map onto the
// converted Anthropic tool, whether it arrived as OpenAI's "web_search"/
// "web_search_preview" shorthand or Anthropic's own versioned type.
func applyWebSearchOptions(tool *AnthropicTool, toolMap map[string]interface{}) {
	if v, ok := toolMap["max_uses"].(float64); ok {
		tool.MaxUses = int(v)
	}
	if v, ok := toolMap["allowed_domains"]; ok {
		tool.AllowedDomains = v
	}
	if v, ok := toolMap["blocked_domains"]; ok {
		tool.BlockedDomains = v
	}
	if v, ok := toolMap["user_location"]; ok {
		tool.UserLocation = v
	}
	if v, ok := toolMap["cache_control"]; ok {
		tool.CacheControl = v
	}
}

// anthropicBuiltinTypePrefixes lists the versioned type prefixes of Anthropic built-in
// tools together with the canonical tool name Anthropic expects next to each type.
var anthropicBuiltinTypePrefixes = []struct {
	prefix string
	name   string
}{
	{"computer_", "computer"},
	{"bash_", "bash"},
	{"text_editor_", "str_replace_editor"},
	{"web_search_", "web_search"},
}

// anthropicBuiltinToolFromVersionedType rebuilds an Anthropic built-in tool from a
// definition that already uses the versioned type identifier. It mirrors the set of
// prefixes MessagesToChat passes through untouched. Returns ok=false for every other
// type, so genuinely unknown tools keep being dropped as before.
func anthropicBuiltinToolFromVersionedType(toolMap map[string]interface{}, toolType string) (AnthropicTool, bool) {
	for _, builtin := range anthropicBuiltinTypePrefixes {
		if !strings.HasPrefix(toolType, builtin.prefix) {
			continue
		}
		tool := AnthropicTool{
			Type: toolType,
			Name: converterutil.GetString(toolMap, "name"),
		}
		if tool.Name == "" {
			tool.Name = builtin.name
		}
		if width, ok := toolMap["display_width_px"].(float64); ok {
			tool.DisplayWidthPx = int(width)
		}
		if height, ok := toolMap["display_height_px"].(float64); ok {
			tool.DisplayHeightPx = int(height)
		}
		tool.CacheControl = toolMap["cache_control"]
		return tool, true
	}
	return AnthropicTool{}, false
}

// expandAllowedTools converts an "allowed_tools" tool_choice into a supported Anthropic/Bedrock
// format by filtering the tools slice to only the allowed subset and returning a plain
// {"type":"auto"} or {"type":"any"} tool_choice based on the mode field.
//
// Bedrock (and the Anthropic API) do not support "allowed_tools" natively; restricting the
// visible tools list achieves the same effect: the model can only call what it sees.
func expandAllowedTools(tc map[string]interface{}, tools []AnthropicTool) (interface{}, []AnthropicTool) {
	// Build set of allowed tool names.
	allowed := map[string]bool{}
	if list, ok := tc["tools"].([]interface{}); ok {
		for _, item := range list {
			if m, ok := item.(map[string]interface{}); ok {
				if name, ok := m["name"].(string); ok && name != "" {
					allowed[name] = true
				}
			}
		}
	}

	// Filter tools to allowed subset (keep all if no names were specified).
	filtered := tools
	if len(allowed) > 0 {
		filtered = make([]AnthropicTool, 0, len(allowed))
		for _, t := range tools {
			if allowed[t.Name] {
				filtered = append(filtered, t)
			}
		}
	}

	// Map mode → tool_choice type.
	choiceType := "auto"
	if mode, _ := tc["mode"].(string); mode == "any" {
		choiceType = "any"
	}
	return map[string]interface{}{"type": choiceType}, filtered
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
