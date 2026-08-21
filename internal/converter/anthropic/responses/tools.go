package anthropicresponses

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// responsesToolsToAnthropic converts Responses API tools to Anthropic tool definitions.
// Unsupported tool types (file_search, code_interpreter, mcp, image_generation) are
// dropped — Anthropic has no equivalent. web_search variants are converted (see below).
func responsesToolsToAnthropic(tools []responses.Tool) ([]anthropic.AnthropicTool, []string) {
	var anthropicTools []anthropic.AnthropicTool
	var betas []string

	for _, t := range tools {
		switch t.Type {
		case "function":
			anthropicTools = append(anthropicTools, anthropic.AnthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.Parameters,
			})

		case "computer_use_preview":
			w := 0
			h := 0
			if t.DisplayWidth != nil {
				w = *t.DisplayWidth
			}
			if t.DisplayHeight != nil {
				h = *t.DisplayHeight
			}
			anthropicTools = append(anthropicTools, anthropic.AnthropicTool{
				Type:            "computer_20241022",
				Name:            "computer",
				DisplayWidthPx:  w,
				DisplayHeightPx: h,
			})
			// Computer use requires the beta header.
			betas = appendUnique(betas, "computer-use-2024-10-22")

		case "web_search", "web_search_preview", "web_search_preview_2025_03_11":
			// Responses API's built-in web search tool → Anthropic's own
			// hosted web_search tool. search_context_size has no Anthropic
			// equivalent and is dropped; user_location carries over as-is.
			tool := anthropic.AnthropicTool{
				Type: "web_search_20250305",
				Name: "web_search",
			}
			if t.UserLocation != nil {
				tool.UserLocation = t.UserLocation
			}
			anthropicTools = append(anthropicTools, tool)

		default:
			// A client may send Anthropic's own versioned web_search type
			// directly (e.g. "web_search_20250305", "web_search_20260209"):
			// pass it through instead of dropping it as unrecognized.
			if strings.HasPrefix(t.Type, "web_search_") {
				anthropicTools = append(anthropicTools, anthropic.AnthropicTool{
					Type: t.Type,
					Name: "web_search",
				})
			}
			// file_search, code_interpreter, mcp, image_generation, etc. are
			// not supported by Anthropic — skip them.
		}
	}

	return anthropicTools, betas
}

// responsesToolChoiceToAnthropic maps Responses API tool_choice to Anthropic tool_choice.
func responsesToolChoiceToAnthropic(toolChoice interface{}) interface{} {
	if toolChoice == nil {
		return nil
	}
	switch tc := toolChoice.(type) {
	case string:
		switch tc {
		case "none":
			return map[string]interface{}{"type": "none"}
		case "auto":
			return map[string]interface{}{"type": "auto"}
		case "required":
			return map[string]interface{}{"type": "any"}
		default:
			return nil
		}
	case map[string]interface{}:
		tcType, _ := tc["type"].(string)
		if tcType == "function" {
			name, _ := tc["name"].(string)
			return map[string]interface{}{
				"type": "tool",
				"name": name,
			}
		}
		if tcType == "web_search" || tcType == "web_search_preview" || tcType == "web_search_preview_2025_03_11" ||
			strings.HasPrefix(tcType, "web_search_") {
			// Force the web_search built-in tool by name, same as Anthropic's
			// {"type":"tool","name":"web_search"} tool_choice form.
			return map[string]interface{}{
				"type": "tool",
				"name": "web_search",
			}
		}
	}
	return nil
}

func appendUnique(slice []string, s string) []string {
	for _, v := range slice {
		if v == s {
			return slice
		}
	}
	return append(slice, s)
}
