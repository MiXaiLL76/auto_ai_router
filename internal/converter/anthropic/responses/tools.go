package anthropicresponses

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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
			// equivalent and is dropped; user_location and the Anthropic-only
			// restriction fields (see applyResponsesWebSearchOptions) carry
			// over as-is.
			tool := anthropic.AnthropicTool{
				Type: "web_search_20250305",
				Name: "web_search",
			}
			applyResponsesWebSearchOptions(&tool, t)
			anthropicTools = append(anthropicTools, tool)

		default:
			if strings.HasPrefix(t.Type, "web_search_") {
				if isAnthropicVersionedWebSearchType(t.Type) {
					// A client may send Anthropic's own versioned web_search type
					// directly (e.g. "web_search_20250305", "web_search_20260209"):
					// pass it through instead of dropping it as unrecognized.
					tool := anthropic.AnthropicTool{
						Type: t.Type,
						Name: "web_search",
					}
					applyResponsesWebSearchOptions(&tool, t)
					anthropicTools = append(anthropicTools, tool)
				} else {
					// An OpenAI-origin web_search variant not covered by the
					// explicit cases above (e.g. a newer dated Responses API
					// release such as "web_search_2025_08_26"). Sending that
					// string to Anthropic verbatim as "type" produces a 400 —
					// Anthropic only recognizes its own versioned type
					// identifiers — so map it to the current stable Anthropic
					// web_search version instead of guessing at the OpenAI type.
					tool := anthropic.AnthropicTool{
						Type: "web_search_20250305",
						Name: "web_search",
					}
					applyResponsesWebSearchOptions(&tool, t)
					anthropicTools = append(anthropicTools, tool)
				}
			}
			// file_search, code_interpreter, mcp, image_generation, etc. are
			// not supported by Anthropic — skip them.
		}
	}

	return anthropicTools, betas
}

// applyResponsesWebSearchOptions copies the Anthropic-only web_search restriction
// fields (max_uses, allowed_domains, blocked_domains, cache_control — no equivalent
// in OpenAI's own Responses API web_search_preview schema, see responses.Tool) plus
// user_location from a Responses API tool definition onto the converted Anthropic
// tool. Mirrors anthropic.applyWebSearchOptions, which does the same for the Chat
// Completions conversion path.
func applyResponsesWebSearchOptions(tool *anthropic.AnthropicTool, t responses.Tool) {
	if t.UserLocation != nil {
		tool.UserLocation = t.UserLocation
	}
	if t.MaxUses != nil {
		tool.MaxUses = *t.MaxUses
	}
	if t.AllowedDomains != nil {
		tool.AllowedDomains = converterutil.OmitEmptySlice(t.AllowedDomains)
	}
	if t.BlockedDomains != nil {
		tool.BlockedDomains = converterutil.OmitEmptySlice(t.BlockedDomains)
	}
	if t.CacheControl != nil {
		tool.CacheControl = t.CacheControl
	}
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

// isAnthropicVersionedWebSearchType reports whether t is one of Anthropic's own
// versioned web_search tool types (e.g. "web_search_20250305",
// "web_search_20260209") — an 8-digit YYYYMMDD date directly after the
// "web_search_" prefix. OpenAI's own dated Responses API web_search tool
// types use underscore-separated date components instead (e.g.
// "web_search_2025_08_26"), so this distinguishes the two without an
// allowlist that needs updating every time Anthropic ships a new version.
func isAnthropicVersionedWebSearchType(t string) bool {
	const prefix = "web_search_"
	suffix := strings.TrimPrefix(t, prefix)
	if len(suffix) != 8 {
		return false
	}
	for _, r := range suffix {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func appendUnique(slice []string, s string) []string {
	for _, v := range slice {
		if v == s {
			return slice
		}
	}
	return append(slice, s)
}
