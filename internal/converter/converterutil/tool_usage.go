package converterutil

import "encoding/json"

// ToolUsageExtensions captures the non-standard usage objects some
// OpenAI-compatible providers use to report built-in tool executions next to
// the token counters:
//
//   - Responses API: usage.x_tools.web_search.count, with the same usage
//     broken down per billing line in usage.x_details[].plugins.web_search.count;
//   - chat-style usage: usage.plugins.search.count.
//
// The fields stay raw and are decoded only on demand, so an unexpected shape
// under one of these keys never fails the decode of the surrounding usage
// object (which would drop the token counters along with it). Embedding the
// struct into a typed usage object also carries the fields through a
// decode/re-encode round trip unchanged.
type ToolUsageExtensions struct {
	XTools   json.RawMessage `json:"x_tools,omitempty"`
	XDetails json.RawMessage `json:"x_details,omitempty"`
	Plugins  json.RawMessage `json:"plugins,omitempty"`
}

type webSearchCount struct {
	WebSearch struct {
		Count int `json:"count"`
	} `json:"web_search"`
}

// WebSearchRequests returns the number of built-in web search executions
// reported through the extensions, or 0. The objects are alternative views of
// the same executions, so they are never added together: x_tools wins, then
// x_details, then plugins.search. x_details lines are not summed either —
// the largest line is taken, so a figure repeated on several lines is not
// billed several times.
func (u ToolUsageExtensions) WebSearchRequests() int {
	if len(u.XTools) > 0 {
		var tools webSearchCount
		if json.Unmarshal(u.XTools, &tools) == nil && tools.WebSearch.Count > 0 {
			return tools.WebSearch.Count
		}
	}

	if len(u.XDetails) > 0 {
		var details []struct {
			Plugins webSearchCount `json:"plugins"`
		}
		if json.Unmarshal(u.XDetails, &details) == nil {
			largest := 0
			for _, detail := range details {
				largest = max(largest, detail.Plugins.WebSearch.Count)
			}
			if largest > 0 {
				return largest
			}
		}
	}

	if len(u.Plugins) > 0 {
		var plugins struct {
			Search struct {
				Count int `json:"count"`
			} `json:"search"`
		}
		if json.Unmarshal(u.Plugins, &plugins) == nil && plugins.Search.Count > 0 {
			return plugins.Search.Count
		}
	}

	return 0
}
