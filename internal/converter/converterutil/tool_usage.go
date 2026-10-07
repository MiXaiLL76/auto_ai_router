package converterutil

import (
	"bytes"
	"encoding/json"
	"math"
)

// ToolUsageExtensions captures the non-standard usage objects some
// OpenAI-compatible providers use to report built-in tool executions next to
// the token counters:
//
//   - Responses API: usage.x_tools.web_search.count, with the same usage
//     broken down per billing line in usage.x_details[].plugins.web_search.count;
//   - chat-style usage: usage.plugins.search.count;
//   - xAI (chat and Responses): usage.server_side_tool_usage_details, one
//     counter per server-side tool category (see ServerSideToolUsage).
//
// The fields stay raw and are decoded only on demand, so an unexpected shape
// under one of these keys never fails the decode of the surrounding usage
// object (which would drop the token counters along with it). Embedding the
// struct into a typed usage object also carries the fields through a
// decode/re-encode round trip unchanged.
type ToolUsageExtensions struct {
	XTools                     json.RawMessage `json:"x_tools,omitempty"`
	XDetails                   json.RawMessage `json:"x_details,omitempty"`
	Plugins                    json.RawMessage `json:"plugins,omitempty"`
	ServerSideToolUsageDetails json.RawMessage `json:"server_side_tool_usage_details,omitempty"`
}

// ServerSideToolUsage is xAI's usage.server_side_tool_usage_details: the
// server-side tool executions that succeeded, which is what xAI bills. Failed
// attempts are not counted there. X Search is billed per fetched post and
// user profile (XPostsFetched/XUsersFetched, not de-duplicated), not per call.
type ServerSideToolUsage struct {
	WebSearchCalls         int
	XSearchCalls           int
	XPostsFetched          int
	XUsersFetched          int
	CodeExecutionCalls     int // code_execution, alias code_interpreter
	AttachmentSearchCalls  int // attachment_search, reported as document_search
	CollectionsSearchCalls int // collections_search, alias file_search
	MCPCalls               int
	ImageGenerationCalls   int
	// WebSearchCallsReported and ImageGenerationCallsReported tell whether
	// the object carried that counter at all. Only a carried counter is
	// authoritative, zero included: these two executions have another
	// source (web_search_call and image_generation_call output items,
	// citations) to fall back to when the counter is missing.
	WebSearchCallsReported       bool
	ImageGenerationCallsReported bool
}

// serverSideToolUsageKeys lists, per category, every counter name that may
// carry it. xAI reports the Responses API tool names (code_interpreter_calls,
// file_search_calls, document_search_calls); the xAI SDK names are accepted
// too. Alternative names are views of the same executions, so the largest one
// is taken rather than their sum.
var serverSideToolUsageKeys = struct {
	web, xSearch, xPosts, xUsers, code, attachment, collections, mcp, image []string
}{
	web:         []string{"web_search_calls"},
	xSearch:     []string{"x_search_calls"},
	xPosts:      []string{"x_posts_fetched"},
	xUsers:      []string{"x_users_fetched"},
	code:        []string{"code_interpreter_calls", "code_execution_calls"},
	attachment:  []string{"document_search_calls", "attachment_search_calls"},
	collections: []string{"file_search_calls", "collections_search_calls"},
	mcp:         []string{"mcp_calls"},
	image:       []string{"image_generation_calls"},
}

// ServerSideToolUsage decodes server_side_tool_usage_details. ok is false when
// the provider sent no such object (absent, null or not an object): there is
// no data then, as opposed to an object reporting zero usage, whose zeros are
// authoritative. A counter missing from a present object counts as zero, and
// so does one whose value is not a non-negative number, so a single malformed
// counter cannot hide the others; neither counts as reported (see
// ServerSideToolUsage.WebSearchCallsReported).
func (u ToolUsageExtensions) ServerSideToolUsage() (usage ServerSideToolUsage, ok bool) {
	raw := bytes.TrimSpace(u.ServerSideToolUsageDetails)
	if len(raw) == 0 || raw[0] != '{' {
		return ServerSideToolUsage{}, false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return ServerSideToolUsage{}, false
	}
	count := func(keys []string) (largest int, reported bool) {
		for _, key := range keys {
			var value *float64 // nil for null
			if json.Unmarshal(fields[key], &value) != nil || value == nil || math.IsNaN(*value) || *value < 0 {
				continue
			}
			reported = true
			largest = max(largest, int(min(*value, math.MaxInt32)))
		}
		return largest, reported
	}
	keys := serverSideToolUsageKeys
	usage.WebSearchCalls, usage.WebSearchCallsReported = count(keys.web)
	usage.XSearchCalls, _ = count(keys.xSearch)
	usage.XPostsFetched, _ = count(keys.xPosts)
	usage.XUsersFetched, _ = count(keys.xUsers)
	usage.CodeExecutionCalls, _ = count(keys.code)
	usage.AttachmentSearchCalls, _ = count(keys.attachment)
	usage.CollectionsSearchCalls, _ = count(keys.collections)
	usage.MCPCalls, _ = count(keys.mcp)
	usage.ImageGenerationCalls, usage.ImageGenerationCallsReported = count(keys.image)
	return usage, true
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
