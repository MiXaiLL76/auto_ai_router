package openai

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Anthropic ids server tools (web search, web fetch, code execution) with
// srvtoolu_: the provider has already run them, the client cannot.
const serverToolCallIDPrefix = "srvtoolu_"

var serverToolCallNeedle = []byte(`"` + serverToolCallIDPrefix)

// StripServerToolCalls drops server tool calls that an aggregator (Requesty)
// relays in a non-streaming Chat Completions body. A choice left without tool
// calls finishes as "stop"; the dropped web searches go to
// usage.server_tool_use, which billing reads.
func StripServerToolCalls(body []byte) []byte {
	if !bytes.Contains(body, serverToolCallNeedle) {
		return body
	}

	var root map[string]any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return body
	}
	choices, ok := root["choices"].([]any)
	if !ok {
		return body
	}

	stripped := false
	webSearches := 0
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		message, ok := choice["message"].(map[string]any)
		if !ok {
			continue
		}
		toolCalls, ok := message["tool_calls"].([]any)
		if !ok {
			continue
		}
		kept := make([]any, 0, len(toolCalls))
		for _, rawCall := range toolCalls {
			call, _ := rawCall.(map[string]any)
			id, _ := call["id"].(string)
			if !strings.HasPrefix(id, serverToolCallIDPrefix) {
				kept = append(kept, rawCall)
				continue
			}
			stripped = true
			function, _ := call["function"].(map[string]any)
			if name, _ := function["name"].(string); strings.EqualFold(name, "web_search") {
				webSearches++
			}
		}
		if len(kept) == len(toolCalls) {
			continue
		}
		if len(kept) > 0 {
			message["tool_calls"] = kept
			continue
		}
		delete(message, "tool_calls")
		if choice["finish_reason"] == "tool_calls" {
			choice["finish_reason"] = "stop"
		}
	}
	if !stripped {
		return body
	}

	if webSearches > 0 {
		recordServerWebSearches(root, webSearches)
	}

	result, err := marshalWithoutHTMLEscape(root)
	if err != nil {
		return body
	}
	return result
}

// recordServerWebSearches keeps a counter the provider reported itself.
func recordServerWebSearches(root map[string]any, webSearches int) {
	usage, ok := root["usage"].(map[string]any)
	if !ok {
		return
	}
	serverToolUse, ok := usage["server_tool_use"].(map[string]any)
	if !ok {
		serverToolUse = map[string]any{}
		usage["server_tool_use"] = serverToolUse
	}
	if reported, ok := serverToolUse["web_search_requests"].(json.Number); ok {
		if count, err := reported.Int64(); err == nil && count > 0 {
			return
		}
	}
	serverToolUse["web_search_requests"] = webSearches
}
