package openai

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Z.AI (Zhipu GLM) Chat Completions take a built-in search tool shaped
//
//	{"type":"web_search","web_search":{"enable":true,"search_engine":"search-prime",...}}
//
// and report the search only through a top-level "web_search" results array in
// the response: there is no usage counter. The array comes back only when
// web_search.search_result is true, and that flag defaults to false, so a
// client that leaves it out gets (and the provider charges for) a search the
// router cannot see. ForceWebSearchResults turns the flag on for such tools so
// the results always come back as billing evidence; the proxy strips them from
// the client response again when WebSearchResultsHidden says the client did not
// ask for them.

var webSearchNeedle = []byte(`"web_search"`)

// WebSearchResultsHidden reports whether a Chat Completions body carries a
// Z.AI-style web_search tool whose results the client did not ask for.
func WebSearchResultsHidden(body []byte) bool {
	_, hidden := forceWebSearchResults(body, false)
	return hidden
}

// ForceWebSearchResults sets web_search.search_result to true on every
// Z.AI-style web_search tool of a Chat Completions body. Bodies without such a
// tool are returned unchanged.
func ForceWebSearchResults(body []byte) []byte {
	forced, _ := forceWebSearchResults(body, true)
	return forced
}

func forceWebSearchResults(body []byte, rewrite bool) ([]byte, bool) {
	if !bytes.Contains(body, webSearchNeedle) {
		return body, false
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		return body, false
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(data["tools"], &tools); err != nil {
		return body, false
	}

	changed := false
	for i, rawTool := range tools {
		var tool map[string]json.RawMessage
		if err := json.Unmarshal(rawTool, &tool); err != nil {
			continue
		}
		var toolType string
		if err := json.Unmarshal(tool["type"], &toolType); err != nil || toolType != "web_search" {
			continue
		}
		var options map[string]json.RawMessage
		if err := json.Unmarshal(tool["web_search"], &options); err != nil || options == nil {
			continue
		}
		if isJSONTrue(options["search_result"]) {
			continue
		}
		changed = true
		if !rewrite {
			return body, true
		}
		options["search_result"] = json.RawMessage("true")
		encodedOptions, err := marshalWithoutHTMLEscape(options)
		if err != nil {
			return body, false
		}
		tool["web_search"] = encodedOptions
		encodedTool, err := marshalWithoutHTMLEscape(tool)
		if err != nil {
			return body, false
		}
		tools[i] = encodedTool
	}
	if !changed {
		return body, false
	}

	encodedTools, err := marshalWithoutHTMLEscape(tools)
	if err != nil {
		return body, false
	}
	data["tools"] = encodedTools
	out, err := marshalWithoutHTMLEscape(data)
	if err != nil {
		return body, false
	}
	return out, true
}

func isJSONTrue(raw json.RawMessage) bool {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	switch v := value.(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(strings.TrimSpace(v), "true")
	default:
		return false
	}
}

func marshalWithoutHTMLEscape(value any) (json.RawMessage, error) {
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}
