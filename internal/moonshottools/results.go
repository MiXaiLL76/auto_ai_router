package moonshottools

import "encoding/json"

// SearchToolResult is what the search tool returns to the model.
type SearchToolResult struct {
	// Results are the pages the provider found.
	Results []SearchResult `json:"results"`
}

// FetchToolResult is what the fetch tool returns to the model.
type FetchToolResult struct {
	Title string `json:"title"`
	// Content is the page's content, as markdown.
	Content string `json:"content"`
}

// EncodeSearchResult encodes the search result for the tool message.
func EncodeSearchResult(result *SearchResponse) ([]byte, error) {
	if result == nil {
		result = &SearchResponse{}
	}
	return json.Marshal(SearchToolResult{Results: result.Results})
}

// EncodeFetchResult encodes the fetch result for the tool message.
func EncodeFetchResult(result *FetchResponse) ([]byte, error) {
	if result == nil {
		result = &FetchResponse{}
	}
	return json.Marshal(FetchToolResult{Title: result.Title, Content: result.Markdown})
}

// EncodeToolError encodes a failed tool call for the tool message. The model
// sees the provider's own error, so it can react to it instead of the loop
// failing outright.
func EncodeToolError(message string) []byte {
	encoded, err := json.Marshal(map[string]any{"error": message})
	if err != nil {
		return []byte(`{"error":"the tool call failed"}`)
	}
	return encoded
}
