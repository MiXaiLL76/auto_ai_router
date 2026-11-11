package deepseektools

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The names of the two function tools the adapter installs in place of the
// built-in ones. They are function tools (not built-ins) so that any
// OpenAI-compatible upstream accepts them: the model decides to call one,
// the router runs it, and the result goes back as an ordinary tool result.
const (
	// SearchToolName is the function name the search tool is installed under.
	SearchToolName = "deepseek_web_search"
	// FetchToolName is the function name the fetch tool is installed under.
	FetchToolName = "deepseek_web_fetch"
)

// Built-in tool type names the adapter intercepts. DeepSeek's flash model
// rejects web_search outright (HTTP 422) and silently ignores web_fetch, so
// both have to be executed by the router.
const (
	// BuiltinWebSearch is the built-in search tool's type name.
	BuiltinWebSearch = "web_search"
	// BuiltinWebSearchPreview is the dated variant of the same tool.
	BuiltinWebSearchPreview = "web_search_preview"
	// BuiltinWebFetch is the built-in fetch tool's type name.
	BuiltinWebFetch = "web_fetch"
)

// ToolPlan is what the adapter decided to do with a request's tools: which
// built-in tools were requested, and the function tools that replace them.
//
// It is computed from the request body before anything is sent upstream, so
// the caller can tell whether the adapter is involved at all (Plan == nil or
// an empty plan means "send the request as it is").
type ToolPlan struct {
	// NeedsSearch is true when the request asked for a built-in search tool.
	NeedsSearch bool
	// NeedsFetch is true when the request asked for a built-in fetch tool.
	NeedsFetch bool
	// SearchContextSize is the requested search context size, carried into
	// the search request's count when the model set one.
	SearchContextSize string
}

// Empty reports whether the plan involves no adapter work.
func (p *ToolPlan) Empty() bool {
	return p == nil || (!p.NeedsSearch && !p.NeedsFetch)
}

// chatTool is the shape of one entry of a Chat Completions tools array.
type chatTool struct {
	Type              string         `json:"type"`
	Function          *chatFunction  `json:"function,omitempty"`
	SearchContextSize string         `json:"search_context_size,omitempty"`
	Raw               map[string]any `json:"-"`
}

// chatFunction is the function part of a tool entry.
type chatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// chatRequest is the part of a Chat Completions request body the adapter
// reads and rewrites. Everything else in the body is preserved verbatim.
type chatRequest struct {
	Model    string     `json:"model,omitempty"`
	Messages []chatMsg  `json:"messages,omitempty"`
	Tools    []chatTool `json:"tools,omitempty"`
}

// chatMsg is one message of the conversation. Only the fields the adapter
// needs are modelled; the rest of the message is carried as raw JSON when
// the body is rewritten.
type chatMsg struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ToolCalls        []chatToolCall  `json:"tool_calls,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
}

// chatToolCall is one tool call the model made.
type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// PlanTools inspects a Chat Completions request body and reports which
// built-in tools the adapter has to run.
//
// It returns a nil plan when the request has no tools at all or none of them
// is a built-in the adapter owns -- the overwhelmingly common case, and the
// one that must cost nothing.
func PlanTools(body []byte) *ToolPlan {
	var request chatRequest
	if err := json.Unmarshal(body, &request); err != nil {
		return nil
	}
	plan := &ToolPlan{}
	for _, tool := range request.Tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview:
			plan.NeedsSearch = true
			if tool.SearchContextSize != "" {
				plan.SearchContextSize = tool.SearchContextSize
			}
		case BuiltinWebFetch:
			plan.NeedsFetch = true
		}
	}
	if plan.Empty() {
		return nil
	}
	return plan
}

// RewriteRequest replaces the built-in tools in a Chat Completions request
// body with the adapter's function tools, so the upstream model can only
// reach them by calling a function.
//
// The original tools are preserved in the function descriptions, so the model
// still learns what the tool was for. Ordinary function tools are left
// untouched: the client asked for them, and the upstream serves them.
//
// The body is returned unchanged when it has no built-in tool the adapter
// owns.
func RewriteRequest(body []byte) ([]byte, *ToolPlan, error) {
	plan := PlanTools(body)
	if plan.Empty() {
		return body, nil, nil
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, nil, fmt.Errorf("deepseektools: parsing request body: %w", err)
	}
	rawTools, ok := fields["tools"]
	if !ok {
		return body, nil, nil
	}
	var tools []chatTool
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return nil, nil, fmt.Errorf("deepseektools: parsing tools: %w", err)
	}

	rewritten := make([]chatTool, 0, len(tools)+2)
	for _, tool := range tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview:
			rewritten = append(rewritten, searchFunctionTool(tool))
		case BuiltinWebFetch:
			rewritten = append(rewritten, fetchFunctionTool(tool))
		default:
			rewritten = append(rewritten, tool)
		}
	}
	encoded, err := json.Marshal(rewritten)
	if err != nil {
		return nil, nil, fmt.Errorf("deepseektools: encoding tools: %w", err)
	}
	fields["tools"] = encoded
	result, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("deepseektools: encoding request body: %w", err)
	}
	return result, plan, nil
}

// searchFunctionTool builds the function tool that replaces the built-in
// search tool.
func searchFunctionTool(_ chatTool) chatTool {
	return chatTool{
		Type: "function",
		Function: &chatFunction{
			Name: SearchToolName,
			Description: "Search the web and return the matching pages. " +
				"Arguments: {\"query\": string, \"count\": integer, " +
				"\"ori_question\": string, \"need_content\": boolean}. " +
				"Use this instead of a built-in web search.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query, 1 to 128 characters.",
					},
					"count": map[string]any{
						"type":        "integer",
						"description": "How many results to return, 1 to 50.",
					},
					"ori_question": map[string]any{
						"type":        "string",
						"description": "The user's original question, to improve query understanding.",
					},
					"need_content": map[string]any{
						"type":        "boolean",
						"description": "Whether to include the page content in the results.",
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

// fetchFunctionTool builds the function tool that replaces the built-in
// fetch tool.
func fetchFunctionTool(_ chatTool) chatTool {
	return chatTool{
		Type: "function",
		Function: &chatFunction{
			Name: FetchToolName,
			Description: "Fetch a web page and return its content as Markdown. " +
				"Arguments: {\"url\": string}. " +
				"Use this instead of a built-in web fetch.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The page URL to fetch.",
					},
				},
				"required": []string{"url"},
			},
		},
	}
}

// SearchArguments are the arguments the model passed to the search tool.
type SearchArguments struct {
	Query        string `json:"query"`
	Count        int    `json:"count"`
	OriQuestion  string `json:"ori_question"`
	Localization string `json:"localization"`
	NeedContent  bool   `json:"need_content"`
}

// FetchArguments are the arguments the model passed to the fetch tool.
type FetchArguments struct {
	URL string `json:"url"`
}

// ParseSearchArguments reads the search tool's arguments. The model's
// arguments are JSON in a string, so an empty or malformed value is an error
// the caller reports to the model as a tool error.
func ParseSearchArguments(raw string) (SearchArguments, error) {
	var args SearchArguments
	if strings.TrimSpace(raw) == "" {
		return args, fmt.Errorf("deepseektools: empty search arguments")
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return args, fmt.Errorf("deepseektools: parsing search arguments: %w", err)
	}
	if strings.TrimSpace(args.Query) == "" {
		return args, fmt.Errorf("deepseektools: search arguments have no query")
	}
	return args, nil
}

// ParseFetchArguments reads the fetch tool's arguments.
func ParseFetchArguments(raw string) (FetchArguments, error) {
	var args FetchArguments
	if strings.TrimSpace(raw) == "" {
		return args, fmt.Errorf("deepseektools: empty fetch arguments")
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return args, fmt.Errorf("deepseektools: parsing fetch arguments: %w", err)
	}
	if strings.TrimSpace(args.URL) == "" {
		return args, fmt.Errorf("deepseektools: fetch arguments have no url")
	}
	return args, nil
}

// SearchToolResult is the tool result the adapter sends back to the model
// after a search.
type SearchToolResult struct {
	Results []SearchResult `json:"results"`
}

// FetchToolResult is the tool result the adapter sends back to the model
// after a fetch.
type FetchToolResult struct {
	Title   string `json:"title"`
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
	return json.Marshal(FetchToolResult{Title: result.Title, Content: result.Content})
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

// responsesTool is one entry of a Responses API tools array.
type responsesTool struct {
	Type              string        `json:"type"`
	Function          *chatFunction `json:"function,omitempty"`
	SearchContextSize string        `json:"search_context_size,omitempty"`
}

// RewriteResponsesRequest replaces the built-in search/fetch tools in a
// Responses API request body with the adapter's function tools.
//
// It is the Responses-shaped counterpart of RewriteRequest: the conversion to
// Chat Completions drops every built-in tool, so the takeover has to happen
// before it -- the adapter's replacement is an ordinary function tool, which
// the conversion keeps.
//
// The body is returned unchanged when it has no built-in tool the adapter
// owns.
func RewriteResponsesRequest(body []byte) ([]byte, *ToolPlan, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, nil, fmt.Errorf("deepseektools: parsing request body: %w", err)
	}
	rawTools, ok := fields["tools"]
	if !ok {
		return body, nil, nil
	}
	var tools []responsesTool
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return nil, nil, fmt.Errorf("deepseektools: parsing tools: %w", err)
	}

	plan := &ToolPlan{}
	rewritten := make([]responsesTool, 0, len(tools)+2)
	for _, tool := range tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview:
			plan.NeedsSearch = true
			if tool.SearchContextSize != "" {
				plan.SearchContextSize = tool.SearchContextSize
			}
			rewritten = append(rewritten, responsesTool{
				Type:     "function",
				Function: searchFunctionTool(chatTool{}).Function,
			})
		case BuiltinWebFetch:
			plan.NeedsFetch = true
			rewritten = append(rewritten, responsesTool{
				Type:     "function",
				Function: fetchFunctionTool(chatTool{}).Function,
			})
		default:
			rewritten = append(rewritten, tool)
		}
	}
	if plan.Empty() {
		return body, nil, nil
	}
	encoded, err := json.Marshal(rewritten)
	if err != nil {
		return nil, nil, fmt.Errorf("deepseektools: encoding tools: %w", err)
	}
	fields["tools"] = encoded
	result, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("deepseektools: encoding request body: %w", err)
	}
	return result, plan, nil
}
