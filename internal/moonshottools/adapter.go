package moonshottools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// The names of the function tools the adapter installs in place of the
// built-in search ones. They are function tools (not built-ins) so that
// any OpenAI-compatible upstream accepts them: the model decides to call
// one, the router runs it, and the result goes back as an ordinary tool
// result.
//
// The names are namespaced so that a request carrying several providers'
// search tools keeps them apart: a Moonshot search is not a DeepSeek one,
// and the router runs the one the model actually asked for.
const (
	// SearchToolName is the function name the Basic search is installed
	// under.
	SearchToolName = "moonshot_web_search"
	// SearchProToolName is the function name the Pro search is installed
	// under.
	SearchProToolName = "moonshot_web_search_pro"
	// FetchToolName is the function name the fetch tool is installed under.
	FetchToolName = "moonshot_web_fetch"
)

// Built-in tool type names the adapter intercepts.
//
// The client asks for the search the ordinary way -- the built-in
// web_search tool, the way any OpenAI-compatible client asks a model that
// can run it itself. The router takes it over: the built-in tool becomes a
// function tool the model can call, and the router runs the REST call
// itself.
//
// The Pro search is a separate built-in type, so that a client can pick
// Basic or Pro explicitly; Basic is the default when only the plain
// web_search is asked for.
const (
	// BuiltinWebSearch is the built-in search tool's type name.
	BuiltinWebSearch = "web_search"
	// BuiltinWebSearchPreview is the dated variant of the same tool.
	BuiltinWebSearchPreview = "web_search_preview"
	// BuiltinMoonshotSearchPro is Moonshot's Pro search tool type.
	BuiltinMoonshotSearchPro = "web_search_pro"
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
	// NeedsSearch is true when the request asked for the Basic search.
	NeedsSearch bool
	// NeedsSearchPro is true when the request asked for the Pro search.
	NeedsSearchPro bool
	// NeedsFetch is true when the request asked for the fetch tool.
	NeedsFetch bool
	// SearchContextSize is the requested search context size, when the client
	// set one.
	SearchContextSize string
}

// Empty reports whether the plan involves no adapter work.
func (p *ToolPlan) Empty() bool {
	if p == nil {
		return true
	}
	return !p.NeedsSearch && !p.NeedsSearchPro && !p.NeedsFetch
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

// responsesTool is the shape of one entry of a Responses API tools array.
//
// The Responses API names a function tool flatly -- {"type":"function",
// "name":"lookup"} -- and not nested under "function" the way the Chat
// Completions API does. Both shapes are accepted, because a client may send
// either and the adapter must not drop the client's own tools.
type responsesTool struct {
	Type              string         `json:"type"`
	Name              string         `json:"name,omitempty"`
	Description       string         `json:"description,omitempty"`
	Parameters        map[string]any `json:"parameters,omitempty"`
	Strict            *bool          `json:"strict,omitempty"`
	Function          *chatFunction  `json:"function,omitempty"`
	SearchContextSize string         `json:"search_context_size,omitempty"`
	Raw               map[string]any `json:"-"`
}

// toolName returns the name of the tool, whichever shape it came in.
func (t responsesTool) toolName() string {
	if t.Function != nil && t.Function.Name != "" {
		return t.Function.Name
	}
	return t.Name
}

// responsesFunctionTool converts a Chat-shaped function tool into the
// Responses-shaped one: the Responses API names a function tool flatly, with
// "name" beside "type" rather than nested under "function".
func responsesFunctionTool(tool chatTool) responsesTool {
	function := tool.Function
	if function == nil {
		return responsesTool{Type: "function"}
	}
	return responsesTool{
		Type:        "function",
		Name:        function.Name,
		Description: function.Description,
		Parameters:  function.Parameters,
	}
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
		case BuiltinMoonshotSearchPro:
			plan.NeedsSearchPro = true
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

// SearchArguments are the arguments of the Basic search tool.
//
// The client may name the query either way: "text_query" is Moonshot's own
// spelling and "query" is what an OpenAI-shaped client would send. Both are
// accepted, and the Moonshot spelling wins when both are present.
type SearchArguments struct {
	TextQuery      string `json:"text_query"`
	Query          string `json:"query"`
	Limit          int    `json:"limit"`
	TimeoutSeconds int    `json:"timeout_seconds"`
	IncludeContent bool   `json:"include_content"`
	NeedContent    bool   `json:"need_content"`
	SearchContext  string `json:"search_context_size"`
}

// SearchProArguments are the arguments of the Pro search tool: the Basic
// ones plus what only the Pro search can narrow by.
type SearchProArguments struct {
	SearchArguments
	Sites      []string `json:"sites"`
	TimeWindow string   `json:"time_window"`
}

// FetchArguments are the arguments of the fetch tool.
type FetchArguments struct {
	URL string `json:"url"`
}

// ToFetchRequest maps the arguments onto Moonshot's fetch body.
func (a FetchArguments) ToFetchRequest() FetchRequest {
	return FetchRequest(a)
}

// ParseSearchArguments reads the Basic search tool's arguments. The
// model's arguments are JSON in a string, so an empty or malformed value is
// an error the caller reports to the model as a tool error.
func ParseSearchArguments(raw string) (SearchArguments, error) {
	var args SearchArguments
	if strings.TrimSpace(raw) == "" {
		return args, fmt.Errorf("moonshottools: empty search arguments")
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return args, fmt.Errorf("moonshottools: parsing search arguments: %w", err)
	}
	if args.QueryText() == "" {
		return args, fmt.Errorf("moonshottools: search arguments have no query")
	}
	return args, nil
}

// ParseSearchProArguments reads the Pro search tool's arguments.
func ParseSearchProArguments(raw string) (SearchProArguments, error) {
	var args SearchProArguments
	if strings.TrimSpace(raw) == "" {
		return args, fmt.Errorf("moonshottools: empty search_pro arguments")
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return args, fmt.Errorf("moonshottools: parsing search_pro arguments: %w", err)
	}
	if args.QueryText() == "" {
		return args, fmt.Errorf("moonshottools: search_pro arguments have no query")
	}
	return args, nil
}

// ParseFetchArguments reads the fetch tool's arguments.
func ParseFetchArguments(raw string) (FetchArguments, error) {
	var args FetchArguments
	if strings.TrimSpace(raw) == "" {
		return args, fmt.Errorf("moonshottools: empty fetch arguments")
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return args, fmt.Errorf("moonshottools: parsing fetch arguments: %w", err)
	}
	if strings.TrimSpace(args.URL) == "" {
		return args, fmt.Errorf("moonshottools: fetch arguments have no url")
	}
	return args, nil
}

// QueryText returns the query, whichever spelling the client used.
func (a SearchArguments) QueryText() string {
	if strings.TrimSpace(a.TextQuery) != "" {
		return strings.TrimSpace(a.TextQuery)
	}
	return strings.TrimSpace(a.Query)
}

// WantContent reports whether the client asked for the page content.
//
// "include_content" is Moonshot's own spelling; "need_content" is what an
// OpenAI-shaped client sends (DeepSeek's spelling). Either one asks for it.
func (a SearchArguments) WantContent() bool {
	return a.IncludeContent || a.NeedContent
}

// ToSearchRequest maps the arguments onto Moonshot's Basic search body.
func (a SearchArguments) ToSearchRequest() SearchRequest {
	return SearchRequest{
		TextQuery:      a.QueryText(),
		Limit:          a.Limit,
		TimeoutSeconds: a.TimeoutSeconds,
		IncludeContent: a.WantContent(),
	}
}

// ToSearchProRequest maps the arguments onto Moonshot's Pro search body.
func (a SearchProArguments) ToSearchProRequest() SearchProRequest {
	return SearchProRequest{
		TextQuery:      a.QueryText(),
		Limit:          a.Limit,
		TimeoutSeconds: a.TimeoutSeconds,
		IncludeContent: a.WantContent(),
		Sites:          a.Sites,
		TimeWindow:     a.TimeWindow,
	}
}

// RewriteRequest replaces the built-in search tools in a Chat Completions
// request body with the adapter's function tools.
//
// Ordinary function tools are left untouched: the client asked for them, and
// the upstream serves them -- the router does not run a client's own tools.
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
		return nil, nil, fmt.Errorf("moonshottools: parsing request body: %w", err)
	}
	rawTools, ok := fields["tools"]
	if !ok {
		return body, nil, nil
	}
	var tools []chatTool
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return nil, nil, fmt.Errorf("moonshottools: parsing tools: %w", err)
	}

	rewritten := make([]chatTool, 0, len(tools)+3)
	for _, tool := range tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview:
			rewritten = append(rewritten, searchFunctionTool())
		case BuiltinMoonshotSearchPro:
			rewritten = append(rewritten, searchProFunctionTool())
		case BuiltinWebFetch:
			rewritten = append(rewritten, fetchFunctionTool())
		default:
			rewritten = append(rewritten, tool)
		}
	}
	encoded, err := json.Marshal(rewritten)
	if err != nil {
		return nil, nil, fmt.Errorf("moonshottools: encoding tools: %w", err)
	}
	fields["tools"] = encoded
	result, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("moonshottools: encoding request body: %w", err)
	}
	return result, plan, nil
}

// HasBuiltInTool reports whether a Responses API request body asks for one of
// the built-in tools the adapter takes over.
//
// It is what the router's passthrough branch asks: a request that carries a
// built-in search must not be forwarded as it is (the provider refuses it),
// while every other /v1/responses request keeps the passthrough path.
func HasBuiltInTool(body []byte) bool {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return false
	}
	rawTools, ok := fields["tools"]
	if !ok {
		return false
	}
	var tools []responsesTool
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return false
	}
	for _, tool := range tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview, BuiltinMoonshotSearchPro, BuiltinWebFetch:
			return true
		}
	}
	return false
}

// ExposesTool reports whether the request offers the model any tool this
// adapter could run: a built-in one the adapter takes over, or a function
// tool it installed in place of one.
//
// The router-run stream loop buffers the whole answer before it can decide
// whether to relay it, because it has to see the model's tool call first. On
// a request that offers no tool of ours there is nothing to see: the model
// cannot ask for a tool the request never declared, so buffering only costs
// the client its stream -- the first byte arrives when the last one is already
// generated, and a client that gives up on a slow first token tears down a
// request that was answering all along.
//
// The body this is called with is the one the loop will run on, which is not
// always the client's: RewriteRequest has already replaced a built-in tool
// with the adapter's function, so the usual case is a function name. The
// built-in case is not dead, though. A proxy-like credential gets its body
// back from sanitizeClientParams unchanged, and a rewrite that fails returns
// the body as it came, so the built-in type can still be what the gate sees.
//
// Only the tools are decoded, and only after a cheap substring check says
// there is something to decode: this runs on the hot path, and the messages
// of a long conversation are the bulk of the body.
func ExposesTool(body []byte) bool {
	// Every name this adapter installs or intercepts contains "web_", and a
	// request that offers none of them cannot contain it either. The check is
	// free and keeps the decode -- and its allocation -- off the requests that
	// are the whole point of the gate: the ones with no tools of ours.
	// TestEveryExposedToolNameContainsThePrefilter pins that assumption to the
	// name lists, so adding a tool that breaks it fails a test rather than
	// silently reopening the buffering.
	if !bytes.Contains(body, []byte("web_")) {
		return false
	}
	var request struct {
		Tools []chatTool `json:"tools,omitempty"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return false
	}
	for _, tool := range request.Tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview, BuiltinMoonshotSearchPro, BuiltinWebFetch:
			return true
		}
		if tool.Function != nil {
			switch tool.Function.Name {
			case SearchToolName, SearchProToolName, FetchToolName:
				return true
			}
		}
	}
	return false
}

// RewriteResponsesRequest replaces the built-in search tools in a Responses
// API request body with the adapter's function tools.
//
// It is the Responses-shaped counterpart of RewriteRequest: the conversion to
// Chat Completions drops every built-in tool, so the takeover has to happen
// before it -- the adapter's replacement is an ordinary function tool, which
// the conversion keeps.
func RewriteResponsesRequest(body []byte) ([]byte, *ToolPlan, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, nil, fmt.Errorf("moonshottools: parsing request body: %w", err)
	}
	rawTools, ok := fields["tools"]
	if !ok {
		return body, nil, nil
	}
	var tools []responsesTool
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return nil, nil, fmt.Errorf("moonshottools: parsing tools: %w", err)
	}

	plan := &ToolPlan{}
	rewritten := make([]responsesTool, 0, len(tools)+3)
	for _, tool := range tools {
		switch tool.Type {
		case BuiltinWebSearch, BuiltinWebSearchPreview:
			plan.NeedsSearch = true
			if tool.SearchContextSize != "" {
				plan.SearchContextSize = tool.SearchContextSize
			}
			rewritten = append(rewritten, responsesFunctionTool(searchFunctionTool()))
		case BuiltinMoonshotSearchPro:
			plan.NeedsSearchPro = true
			if tool.SearchContextSize != "" {
				plan.SearchContextSize = tool.SearchContextSize
			}
			rewritten = append(rewritten, responsesFunctionTool(searchProFunctionTool()))
		case BuiltinWebFetch:
			plan.NeedsFetch = true
			rewritten = append(rewritten, responsesFunctionTool(fetchFunctionTool()))
		default:
			rewritten = append(rewritten, tool)
		}
	}
	encoded, err := json.Marshal(rewritten)
	if err != nil {
		return nil, nil, fmt.Errorf("moonshottools: encoding tools: %w", err)
	}
	fields["tools"] = encoded
	result, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("moonshottools: encoding request body: %w", err)
	}
	return result, plan, nil
}

// searchFunctionTool builds the function tool that replaces the built-in
// Basic search tool.
func searchFunctionTool() chatTool {
	return chatTool{
		Type: "function",
		Function: &chatFunction{
			Name: SearchToolName,
			Description: "Search the web and return the matching pages. " +
				"Arguments: {\"text_query\": string, \"limit\": integer, " +
				"\"timeout_seconds\": integer, \"include_content\": boolean}. " +
				"Use this instead of a built-in web search.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text_query": map[string]any{
						"type":        "string",
						"description": "The search query.",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "How many results to return.",
					},
					"timeout_seconds": map[string]any{
						"type":        "integer",
						"description": "How long the provider may search, in seconds.",
					},
					"include_content": map[string]any{
						"type":        "boolean",
						"description": "Whether to include the page content in the results.",
					},
				},
				"required": []string{"text_query"},
			},
		},
	}
}

// searchProFunctionTool builds the function tool that replaces Moonshot's
// Pro search tool.
func searchProFunctionTool() chatTool {
	return chatTool{
		Type: "function",
		Function: &chatFunction{
			Name: SearchProToolName,
			Description: "Search the web with the Pro search, which can be narrowed to " +
				"particular sites and to a time window. " +
				"Arguments: {\"text_query\": string, \"limit\": integer, " +
				"\"timeout_seconds\": integer, \"include_content\": boolean, " +
				"\"sites\": array of strings, \"time_window\": string}. " +
				"Use this instead of a built-in web search.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text_query": map[string]any{
						"type":        "string",
						"description": "The search query.",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "How many results to return.",
					},
					"timeout_seconds": map[string]any{
						"type":        "integer",
						"description": "How long the provider may search, in seconds.",
					},
					"include_content": map[string]any{
						"type":        "boolean",
						"description": "Whether to include the page content in the results.",
					},
					"sites": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "The sites to search in; every site when empty.",
					},
					"time_window": map[string]any{
						"type":        "string",
						"description": "The time window to search in; every window when empty.",
					},
				},
				"required": []string{"text_query"},
			},
		},
	}
}

// fetchFunctionTool builds the function tool that replaces the built-in
// fetch tool.
func fetchFunctionTool() chatTool {
	return chatTool{
		Type: "function",
		Function: &chatFunction{
			Name: FetchToolName,
			Description: "Fetch a page and return it as markdown. " +
				"Arguments: {\"url\": string}. " +
				"Use this instead of a built-in web fetch.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The page's URL.",
					},
				},
				"required": []string{"url"},
			},
		},
	}
}
