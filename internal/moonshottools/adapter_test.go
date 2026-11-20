package moonshottools

import (
	"encoding/json"
	"testing"
)

func TestPlanToolsDetectsBuiltins(t *testing.T) {
	cases := []struct {
		name          string
		body          string
		wantSearch    bool
		wantSearchPro bool
		wantFetch     bool
		wantNil       bool
	}{
		{
			name:    "no tools",
			body:    `{"model":"kimi-k2","messages":[]}`,
			wantNil: true,
		},
		{
			name:    "only function tools",
			body:    `{"tools":[{"type":"function","function":{"name":"lookup"}}]}`,
			wantNil: true,
		},
		{
			name:       "web_search",
			body:       `{"tools":[{"type":"web_search"}]}`,
			wantSearch: true,
		},
		{
			name:       "web_search_preview",
			body:       `{"tools":[{"type":"web_search_preview"}]}`,
			wantSearch: true,
		},
		{
			name:          "web_search_pro",
			body:          `{"tools":[{"type":"web_search_pro"}]}`,
			wantSearchPro: true,
		},
		{
			name:      "web_fetch",
			body:      `{"tools":[{"type":"web_fetch"}]}`,
			wantFetch: true,
		},
		{
			name:       "basic and fetch",
			body:       `{"tools":[{"type":"web_search"},{"type":"web_fetch"}]}`,
			wantSearch: true,
			wantFetch:  true,
		},
		{
			name:          "basic and pro",
			body:          `{"tools":[{"type":"web_search"},{"type":"web_search_pro"}]}`,
			wantSearch:    true,
			wantSearchPro: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := PlanTools([]byte(tc.body))
			if tc.wantNil {
				if plan != nil {
					t.Fatalf("expected no plan, got %+v", plan)
				}
				return
			}
			if plan == nil {
				t.Fatal("expected a plan")
			}
			if plan.NeedsSearch != tc.wantSearch {
				t.Errorf("NeedsSearch = %v, want %v", plan.NeedsSearch, tc.wantSearch)
			}
			if plan.NeedsSearchPro != tc.wantSearchPro {
				t.Errorf("NeedsSearchPro = %v, want %v", plan.NeedsSearchPro, tc.wantSearchPro)
			}
			if plan.NeedsFetch != tc.wantFetch {
				t.Errorf("NeedsFetch = %v, want %v", plan.NeedsFetch, tc.wantFetch)
			}
		})
	}
}

func TestRewriteRequestReplacesBuiltinsWithFunctions(t *testing.T) {
	body := []byte(`{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "web_search", "search_context_size": "high"},
			{"type": "web_search_pro"},
			{"type": "web_fetch"},
			{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
		]
	}`)

	rewritten, plan, err := RewriteRequest(body)
	if err != nil {
		t.Fatalf("RewriteRequest: %v", err)
	}
	if plan == nil || !plan.NeedsSearch || !plan.NeedsSearchPro || !plan.NeedsFetch {
		t.Fatalf("unexpected plan: %+v", plan)
	}
	if plan.SearchContextSize != "high" {
		t.Errorf("SearchContextSize = %q, want %q", plan.SearchContextSize, "high")
	}

	var request struct {
		Tools []chatTool `json:"tools"`
	}
	if err := json.Unmarshal(rewritten, &request); err != nil {
		t.Fatalf("unmarshalling the rewritten body: %v", err)
	}
	if len(request.Tools) != 4 {
		t.Fatalf("got %d tools, want 4", len(request.Tools))
	}
	wantNames := []string{SearchToolName, SearchProToolName, FetchToolName, "lookup"}
	for i, want := range wantNames {
		if request.Tools[i].Type != "function" {
			t.Errorf("tool %d has the type %q, want function", i, request.Tools[i].Type)
		}
		if request.Tools[i].Function == nil {
			t.Fatalf("tool %d has no function", i)
		}
		if request.Tools[i].Function.Name != want {
			t.Errorf("tool %d is named %q, want %q", i, request.Tools[i].Function.Name, want)
		}
	}
	// The rest of the body is untouched.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(rewritten, &fields); err != nil {
		t.Fatalf("unmarshalling the rewritten body: %v", err)
	}
	if _, ok := fields["messages"]; !ok {
		t.Error("the messages were dropped")
	}
	if string(fields["model"]) != `"kimi-k2"` {
		t.Errorf("the model was changed: %s", fields["model"])
	}
}

func TestRewriteRequestKeepsBodyWithoutBuiltins(t *testing.T) {
	body := []byte(`{"model":"kimi-k2","tools":[{"type":"function","function":{"name":"lookup"}}]}`)
	rewritten, plan, err := RewriteRequest(body)
	if err != nil {
		t.Fatalf("RewriteRequest: %v", err)
	}
	if plan != nil {
		t.Errorf("expected no plan, got %+v", plan)
	}
	if string(rewritten) != string(body) {
		t.Errorf("the body was changed:\n%s", rewritten)
	}
}

func TestRewriteResponsesRequestReplacesBuiltinsWithFunctions(t *testing.T) {
	body := []byte(`{
		"model": "kimi-k2",
		"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
		"tools": [
			{"type": "web_search"},
			{"type": "web_search_pro"},
			{"type": "web_fetch"},
			{"type": "function", "name": "lookup", "parameters": {"type": "object"}}
		]
	}`)

	rewritten, plan, err := RewriteResponsesRequest(body)
	if err != nil {
		t.Fatalf("RewriteResponsesRequest: %v", err)
	}
	if plan == nil || !plan.NeedsSearch || !plan.NeedsSearchPro || !plan.NeedsFetch {
		t.Fatalf("unexpected plan: %+v", plan)
	}

	var request struct {
		Tools []responsesTool `json:"tools"`
	}
	if err := json.Unmarshal(rewritten, &request); err != nil {
		t.Fatalf("unmarshalling the rewritten body: %v", err)
	}
	if len(request.Tools) != 4 {
		t.Fatalf("got %d tools, want 4", len(request.Tools))
	}
	wantNames := []string{SearchToolName, SearchProToolName, FetchToolName, "lookup"}
	for i, want := range wantNames {
		if request.Tools[i].Type != "function" {
			t.Errorf("tool %d has the type %q, want function", i, request.Tools[i].Type)
		}
		// The Responses API names a function tool flatly.
		if request.Tools[i].toolName() != want {
			t.Errorf("tool %d is named %q, want %q", i, request.Tools[i].toolName(), want)
		}
	}
	// The client's own tool is preserved as it was: flat, with its parameters.
	if request.Tools[3].Parameters == nil {
		t.Error("the client's function tool lost its parameters")
	}
	// The input is untouched.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(rewritten, &fields); err != nil {
		t.Fatalf("unmarshalling the rewritten body: %v", err)
	}
	if _, ok := fields["input"]; !ok {
		t.Error("the input was dropped")
	}
}

func TestParseArguments(t *testing.T) {
	search, err := ParseSearchArguments(`{"text_query":"kimi k2","limit":5,"timeout_seconds":10,"include_content":true}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments: %v", err)
	}
	if search.TextQuery != "kimi k2" || search.Limit != 5 || search.TimeoutSeconds != 10 || !search.IncludeContent {
		t.Errorf("unexpected search arguments: %+v", search)
	}
	if _, err := ParseSearchArguments(`{"limit":5}`); err == nil {
		t.Error("a search without a query must be an error")
	}
	if _, err := ParseSearchArguments(``); err == nil {
		t.Error("empty search arguments must be an error")
	}
	if _, err := ParseSearchArguments(`not json`); err == nil {
		t.Error("malformed search arguments must be an error")
	}

	// The OpenAI-shaped spelling of the query is accepted too.
	alt, err := ParseSearchArguments(`{"query":"kimi k2"}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments with the query spelling: %v", err)
	}
	if alt.QueryText() != "kimi k2" {
		t.Errorf("the query = %q, want kimi k2", alt.QueryText())
	}

	pro, err := ParseSearchProArguments(`{"text_query":"kimi k2","sites":["arxiv.org"],"time_window":"week"}`)
	if err != nil {
		t.Fatalf("ParseSearchProArguments: %v", err)
	}
	if len(pro.Sites) != 1 || pro.Sites[0] != "arxiv.org" || pro.TimeWindow != "week" {
		t.Errorf("unexpected search_pro arguments: %+v", pro)
	}
	if _, err := ParseSearchProArguments(`{"sites":["arxiv.org"]}`); err == nil {
		t.Error("a search_pro without a query must be an error")
	}

	fetch, err := ParseFetchArguments(`{"url":"https://example.com/a"}`)
	if err != nil {
		t.Fatalf("ParseFetchArguments: %v", err)
	}
	if fetch.URL != "https://example.com/a" {
		t.Errorf("unexpected fetch arguments: %+v", fetch)
	}
	if _, err := ParseFetchArguments(`{}`); err == nil {
		t.Error("a fetch without a url must be an error")
	}
	if _, err := ParseFetchArguments(`{"url":"   "}`); err == nil {
		t.Error("a fetch with a blank url must be an error")
	}
}

// TestArgumentsMapOntoTheProviderRequest checks the mapping from the model's
// arguments onto the REST body: the fields the client set are carried over,
// the ones it did not are left to the provider's defaults.
func TestArgumentsMapOntoTheProviderRequest(t *testing.T) {
	args, err := ParseSearchArguments(`{"text_query":"kimi k2","limit":7,"timeout_seconds":15,"include_content":true}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments: %v", err)
	}
	request := args.ToSearchRequest()
	if request.TextQuery != "kimi k2" || request.Limit != 7 ||
		request.TimeoutSeconds != 15 || !request.IncludeContent {
		t.Errorf("unexpected search request: %+v", request)
	}

	pro, err := ParseSearchProArguments(`{"text_query":"kimi k2","sites":["arxiv.org"],"time_window":"week"}`)
	if err != nil {
		t.Fatalf("ParseSearchProArguments: %v", err)
	}
	proRequest := pro.ToSearchProRequest()
	if proRequest.TextQuery != "kimi k2" || len(proRequest.Sites) != 1 ||
		proRequest.TimeWindow != "week" {
		t.Errorf("unexpected search_pro request: %+v", proRequest)
	}
}

// TestWantContentAcceptsBothSpellings checks that either spelling of the
// "give me the page content" flag asks for the content.
func TestWantContentAcceptsBothSpellings(t *testing.T) {
	for _, raw := range []string{
		`{"text_query":"q","include_content":true}`,
		`{"text_query":"q","need_content":true}`,
	} {
		args, err := ParseSearchArguments(raw)
		if err != nil {
			t.Fatalf("ParseSearchArguments(%s): %v", raw, err)
		}
		if !args.WantContent() {
			t.Errorf("%s: WantContent = false, want true", raw)
		}
	}
	args, err := ParseSearchArguments(`{"text_query":"q"}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments: %v", err)
	}
	if args.WantContent() {
		t.Error("WantContent = true, want false")
	}
}

func TestEndpointPathForBothBaseURLForms(t *testing.T) {
	cases := []struct {
		base string
		path string
		want string
	}{
		{"https://api.moonshot.ai", SearchPath, "https://api.moonshot.ai/v1/tools/search"},
		{"https://api.moonshot.ai/", SearchPath, "https://api.moonshot.ai/v1/tools/search"},
		{"https://api.moonshot.ai/v1", SearchPath, "https://api.moonshot.ai/v1/tools/search"},
		{"https://api.moonshot.ai/v1/", SearchPath, "https://api.moonshot.ai/v1/tools/search"},
		{"https://api.moonshot.ai", SearchProPath, "https://api.moonshot.ai/v1/tools/search_pro"},
		{"https://api.moonshot.ai/v1", SearchProPath, "https://api.moonshot.ai/v1/tools/search_pro"},
		{"https://api.moonshot.ai", FetchPath, "https://api.moonshot.ai/v1/tools/fetch"},
		{"https://api.moonshot.ai/v1", FetchPath, "https://api.moonshot.ai/v1/tools/fetch"},
	}
	for _, tc := range cases {
		if got := endpointPath(tc.base, tc.path); got != tc.want {
			t.Errorf("endpointPath(%q, %q) = %q, want %q", tc.base, tc.path, got, tc.want)
		}
	}
}

// TestToolResultsAreEncodedForTheModel checks what the model receives: the
// results the provider found, and the provider's error when there was one.
func TestToolResultsAreEncodedForTheModel(t *testing.T) {
	encoded, err := EncodeSearchResult(&SearchResponse{Results: []SearchResult{{
		Title: "Kimi K2 released",
		URL:   "https://example.com/k2",
	}}})
	if err != nil {
		t.Fatalf("EncodeSearchResult: %v", err)
	}
	var result SearchToolResult
	if err := json.Unmarshal(encoded, &result); err != nil {
		t.Fatalf("decoding the tool result: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].Title != "Kimi K2 released" {
		t.Errorf("unexpected tool result: %+v", result)
	}

	fetched, err := EncodeFetchResult(&FetchResponse{Title: "Example", Markdown: "# body"})
	if err != nil {
		t.Fatalf("EncodeFetchResult: %v", err)
	}
	var page FetchToolResult
	if err := json.Unmarshal(fetched, &page); err != nil {
		t.Fatalf("decoding the tool result: %v", err)
	}
	if page.Content != "# body" {
		t.Errorf("the fetched content = %q, want the page's markdown", page.Content)
	}

	failed := EncodeToolError("the search returned no results")
	var failure map[string]any
	if err := json.Unmarshal(failed, &failure); err != nil {
		t.Fatalf("decoding the tool error: %v", err)
	}
	if failure["error"] != "the search returned no results" {
		t.Errorf("unexpected tool error: %v", failure)
	}
}
