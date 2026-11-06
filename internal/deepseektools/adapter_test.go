package deepseektools

import (
	"encoding/json"
	"testing"
)

func TestPlanToolsDetectsBuiltins(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantSearch bool
		wantFetch  bool
		wantNil    bool
	}{
		{
			name:    "no tools",
			body:    `{"model":"deepseek-flash","messages":[]}`,
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
			name:      "web_fetch",
			body:      `{"tools":[{"type":"web_fetch"}]}`,
			wantFetch: true,
		},
		{
			name:       "both",
			body:       `{"tools":[{"type":"web_search"},{"type":"web_fetch"}]}`,
			wantSearch: true,
			wantFetch:  true,
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
			if plan.NeedsFetch != tc.wantFetch {
				t.Errorf("NeedsFetch = %v, want %v", plan.NeedsFetch, tc.wantFetch)
			}
		})
	}
}

func TestRewriteRequestReplacesBuiltinsWithFunctions(t *testing.T) {
	body := []byte(`{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "web_search", "search_context_size": "high"},
			{"type": "web_fetch"},
			{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
		]
	}`)

	rewritten, plan, err := RewriteRequest(body)
	if err != nil {
		t.Fatalf("RewriteRequest: %v", err)
	}
	if plan == nil || !plan.NeedsSearch || !plan.NeedsFetch {
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
	if len(request.Tools) != 3 {
		t.Fatalf("got %d tools, want 3", len(request.Tools))
	}
	if request.Tools[0].Type != "function" || request.Tools[0].Function == nil ||
		request.Tools[0].Function.Name != SearchToolName {
		t.Errorf("the search tool was not replaced: %+v", request.Tools[0])
	}
	if request.Tools[1].Type != "function" || request.Tools[1].Function == nil ||
		request.Tools[1].Function.Name != FetchToolName {
		t.Errorf("the fetch tool was not replaced: %+v", request.Tools[1])
	}
	// An ordinary function tool is the client's own and must survive.
	if request.Tools[2].Type != "function" || request.Tools[2].Function == nil ||
		request.Tools[2].Function.Name != "lookup" {
		t.Errorf("the client's function tool was not preserved: %+v", request.Tools[2])
	}
	// The rest of the body is untouched.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(rewritten, &fields); err != nil {
		t.Fatalf("unmarshalling the rewritten body: %v", err)
	}
	if _, ok := fields["messages"]; !ok {
		t.Error("the messages were dropped")
	}
	if string(fields["model"]) != `"deepseek-flash"` {
		t.Errorf("the model was changed: %s", fields["model"])
	}
}

func TestRewriteRequestKeepsBodyWithoutBuiltins(t *testing.T) {
	body := []byte(`{"model":"deepseek-flash","tools":[{"type":"function","function":{"name":"lookup"}}]}`)
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

func TestParseArguments(t *testing.T) {
	search, err := ParseSearchArguments(`{"query":"deepseek v4","count":5,"need_content":true}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments: %v", err)
	}
	if search.Query != "deepseek v4" || search.Count != 5 || !search.NeedContent {
		t.Errorf("unexpected search arguments: %+v", search)
	}
	if _, err := ParseSearchArguments(`{"count":5}`); err == nil {
		t.Error("a search without a query must be an error")
	}
	if _, err := ParseSearchArguments(``); err == nil {
		t.Error("empty search arguments must be an error")
	}
	if _, err := ParseSearchArguments(`not json`); err == nil {
		t.Error("malformed search arguments must be an error")
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
}
