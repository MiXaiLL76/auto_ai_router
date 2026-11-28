package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// requestReasoningDetails and extractWebSearchRequestUsage used to decode the
// whole request body. They now read it through jsonscan and decode only the
// fields they use. The functions below are the previous implementations,
// kept as the reference the new ones are held to.

func legacyRequestReasoningDetails(body []byte) (bool, string, string) {
	var request map[string]interface{}
	if json.Unmarshal(body, &request) != nil {
		return false, "", thinkingModeUnspecified
	}
	requested, source := mapReasoningSource(request, "")
	return requested, source, mapThinkingMode(request)
}

func legacyExtractWebSearchRequestUsage(body []byte, contentType string) (bool, string) {
	if len(body) == 0 || strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		return false, ""
	}
	var req struct {
		WebSearchOptions map[string]interface{}   `json:"web_search_options,omitempty"`
		Tools            []map[string]interface{} `json:"tools,omitempty"`
		Plugins          []map[string]interface{} `json:"plugins,omitempty"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false, ""
	}
	if req.WebSearchOptions != nil {
		return true, webSearchContextSizeFromMap(req.WebSearchOptions)
	}
	for _, tool := range req.Tools {
		if !isWebSearchTool(tool) {
			continue
		}
		return true, webSearchContextSizeFromMap(tool)
	}
	for _, plugin := range req.Plugins {
		if !isWebSearchPlugin(plugin) {
			continue
		}
		return true, webSearchContextSizeFromMap(plugin)
	}
	return false, ""
}

var requestScanEquivalenceBodies = []string{
	``, `null`, `[]`, `"x"`, `{`, `{"tools":[}`, `{} trailing`, `{}`,
	`{"model":"m","messages":[{"role":"user","content":"line\nbreak, reasoning and thinking as words"}]}`,
	// reasoning
	`{"reasoning_effort":"high"}`, `{"reasoning_effort":"none"}`, `{"reasoning_effort":" Off "}`, `{"reasoning_effort":null}`,
	`{"reasoning":{"effort":"medium"}}`, `{"reasoning":{"effort":null,"type":"enabled"}}`, `{"reasoning":{}}`, `{"reasoning":{"summary":"auto"}}`,
	`{"reasoning":true}`, `{"reasoning":0}`, `{"reasoning":1.5}`, `{"reasoning":[1]}`,
	`{"thinking":{"type":"enabled","budget_tokens":1024}}`, `{"thinking":{"type":"disabled"}}`, `{"thinking":{"type":"adaptive"}}`,
	`{"thinking":{"type":"between_tools"}}`, `{"thinking":{"budget_tokens":0}}`, `{"thinking":{"budget_tokens":"2048"}}`,
	`{"thinking":"enabled"}`, `{"thinking":7}`, `{"thinking":{}}`, `{"thinking":{"type":5}}`,
	`{"thinking_budget":0}`, `{"thinking_budget":"12"}`, `{"thinkingBudget":512}`, `{"thinking_level":"low"}`, `{"thinkingLevel":"none"}`,
	`{"thinking_config":{"thinking_budget":100}}`, `{"thinkingConfig":{"thinkingLevel":"high"}}`, `{"thinking_config":"x"}`,
	`{"extra_body":{"reasoning_effort":"low"}}`, `{"extra_body":{"thinking":{"type":"enabled"}}}`,
	`{"extra_body":{"extra_body":{"thinking":{"type":"disabled"}}}}`,
	`{"extra_body":{"generation_config":{"thinking_config":{"thinking_budget":1}}}}`,
	`{"extra_body":{"generation_config":{"thinkingConfig":{"thinking_level":"off"}}}}`,
	`{"extra_body":"x"}`, `{"extra_body":null,"reasoning_effort":"high"}`,
	`{"reasoning_effort":"none","reasoning":{"effort":"high"},"thinking":{"type":"disabled"}}`,
	`{"reasoning_effort":"low","reasoning_effort":"none"}`,
	`{"reasoning_effort":"high"}`, `{"thinking":{"type":"enabled"}}`, `{"reasoning_effort":"high"}`,
	`{"messages":[{"reasoning_effort":"high","thinking":{"type":"enabled"}}]}`,
	`{"metadata":{"thinking":"enabled"},"reasoning_effort":"none"}`,
	// web search
	`{"web_search_options":{}}`, `{"web_search_options":{"search_context_size":"high"}}`, `{"web_search_options":null}`,
	`{"web_search_options":"x"}`, `{"web_search_options":{"search_context_size":3}}`,
	`{"tools":[{"type":"web_search"}]}`, `{"tools":[{"type":"web_search_preview","search_context_size":"low"}]}`,
	`{"tools":[{"type":"function","function":{"name":"web_search"}},{"type":"google_search"}]}`,
	`{"tools":[{"type":"function"}]}`, `{"tools":[]}`, `{"tools":null}`, `{"tools":"x"}`, `{"tools":[1]}`, `{"tools":[null]}`,
	`{"tools":[{"type":5}]}`,
	`{"plugins":[{"id":"web"}]}`, `{"plugins":[{"id":"web","search_context_size":"medium"}]}`, `{"plugins":[{"id":"other"}]}`, `{"plugins":{}}`,
	`{"web_search_options":{"search_context_size":"low"},"tools":[{"type":"web_search","search_context_size":"high"}]}`,
	`{"tools":[{"type":"web_search"}],"plugins":"broken"}`,
	// encoding/json matches struct fields case-insensitively and merges repeats.
	`{"Tools":[{"type":"web_search"}]}`, `{"TOOLS":[{"type":"function"}],"tools":[{"type":"web_search"}]}`,
	`{"tools":[{"type":"web_search"}],"Tools":[{"type":"function"}]}`,
	`{"tools":[{"type":"web_search","search_context_size":"high"}],"Tools":[{"search_context_size":"low"}]}`,
	`{"web_search_options":{"search_context_size":"high"},"WEB_SEARCH_OPTIONS":{"x":1}}`,
	`{"Web_Search_Options":{},"web_search_options":null}`,
	`{"tools":[{"type":"web_search"}]}`,
	`{"messages":[{"tools":[{"type":"web_search"}]}]}`,
	`{"tools":[{"type":"web_search"}],"temperature":1e999}`,
	"{\"tools\":[{\"type\":\"web_search\"}],\"x\":\"\xff\"}",
}

func TestScanBasedRequestHelpersMatchFullDecode(t *testing.T) {
	for _, body := range requestScanEquivalenceBodies {
		b := []byte(body)
		gotReq, gotSrc, gotMode := requestReasoningDetails(b)
		wantReq, wantSrc, wantMode := legacyRequestReasoningDetails(b)
		if gotReq != wantReq || gotSrc != wantSrc || gotMode != wantMode {
			// The single allowed difference, see the test below.
			if !strings.Contains(body, "1e999") {
				t.Errorf("requestReasoningDetails(%q) = (%v, %q, %q), full decode says (%v, %q, %q)",
					body, gotReq, gotSrc, gotMode, wantReq, wantSrc, wantMode)
			}
		}
		gotWS, gotSize := extractWebSearchRequestUsage(b, "application/json")
		wantWS, wantSize := legacyExtractWebSearchRequestUsage(b, "application/json")
		if gotWS != wantWS || gotSize != wantSize {
			t.Errorf("extractWebSearchRequestUsage(%q) = (%v, %q), full decode says (%v, %q)",
				body, gotWS, gotSize, wantWS, wantSize)
		}
	}
}

func TestScanBasedRequestHelpersFindSomething(t *testing.T) {
	// Guards the equivalence test against both sides being trivially empty.
	if ok, source, mode := requestReasoningDetails([]byte(`{"extra_body":{"thinking":{"type":"enabled"}}}`)); !ok || source != "extra_body.thinking" || mode != thinkingModeEnabled {
		t.Fatalf("reasoning = (%v, %q, %q)", ok, source, mode)
	}
	if ok, size := extractWebSearchRequestUsage([]byte(`{"tools":[{"type":"web_search","search_context_size":"high"}]}`), "application/json"); !ok || size != "high" {
		t.Fatalf("web search = (%v, %q)", ok, size)
	}
	if ok, _ := extractWebSearchRequestUsage([]byte(`{"tools":[{"type":"web_search"}]}`), "multipart/form-data; boundary=x"); ok {
		t.Fatal("multipart bodies are not inspected")
	}
}

// The one deliberate difference: a number outside the float64 range in a
// field the reasoning check does not read. A decode into interface{} fails on
// it, so the old check reported "no reasoning" for the whole request; the new
// one never reads that number. extractWebSearchRequestUsage decoded into a
// struct and never converted unrelated numbers, so it has no such difference.
func TestRequestReasoningDetailsOutOfRangeNumberElsewhere(t *testing.T) {
	body := []byte(`{"reasoning_effort":"high","temperature":1e999}`)
	if ok, _, _ := legacyRequestReasoningDetails(body); ok {
		t.Fatal("reference changed: a full decode used to fail on 1e999")
	}
	if ok, source, _ := requestReasoningDetails(body); !ok || source != "reasoning_effort" {
		t.Fatalf("requestReasoningDetails = (%v, %q), want the reasoning_effort it was sent", ok, source)
	}
	// In a field it does read, the value still fails to decode.
	if ok, _, _ := requestReasoningDetails([]byte(`{"reasoning_effort":"high","thinking_budget":1e999}`)); ok {
		t.Fatal("an undecodable reasoning field must keep reporting no reasoning")
	}
}

func FuzzScanBasedRequestHelpers(f *testing.F) {
	for _, body := range requestScanEquivalenceBodies {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		gotWS, gotSize := extractWebSearchRequestUsage(body, "application/json")
		wantWS, wantSize := legacyExtractWebSearchRequestUsage(body, "application/json")
		if gotWS != wantWS || gotSize != wantSize {
			t.Fatalf("extractWebSearchRequestUsage(%q) = (%v, %q), full decode says (%v, %q)", body, gotWS, gotSize, wantWS, wantSize)
		}
		gotReq, gotSrc, gotMode := requestReasoningDetails(body)
		wantReq, wantSrc, wantMode := legacyRequestReasoningDetails(body)
		if gotReq == wantReq && gotSrc == wantSrc && gotMode == wantMode {
			return
		}
		var asRaw map[string]json.RawMessage
		var full map[string]interface{}
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil && !wantReq {
			return // out-of-range number outside the reasoning fields
		}
		t.Fatalf("requestReasoningDetails(%q) = (%v, %q, %q), full decode says (%v, %q, %q)", body, gotReq, gotSrc, gotMode, wantReq, wantSrc, wantMode)
	})
}
