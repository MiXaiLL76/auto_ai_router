package openai

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A run of rules applied in one go must give the JSON the rules give one
// after another. Every combination of rules is checked on every body.

func allRequestRules() []RequestRules {
	var out []RequestRules
	for bits := 0; bits < 64; bits++ {
		out = append(out, RequestRules{
			ChatTools:                     bits&1 != 0,
			StripCacheSalt:                bits&2 != 0,
			StreamOptionsIncludeUsageOnly: bits&4 != 0,
			StripOpenRouterOnlyFields:     bits&8 != 0,
			StripVLLMOnlySamplingParams:   bits&16 != 0,
			StripStreamOptions:            bits&32 != 0,
		})
	}
	return out
}

func upperEffort(v string) (string, bool) {
	if v == "" || v == strings.ToUpper(v) {
		return v, false
	}
	return strings.ToUpper(v), true
}

func allClientParamRules() []ClientParamRules {
	var out []ClientParamRules
	for _, model := range []string{"gpt-4o", "gpt-5", "gpt-5.2", "o3-mini"} {
		for _, chat := range []bool{true, false} {
			out = append(out,
				ClientParamRules{ModelID: model, Chat: chat},
				ClientParamRules{ModelID: model, Chat: chat, MapEffort: upperEffort},
				ClientParamRules{ModelID: model, Chat: chat, MapEffort: func(v string) (string, bool) { return v, false }},
			)
		}
	}
	return out
}

func assertRuleRunsMatchOneByOne(t testing.TB, body []byte) {
	t.Helper()
	for _, rules := range allRequestRules() {
		got, want := rules.Apply(body), rules.applyOneByOne(body)
		if !sameDecodedJSON(got, want) {
			t.Fatalf("%+v\n got %.500q\nwant %.500q\n for %.400q", rules, got, want, body)
		}
	}
	for _, rules := range allClientParamRules() {
		got, want := rules.Apply(body), rules.applyOneByOne(body)
		if !sameDecodedJSON(got, want) {
			t.Fatalf("model %s chat=%v effort=%v\n got %.500q\nwant %.500q\n for %.400q",
				rules.ModelID, rules.Chat, rules.MapEffort != nil, got, want, body)
		}
	}
}

var ruleRunBodies = []string{
	`{"model":"gpt-5","input":"hi","tools":[],"tool_choice":"auto","parallel_tool_calls":true,"reasoning":{"effort":"low"},"temperature":0.2,"top_p":1,"top_logprobs":2,"include":["message.output_text.logprobs","reasoning.encrypted_content"]}`,
	`{"model":"gpt-5","input":"hi","include":["message.output_text.logprobs"],"temperature":1,"reasoning":{"effort":"none"}}`,
	`{"model":"gpt-5.2","input":"hi","include":"notalist","temperature":1,"reasoning":"x"}`,
	`{"model":"gpt-5","messages":[],"max_tokens":9,"max_completion_tokens":7,"reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"low","x":1},"logprobs":true}`,
	`{"model":"glm-4","messages":[],"tools":[{"type":"web_search","web_search":{"enable":true}},{"type":"web_search","web_search":{"search_result":true}},{"type":"web_search","web_search":{"search_result":"TRUE"}},{"type":"web_search"},{"type":"function","function":{"name":"f"}}],"stream_options":{"include_usage":true,"continuous_usage_stats":true},"cache_salt":"s"}`,
	`{"model":"gpt-4o-search-preview","messages":[],"tools":[{"type":"web_search","web_search":{"enable":true},"search_context_size":"low"}],"plugins":[{"id":"web"}],"provider":{"order":["x"]},"repetition_penalty":1.2,"length_penalty":1,"chat_template_kwargs":{}}`,
	`{"model":"m","messages":[{"role":"user","content":"\"tools\" \"web_search\" \"cache_salt\" \"stream_options\" effort"}]}`,
}

func TestRuleRunsMatchOneByOne(t *testing.T) {
	bodies := append(append([]string{}, requestRuleBodies...), ruleRunBodies...)
	bodies = append(bodies, realEscapes(
		`{"model":"glm-4","messages":[],"tools":[{"type":"web%u005fsearch","web_search":{"enable":true}}],"cache%u005fsalt":1,"stream_options":{"a":1}}`,
		`{"model":"gpt-5","messages":[{"content":"caf%u00e9 %u001b[0m"}],"max_tokens":3,"tools":[],"cache_salt":1,"stream_options":{"a":1},"plugins":[],"temperature":1}`,
		`{"model":"gpt-5","input":"x","reasoning":{"eff%u006frt":"low"},"temperature":1,"t%u006fols":[]}`,
	)...)
	for _, body := range bodies {
		assertRuleRunsMatchOneByOne(t, []byte(body))
	}
}

func TestRuleRunsMatchOneByOneOnCorpus(t *testing.T) {
	dir := os.Getenv("AIR_BODY_CORPUS")
	if dir == "" {
		t.Skip("set AIR_BODY_CORPUS to a directory of request bodies")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	for _, file := range files {
		body, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		assertRuleRunsMatchOneByOne(t, body)
		var m map[string]any
		if json.Unmarshal(body, &m) != nil {
			continue
		}
		m["cache_salt"], m["plugins"], m["provider"] = "s", []any{map[string]any{"id": "web"}}, map[string]any{"order": []any{"x"}}
		m["stream_options"] = map[string]any{"include_usage": true, "continuous_usage_stats": true}
		m["chat_template_kwargs"], m["max_tokens"], m["temperature"], m["logprobs"] = map[string]any{"reasoning_effort": "low"}, 10, 0.3, true
		m["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "f"}}, map[string]any{"type": "web_search", "web_search": map[string]any{"enable": true}}, map[string]any{"type": "computer_use"}}
		m["tool_choice"], m["reasoning_effort"] = map[string]any{"type": "computer_use"}, "medium"
		full, _ := json.Marshal(m)
		assertRuleRunsMatchOneByOne(t, full)
	}
}

// With nothing for any rule to do, the body comes back as the same slice.
func TestRuleRunsLeaveUnaffectedBodyAlone(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"line\nbreak"}],"tools":[{"type":"function","function":{"name":"f"}}],"stream":true,"stream_options":{"include_usage":true}}`)
	everything := RequestRules{ChatTools: true, StripCacheSalt: true, StreamOptionsIncludeUsageOnly: true, StripOpenRouterOnlyFields: true, StripVLLMOnlySamplingParams: true}
	if got := everything.Apply(body); len(got) != len(body) || &got[0] != &body[0] {
		t.Fatalf("copied or rewrote a body no rule changes: %s", got)
	}
	if got := (ClientParamRules{ModelID: "gpt-4o", Chat: true, MapEffort: upperEffort}).Apply(body); len(got) != len(body) || &got[0] != &body[0] {
		t.Fatalf("copied or rewrote a body no rule changes: %s", got)
	}
	if got := (RequestRules{}).Apply([]byte(`not json`)); string(got) != `not json` {
		t.Fatalf("no rules: %s", got)
	}
}

func FuzzRuleRuns(f *testing.F) {
	for _, body := range append(append([]string{}, requestRuleBodies...), ruleRunBodies...) {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var asRaw map[string]json.RawMessage
		var full map[string]any
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil {
			t.Skip() // a number outside float64, see FuzzRequestRules
		}
		assertRuleRunsMatchOneByOne(t, body)
	})
}
