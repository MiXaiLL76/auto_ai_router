package openai

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"unicode/utf8"
)

// The request rules in this package used to decode the body (into
// map[string]any, or into map[string]json.RawMessage) and encode it again.
// They now edit it through jsonscan and leave untouched members as they are.
// The functions below are the previous implementations; every rule must keep
// producing the same JSON they do.
//
// "The same JSON" is compared on decoded values. The old output differed from
// its input in layout even where no rule applied (sorted keys, HTML-escaped
// strings, numbers rounded through float64 in the map[string]any ones); none
// of that is something a rule meant to do.

func legacyUpdateJSONField(body []byte, mapping ModelParamsMapping) []byte {
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	applyMapping(data, mapping)
	updatedBody, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return updatedBody
}

func legacyRebuildStreamOptionsIncludeUsageOnly(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"stream_options"`)) {
		return body
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	if _, exists := data["stream_options"]; !exists {
		return body
	}
	data["stream_options"] = streamOptionsIncludeUsageOnly
	marshaled, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return marshaled
}

func legacyDropEmptyTools(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"tools"`)) {
		return body
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return body
	}
	var tools []json.RawMessage
	raw, ok := top["tools"]
	if !ok || json.Unmarshal(raw, &tools) != nil || tools == nil || len(tools) > 0 {
		return body
	}
	delete(top, "tools")
	delete(top, "tool_choice")
	delete(top, "parallel_tool_calls")
	out, err := json.Marshal(top)
	if err != nil {
		return body
	}
	return out
}

func legacyReplaceBodyParam(modelID string, body []byte) []byte {
	if !IsReasoningModel(modelID) {
		return body
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	mapping := ModelParamsMapping{
		KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"},
		KeysToRemove:  chatLogprobsParams,
	}
	if !reasoningOff(modelID, chatReasoningEffort(data)) {
		mapping.KeysToRemove = slices.Concat(chatLogprobsParams, samplingParams)
	}
	applyMapping(data, mapping)
	updated, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return updated
}

func legacyConvertWebSearchTools(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"tools"`)) &&
		!bytes.Contains(body, []byte(`"tool_choice"`)) &&
		bytes.IndexByte(body, '\\') < 0 {
		return body
	}
	return convertWebSearchToolsDecode(body)
}

func sameDecodedJSON(a, b []byte) bool {
	if bytes.Equal(a, b) {
		return true
	}
	// Decoding replaces invalid UTF-8, so two bodies can decode alike while
	// only one of them is valid text on the wire: that must match too.
	if utf8.Valid(a) != utf8.Valid(b) {
		return false
	}
	var av, bv any
	if json.Unmarshal(a, &av) != nil || json.Unmarshal(b, &bv) != nil {
		return false
	}
	return reflect.DeepEqual(av, bv)
}

type requestRule struct {
	name        string
	now, legacy func([]byte) []byte
}

var requestRules = []requestRule{
	{"StripCacheSalt", StripCacheSalt, func(b []byte) []byte {
		if !bytes.Contains(b, []byte(`"cache_salt"`)) {
			return b
		}
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToRemove: []string{"cache_salt"}})
	}},
	{"StripStreamOptions", StripStreamOptions, func(b []byte) []byte {
		if !bytes.Contains(b, []byte(`"stream_options"`)) {
			return b
		}
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToRemove: []string{"stream_options"}})
	}},
	{"StripOpenRouterOnlyFields", StripOpenRouterOnlyFields, func(b []byte) []byte {
		if !bytes.Contains(b, []byte(`"plugins"`)) && !bytes.Contains(b, []byte(`"provider"`)) {
			return b
		}
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToRemove: []string{"plugins", "provider"}})
	}},
	{"StripVLLMOnlySamplingParams", StripVLLMOnlySamplingParams, func(b []byte) []byte {
		if !bytes.Contains(b, []byte(`"chat_template_kwargs"`)) && !bytes.Contains(b, []byte(`"repetition_penalty"`)) && !bytes.Contains(b, []byte(`"length_penalty"`)) {
			return b
		}
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToRemove: []string{"chat_template_kwargs", "repetition_penalty", "length_penalty"}})
	}},
	{"StripResponseFormat", StripResponseFormat, func(b []byte) []byte {
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToRemove: []string{"response_format"}})
	}},
	{"UpdateJSONField(rename)", func(b []byte) []byte {
		return UpdateJSONField(b, ModelParamsMapping{KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"}, KeysToRemove: []string{"logprobs"}})
	}, func(b []byte) []byte {
		return legacyUpdateJSONField(b, ModelParamsMapping{KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"}, KeysToRemove: []string{"logprobs"}})
	}},
	{"RebuildStreamOptionsIncludeUsageOnly", RebuildStreamOptionsIncludeUsageOnly, legacyRebuildStreamOptionsIncludeUsageOnly},
	{"DropEmptyTools", DropEmptyTools, legacyDropEmptyTools},
	{"ReplaceBodyParam(gpt-5)", func(b []byte) []byte { return ReplaceBodyParam("gpt-5", b) }, func(b []byte) []byte { return legacyReplaceBodyParam("gpt-5", b) }},
	{"ReplaceBodyParam(gpt-5.2)", func(b []byte) []byte { return ReplaceBodyParam("gpt-5.2", b) }, func(b []byte) []byte { return legacyReplaceBodyParam("gpt-5.2", b) }},
	{"ReplaceBodyParam(gpt-4o)", func(b []byte) []byte { return ReplaceBodyParam("gpt-4o", b) }, func(b []byte) []byte { return legacyReplaceBodyParam("gpt-4o", b) }},
	{"ConvertWebSearchTools", ConvertWebSearchTools, legacyConvertWebSearchTools},
	{"NormalizeDeveloperRole", NormalizeDeveloperRole, normalizeDeveloperRoleDecode},
}

var requestRuleBodies = []string{
	``, `null`, `[]`, `{`, `{"tools":[}`, `{} x`, `{}`,
	`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`,
	"{\n \"model\" : \"gpt-4o\",\n \"messages\" : [ { \"role\" : \"developer\", \"content\" : \"<be> & brief\" }, { \"role\": \"user\", \"content\": \"a developer asks\" } ],\n \"seed\": 12345678901234567890, \"temperature\": 1.0 }",
	`{"model":"gpt-4o","messages":[{"role":"developer","content":"x"},{"role":"developer","content":"y"},{"role":"system","content":"developer"},"str",7,{"role":7},{"role":"developer","role":"user"},{"role":"user","role":"developer"}]}`,
	`{"model":"gpt-4o","messages":"developer"}`, `{"model":"gpt-4o","messages":{"role":"developer"}}`, `{"model":"gpt-4o","messages":null,"note":"developer"}`,
	`{"model":"gpt-4o","messages":[{"role":"user","content":"café"}],"messages":[{"role":"developer","content":"last wins"}]}`,
	`{"model":"gpt-4o","messages":[],"cache_salt":"s","stream_options":{"include_usage":true,"continuous_usage_stats":true},"plugins":[{"id":"web"}],"provider":{"order":["a"]},"chat_template_kwargs":{"enable_thinking":false},"repetition_penalty":1.1,"length_penalty":0.5,"response_format":{"type":"json_object"}}`,
	`{"model":"gpt-4o","messages":[],"stream_options":{"include_usage":true}}`, `{"model":"gpt-4o","messages":[],"stream_options": { "include_usage" : true } }`,
	`{"model":"gpt-4o","messages":[],"stream_options":null}`, `{"model":"gpt-4o","messages":[],"stream_options":"x","stream_options":{"a":1}}`,
	`{"model":"gpt-4o","messages":[{"content":"\"stream_options\" \"cache_salt\" \"tools\" \"plugins\" in text"}]}`,
	`{"model":"gpt-5","messages":[],"max_tokens":100,"temperature":0.5,"top_p":1,"logprobs":true,"top_logprobs":3,"presence_penalty":0,"frequency_penalty":0}`,
	`{"model":"gpt-5","messages":[],"max_tokens":100,"max_completion_tokens":200,"reasoning_effort":"none"}`,
	`{"model":"gpt-5.2","messages":[],"max_tokens":100,"temperature":0.5,"reasoning_effort":"none"}`,
	`{"model":"gpt-5.2","messages":[],"temperature":0.5,"reasoning":{"effort":"high"}}`, `{"model":"gpt-5.2","messages":[],"temperature":0.5,"reasoning":"x","reasoning_effort":null}`,
	`{"model":"gpt-4o","messages":[],"tools":[]}`, `{"model":"gpt-4o","messages":[],"tools":[],"tool_choice":"auto","parallel_tool_calls":true}`,
	`{"model":"gpt-4o","messages":[],"tools":null,"tool_choice":"auto"}`, `{"model":"gpt-4o","messages":[],"tools":"x"}`, `{"model":"gpt-4o","messages":[],"tools":{}}`,
	`{"model":"gpt-4o","messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"}`,
	`{"model":"gpt-4o","messages":[],"tools":[{"type":"function","function":{"name":"f"}},{"type":"web_search"},{"type":"web_search_preview","search_context_size":"high","user_location":{"type":"approximate"}},{"type":"computer_use"},"str",{"notype":1}]}`,
	`{"model":"gpt-4o-search-preview","messages":[],"tools":[{"type":"function","function":{"name":"f"}},{"type":"web_search","search_context_size":"low"},{"type":"web_search_preview","search_context_size":"high","user_location":{"type":"approximate"}},{"type":"code_execution"}],"tool_choice":{"type":"web_search"}}`,
	`{"model":"gpt-4o-search-preview","messages":[],"tools":[{"type":"web_search","search_context_size":"low"}],"web_search_options":{"user_location":{"x":1}},"tool_choice":{"type":"function","function":{"name":"f"}}}`,
	`{"model":"gpt-4o-search-preview","messages":[],"tools":[{"type":"web_search"}],"web_search_options":"notobj"}`,
	`{"model":"openai/gpt-4o-search-preview","messages":[],"tools":[{"type":"web_search_preview"}],"tool_choice":"auto"}`,
	`{"model":"grok-4","messages":[],"tools":[{"type":"web_search"},{"type":"bash"}],"tool_choice":{"type":"web_search_preview"}}`,
	`{"model":"grok-4","messages":[],"tools":[{"type":"text_editor"}],"tool_choice":{"type":"text_editor"}}`,
	`{"model":"gpt-4o","messages":[],"tool_choice":{"type":"computer_use"}}`, `{"model":"gpt-4o","messages":[],"tool_choice":{"type":"function"}}`, `{"model":"gpt-4o","messages":[],"tool_choice":"required"}`,
	`{"model":7,"messages":[],"tools":[{"type":"web_search"}]}`, `{"messages":[],"tools":[{"type":"mcp"}],"tool_choice":{"type":"mcp"}}`,
	`{"model":"gpt-4o","messages":[],"tools":[{"type":"computer_use"}]}`,
	`{"model":"gpt-4o","messages":[],"tools":[{"type":"function"}],"tools":[{"type":"computer_use"}],"cache_salt":1,"cache_salt":2}`,
	"{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"developer\",\"content\":\"\xff\xfe\"}],\"tools\":[{\"type\":\"bash\"}],\"cache_salt\":\"x\"}",
}

func assertRequestRulesMatchLegacy(t testing.TB, body []byte) {
	t.Helper()
	for _, rule := range requestRules {
		got, want := rule.now(bytes.Clone(body)), rule.legacy(bytes.Clone(body))
		if !sameDecodedJSON(got, want) {
			t.Fatalf("%s\n got %.500q\nwant %.500q\n for %.400q", rule.name, got, want, body)
		}
	}
}

func TestRequestRulesMatchLegacy(t *testing.T) {
	for _, body := range requestRuleBodies {
		assertRequestRulesMatchLegacy(t, []byte(body))
	}
}

func TestRequestRulesMatchLegacyOnCorpus(t *testing.T) {
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
		assertRequestRulesMatchLegacy(t, body)
		// The same session carrying every field a rule acts on.
		var m map[string]any
		if json.Unmarshal(body, &m) != nil {
			continue
		}
		m["cache_salt"], m["plugins"], m["provider"] = "s", []any{map[string]any{"id": "web"}}, map[string]any{"order": []any{"x"}}
		m["stream_options"] = map[string]any{"include_usage": true, "continuous_usage_stats": true}
		m["chat_template_kwargs"], m["max_tokens"], m["temperature"], m["logprobs"] = map[string]any{"k": 1}, 10, 0.3, true
		m["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "f"}}, map[string]any{"type": "web_search_preview"}, map[string]any{"type": "computer_use"}}
		m["tool_choice"] = map[string]any{"type": "computer_use"}
		if msgs, ok := m["messages"].([]any); ok {
			m["messages"] = append([]any{map[string]any{"role": "developer", "content": "rules"}}, msgs...)
		}
		full, _ := json.Marshal(m)
		assertRequestRulesMatchLegacy(t, full)
	}
}

// A body no rule applies to must come back as the same slice: no decode, no
// copy.
func TestRequestRulesLeaveUnaffectedBodyAlone(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"line\nbreak"}],"tools":[{"type":"function","function":{"name":"f"}}],"stream":true,"stream_options":{"include_usage":true}}`)
	for _, rule := range requestRules {
		if rule.name == "StripStreamOptions" {
			continue // the one rule this body does give something to do
		}
		got := rule.now(body)
		if len(got) != len(body) || &got[0] != &body[0] {
			t.Errorf("%s copied or rewrote a body it has nothing to change in: %s", rule.name, got)
		}
	}
}

func FuzzRequestRules(f *testing.F) {
	for _, body := range requestRuleBodies {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var asRaw map[string]json.RawMessage
		var full map[string]any
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil {
			t.Skip() // a number outside float64: the map[string]any rules used to leave such a body alone
		}
		assertRequestRulesMatchLegacy(t, body)
	})
}
