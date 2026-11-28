package responses

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// PrepareCodexPassthrough no longer decodes the body. Its output must be the
// same JSON prepareCodexPassthroughDecode (the previous implementation)
// produces. "The same JSON" is checked on decoded values: the decode path
// also sorts keys, re-escapes strings and rounds every number through
// float64, none of which the provider can observe except the last -- and
// there the new path is the faithful one (it forwards the number as sent).

func sameDecodedJSON(a, b []byte) bool {
	if bytes.Equal(a, b) {
		return true
	}
	// Decoding replaces invalid UTF-8, so two bodies can decode alike while
	// only one of them is valid text on the wire: that must match too.
	if utf8.Valid(a) != utf8.Valid(b) {
		return false
	}
	var av, bv interface{}
	if json.Unmarshal(a, &av) != nil || json.Unmarshal(b, &bv) != nil {
		return false
	}
	return reflect.DeepEqual(av, bv)
}

func assertCodexPassthroughMatchesDecode(t testing.TB, body []byte) {
	t.Helper()
	for _, prevHandled := range []bool{false, true} {
		got := PrepareCodexPassthrough(body, prevHandled)
		want := prepareCodexPassthroughDecode(body, prevHandled)
		if !sameDecodedJSON(got, want) {
			t.Fatalf("prevEntryHandled=%v\n got %.600q\nwant %.600q\n for %.400q", prevHandled, got, want, body)
		}
	}
}

var codexPassthroughBodies = []string{
	``, `null`, `[]`, `{`, `{"input":[}`, `{} x`, `{}`,
	`{"model":"gpt-5","input":"hi"}`, `{"model":"gpt-5","input":null}`, `{"model":"gpt-5","input":7}`,
	`{"model":"gpt-5","input":"hi","store":true,"metadata":{"a":"b"},"ttl":60,"previous_response_id":"resp_1"}`,
	`{"model":"gpt-5","input":"hi","store":null,"metadata":null,"ttl":null,"previous_response_id":null}`,
	`{"model":"gpt-5","input":{"role":"user","content":"single"}}`,
	`{"model":"gpt-5","input":{"type":"compaction","encrypted_content":"sum"}}`,
	`{"model":"gpt-5","input":{"type":"compaction"}}`,
	`{"model":"gpt-5","input":{"type":"reasoning","id":"rs_1"}}`,
	`{"model":"gpt-5","input":[]}`,
	`{"model":"gpt-5","input":[{"role":"user","content":"a"},"plain string",7,null,[1,2]]}`,
	`{"model":"gpt-5","input":[{"type":"compaction","id":"c1","encrypted_content":"summary"},{"type":"compaction","id":"c2"},{"type":"compaction","encrypted_content":""},{"type":"compaction","encrypted_content":5}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","summary":[{"type":"summary_text","text":"s"}]},{"type":"reasoning","id":"r2","summary":[]},{"type":"reasoning","id":"r3","summary":"not a list","encrypted_content":"e"}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","content":[{"type":"reasoning_text","text":"t1"},{"type":"reasoning_text","text":""},"x",{"text":"t2"}]}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","content":[{"type":"reasoning_text","text":""}],"encrypted_content":"enc"}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","content":[]},{"type":"reasoning","id":"r2","encrypted_content":""},{"type":"reasoning","id":"r3","encrypted_content":7},{"type":"reasoning"}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1","summary":null,"content":"text","encrypted_content":"enc"}]}`,
	`{"model":"gpt-5","input":[{"type":"function_call_output","call_id":"c","output":"the word reasoning and compaction in tool output"},{"type":"message","role":"user","content":[{"type":"input_text","text":"reasoning"}]}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","id":"r1"},{"type":"compaction","encrypted_content":"s"},{"type":"compaction","encrypted_content":"s2"}]}`,
	`{"model":"gpt-5","input":[{"type":"message","content":"café"},{"type":"reasoning","encrypted_content":"e"}]}`,
	`{"model":"gpt-5","input":[{"type":"message","type":"compaction","encrypted_content":"dup"},{"type":"compaction","type":"message","encrypted_content":"dup2"}]}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","summary":"a","summary":[],"encrypted_content":"e"}]}`,
	`{"model":"gpt-5","input":"a","input":[{"type":"compaction","encrypted_content":"last wins"}]}`,
	`{"model":"gpt-5","input":"hi","instructions":"plain"}`, `{"model":"gpt-5","input":"hi","instructions":null}`,
	`{"model":"gpt-5","input":"hi","instructions":[{"role":"system","content":"a"},{"role":"developer","content":"b"}]}`,
	`{"model":"gpt-5","input":"hi","instructions":[{"content":""},{"content":7},"x",{"other":1}]}`, `{"model":"gpt-5","input":"hi","instructions":[]}`,
	`{"model":"gpt-4o-mini","input":"hi","reasoning":{"effort":"none"}}`, `{"model":"gpt-4o-mini","input":"hi","reasoning":{"effort":"low"}}`,
	`{"model":"gpt-5.1","input":"hi","reasoning":{"effort":"none"}}`, `{"input":"hi","reasoning":{"effort":"none"}}`, `{"model":7,"input":"hi","reasoning":{"effort":"none","summary":"auto"}}`,
	`{"model":"gpt-4o-mini","input":"hi","reasoning":"none"}`, `{"model":"gpt-4o-mini","input":"hi","reasoning":null}`,
	`{"model":"gpt-5","input":"hi","tools":[]}`, `{"model":"gpt-5","input":"hi","tools":[],"tool_choice":"auto"}`,
	`{"model":"gpt-5","input":"hi","tools":[],"tool_choice":"auto","additional_tools":[{"type":"web_search"}]}`,
	`{"model":"gpt-5","input":"hi","tools":[],"tool_choice":"auto","additional_tools":[]}`, `{"model":"gpt-5","input":"hi","tools":null,"tool_choice":"auto"}`,
	`{"model":"gpt-5","input":"hi","tools":"x","tool_choice":"auto"}`,
	`{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"flat","parameters":{"type":"object"}}]}`,
	`{"model":"gpt-5","input":"hi","tools":[{"type":"function","function":{"name":"nested","description":"d","parameters":{"type":"object"}},"strict":true}]}`,
	`{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"outer","function":{"name":"inner"}},{"type":"web_search"},"str",{"type":"function","function":"notobj"},{"type":"custom","function":{"name":"n"}}]}`,
	`{"model":"gpt-5","input":[{"role":"user","content":"x"}],"stream":true,"seed":12345678901234567890,"temperature":1.0,"top_p":0.10}`,
	"{\n  \"model\": \"gpt-5\",\n  \"input\": [\n    { \"type\": \"message\", \"role\": \"user\", \"content\": \"<b>a & b</b>\" },\n    { \"type\": \"reasoning\", \"id\": \"r\" , \"encrypted_content\": \"e\" }\n  ],\n  \"store\": false\n}",
	`{"model":"gpt-5","input":[{"type":"reasoning","encrypted_content":"e"}],"store":false,"store":true,"model":"gpt-4o-mini","reasoning":{"effort":"none"}}`,
	"{\"model\":\"gpt-5\",\"input\":[{\"type\":\"message\",\"content\":\"\xff\xfe\"},{\"type\":\"reasoning\",\"encrypted_content\":\"e\"}]}",
}

func TestPrepareCodexPassthroughMatchesDecodePath(t *testing.T) {
	for _, body := range codexPassthroughBodies {
		assertCodexPassthroughMatchesDecode(t, []byte(body))
	}
}

func TestPrepareCodexPassthroughForwardsUntouchedBodyAsIs(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":[{"type":"message","role":"user","content":"hi"},{"type":"reasoning","id":"r","summary":[],"encrypted_content":"e"}],"tools":[{"type":"function","name":"f"}],"stream":true}`)
	got := PrepareCodexPassthrough(body, false)
	if &got[0] != &body[0] || len(got) != len(body) {
		t.Fatalf("a body no rule applies to must be forwarded without a copy, got %s", got)
	}
	// Only the changed members are rewritten; the conversation keeps its bytes.
	body = []byte(`{"model":"gpt-5","input":[ {"type":"message","content":"a  <b>"} ],"store":false,"stream":true}`)
	want := `{"model":"gpt-5","input":[ {"type":"message","content":"a  <b>"} ],"stream":true}`
	if got := string(PrepareCodexPassthrough(body, false)); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestPrepareCodexPassthroughMatchesDecodePathOnCorpus(t *testing.T) {
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
		assertCodexPassthroughMatchesDecode(t, body)
		// The same session with the shapes the rules rewrite mixed in.
		var m map[string]interface{}
		if json.Unmarshal(body, &m) != nil {
			continue
		}
		if in, ok := m["input"].([]interface{}); ok {
			in = append([]interface{}{map[string]interface{}{"type": "compaction", "encrypted_content": "earlier turns"}}, in...)
			in = append(in, map[string]interface{}{"type": "reasoning", "id": "rs_tail", "content": []interface{}{map[string]interface{}{"type": "reasoning_text", "text": "t"}}},
				map[string]interface{}{"type": "reasoning", "id": "rs_drop"})
			m["input"] = in
		}
		m["store"], m["metadata"], m["previous_response_id"] = true, map[string]interface{}{"k": "v"}, "resp_1"
		m["tools"] = []interface{}{map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "n", "parameters": map[string]interface{}{"type": "object"}}}}
		mixed, _ := json.Marshal(m)
		assertCodexPassthroughMatchesDecode(t, mixed)
	}
}

func FuzzPrepareCodexPassthrough(f *testing.F) {
	for _, body := range codexPassthroughBodies {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var asRaw map[string]json.RawMessage
		var full map[string]interface{}
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil {
			t.Skip() // a number outside float64: the decode path leaves such a body alone
		}
		if strings.Contains(string(body), "\x00") {
			t.Skip()
		}
		assertCodexPassthroughMatchesDecode(t, body)
	})
}
