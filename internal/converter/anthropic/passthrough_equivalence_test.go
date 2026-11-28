package anthropic

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"unicode/utf8"
)

// injectCacheControl and NormalizeMessagesForPassthrough no longer decode the
// conversation. injectCacheControlDecode and
// normalizeMessagesForPassthroughDecode are the previous, whole-body
// implementations; outcome, error and resulting JSON must match them.

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

func assertInjectCacheControlMatchesDecode(t testing.TB, body []byte) {
	t.Helper()
	got, gotOutcome := injectCacheControl(bytes.Clone(body))
	want, wantOutcome := injectCacheControlDecode(bytes.Clone(body))
	if gotOutcome != wantOutcome || !sameDecodedJSON(got, want) {
		t.Fatalf("injectCacheControl: outcome %q, decode path %q\n got %.600q\nwant %.600q\n for %.400q", gotOutcome, wantOutcome, got, want, body)
	}
}

var passthroughModels = []struct {
	model string
	real  bool
}{
	{"claude-sonnet-4-5", true}, {"claude-sonnet-5-5", true}, {"claude-opus-4-7", true}, {"claude-haiku-5-5", false},
	{"claude-3-5-sonnet", true}, {"glm-4.6", false}, {"", false},
}

// normalizeOutcome runs one implementation and records a panic as an outcome:
// the whole-body implementation writes into the decoded map, which is nil for
// the document `null`, and that body is handed to it unchanged. (Such a
// request never gets this far: it has no model and is refused at ingress.)
func normalizeOutcome(fn func([]byte, string, bool) ([]byte, error), body []byte, model string, real bool) (out []byte, err error, panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	out, err = fn(bytes.Clone(body), model, real)
	return out, err, false
}

func assertNormalizePassthroughMatchesDecode(t testing.TB, body []byte) {
	t.Helper()
	for _, m := range passthroughModels {
		got, gotErr, gotPanic := normalizeOutcome(NormalizeMessagesForPassthrough, body, m.model, m.real)
		want, wantErr, wantPanic := normalizeOutcome(normalizeMessagesForPassthroughDecode, body, m.model, m.real)
		if gotPanic != wantPanic {
			t.Fatalf("NormalizeMessagesForPassthrough(%q, real=%v): panicked=%v, decode path panicked=%v\n for %.400q", m.model, m.real, gotPanic, wantPanic, body)
		}
		if gotPanic {
			continue
		}
		if (gotErr == nil) != (wantErr == nil) || (gotErr != nil && gotErr.Error() != wantErr.Error()) {
			t.Fatalf("NormalizeMessagesForPassthrough(%q, real=%v): error %v, decode path %v\n for %.400q", m.model, m.real, gotErr, wantErr, body)
		}
		if gotErr == nil && !sameDecodedJSON(got, want) {
			t.Fatalf("NormalizeMessagesForPassthrough(%q, real=%v)\n got %.600q\nwant %.600q\n for %.400q", m.model, m.real, got, want, body)
		}
	}
}

var cacheInjectionBodies = []string{
	``, `null`, `[]`, `{`, `{} x`, `{}`, `{"model":"m"}`,
	`{"messages":[]}`, `{"messages":null}`, `{"messages":"x"}`, `{"messages":{"role":"user"}}`, `{"messages":[],"system":null}`, `{"messages":[],"system":""}`,
	`{"messages":[],"system":"You are helpful."}`, `{"messages":[],"system":7}`, `{"messages":[],"system":{"type":"text"}}`, `{"messages":[],"system":[]}`,
	`{"messages":[],"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`, `{"messages":[],"system":["a","b"]}`, `{"messages":[],"system":[{"type":"text","text":"a"},"tail"]}`,
	`{"system":"s","messages":[{"role":"user","content":"u1"}]}`,
	`{"system":"s","messages":[{"role":"user","content":"u1"},{"role":"assistant","content":"a1"},{"role":"user","content":"u2"}]}`,
	`{"messages":[{"role":"system","content":"sys"},{"role":"developer","content":[{"type":"text","text":"dev"}]},{"role":"user","content":"u1"},{"role":"assistant","content":"a"},{"role":"user","content":[{"type":"text","text":"u2"},{"type":"image_url","image_url":{"url":"x"}}]},{"role":"user","content":"u3"}]}`,
	`{"messages":[{"role":"system","content":""},{"role":"system","content":[]},{"role":"system","content":["str"]},{"role":"system","content":null},{"role":"system"},{"role":"user","content":""},{"role":"user","content":"last"}]}`,
	`{"messages":["str",7,null,[1],{"role":"user","content":"u1"},{"role":7,"content":"x"},{"content":"norole"},{"role":"user","content":"u2"},{"role":"user","content":"u3"}]}`,
	// client-supplied markers, in every place the check looks ...
	`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"messages":[{"role":"tool","tool_call_id":"t","content":"r","cache_control":{"type":"ephemeral","ttl":"1h"}},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	// ... and in places it does not look, or that are not markers.
	`{"messages":[{"role":"user","content":"a","cache_control":{"type":"ephemeral"}},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"messages":[{"role":"user","content":[{"type":"tool_result","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"other"}},{"cache_control":null},{"cache_control":"ephemeral"},{"cache_control":{}}]},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"tools":[{"name":"t","cache_control":{"type":"ephemeral"}}],"metadata":{"cache_control":{"type":"ephemeral"}},"system":"s","messages":[{"role":"user","content":"the text cache_control"},{"role":"user","content":"c"}]}`,
	`{"system":{"cache_control":{"type":"ephemeral"}},"messages":[{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"messages":[{"role":"tool","content":"r","cache_control":{"type":"other"}},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	// duplicates: a decode keeps the last of each name
	`{"system":"a","system":null,"messages":[{"role":"user","content":"b"}]}`, `{"messages":[{"role":"user","content":"x"}],"messages":[{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
	`{"messages":[{"role":"assistant","role":"user","content":"b","content":"b2"},{"role":"user","content":[{"cache_control":{"type":"ephemeral"},"cache_control":null}]},{"role":"user","content":"c"}]}`,
	"{\n \"system\" : [ { \"type\" : \"text\" , \"text\" : \"<s> & s\" } ] ,\n \"messages\" : [ { \"role\" : \"user\" , \"content\" : \"u1\" } , { \"role\" : \"user\" , \"content\" : \"u2\" } ] ,\n \"max_tokens\" : 1024 , \"seed\" : 12345678901234567890 }",
	"{\"system\":\"s\",\"messages\":[{\"role\":\"user\",\"content\":\"\xff\xfe\"},{\"role\":\"user\",\"content\":\"c\"}]}",
}

var passthroughBodies = []string{
	``, `null`, `[]`, `{`, `{"model":"m"`, `{} x`, `{}`,
	`{"model":"m","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`,
	`{"model":"m","max_tokens":1024,"temperature":0.3,"top_p":0.9,"top_k":40,"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":2000},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":20000},"temperature":0.2,"messages":[]}`,
	`{"model":"m","max_tokens":64000,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"adaptive"},"output_config":{"format":{"type":"json_schema"}},"anthropic_beta":["a","b"],"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"disabled"},"messages":[]}`, `{"model":"m","max_tokens":1024,"thinking":{"type":"between_tools"},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"between_tools"},"output_config":{"effort":"max"},"messages":[]}`,
	`{"model":"m","thinking":{"type":"enabled","budget_tokens":1024},"messages":[]}`, `{"model":"m","max_tokens":"lots","thinking":{"type":"enabled","budget_tokens":1024},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":"enabled","messages":[]}`, `{"model":"m","max_tokens":1024,"thinking":null,"messages":[]}`, `{"model":"m","max_tokens":1024,"thinking":{"type":"weird"},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"enabled"},"output_config":"x","anthropic_beta":"single-beta","messages":[]}`,
	`{"model":"m","max_tokens":1024,"output_config":{"effort":"low"},"messages":[]}`, `{"model":"m","max_tokens":1024,"anthropic_beta":[1,"x"," y ",""],"thinking":{"type":"adaptive"},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"temperature":null,"top_p":null,"thinking":{"type":"enabled","budget_tokens":5000,"extra":true},"messages":[]}`,
	`{"model":"m","max_tokens":1024,"thinking":{"type":"disabled"},"thinking":{"type":"enabled","budget_tokens":3000},"max_tokens":2048,"messages":[]}`,
	"{\n \"model\" : \"m\" , \"max_tokens\" : 4096 , \"thinking\" : { \"type\" : \"enabled\" , \"budget_tokens\" : 2048 } ,\n \"messages\" : [ { \"role\" : \"user\" , \"content\" : \"<a> & b\" } ] , \"seed\" : 12345678901234567890 , \"stream\" : true }",
	"{\"model\":\"m\",\"max_tokens\":10,\"thinking\":{\"type\":\"adaptive\"},\"messages\":[{\"role\":\"user\",\"content\":\"\xff\xfe\"}]}",
}

func TestInjectCacheControlMatchesDecodePath(t *testing.T) {
	for _, body := range append(append([]string{}, cacheInjectionBodies...), cacheMarkerEquivalenceBodies...) {
		assertInjectCacheControlMatchesDecode(t, []byte(body))
	}
	for _, body := range realEscapes(
		`{"system":"s","messages":[{"role":"us%u0065r","content":"u1"},{"r%u006fle":"user","content":"u2"},{"role":"user","content":"u3"}]}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache%u005fcontrol":{"type":"ephemeral"}}]},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
		`{"messages":[{"role":"t%u006fol","content":"r","cache_control":{"type":"ephem%u0065ral"}},{"role":"user","content":"b"},{"role":"user","content":"c"}]}`,
		`{"syst%u0065m":"s","mess%u0061ges":[{"role":"user","content":"caf%u00e9 %u001b[0m"},{"role":"user","c%u006fntent":"c"},{"role":"user","content":"d"}]}`,
		`{"system":"s","messages":[{"role":"system","content":"only control %u001b[0m and latin %u00e9"},{"role":"user","content":"a"},{"role":"user","content":"b"}]}`,
	) {
		assertInjectCacheControlMatchesDecode(t, []byte(body))
	}
}

func TestNormalizeMessagesForPassthroughMatchesDecodePath(t *testing.T) {
	for _, body := range append(append([]string{}, passthroughBodies...), cacheInjectionBodies...) {
		assertNormalizePassthroughMatchesDecode(t, []byte(body))
	}
	for _, body := range realEscapes(
		`{"model":"m","max_tokens":1024,"th%u0069nking":{"type":"en%u0061bled","budget%u005ftokens":4000},"temper%u0061ture":0.5,"messages":[]}`,
		`{"model":"m","max%u005ftokens":1024,"thinking":{"type":"adaptive"},"output%u005fconfig":{"eff%u006frt":"low"},"anthropic%u005fbeta":["x"],"messages":[{"role":"user","content":"caf%u00e9"}]}`,
	) {
		assertNormalizePassthroughMatchesDecode(t, []byte(body))
	}
}

// What these two functions exist to avoid: a request they have nothing to do
// with comes back as the same slice.
func TestPassthroughHelpersLeaveUnaffectedBodyAlone(t *testing.T) {
	supplied := []byte(`{"model":"claude-3-5-sonnet","max_tokens":10,"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}`)
	got, outcome := injectCacheControl(supplied)
	if outcome != CacheControlClientSupplied || &got[0] != &supplied[0] {
		t.Fatalf("outcome %q, copied=%v", outcome, &got[0] != &supplied[0])
	}
	plain := []byte(`{"model":"claude-3-5-sonnet","max_tokens":10,"messages":[{"role":"user","content":"a"}]}`)
	out, err := NormalizeMessagesForPassthrough(plain, "claude-3-5-sonnet", true)
	if err != nil || &out[0] != &plain[0] {
		t.Fatalf("err %v, copied=%v: %s", err, &out[0] != &plain[0], out)
	}
	// And when a marker is injected, only the marked messages are rewritten.
	body := []byte(`{"messages":[ {"role":"user","content":"keep  spacing <1>"} , {"role":"user","content":"mark"} , {"role":"user","content":"keep  spacing <2>"} ]}`)
	got, outcome = injectCacheControl(body)
	want := `{"messages":[{"role":"user","content":"keep  spacing <1>"},{"content":[{"cache_control":{"type":"ephemeral"},"text":"mark","type":"text"}],"role":"user"},{"role":"user","content":"keep  spacing <2>"}]}`
	if outcome != CacheControlRouterInjected || string(got) != want {
		t.Fatalf("outcome %q\n got %s\nwant %s", outcome, got, want)
	}
}

func TestPassthroughHelpersMatchDecodePathOnCorpus(t *testing.T) {
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
		assertInjectCacheControlMatchesDecode(t, body)
		assertNormalizePassthroughMatchesDecode(t, body)
		// The same session without the client's own markers, so the router
		// injects, and with every field the passthrough rules act on.
		stripped := bytes.ReplaceAll(body, []byte(`"cache_control"`), []byte(`"cache_note"`))
		assertInjectCacheControlMatchesDecode(t, stripped)
		var m map[string]interface{}
		if json.Unmarshal(stripped, &m) != nil {
			continue
		}
		m["thinking"] = map[string]interface{}{"type": "enabled", "budget_tokens": 8000}
		m["temperature"], m["top_p"], m["max_tokens"] = 0.4, 0.9, 16000
		m["output_config"], m["anthropic_beta"] = map[string]interface{}{"format": map[string]interface{}{"type": "text"}}, []interface{}{"x-beta"}
		full, _ := json.Marshal(m)
		assertInjectCacheControlMatchesDecode(t, full)
		assertNormalizePassthroughMatchesDecode(t, full)
	}
}

func FuzzPassthroughHelpers(f *testing.F) {
	for _, body := range append(append([]string{}, cacheInjectionBodies...), passthroughBodies...) {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var asRaw map[string]json.RawMessage
		var full map[string]interface{}
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &full) != nil {
			t.Skip() // a number outside float64: the decode paths refuse such a body outright
		}
		assertInjectCacheControlMatchesDecode(t, body)
		assertNormalizePassthroughMatchesDecode(t, body)
	})
}
