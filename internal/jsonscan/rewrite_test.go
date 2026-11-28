package jsonscan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// mapRoundTrip is the operation Rewrite replaces: decode the whole document
// into a map, apply the edits, encode it again.
func mapRoundTrip(t testing.TB, doc []byte, edits []Edit) map[string]interface{} {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(doc, &m); err != nil {
		t.Fatalf("reference decode of %q: %v", doc, err)
	}
	for _, e := range edits {
		if e.Raw == nil {
			delete(m, e.Name)
		} else {
			m[e.Name] = json.RawMessage(e.Raw)
		}
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("reference encode: %v", err)
	}
	return decodeAny(t, encoded)
}

func decodeAny(t testing.TB, doc []byte) map[string]interface{} {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(string(doc)))
	dec.UseNumber() // number literals must survive untouched, not merely round to the same float
	var m map[string]interface{}
	if err := dec.Decode(&m); err != nil {
		t.Fatalf("decode of %q: %v", doc, err)
	}
	if dec.More() {
		t.Fatalf("trailing data after %q", doc)
	}
	return m
}

func TestRewriteMatchesMapRoundTrip(t *testing.T) {
	docs := []string{
		`{}`,
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi <b>&</b>"}]}`,
		"{\n  \"model\": \"m\",\n  \"stream\": true,\n  \"messages\": [ { \"role\": \"user\" } ]\n}",
		`{"a":1,"a":2,"b":3,"a":4}`,
		`{"stream_options":{"include_usage":false},"seed":12345678901234567890,"t":1.0,"big":1e400}`,
		`{"key":"v","к":"привет","q\"uote":1}`,
		`{"service_tier":"priority","extra_body":{"service_tier":"flex"},"session_id":"s"}`,
	}
	editSets := [][]Edit{
		nil,
		{{Name: "stream_options", Raw: []byte(`{"include_usage":true}`)}},
		{{Name: "service_tier"}, {Name: "session_id"}, {Name: "absent"}},
		{{Name: "a", Raw: []byte(`"replaced"`)}},
		{{Name: "a"}},
		{{Name: "new", Raw: []byte(`[1,2]`)}, {Name: "model", Raw: []byte(`"other"`)}},
		{{Name: "x", Raw: []byte(`1`)}, {Name: "x", Raw: []byte(`2`)}},
		{{Name: "x", Raw: []byte(`1`)}, {Name: "x"}},
		{{Name: "model"}, {Name: "model", Raw: []byte(`"back"`)}},
		{{Name: "model", Raw: []byte(`"gone"`)}, {Name: "model"}},
		{{Name: "q\"uote", Raw: []byte(`null`)}, {Name: "new \"quoted\" \\ key", Raw: []byte(`true`)}},
	}
	for _, doc := range docs {
		for _, edits := range editSets {
			top, ok := ScanTop([]byte(doc))
			if !ok {
				t.Fatalf("ScanTop refused %q", doc)
			}
			got := decodeAny(t, top.Rewrite(edits...))
			want := mapRoundTrip(t, []byte(doc), edits)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("Rewrite(%q, %v)\n got %v\nwant %v", doc, edits, got, want)
			}
		}
	}
}

func TestRewriteCopiesUntouchedMembersVerbatim(t *testing.T) {
	doc := []byte(`{ "model" : "m" , "messages" : [ { "content" : "a b <tag>" , "n" : 1.50 } ] , "stream" : true }`)
	top, _ := ScanTop(doc)
	got := string(top.Rewrite(Edit{Name: "stream"}, Edit{Name: "stream_options", Raw: []byte(`{"include_usage":true}`)}))
	want := `{"model":"m","messages":[ { "content" : "a b <tag>" , "n" : 1.50 } ],"stream_options":{"include_usage":true}}`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	if string(doc) != `{ "model" : "m" , "messages" : [ { "content" : "a b <tag>" , "n" : 1.50 } ] , "stream" : true }` {
		t.Fatal("Rewrite must not modify the scanned body")
	}
}

func TestRewriteAllocatesOnce(t *testing.T) {
	doc := []byte(`{"model":"m","messages":[` + strings.Repeat(`{"role":"user","content":"0123456789"},`, 5000) + `{}],"stream":true}`)
	top, _ := ScanTop(doc)
	edit := Edit{Name: "stream_options", Raw: []byte(`{"include_usage":true}`)}
	// One allocation for the document; the rest are the few dozen bytes of its layout.
	if allocs := testing.AllocsPerRun(20, func() { _ = top.Rewrite(edit) }); allocs > 4 {
		t.Fatalf("%v allocations to rewrite one small member", allocs)
	}
}

func TestHasDuplicatesAndRange(t *testing.T) {
	if scanTopMust(t, `{"a":1,"b":2}`).HasDuplicates() {
		t.Fatal("no duplicates")
	}
	top := scanTopMust(t, `{"a":1,"b":2,"a":3}`)
	if !top.HasDuplicates() {
		t.Fatal("duplicate not reported")
	}
	var seen []string
	top.Range(func(name string, raw []byte) { seen = append(seen, name+"="+string(raw)) })
	if got := strings.Join(seen, ","); got != "a=1,b=2,a=3" {
		t.Fatalf("Range order: %s", got)
	}
	var nilTop *Top
	nilTop.Range(func(string, []byte) { t.Fatal("nil Top has no members") })
	if nilTop.HasDuplicates() || nilTop.Rewrite() != nil {
		t.Fatal("nil Top")
	}
}

func FuzzRewrite(f *testing.F) {
	f.Add([]byte(`{"a":1,"b":[1,2],"a":"x"}`), "a", []byte(`true`), "c", false)
	f.Add([]byte(`{"model":"m","stream":true}`), "stream_options", []byte(`{"include_usage":true}`), "model", true)
	f.Fuzz(func(t *testing.T, doc []byte, name string, raw []byte, name2 string, deleteSecond bool) {
		top, ok := ScanTop(doc)
		// Edit names come from the router's own code and are plain ASCII; a name
		// that is not valid UTF-8 has no single JSON spelling to compare.
		if !ok || !json.Valid(raw) || !utf8.ValidString(name) || !utf8.ValidString(name2) {
			t.Skip()
		}
		var probe map[string]interface{}
		if json.Unmarshal(doc, &probe) != nil || json.Unmarshal(raw, new(interface{})) != nil {
			t.Skip() // number outside float64: nothing to compare against
		}
		edits := []Edit{{Name: name, Raw: raw}}
		if deleteSecond {
			edits = append(edits, Edit{Name: name2})
		}
		got := decodeAny(t, top.Rewrite(edits...))
		want := mapRoundTrip(t, doc, edits)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("Rewrite(%q, %q=%q, del %q=%v)\n got %v\nwant %v", doc, name, raw, name2, deleteSecond, got, want)
		}
	})
}

func TestScanArrayMatchesEncodingJSON(t *testing.T) {
	valid := []string{`[]`, ` [ ] `, `[1]`, `[1,"a",null,true,{"a":[1,2]},[3,[4]]]`, "[\n  {\"a\": 1} ,\n  \"x,y\" , [ ] \n]", `["]","[",",","\"]"]`}
	for _, doc := range valid {
		var want []json.RawMessage
		if err := json.Unmarshal([]byte(doc), &want); err != nil {
			t.Fatal(err)
		}
		got, ok := ScanArray([]byte(doc))
		if !ok || len(got) != len(want) {
			t.Fatalf("%q: got %d elements (%v), want %d", doc, len(got), ok, len(want))
		}
		for i := range want {
			if string(got[i]) != string(want[i]) {
				t.Fatalf("%q[%d] = %q, want %q", doc, i, got[i], want[i])
			}
		}
		var back, orig interface{}
		if err := json.Unmarshal(JoinArray(got), &back); err != nil {
			t.Fatal(err)
		}
		_ = json.Unmarshal([]byte(doc), &orig)
		if !reflect.DeepEqual(back, orig) {
			t.Fatalf("JoinArray(ScanArray(%q)) = %v", doc, back)
		}
	}
	for _, doc := range []string{``, `{}`, `null`, `"a"`, `[`, `[1,]`, `[1 2]`, `[1] x`, `[1]]`, `[,1]`} {
		if got, ok := ScanArray([]byte(doc)); ok {
			t.Fatalf("ScanArray accepted %q: %q", doc, got)
		}
	}
}

func FuzzScanArray(f *testing.F) {
	f.Add([]byte(`[1,"a",{"b":[2]}]`))
	f.Add([]byte(`[ ]`))
	f.Fuzz(func(t *testing.T, doc []byte) {
		var want []json.RawMessage
		err := json.Unmarshal(doc, &want)
		got, ok := ScanArray(doc)
		if ok != (err == nil && want != nil) {
			t.Fatalf("ScanArray ok=%v, encoding/json err=%v for %q", ok, err, doc)
		}
		if !ok {
			return
		}
		if len(got) != len(want) {
			t.Fatalf("%q: %d elements, want %d", doc, len(got), len(want))
		}
		for i := range want {
			if string(got[i]) != string(want[i]) {
				t.Fatalf("%q[%d] = %q, want %q", doc, i, got[i], want[i])
			}
		}
	})
}

// RewriteTop must describe its own output exactly as a fresh scan would.
func TestRewriteTopLayoutMatchesFreshScan(t *testing.T) {
	docs := []string{`{}`, `{"a":1,"b":[1,2],"a":"x","c":{"k":null}}`, "{ \"model\" : \"m\" , \"messages\" : [ { } ] }"}
	editSets := [][]Edit{nil, {{Name: "a"}}, {{Name: "b", Raw: []byte(`"new"`)}, {Name: "zz", Raw: []byte(`[true]`)}}, {{Name: "model"}, {Name: "messages"}}}
	for _, doc := range docs {
		for _, edits := range editSets {
			got := scanTopMust(t, doc).RewriteTop(edits...)
			fresh, ok := ScanTop(got.Bytes())
			if !ok {
				t.Fatalf("RewriteTop produced an invalid document: %s", got.Bytes())
			}
			var a, b []string
			got.Range(func(name string, raw []byte) { a = append(a, name+"="+string(raw)) })
			fresh.Range(func(name string, raw []byte) { b = append(b, name+"="+string(raw)) })
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("layout of %s\n got %v\nwant %v", got.Bytes(), a, b)
			}
			if !got.Describes(got.Bytes()) || got.Describes([]byte(string(got.Bytes()))) || fresh.Describes(nil) {
				t.Fatal("Describes is about the slice, not its contents")
			}
		}
	}
}
