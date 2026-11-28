package jsonscan

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// corpus is shared by the tests below: documents that must index, and
// documents that must be refused.
var validDocs = []string{
	`{}`,
	` { } `,
	`{"model":"m","stream":true}`,
	"{\n  \"model\" : \"m\" ,\n\t\"input\" :\r\n [1,2,{\"a\":null}] }\n",
	`{"a":null,"b":false,"c":0,"d":-1.5e3,"e":"","f":[],"g":{}}`,
	`{"a":{"a":{"a":[{"a":"}"}]}},"b":"\"}{\\"}`,
	`{"model":"first","model":"second"}`,
	`{"input":[],"messages":1}`,
	`{"text":"` + "\xff\xfe" + `","k":1}`,
	`{"big":1e999,"k":"v"}`,
	`{"Tools":[1],"tools":[2],"TOOLS":[3]}`,
}

var invalidDocs = []string{
	``, ` `, `null`, `[]`, `"s"`, `1`, `true`,
	`{`, `}`, `{"a"}`, `{"a":}`, `{"a":1,}`, `{a:1}`, `{"a":1}{"b":2}`, `{"a":1} x`,
	`{"a":01}`, `{"a":"unterminated}`, `{"a":[1,2}`, `{'a':1}`, `{"a":tru}`,
}

func TestScanTopAgreesWithEncodingJSON(t *testing.T) {
	for _, doc := range validDocs {
		top, ok := ScanTop([]byte(doc))
		if !ok {
			t.Fatalf("ScanTop refused valid document %q", doc)
		}
		var want map[string]json.RawMessage
		if err := json.Unmarshal([]byte(doc), &want); err != nil {
			t.Fatalf("test corpus: encoding/json refused %q: %v", doc, err)
		}
		seen := map[string]bool{}
		for _, e := range top.entries {
			seen[e.name] = true
		}
		if len(seen) != len(want) {
			t.Fatalf("%q: keys %v, want %d keys", doc, seen, len(want))
		}
		for name, wantRaw := range want {
			got, ok := top.Raw(name)
			if !ok {
				t.Fatalf("%q: key %q missing", doc, name)
			}
			// encoding/json returns the raw value verbatim as well.
			if string(got) != string(wantRaw) {
				t.Fatalf("%q: key %q = %q, want %q", doc, name, got, wantRaw)
			}
			if !top.Has(name) {
				t.Fatalf("%q: Has(%q) = false", doc, name)
			}
		}
	}
}

func TestScanTopRefusesWhatEncodingJSONRefuses(t *testing.T) {
	for _, doc := range invalidDocs {
		if top, ok := ScanTop([]byte(doc)); ok || top != nil {
			t.Fatalf("ScanTop accepted %q", doc)
		}
		var m map[string]interface{}
		err := json.Unmarshal([]byte(doc), &m)
		if err == nil && m != nil {
			t.Fatalf("test corpus: encoding/json decoded %q into a map", doc)
		}
	}
}

func TestScanTopDoesNotModifyOrCopyBody(t *testing.T) {
	body := []byte(`{"model":"m","input":[{"role":"user","content":"hi"}]}`)
	before := string(body)
	top, ok := ScanTop(body)
	if !ok {
		t.Fatal("ScanTop failed")
	}
	if string(body) != before {
		t.Fatalf("body modified: %q", body)
	}
	raw, _ := top.Raw("input")
	if len(raw) == 0 || &raw[0] != &body[strings.Index(before, "[")] {
		t.Fatal("Raw must return a sub-slice of the scanned body")
	}
}

func TestNilTop(t *testing.T) {
	var top *Top
	if top.Has("a") {
		t.Fatal("nil Top has no keys")
	}
	if _, ok := top.Raw("a"); ok {
		t.Fatal("nil Top has no values")
	}
	if got := top.RawFoldAll("a"); got != nil {
		t.Fatal("nil Top has no values")
	}
	if fields, ok := top.Project("a", "b"); !ok || len(fields) != 0 {
		t.Fatalf("nil Top projects to an empty map, got %v %v", fields, ok)
	}
}

// Decoding RawFoldAll's values in order into one destination must give what
// encoding/json gives for a struct field of that name, including its merging
// of repeated keys.
func TestRawFoldAllReproducesStructDecoding(t *testing.T) {
	docs := []string{
		`{"Tools":[1],"tools":[2],"TOOLS":[3],"other":0}`,
		`{"opts":{"a":1},"OPTS":{"b":2},"Opts":{"a":3}}`,
		`{"tools":[{"a":1},{"b":2}],"Tools":[{"c":3}]}`,
		`{"tools":null,"opts":null}`,
		`{"other":1}`,
	}
	type fields struct {
		Tools []interface{}          `json:"tools"`
		Opts  map[string]interface{} `json:"opts"`
	}
	for _, doc := range docs {
		var want fields
		if err := json.Unmarshal([]byte(doc), &want); err != nil {
			t.Fatal(err)
		}
		top, ok := ScanTop([]byte(doc))
		if !ok {
			t.Fatalf("ScanTop refused %q", doc)
		}
		var got fields
		for _, raw := range top.RawFoldAll("tools") {
			if err := json.Unmarshal(raw, &got.Tools); err != nil {
				t.Fatal(err)
			}
		}
		for _, raw := range top.RawFoldAll("opts") {
			if err := json.Unmarshal(raw, &got.Opts); err != nil {
				t.Fatal(err)
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%q: got %#v, want %#v", doc, got, want)
		}
	}
	if raw, _ := scanTopMust(t, `{"Tools":[1],"tools":[2]}`).Raw("tools"); string(raw) != `[2]` {
		t.Fatalf("Raw is exact-match only, got %s", raw)
	}
}

func scanTopMust(t *testing.T, doc string) *Top {
	t.Helper()
	top, ok := ScanTop([]byte(doc))
	if !ok {
		t.Fatalf("ScanTop refused %q", doc)
	}
	return top
}

// Project must hand back exactly what a full decode holds under the same keys.
func TestProjectMatchesFullDecode(t *testing.T) {
	docs := []string{
		`{"store":true,"metadata":{"a":"b","n":1},"ttl":60,"input":[1,2,3],"text":{"format":{"type":"text"}}}`,
		`{"store":null,"metadata":null}`,
		`{"reasoning":{"effort":"high"},"reasoning":{"effort":"none"},"extra_body":{"thinking":{"type":"enabled"}}}`,
		`{"metadata":{"a":1},"metadata":{"b":2}}`,
		`{"store":true}`,
		`{}`,
	}
	keys := []string{"store", "metadata", "ttl", "text", "reasoning", "extra_body", "absent"}
	for _, doc := range docs {
		var full map[string]interface{}
		if err := json.Unmarshal([]byte(doc), &full); err != nil {
			t.Fatal(err)
		}
		want := map[string]interface{}{}
		for _, k := range keys {
			if v, ok := full[k]; ok {
				want[k] = v
			}
		}
		top, ok := ScanTop([]byte(doc))
		if !ok {
			t.Fatalf("ScanTop refused %q", doc)
		}
		got, ok := top.Project(keys...)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Fatalf("%q: Project = %#v (%v), want %#v", doc, got, ok, want)
		}
	}
}

func TestProjectReportsUndecodableValue(t *testing.T) {
	top, ok := ScanTop([]byte(`{"ttl":1e999,"store":true}`))
	if !ok {
		t.Fatal("ScanTop validates syntax only and must accept an out-of-range number")
	}
	if _, ok := top.Project("store"); !ok {
		t.Fatal("a key that decodes must project")
	}
	if _, ok := top.Project("store", "ttl"); ok {
		t.Fatal("an out-of-range number must fail the projection, as it fails json.Unmarshal")
	}
}

func FuzzScanTop(f *testing.F) {
	for _, doc := range append(append([]string{}, validDocs...), invalidDocs...) {
		f.Add([]byte(doc))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		top, ok := ScanTop(body)
		var want map[string]json.RawMessage
		err := json.Unmarshal(body, &want)
		isObject := err == nil && want != nil
		if ok != isObject {
			t.Fatalf("ScanTop ok=%v, encoding/json object=%v (err %v) for %q", ok, isObject, err, body)
		}
		for name, wantRaw := range want {
			got, found := top.Raw(name)
			if !found || string(got) != string(wantRaw) {
				t.Fatalf("key %q: got %q (%v), want %q in %q", name, got, found, wantRaw, body)
			}
		}
	})
}

// Every key or word the pre-checks look for is made of printable ASCII, so
// any escaped spelling of one must trip HasEscapedASCII.
func TestHasEscapedASCIICoversEveryEscapedSpelling(t *testing.T) {
	for _, word := range []string{"cache_control", "input", "reasoning", "thinking", "tools", "tool_choice", "compaction", "developer", "stream_options"} {
		for i := range word {
			for _, format := range []string{`%s\u%04x%s`, `%s\u%04X%s`} {
				escaped := fmt.Sprintf(format, word[:i], word[i], word[i+1:])
				var decoded string
				if err := json.Unmarshal([]byte(`"`+escaped+`"`), &decoded); err != nil || decoded != word {
					t.Fatalf("test bug: %s decodes to %q (%v)", escaped, decoded, err)
				}
				if !HasEscapedASCII([]byte(escaped)) {
					t.Errorf("escaped spelling %s of %q not detected", escaped, word)
				}
			}
		}
	}
}
