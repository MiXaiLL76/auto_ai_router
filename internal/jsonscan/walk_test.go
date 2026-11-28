package jsonscan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

// findKeyByDecoding is what FindKey replaces: decode the value, search the
// result for the key at any depth, not descending into matches. It returns
// the matches re-encoded, in the order a decode can still tell (sorted
// encoding), and false when the value does not decode.
func findKeyByDecoding(raw []byte, key string) ([]string, bool) {
	var v interface{}
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, false
	}
	var out []string
	var visit func(v interface{})
	visit = func(v interface{}) {
		switch t := v.(type) {
		case map[string]interface{}:
			for k, child := range t {
				if k == key {
					b, _ := json.Marshal(child)
					out = append(out, string(b))
					continue
				}
				visit(child)
			}
		case []interface{}:
			for _, child := range t {
				visit(child)
			}
		}
	}
	visit(v)
	return out, true
}

func normalizeFound(t testing.TB, values [][]byte) []string {
	t.Helper()
	out := make([]string, 0, len(values))
	for _, raw := range values {
		var v interface{}
		if err := json.Unmarshal(raw, &v); err != nil {
			t.Fatalf("FindKey returned an undecodable value %q: %v", raw, err)
		}
		b, _ := json.Marshal(v)
		out = append(out, string(b))
	}
	return out
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	for i := range out {
		for j := i + 1; j < len(out); j++ {
			if out[j] < out[i] {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out
}

// Whenever FindKey answers, the answer is the decode-and-search answer.
// Whenever it declines, that is allowed (the caller decodes) -- but it must
// not decline documents of the ordinary shape, checked separately below.
func assertFindKeyMatchesDecoding(t testing.TB, raw []byte, key string) (answered bool) {
	t.Helper()
	got, ok := FindKey(raw, key)
	want, decodes := findKeyByDecoding(raw, key)
	if !ok {
		return false
	}
	if !decodes {
		t.Fatalf("FindKey answered for a value that does not decode: %q", raw)
	}
	if !reflect.DeepEqual(sortedCopy(normalizeFound(t, got)), sortedCopy(want)) {
		t.Fatalf("FindKey(%q, %q)\n got %v\nwant %v", raw, key, normalizeFound(t, got), want)
	}
	return true
}

var findKeyCases = []string{
	`null`, `1`, `"cc"`, `[]`, `{}`,
	`{"cc":1}`, `{"a":{"cc":{"cc":2,"x":[{"cc":3}]}}}`, `[{"cc":1},[{"cc":[1,2]}],{"b":{"c":{"cc":null}}}]`,
	`{"a":"cc","b":["cc"],"ccc":1,"c":{"Cc":2}}`,
	"[ {\n \"a\" : 1 ,\n \"cc\" : { \"type\" : \"ephemeral\" } } ,\n { \"cc\" : \"x\" } ]",
	`[{"n":1e308,"cc":1},{"n":-0.0e-400,"cc":2},{"n":12345678901234567890,"cc":3}]`,
	"[{\"s\":\"\xff\xfe\",\"cc\":1}]",
}

var findKeyDeclined = []string{
	``, `{`, `[1,]`, `{"cc":1} x`,
	`{"cc":1,"cc":2}`, `{"a":{"x":{"cc":1}},"a":{"x":2}}`, `[{"b":1,"b":2,"cc":3}]`,
	`[{"n":1e999,"cc":1}]`, `[{"cc":{"n":1e999}}]`,
	"[{\"\xff\":1,\"cc\":2}]",
}

func TestFindKeyMatchesDecodeAndSearch(t *testing.T) {
	for _, doc := range findKeyCases {
		if !assertFindKeyMatchesDecoding(t, []byte(doc), "cc") {
			t.Errorf("FindKey declined an ordinary document: %q", doc)
		}
	}
	for _, doc := range findKeyDeclined {
		if got, ok := FindKey([]byte(doc), "cc"); ok {
			t.Errorf("FindKey must decline %q, answered %q", doc, got)
		}
	}
	// A name written with an escape is declined whether or not it spells the key.
	for _, doc := range realEscapes(`[{"c%u0063":1}]`, `[{"%u0061":1,"cc":2}]`, `[{"a":{"k%u0065y":1},"cc":2}]`) {
		if _, ok := FindKey([]byte(doc), "cc"); ok {
			t.Errorf("FindKey must decline a document with an escaped name: %s", doc)
		}
	}
	// Escapes in values are fine.
	for _, doc := range realEscapes(`[{"s":"%u0063c %u001b","cc":"%u0041"}]`) {
		if !assertFindKeyMatchesDecoding(t, []byte(doc), "cc") {
			t.Errorf("FindKey declined an escaped value: %s", doc)
		}
	}
}

func TestFindKeyObjectSizeLimit(t *testing.T) {
	build := func(n int) []byte {
		var b strings.Builder
		b.WriteString(`{`)
		for i := 0; i < n; i++ {
			b.WriteString(`"k` + strings.Repeat("x", i%7) + string(rune('a'+i%26)) + strings.Repeat("y", i/26) + `":0,`)
		}
		b.WriteString(`"cc":1}`)
		return []byte(b.String())
	}
	if !assertFindKeyMatchesDecoding(t, build(maxWalkObjectKeys-1), "cc") {
		t.Fatal("an object within the limit must be walked")
	}
	if _, ok := FindKey(build(maxWalkObjectKeys+5), "cc"); ok {
		t.Fatal("an object over the limit must be declined")
	}
}

func TestFindKeyDoesNotAllocateForNonMatches(t *testing.T) {
	doc := []byte(`[` + strings.Repeat(`{"role":"user","content":[{"type":"text","text":"0123456789 0123456789"}]},`, 2000) + `{"cc":1}]`)
	allocs := testing.AllocsPerRun(10, func() {
		if got, ok := FindKey(doc, "cc"); !ok || len(got) != 1 {
			t.Fatal("unexpected result")
		}
	})
	if allocs > 12 {
		t.Fatalf("%v allocations walking %d bytes with one match", allocs, len(doc))
	}
}

func FuzzFindKey(f *testing.F) {
	for _, doc := range append(append([]string{}, findKeyCases...), findKeyDeclined...) {
		f.Add([]byte(doc), "cc")
	}
	f.Add([]byte(`{"messages":[{"content":[{"cache_control":{"type":"ephemeral"}}]}]}`), "cache_control")
	f.Fuzz(func(t *testing.T, raw []byte, key string) {
		assertFindKeyMatchesDecoding(t, raw, key)
	})
}

func TestScanArrayMembersMatchesDecoding(t *testing.T) {
	docs := append([]string{
		`[]`, `[1,"type",null,[{"type":"x"}]]`,
		`[{"type":"message","content":"a"},{"content":"b"},{"type":"reasoning","type":"compaction"},{"x":{"type":"nested"}},{}]`,
		"[ { \"a\" : 1 , \"type\" : \"spaced\" } , { \"type\" : [ 1 , { \"type\" : 2 } ] } ]",
		`[{"type":null},{"type":7},{"Type":"case"},{"typ":"short"},{"types":"long"}]`,
	}, realEscapes(`[{"t%u0079pe":"escaped key"},{"type":"reas%u006fning"},{"%u0074ype":"a","type":"b"},{"type":"a","typ%u0065":"b"}]`)...)
	for _, doc := range docs {
		var want []json.RawMessage
		if err := json.Unmarshal([]byte(doc), &want); err != nil {
			t.Fatal(err)
		}
		elements, values, ok := ScanArrayMembers([]byte(doc), "type")
		if !ok || len(elements) != len(want) || len(values) != len(want) {
			t.Fatalf("%s: %d elements, %d values (%v), want %d", doc, len(elements), len(values), ok, len(want))
		}
		for i := range want {
			if string(elements[i]) != string(want[i]) {
				t.Fatalf("%s[%d] = %s, want %s", doc, i, elements[i], want[i])
			}
			var fields map[string]json.RawMessage
			_ = json.Unmarshal(want[i], &fields) // nil for a non-object
			wantValue, has := fields["type"]
			if has != (values[i] != nil) || string(values[i]) != string(wantValue) {
				t.Fatalf("%s[%d].type = %q, want %q (present %v)", doc, i, values[i], wantValue, has)
			}
		}
	}
	for _, doc := range []string{``, `{}`, `[`, `[{"type":}]`, `[{"type":1,}]`, `[{}] x`, `[{"a"}]`} {
		if _, _, ok := ScanArrayMembers([]byte(doc), "type"); ok {
			t.Fatalf("ScanArrayMembers accepted %q", doc)
		}
	}
}

func FuzzScanArrayMembers(f *testing.F) {
	f.Add([]byte(`[{"type":"a","x":[1,{"type":2}]},3,{"type":"b","type":"c"}]`), "type")
	f.Fuzz(func(t *testing.T, doc []byte, member string) {
		if !utf8.ValidString(member) {
			t.Skip() // callers pass key names from their own source; see ScanArrayMembers
		}
		var want []json.RawMessage
		err := json.Unmarshal(doc, &want)
		elements, values, ok := ScanArrayMembers(doc, member)
		if ok != (err == nil && want != nil) {
			t.Fatalf("ScanArrayMembers ok=%v, encoding/json err=%v for %q", ok, err, doc)
		}
		if !ok {
			return
		}
		if len(elements) != len(want) {
			t.Fatalf("%q: %d elements, want %d", doc, len(elements), len(want))
		}
		for i := range want {
			if string(elements[i]) != string(want[i]) {
				t.Fatalf("%q[%d] = %q, want %q", doc, i, elements[i], want[i])
			}
			var fields map[string]json.RawMessage
			_ = json.Unmarshal(want[i], &fields)
			wantValue, has := fields[member]
			if has != (values[i] != nil) || string(values[i]) != string(wantValue) {
				t.Fatalf("%q[%d].%q = %q, want %q (present %v)", doc, i, member, values[i], wantValue, has)
			}
		}
	})
}

func TestTopEscapedASCIIMatchesByteSearch(t *testing.T) {
	docs := append(append([]string{}, validDocs...), realEscapes(
		`{"a":"%u001b[0m","b":"caf%u00e9"}`, `{"a":[{"k":"x%u0041"}]}`, `{"a":{"k%u0065y":1}}`, `{"%u0020":1}`, `{"a":"%u007f%u0080"}`,
	)...)
	for _, doc := range docs {
		top, ok := ScanTop([]byte(doc))
		if !ok {
			continue
		}
		if got, want := top.EscapedASCII(), HasEscapedASCII([]byte(doc)); got != want {
			// The byte search also fires on an escaped backslash followed by
			// the text u00XX, which is not an escape; the scan knows better.
			if !want || got {
				t.Errorf("EscapedASCII(%s) = %v, byte search says %v", doc, got, want)
			}
		}
	}
}
