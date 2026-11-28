package jsonscan

import (
	"encoding/json"
	"reflect"
	"testing"
	"unicode/utf8"
)

// mutation is one step of a caller's edit of the top-level members.
type mutation struct {
	name string
	raw  string // "" deletes
}

func applyMutations(fields map[string]json.RawMessage, steps []mutation) {
	for _, s := range steps {
		if s.raw == "" {
			delete(fields, s.name)
		} else {
			fields[s.name] = json.RawMessage(s.raw)
		}
	}
}

// Object must be interchangeable with the Unmarshal / edit / Marshal pattern
// it replaces: same members on the way in, same JSON value on the way out.
func assertObjectMatchesStdlib(t testing.TB, body []byte, steps []mutation) {
	t.Helper()
	var want map[string]json.RawMessage
	err := json.Unmarshal(body, &want)
	obj, ok := ParseObject(body)
	if ok != (err == nil && want != nil) {
		t.Fatalf("ParseObject ok=%v, encoding/json err=%v nil=%v for %q", ok, err, want == nil, body)
	}
	if !ok {
		return
	}
	if len(obj.Fields) != len(want) {
		t.Fatalf("%q: %d fields, encoding/json has %d", body, len(obj.Fields), len(want))
	}
	for name, raw := range want {
		if string(obj.Fields[name]) != string(raw) {
			t.Fatalf("%q: field %q = %q, encoding/json has %q", body, name, obj.Fields[name], raw)
		}
	}
	applyMutations(want, steps)
	applyMutations(obj.Fields, steps)
	wantBytes, err := json.Marshal(want)
	if err != nil {
		t.Skip("a mutation is not valid JSON")
	}
	got := obj.Encode()
	gotValue, wantValue := decodeAny(t, got), decodeAny(t, wantBytes)
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("Encode after %v on %q\n got %s\nwant %s", steps, body, got, wantBytes)
	}
}

func TestObjectMatchesStdlibRoundTrip(t *testing.T) {
	bodies := []string{
		``, `null`, `[]`, `{`, `{} x`, `{}`,
		`{"model":"m","messages":[{"role":"user","content":"<hi> & bye"}],"stream":true}`,
		"{ \"model\" : \"m\" ,\n \"messages\" : [ ] , \"n\" : 1.50 , \"seed\" : 12345678901234567890 }",
		`{"a":1,"b":2,"a":3}`,
		`{"key":1,"ключ":"з"}`,
		"{\"text\":\"\xff\xfe\",\"k\":1}",
	}
	scripts := [][]mutation{
		nil,
		{{"stream", ""}},
		{{"stream_options", `{"include_usage":true}`}},
		{{"model", `"other"`}, {"absent", ""}, {"zz", `[1]`}, {"aa", `null`}},
		{{"a", ""}, {"a", `"back"`}},
		{{"a", `7`}, {"a", ""}},
		{{"model", `"m"`}},                        // replaced with an equal value
		{{"max_tokens", `1`}, {"max_tokens", ""}}, // added then removed
		{{"key", `2`}, {"ключ", `"x"`}},
	}
	for _, body := range bodies {
		for _, steps := range scripts {
			assertObjectMatchesStdlib(t, []byte(body), steps)
		}
	}
}

func TestObjectEncodeReturnsBodyWhenUnchanged(t *testing.T) {
	body := []byte(`{"model":"m","messages":[1,2,3]}`)
	obj, _ := ParseObject(body)
	if obj.Changed() {
		t.Fatal("nothing changed yet")
	}
	// Reading, and writing back a value equal to the one already there, is not a change.
	obj.Fields["model"] = json.RawMessage(`"m"`)
	out := obj.Encode()
	if &out[0] != &body[0] || len(out) != len(body) {
		t.Fatal("an unchanged object must encode to the original body, not a copy")
	}
	obj.Fields["model"] = json.RawMessage(`"n"`)
	if !obj.Changed() || string(obj.Encode()) != `{"model":"n","messages":[1,2,3]}` {
		t.Fatalf("got %s", obj.Encode())
	}
	if string(body) != `{"model":"m","messages":[1,2,3]}` {
		t.Fatal("Encode must not modify the original body")
	}
	// Duplicates are resolved even without edits, as a decode would.
	dup, _ := ParseObject([]byte(`{"a":1,"a":2}`))
	if string(dup.Encode()) != `{"a":2}` {
		t.Fatalf("got %s", dup.Encode())
	}
}

// encoding/json repairs a name that is not valid UTF-8 whenever it encodes
// the object again, edits or not; Encode does the same, and says so.
func TestObjectEncodeRepairsInvalidNames(t *testing.T) {
	body := []byte("{\"\xf3\":[1],\"ok\":\"v\\u0041 \xff\"}")
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	want, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	obj, ok := ParseObject(body)
	if !ok || !obj.Top().HasInvalidNames() || obj.Changed() {
		t.Fatalf("ok=%v invalidNames=%v changed=%v", ok, obj.Top().HasInvalidNames(), obj.Changed())
	}
	got := obj.Encode()
	// Same members as the round trip gives (in the body's own order, not
	// sorted): the name repaired, the value's own bytes untouched -- a raw
	// value keeps even its invalid UTF-8 through encoding/json.
	if !reflect.DeepEqual(decodeAny(t, got), decodeAny(t, want)) {
		t.Fatalf("Encode\n got %q\nwant %q", got, want)
	}
	gotTop, _ := ScanTop(got)
	wantTop, _ := ScanTop(want)
	if gotTop.HasInvalidNames() || wantTop.HasInvalidNames() {
		t.Fatalf("a name is still invalid UTF-8\n got %q\nwant %q", got, want)
	}
	gotValue, _ := gotTop.Raw("ok")
	wantValue, _ := wantTop.Raw("ok")
	if string(gotValue) != string(wantValue) || utf8.Valid(gotValue) {
		t.Fatalf("the value must keep its bytes: got %q, want %q", gotValue, wantValue)
	}
	if valid, _ := ParseObject([]byte(`{"ok":1}`)); valid.Top().HasInvalidNames() {
		t.Fatal("a valid name reported as invalid")
	}
}

// A caller appending to a field value must not write into the body.
func TestObjectFieldsCannotGrowIntoBody(t *testing.T) {
	body := []byte(`{"a":[1],"b":2}`)
	obj, _ := ParseObject(body)
	_ = append(obj.Fields["a"], "XXXX"...)
	if string(body) != `{"a":[1],"b":2}` {
		t.Fatalf("append to a field value overwrote the body: %s", body)
	}
}

func FuzzObject(f *testing.F) {
	f.Add([]byte(`{"a":1,"b":[1,2],"a":"x"}`), "a", `true`, "c", ``, "b", `{"k":null}`)
	f.Add([]byte(`{"model":"m","stream":true}`), "stream_options", `{"include_usage":true}`, "model", ``, "model", `"z"`)
	f.Fuzz(func(t *testing.T, body []byte, n1, r1, n2, r2, n3, r3 string) {
		var probe map[string]interface{}
		var asRaw map[string]json.RawMessage
		if json.Unmarshal(body, &asRaw) == nil && json.Unmarshal(body, &probe) != nil {
			t.Skip() // number outside float64: decodeAny keeps literals, nothing else to check
		}
		if !utf8.ValidString(n1) || !utf8.ValidString(n2) || !utf8.ValidString(n3) {
			t.Skip() // see FuzzRewrite
		}
		steps := []mutation{{n1, r1}, {n2, r2}, {n3, r3}}
		for _, s := range steps {
			if s.raw != "" && !json.Valid([]byte(s.raw)) {
				t.Skip()
			}
		}
		assertObjectMatchesStdlib(t, body, steps)
	})
}
