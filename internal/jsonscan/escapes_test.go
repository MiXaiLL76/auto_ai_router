package jsonscan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// Test inputs with escaped spellings are written with a %u marker and turned
// into real backslash-u escapes here. Writing the escape directly in a source
// literal is easy to get wrong without noticing (an editor or tool that
// decodes it leaves a test that still passes but no longer tests an escape),
// so the marker form is used throughout and checked below.
const backslash = "\\"

func realEscapes(cases ...string) []string {
	out := make([]string, len(cases))
	for i, c := range cases {
		out[i] = strings.ReplaceAll(c, "%u", backslash+"u")
	}
	return out
}

var escapedValidDocs = realEscapes(
	`{"%u0069nput":[],"mess%u0061ges":1}`,
	`{"%u0073tore":true,"k%u0065y":"v%u0061lue","%u043a":"%u043f"}`,
	`{"a":"%u001b[0m","b":"caf%u00e9","c":"%ud83d%ude00","d":"%u0022"}`,
	`{"%u0061":1,"a":2}`,
)

func init() {
	validDocs = append(validDocs, escapedValidDocs...)
	invalidDocs = append(invalidDocs, realEscapes(`{"a":"%u00"}`, `{"a":"%u12G4"}`, `{"%u00":1}`)...)
}

func TestEscapedTestInputsAreReallyEscaped(t *testing.T) {
	if !strings.Contains(escapedValidDocs[0], backslash+"u0069nput") || strings.Contains(escapedValidDocs[0], "input") {
		t.Fatalf("test inputs lost their escapes: %s", escapedValidDocs[0])
	}
	top := scanTopMust(t, escapedValidDocs[0])
	if !top.Has("input") || !top.Has("messages") {
		t.Fatal("an escaped key must be found under its decoded name")
	}
	if raw, _ := scanTopMust(t, escapedValidDocs[3]).Raw("a"); string(raw) != "2" {
		t.Fatalf("an escaped and a plain spelling are the same key, last wins; got %s", raw)
	}
}

func TestHasEscapedASCII(t *testing.T) {
	yes := realEscapes(`"%u0069nput"`, `x%u005fy`, `%u001b[0m then %u0041`, `%u007F`, `%u0020`, `%u007a%u0030`)
	no := realEscapes(``, `plain`, `%u001b[31mred%u001b[0m`, `%u0000%u0009%u000a%u001f`, `caf%u00e9 %u00a0`, `%u043f%u0440`, `%u00`, `%u008f`, `%u00ff`)
	for _, s := range yes {
		if !HasEscapedASCII([]byte(s)) {
			t.Errorf("missed an ASCII escape in %s", s)
		}
	}
	for _, s := range no {
		if HasEscapedASCII([]byte(s)) {
			t.Errorf("false positive on %s", s)
		}
	}
}

func TestProjectRewriteAndObjectWithEscapedKeys(t *testing.T) {
	doc := escapedValidDocs[1]
	var full map[string]interface{}
	if err := json.Unmarshal([]byte(doc), &full); err != nil {
		t.Fatal(err)
	}
	if _, ok := full["store"]; !ok {
		t.Fatal("test bug: the escaped key does not decode to store")
	}
	top := scanTopMust(t, doc)
	got, ok := top.Project("store", "key")
	if !ok || !reflect.DeepEqual(got, map[string]interface{}{"store": full["store"], "key": full["key"]}) {
		t.Fatalf("Project = %v (%v)", got, ok)
	}
	edits := []Edit{{Name: "store"}, {Name: "key", Raw: []byte(`"new"`)}}
	if rewritten := top.Rewrite(edits...); !reflect.DeepEqual(decodeAny(t, rewritten), mapRoundTrip(t, []byte(doc), edits)) {
		t.Fatalf("Rewrite = %s", rewritten)
	}
	// An untouched value keeps its escapes exactly as written.
	keep := string(scanTopMust(t, escapedValidDocs[2]).Rewrite(Edit{Name: "d"}))
	if !strings.Contains(keep, backslash+"u001b[0m") || !strings.Contains(keep, "caf"+backslash+"u00e9") {
		t.Fatalf("escapes in untouched members must survive: %s", keep)
	}
	for _, d := range escapedValidDocs {
		assertObjectMatchesStdlib(t, []byte(d), []mutation{{"input", ""}, {"a", "5"}, {"store", "false"}})
		assertObjectMatchesStdlib(t, []byte(d), nil)
	}
}

func TestScanArrayWithEscapes(t *testing.T) {
	doc := realEscapes(`["%u0041","]%u005d",{"k%u0065y":"%u005b"}]`)[0]
	var want []json.RawMessage
	if err := json.Unmarshal([]byte(doc), &want); err != nil {
		t.Fatal(err)
	}
	got, ok := ScanArray([]byte(doc))
	if !ok || len(got) != len(want) {
		t.Fatalf("ScanArray = %q (%v)", got, ok)
	}
	for i := range want {
		if string(got[i]) != string(want[i]) {
			t.Fatalf("[%d] = %s, want %s", i, got[i], want[i])
		}
	}
}
