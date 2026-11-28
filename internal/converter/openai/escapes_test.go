package openai

import (
	"strings"
	"testing"
)

// Escaped spellings are written with a %u marker and turned into real
// backslash-u escapes here, so the inputs are certain to contain the escape
// sequence itself and not the character it stands for.
const escapeBackslash = "\\"

func realEscapes(cases ...string) []string {
	out := make([]string, len(cases))
	for i, c := range cases {
		out[i] = strings.ReplaceAll(c, "%u", escapeBackslash+"u")
	}
	return out
}

var escapedRequestRuleBodies = realEscapes(
	`{"model":"gpt-4o","messages":[{"role":"devel%u006fper","content":"y"},{"r%u006fle":"developer","content":"z"},{"role":"user","content":"caf%u00e9 %u001b[0m"}]}`,
	`{"model":"gpt-4o","messages":[],"t%u006fols":[{"type":"computer_use"}],"t%u006fol_choice":{"type":"computer_use"}}`,
	`{"model":"gpt-4o-search-preview","messages":[],"tools":[{"type":"web%u005fsearch","search_context_size":"low"}],"web_search%u005foptions":{"a":1}}`,
	`{"model":"gpt-4o","messages":[],"cache%u005fsalt":"s","stream%u005foptions":{"include_usage":true,"x":1},"plug%u0069ns":[],"prov%u0069der":{}}`,
	`{"model":"gpt-5","messages":[],"max%u005ftokens":5,"temper%u0061ture":1,"reasoning%u005feffort":"n%u006fne"}`,
	`{"model":"gpt-4o","messages":[],"t%u006fols":[],"tool_choice":"auto"}`,
	`{"model":"gpt-4o","messages":[{"role":"user","content":"terminal %u001b[0m only, latin %u00e9"}],"tools":[{"type":"function","function":{"name":"f"}}]}`,
)

// Names that are not valid UTF-8: encoding/json repairs them whenever it
// encodes the object again, so what happens to them depends on whether a
// rule re-encodes the body.
var invalidNameBodies = []string{
	"{\"\xf3\":[]}",
	// A repeated name whose discarded value is invalid UTF-8: whether it
	// survives depends on whether the body is written out again.
	"{\"\":\"\x80\",\"\":{}}",
	"{\"model\":\"gpt-5\",\"stream_options\":\"\x80\",\"stream_options\":{\"include_usage\":true},\"tools\":\"\xff\",\"tools\":[],\"messages\":[]}",
	"{\"\xf3\":[],\"model\":\"gpt-5\",\"messages\":[],\"stream\":true,\"stream_options\":{\"a\":1},\"max_tokens\":5,\"tools\":[],\"cache_salt\":1}",
	"{\"model\":\"m\",\"\xce\":[],\"stream_options\":\"\",\"stream_options\":{},\"input\":\"x\",\"store\":true}",
	"{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"\xd4\":1,\"content\":\"a\"},{\"role\":\"user\",\"content\":\"b\"}],\"system\":\"s\",\"thinking\":{\"type\":\"adaptive\"},\"max_tokens\":9}",
}

func init() {
	requestRuleBodies = append(requestRuleBodies, escapedRequestRuleBodies...)
	requestRuleBodies = append(requestRuleBodies, invalidNameBodies...)
}

func TestEscapedTestInputsAreReallyEscaped(t *testing.T) {
	if !strings.Contains(escapedRequestRuleBodies[0], escapeBackslash+"u006fper") || strings.Contains(escapedRequestRuleBodies[0], `"developer","content":"y"`) {
		t.Fatalf("test inputs lost their escapes: %s", escapedRequestRuleBodies[0])
	}
	// A role spelled with an escape is still the developer role.
	out := string(NormalizeDeveloperRole([]byte(escapedRequestRuleBodies[0])))
	if strings.Count(out, `"system"`) != 2 {
		t.Fatalf("escaped developer roles not renamed: %s", out)
	}
}
