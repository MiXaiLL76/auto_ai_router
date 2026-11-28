package responses

import (
	"strings"
	"testing"
)

// Escaped spellings are written with a %u marker and turned into real
// backslash-u escapes here, so the inputs are certain to contain the escape
// sequence itself and not the character it stands for.
const backslash = "\\"

func realEscapes(cases ...string) []string {
	out := make([]string, len(cases))
	for i, c := range cases {
		out[i] = strings.ReplaceAll(c, "%u", backslash+"u")
	}
	return out
}

var escapedScanBodies = realEscapes(
	`{"%u0069nput":[1]}`,
	`{"input":[1],"mess%u0061ges":[]}`,
	`{"input":[],"%u0073tore":true,"metad%u0061ta":{"a":"b"},"previous_response%u005fid":"resp_1","ttl":5}`,
	`{"input":[],"store":false,"%u0073tore":true}`,
	`{"input":[],"text":{"form%u0061t":{"type":"t%u0065xt"}}}`,
)

var escapedCodexBodies = realEscapes(
	`{"model":"gpt-5","input":[{"type":"reas%u006fning","id":"r1"},{"type":"comp%u0061ction","encrypted_content":"s"},{"t%u0079pe":"compaction","encrypted_content":"s2"}]}`,
	`{"model":"gpt-5","input":[{"type":"message","content":"caf%u00e9 %u001b[0m plain"},{"type":"reasoning","encrypted_content":"e"}]}`,
	`{"model":"gpt-5","input":[{"type":"message","content":"only control %u001b[31m and latin %u00e9 escapes"}]}`,
	`{"model":"gpt-5","%u0073tore":true,"input":"x","t%u006fols":[],"tool_choice":"auto"}`,
	`{"model":"gpt-4o-mini","input":"x","re%u0061soning":{"eff%u006frt":"n%u006fne"}}`,
	`{"model":"gpt-5","input":[{"type":"reasoning","summ%u0061ry":[],"encrypted_content":"e"},{"type":"reasoning","c%u006fntent":[{"text":"t"}]}]}`,
	`{"model":"gpt-5","input":"x","instructi%u006fns":[{"content":"a"}]}`,
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
	scanEquivalenceBodies = append(scanEquivalenceBodies, escapedScanBodies...)
	codexPassthroughBodies = append(codexPassthroughBodies, escapedCodexBodies...)
	scanEquivalenceBodies = append(scanEquivalenceBodies, invalidNameBodies...)
	codexPassthroughBodies = append(codexPassthroughBodies, invalidNameBodies...)
}

func TestEscapedTestInputsAreReallyEscaped(t *testing.T) {
	if !strings.Contains(escapedScanBodies[0], backslash+"u0069nput") || strings.Contains(escapedScanBodies[0], "input") {
		t.Fatalf("test inputs lost their escapes: %s", escapedScanBodies[0])
	}
	// And they mean what the plain spelling means.
	if !IsResponsesAPI([]byte(escapedScanBodies[0])) || IsResponsesAPI([]byte(escapedScanBodies[1])) {
		t.Fatal("escaped input/messages keys must be recognised")
	}
	if !ExtractResponsesMetadata([]byte(escapedScanBodies[2])).Store {
		t.Fatal("escaped store key must be recognised")
	}
}
