package proxy

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

var escapedRequestScanBodies = realEscapes(
	`{"reasoning%u005feffort":"high"}`,
	`{"%u0074hinking":{"type":"enabled"}}`,
	`{"reasoning_effort":"hi%u0067h"}`,
	`{"re%u0061soning":{"eff%u006frt":"n%u006fne"}}`,
	`{"extra%u005fbody":{"th%u0069nking":{"type":"dis%u0061bled"}}}`,
	`{"%u0072%u0065%u0061%u0073%u006f%u006e%u0069%u006e%u0067_effort":"low"}`,
	`{"messages":[{"content":"terminal %u001b[0m output, caf%u00e9"}]}`,
	`{"t%u006fols":[{"type":"web_search"}]}`,
	`{"tools":[{"t%u0079pe":"web%u005fsearch","search_context%u005fsize":"hi%u0067h"}]}`,
	`{"web_search%u005foptions":{"search_context_size":"low"}}`,
	`{"plug%u0069ns":[{"%u0069d":"web"}]}`,
)

var escapedSanitizeBodies = realEscapes(
	`{"mo%u0064el":"m","messages":[],"str%u0065am":true}`,
	`{"model":"%u006d","messages":[],"session%u005fid":"s%u0031","stream":true}`,
	`{"model":"m","messages":[],"stream":true,"stream%u005foptions":{"include%u005fusage":true}}`,
	`{"model":"m","messages":[],"service%u005ftier":"prio%u0072ity","extra%u005fbody":{"service_ti%u0065r":"flex"}}`,
	`{"model":"m","%u0069nput":"hi","stream":true,"b%u0061ckground":true}`,
	`{"model":"m","messages":[{"content":"terminal %u001b[0m, caf%u00e9, emoji %ud83d%ude00"}],"stream":true,"us%u0065r":"u1"}`,
)

// Names that are not valid UTF-8: encoding/json repairs them whenever it
// encodes the object again, so what happens to them depends on whether a
// rule re-encodes the body.
var invalidNameBodies = []string{
	"{\"\xf3\":[]}",
	// A repeated name whose discarded value is invalid UTF-8: whether it
	// survives depends on whether the body is written out again.
	"{\"\":\"\x80\",\"\":{}}",
	"{\"model\":\"m\",\"messages\":[],\"stream\":true,\"stream_options\":{\"include_usage\":true,\"\xff\":true}}",
	"{\"model\":\"m\",\"messages\":[],\"stream\":true,\"stream_options\":{\"include_usage\":true,\"x\":\"\xff\"},\"extra_body\":{\"service_tier\":\"a\",\"\xfe\":1}}",
	"{\"model\":\"gpt-5\",\"stream_options\":\"\x80\",\"stream_options\":{\"include_usage\":true},\"tools\":\"\xff\",\"tools\":[],\"messages\":[]}",
	"{\"\xf3\":[],\"model\":\"gpt-5\",\"messages\":[],\"stream\":true,\"stream_options\":{\"a\":1},\"max_tokens\":5,\"tools\":[],\"cache_salt\":1}",
	"{\"model\":\"m\",\"\xce\":[],\"stream_options\":\"\",\"stream_options\":{},\"input\":\"x\",\"store\":true}",
	"{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"\xd4\":1,\"content\":\"a\"},{\"role\":\"user\",\"content\":\"b\"}],\"system\":\"s\",\"thinking\":{\"type\":\"adaptive\"},\"max_tokens\":9}",
}

func init() {
	requestScanEquivalenceBodies = append(requestScanEquivalenceBodies, escapedRequestScanBodies...)
	sanitizeEquivalenceBodies = append(sanitizeEquivalenceBodies, escapedSanitizeBodies...)
	requestScanEquivalenceBodies = append(requestScanEquivalenceBodies, invalidNameBodies...)
	sanitizeEquivalenceBodies = append(sanitizeEquivalenceBodies, invalidNameBodies...)
}

func TestEscapedTestInputsAreReallyEscaped(t *testing.T) {
	if !strings.Contains(escapedRequestScanBodies[0], escapeBackslash+"u005feffort") || strings.Contains(escapedRequestScanBodies[0], "reasoning_effort") {
		t.Fatalf("test inputs lost their escapes: %s", escapedRequestScanBodies[0])
	}
	// An escaped key means what the plain one means.
	if ok, source, _ := requestReasoningDetails([]byte(escapedRequestScanBodies[0])); !ok || source != "reasoning_effort" {
		t.Fatalf("escaped reasoning_effort key not recognised: %v %q", ok, source)
	}
	if ok, source, _ := requestReasoningDetails([]byte(escapedRequestScanBodies[5])); !ok || source != "reasoning_effort" {
		t.Fatalf("fully escaped key not recognised: %v %q", ok, source)
	}
	if ok, _ := extractWebSearchRequestUsage([]byte(escapedRequestScanBodies[7]), "application/json"); !ok {
		t.Fatal("escaped tools key not recognised")
	}
	got, err := sanitizeJSONRequestBody([]byte(escapedSanitizeBodies[0]), false)
	if err != nil || got.ModelID != "m" || !got.Streaming {
		t.Fatalf("escaped model/stream keys not recognised: %+v, %v", got, err)
	}
}
