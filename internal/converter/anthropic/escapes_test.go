package anthropic

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

var escapedCacheMarkerBodies = realEscapes(
	`{"messages":[{"cache%u005fcontrol":{"type":"ephemeral"}}]}`,
	`{"messages":[{"%u0063ache_control":{"t%u0079pe":"ephemer%u0061l","ttl":"1%u0068"}}]}`,
	`{"mess%u0061ges":[{"cache_control":{"type":"ephemeral"}}]}`,
	`{"syst%u0065m":[{"cache_control":{"type":"ephemeral"}}],"t%u006fols":[{"%u0063%u0061%u0063%u0068%u0065%u005f%u0063%u006f%u006e%u0074%u0072%u006f%u006c":{"type":"ephemeral","ttl":"1h"}}]}`,
	`{"messages":[{"content":"terminal %u001b[0m output, caf%u00e9"},{"cache_control":{"type":"ephemeral"}}]}`,
	`{"messages":[{"content":"only control %u001b[0m and latin %u00e9 escapes, no marker"}]}`,
	`{"messages":[{"content":"the text cache%u005fcontrol is not a key"}]}`,
	`{"messages":[{"cache%u005fcontrol":{"type":"ephemeral"},"cache_control":{"type":"other"}}]}`,
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
	cacheMarkerEquivalenceBodies = append(cacheMarkerEquivalenceBodies, escapedCacheMarkerBodies...)
	cacheMarkerEquivalenceBodies = append(cacheMarkerEquivalenceBodies, invalidNameBodies...)
	cacheInjectionBodies = append(cacheInjectionBodies, invalidNameBodies...)
	passthroughBodies = append(passthroughBodies, invalidNameBodies...)
}

func TestEscapedTestInputsAreReallyEscaped(t *testing.T) {
	if !strings.Contains(escapedCacheMarkerBodies[0], escapeBackslash+"u005fcontrol") || strings.Contains(escapedCacheMarkerBodies[0], "cache_control") {
		t.Fatalf("test inputs lost their escapes: %s", escapedCacheMarkerBodies[0])
	}
	// A marker under an escaped key is a marker.
	if got := CountCacheMarkers([]byte(escapedCacheMarkerBodies[0])); got.Messages != 1 {
		t.Fatalf("escaped cache_control key not counted: %+v", got)
	}
	if got := CountCacheMarkers([]byte(escapedCacheMarkerBodies[3])); got.System != 1 || got.Tools != 1 || got.TTL1h != 1 {
		t.Fatalf("fully escaped key not counted: %+v", got)
	}
	// Text that merely contains an escape is not.
	if got := CountCacheMarkers([]byte(escapedCacheMarkerBodies[5])); got.Total() != 0 {
		t.Fatalf("counted a marker that is not there: %+v", got)
	}
}
