package proxy

import (
	"fmt"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// TestEveryToolTheRunnerOwnsIsExposed pins the two lists together: the gate
// asks ExposesTool whether the request offers a tool the loop could run, and
// the runner is the one that runs it. A name the runner owns but the gate does
// not know is a request whose stream skips the loop and whose client is shown
// a function call it never declared -- exactly the Moonshot Pro search bug.
func TestEveryToolTheRunnerOwnsIsExposed(t *testing.T) {
	cases := []struct {
		name   string
		owns   func(string) bool
		expose func([]byte) bool
		names  []string
	}{
		{
			name:   "deepseek",
			owns:   (&deepSeekToolRunner{}).Owns,
			expose: deepseektools.ExposesTool,
			names:  []string{deepseektools.SearchToolName, deepseektools.FetchToolName},
		},
		{
			name:   "moonshot",
			owns:   (&moonshotToolRunner{}).Owns,
			expose: moonshottools.ExposesTool,
			names: []string{
				moonshottools.SearchToolName,
				moonshottools.SearchProToolName,
				moonshottools.FetchToolName,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, name := range tc.names {
				if !tc.owns(name) {
					t.Fatalf("the %s runner does not own %q: the test lists a name it no longer runs", tc.name, name)
				}
				body := fmt.Sprintf(`{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":%q}}]}`, name)
				if !tc.expose([]byte(body)) {
					t.Errorf("ExposesTool does not see %q, but the runner runs it: the stream would skip the loop", name)
				}
			}
		})
	}
}

// TestExposesToolIgnoresAClientToolAndAnEmptyRequest is the other side: the
// gate must not keep the loop for a request that offers nothing of ours, or
// the buffering comes back.
func TestExposesToolIgnoresAClientToolAndAnEmptyRequest(t *testing.T) {
	bodies := map[string]string{
		"no tools":        `{"messages":[{"role":"user","content":"hi"}]}`,
		"empty tools":     `{"messages":[{"role":"user","content":"hi"}],"tools":[]}`,
		"a client tool":   `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather"}}]}`,
		"a builtin other": `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"code_interpreter"}]}`,
		"not json":        `{"messages":`,
	}
	for name, body := range bodies {
		if deepseektools.ExposesTool([]byte(body)) {
			t.Errorf("deepseek ExposesTool = true for %s, want false", name)
		}
		if moonshottools.ExposesTool([]byte(body)) {
			t.Errorf("moonshot ExposesTool = true for %s, want false", name)
		}
	}
}
