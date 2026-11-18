package toolloop

import (
	"testing"
)

// TestStreamToolCallPiecesAreFolded checks that a tool call whose arguments
// arrive in pieces is one call, not one per piece: a provider sends the id
// and the name with the first piece and the arguments split across the rest,
// under the same index.
func TestStreamToolCallPiecesAreFolded(t *testing.T) {
	raw := []byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"router_web_search","arguments":"{\"query\":"}}]}}]}` + "\n\n" +
		"data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"deepseek"}}]}}]}` + "\n\n" +
		"data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" v4\"}"}}]}}]}` + "\n\n" +
		"data: [DONE]\n\n")

	calls, _ := callsFromStream(raw)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1: %+v", len(calls), calls)
	}
	if calls[0].ID != "call_1" {
		t.Errorf("the call's id = %q, want call_1", calls[0].ID)
	}
	if calls[0].Name != "router_web_search" {
		t.Errorf("the call's name = %q, want router_web_search", calls[0].Name)
	}
	if calls[0].Arguments != `{"query":"deepseek v4"}` {
		t.Errorf("the call's arguments = %q, want the whole argument string", calls[0].Arguments)
	}
}

// TestStreamToolCallsAreFilteredByTheRunner checks that only the tools the
// runner owns come back: a client's own function tools are the upstream's to
// serve, and the router does not run them.
func TestStreamToolCallsAreFilteredByTheRunner(t *testing.T) {
	raw := []byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"router_web_search","arguments":"{}"}}]}}]}` + "\n\n" +
		"data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}}]}` + "\n\n" +
		"data: [DONE]\n\n")

	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	calls := CallsFromStream(raw, runner)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1: %+v", len(calls), calls)
	}
	if calls[0].Name != "router_web_search" {
		t.Errorf("the call = %q, want the router's own tool", calls[0].Name)
	}
}
