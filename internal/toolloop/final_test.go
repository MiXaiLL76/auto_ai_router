package toolloop

import (
	"context"
	"encoding/json"
	"testing"
)

// fakeRunner owns one tool and records what it was asked.
type fakeRunner struct {
	owned map[string]bool
	calls []string
}

func (r *fakeRunner) Owns(name string) bool { return r.owned[name] }

func (r *fakeRunner) Run(_ context.Context, name, _ string) string {
	r.calls = append(r.calls, name)
	return `{"results":["found"]}`
}

func (r *fakeRunner) Billable() map[string]int { return nil }

// TestFinalBodyTakesTheToolsAway checks the loop's last request: the model is
// asked to answer with what it has found, and the tools are gone -- so it
// cannot ask for another one.
func TestFinalBodyTakesTheToolsAway(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	cycle, err := NewCycle([]byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "q"}],
		"tools": [{"type": "function", "function": {"name": "router_web_search"}}]
	}`), runner)
	if err != nil {
		t.Fatalf("NewCycle: %v", err)
	}

	// The model's answer asks for the tool.
	answer := `{
		"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "",
			"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "router_web_search", "arguments": "{\"text_query\":\"q\"}"}}]}}]
	}`

	final, err := cycle.FinalBody(context.Background(), []byte(answer))
	if err != nil {
		t.Fatalf("FinalBody: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(final, &body); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if _, ok := body["tools"]; ok {
		t.Errorf("the last request still carries the tools: %s", final)
	}
	if _, ok := body["tool_choice"]; ok {
		t.Errorf("the last request still carries the tool choice: %s", final)
	}
	// The tool the model asked for was still run: its result is what the
	// model needs to answer with.
	if len(runner.calls) != 1 || runner.calls[0] != "router_web_search" {
		t.Errorf("the tool was not run: %v", runner.calls)
	}
	// And the result is in the conversation.
	messages, _ := body["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("the conversation has %d messages, want 3", len(messages))
	}
	result, _ := messages[2].(map[string]any)
	if result["role"] != "tool" {
		t.Errorf("the third message is %v, want the tool result", result["role"])
	}
}

// TestFinalBodyFromToolCallsTakesTheToolsAway is the streaming counterpart.
func TestFinalBodyFromToolCallsTakesTheToolsAway(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	cycle, err := NewCycle([]byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "q"}],
		"tools": [{"type": "function", "function": {"name": "router_web_search"}}]
	}`), runner)
	if err != nil {
		t.Fatalf("NewCycle: %v", err)
	}

	final, err := cycle.FinalBodyFromToolCalls(context.Background(), []ToolCall{{
		ID: "call_1", Name: "router_web_search", Arguments: `{"text_query":"q"}`,
	}})
	if err != nil {
		t.Fatalf("FinalBodyFromToolCalls: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(final, &body); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if _, ok := body["tools"]; ok {
		t.Errorf("the last request still carries the tools: %s", final)
	}
	if len(runner.calls) != 1 {
		t.Errorf("the tool was not run: %v", runner.calls)
	}
}
