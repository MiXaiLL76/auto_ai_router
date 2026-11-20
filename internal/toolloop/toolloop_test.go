package toolloop

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
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

// TestNextBodyWithOnlyAClientToolCallIsNothingToRun checks the guard that
// keeps the loop from sending the very same request again.
//
// The model's answer is the CLIENT's own tool call -- a coding agent's
// read_file. There is nothing of the router's to run and nothing to add to
// the conversation, so the step stops instead of returning a request the
// provider would answer the same way: the client would be billed for both
// sends.
func TestNextBodyWithOnlyAClientToolCallIsNothingToRun(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	cycle, err := NewCycle([]byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "q"}]
	}`), runner)
	if err != nil {
		t.Fatalf("NewCycle: %v", err)
	}

	answer := `{
		"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "",
			"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"main.go\"}"}}]}}]
	}`

	if _, err := cycle.NextBody(context.Background(), []byte(answer)); err != ErrNothingToRun {
		t.Fatalf("NextBody's error = %v, want ErrNothingToRun", err)
	}
	if len(runner.calls) != 0 {
		t.Errorf("a tool was run: %v", runner.calls)
	}
}

// TestNextBodyFromToolCallsWithOnlyAClientToolCallIsNothingToRun is the
// streaming counterpart of the guard.
func TestNextBodyFromToolCallsWithOnlyAClientToolCallIsNothingToRun(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	cycle, err := NewCycle([]byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "q"}]
	}`), runner)
	if err != nil {
		t.Fatalf("NewCycle: %v", err)
	}

	if _, err := cycle.NextBodyFromToolCalls(context.Background(), []ToolCall{{
		ID: "call_1", Name: "read_file", Arguments: `{"path":"main.go"}`,
	}}); err != ErrNothingToRun {
		t.Fatalf("NextBodyFromToolCalls's error = %v, want ErrNothingToRun", err)
	}
	if len(runner.calls) != 0 {
		t.Errorf("a tool was run: %v", runner.calls)
	}
}

// TestTheCollectorHandsOutOnlyTheRoutersOwnCalls checks the collector's
// contract: the calls it returns are the router's own.
//
// A caller that got the raw list could mistake a client's tool call for the
// router's and loop on a request it has no tool to run -- sending the very
// same request upstream a second time and billing the client for both sends.
func TestTheCollectorHandsOutOnlyTheRoutersOwnCalls(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	stream := "data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"router_web_search","arguments":"{}"}}]}}]}` + "\n\n" +
		"data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}]}` + "\n\n" +
		"data: [DONE]\n\n"

	response := &http.Response{Body: io.NopCloser(strings.NewReader(stream))}
	_, calls, _, err := CollectStreamToolCalls(response, runner)
	if err != nil {
		t.Fatalf("CollectStreamToolCalls: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1: %+v", len(calls), calls)
	}
	if calls[0].Name != "router_web_search" {
		t.Errorf("the call is %q, want the router's own tool", calls[0].Name)
	}
}

// TestTheUsageIsSummedOverTheSteps checks the accumulator: every step's
// tokens are added, because every step is a separate upstream request the
// client is charged for.
func TestTheUsageIsSummedOverTheSteps(t *testing.T) {
	total := &Usage{}
	total.Add(&Usage{
		PromptTokens: 100, CompletionTokens: 10, CachedInputTokens: 40,
		ReasoningTokens: 7, TotalTokens: 110,
		ToolCalls: map[string]int{"router_web_search": 1},
	})
	total.Add(&Usage{
		PromptTokens: 200, CompletionTokens: 20, CachedInputTokens: 60,
		ReasoningTokens: 9, TotalTokens: 220,
		ToolCalls: map[string]int{"router_web_search": 2},
	})
	if total.PromptTokens != 300 || total.CompletionTokens != 30 {
		t.Errorf("the tokens = %d/%d, want 300/30", total.PromptTokens, total.CompletionTokens)
	}
	if total.CachedInputTokens != 100 {
		t.Errorf("the cached tokens = %d, want 100", total.CachedInputTokens)
	}
	if total.ReasoningTokens != 16 {
		t.Errorf("the reasoning tokens = %d, want 16", total.ReasoningTokens)
	}
	if total.TotalTokens != 330 {
		t.Errorf("the total tokens = %d, want 330", total.TotalTokens)
	}
	if total.ToolCalls["router_web_search"] != 3 {
		t.Errorf("the tool calls = %v, want 3", total.ToolCalls)
	}
	// A step without usage changes nothing.
	total.Add(nil)
	if total.PromptTokens != 300 {
		t.Errorf("the tokens = %d, want 300", total.PromptTokens)
	}
}

// TestTheUsageIsReadOutOfTheProvidersAnswer checks the reading of the
// provider's own figures.
func TestTheUsageIsReadOutOfTheProvidersAnswer(t *testing.T) {
	body := []byte(`{
		"usage": {
			"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
			"prompt_tokens_details": {"cached_tokens": 40},
			"completion_tokens_details": {"reasoning_tokens": 7}
		}
	}`)
	usage := UsageFromResponse(body)
	if usage.PromptTokens != 100 || usage.CompletionTokens != 10 || usage.TotalTokens != 110 {
		t.Errorf("the tokens = %+v", usage)
	}
	if usage.CachedInputTokens != 40 {
		t.Errorf("the cached tokens = %d, want 40", usage.CachedInputTokens)
	}
	if usage.ReasoningTokens != 7 {
		t.Errorf("the reasoning tokens = %d, want 7", usage.ReasoningTokens)
	}
	// A body without usage reads as zeroes, not as an error.
	if empty := UsageFromResponse([]byte(`{}`)); empty.PromptTokens != 0 {
		t.Errorf("the tokens = %+v, want zeroes", empty)
	}
}

// TestTheUsageIsReadOutOfAStream checks the reading of a stream's figures.
func TestTheUsageIsReadOutOfAStream(t *testing.T) {
	usage := UsageFromStreamUsage(&openai.OpenAIUsage{
		PromptTokens: 200, CompletionTokens: 20, TotalTokens: 220,
		PromptTokensDetails:     &openai.TokenDetails{CachedTokens: 60},
		CompletionTokensDetails: &openai.CompletionTokenDetails{ReasoningTokens: 9},
	})
	if usage.PromptTokens != 200 || usage.CachedInputTokens != 60 || usage.ReasoningTokens != 9 {
		t.Errorf("the usage = %+v", usage)
	}
	if empty := UsageFromStreamUsage(nil); empty.PromptTokens != 0 {
		t.Errorf("the usage = %+v, want zeroes", empty)
	}
}

// TestTheCallsAreReadOutOfTheProvidersAnswer checks the reading of the tool
// calls, and that only the router's own come back.
func TestTheCallsAreReadOutOfTheProvidersAnswer(t *testing.T) {
	runner := &fakeRunner{owned: map[string]bool{"router_web_search": true}}
	body := []byte(`{
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "",
			"tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "router_web_search", "arguments": "{}"}},
				{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
			]}}]
	}`)
	calls := CallsFromResponse(body, runner)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1: %+v", len(calls), calls)
	}
	if calls[0].Name != "router_web_search" || calls[0].ID != "call_1" {
		t.Errorf("the call is %+v", calls[0])
	}
	if empty := CallsFromResponse([]byte(`{}`), runner); len(empty) != 0 {
		t.Errorf("got %d calls, want 0", len(empty))
	}
}

// TestTheStreamIsReadWhole checks the reading of a step's answer: the stream
// comes back whole, with its figures.
func TestTheStreamIsReadWhole(t *testing.T) {
	stream := "data: " + `{"choices":[{"index":0,"delta":{"content":"hi"}}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n" +
		"data: [DONE]\n\n"
	response := &http.Response{Body: io.NopCloser(strings.NewReader(stream))}
	raw, usage, err := ReadStream(response)
	if err != nil {
		t.Fatalf("ReadStream: %v", err)
	}
	if !strings.Contains(string(raw), "hi") {
		t.Errorf("the stream = %s", raw)
	}
	if usage.PromptTokens != 100 {
		t.Errorf("the usage = %+v", usage)
	}
	// A response without a body is an error, not a panic.
	if _, _, err := ReadStream(&http.Response{}); err == nil {
		t.Error("a response without a body must be an error")
	}
}
