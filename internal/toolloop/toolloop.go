// Package toolloop drives the router's own tool loop: the model asks for a
// tool the router runs, the router runs it, the result goes back to the model
// as an ordinary tool result, and the model continues.
//
// The loop is the same for every provider whose tools the router runs: only
// the tools themselves and their prices differ. So the loop lives here, and a
// provider plugs in by implementing Runner -- which tools it owns, how to run
// one, and what the call cost.
package toolloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// ErrNothingToRun is returned when a step's tool calls are none of the
// router's own.
//
// The step cannot go on: there is no tool to run and nothing to add to the
// conversation, so the request would be sent again unchanged -- and the
// client would pay for it twice.
var ErrNothingToRun = errors.New("toolloop: no tool of the router's to run")

// ErrAborted ends the cycle when the client went away.
//
// It is one sentinel for every provider, so the dispatch in the router
// checks it once: a disconnect is not a failure of the loop, and the router
// must not report it as one.
var ErrAborted = errors.New("toolloop: client disconnected")

// Runner executes the tools the router runs on the model's behalf.
//
// A provider implements it: which tools it owns (the names the model calls),
// how to run one, and how the call is billed.
type Runner interface {
	// Owns reports whether the tool with this name is one the router runs.
	// The router runs only those: a client's own function tools are the
	// upstream's to serve, and the router does not run them.
	Owns(name string) bool

	// Run executes the tool and returns the result to send back to the model.
	//
	// A failed call is not an error: the provider's own message is handed to
	// the model as the tool result, so the model can retry or answer without
	// the router inventing a price.
	Run(ctx context.Context, name, arguments string) string

	// Billable counts the successful calls, per tool. The loop folds it into
	// the request's usage, so the calls the router made are billed even when
	// the model that followed them failed.
	Billable() map[string]int
}

// Usage is the usage a loop accumulated over its steps.
//
// The router's own merge helpers replace a counter with the later step's
// value rather than adding them up -- correct for a stream, where the last
// chunk carries the total, and wrong for a tool loop, where every step is a
// separate upstream request that the client is charged for. So the loop sums
// the token counts itself.
type Usage struct {
	// PromptTokens is the sum of every step's prompt tokens.
	PromptTokens int
	// CompletionTokens is the sum of every step's completion tokens.
	CompletionTokens int
	// CachedInputTokens is the sum of every step's cached input tokens.
	CachedInputTokens int
	// ReasoningTokens is the sum of every step's reasoning tokens.
	ReasoningTokens int
	// TotalTokens is the sum of every step's total tokens.
	TotalTokens int
	// ToolCalls counts the billable calls per tool name.
	ToolCalls map[string]int
}

// Add folds one step's usage into the accumulator.
func (u *Usage) Add(other *Usage) {
	if other == nil {
		return
	}
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.CachedInputTokens += other.CachedInputTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.TotalTokens += other.TotalTokens
	for name, count := range other.ToolCalls {
		if u.ToolCalls == nil {
			u.ToolCalls = make(map[string]int)
		}
		u.ToolCalls[name] += count
	}
}

// UsageFromResponse reads the token counters out of a Chat Completions
// response body.
func UsageFromResponse(body []byte) *Usage {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return &Usage{}
	}
	return usageFromOpenAIUsage(response.Usage)
}

// UsageFromStreamUsage reads the token counters out of a stream's usage.
func UsageFromStreamUsage(usage *openai.OpenAIUsage) *Usage {
	return usageFromOpenAIUsage(usage)
}

// usageFromOpenAIUsage converts the provider's usage into the loop's shape.
func usageFromOpenAIUsage(usage *openai.OpenAIUsage) *Usage {
	out := &Usage{}
	if usage == nil {
		return out
	}
	out.PromptTokens = usage.PromptTokens
	out.CompletionTokens = usage.CompletionTokens
	out.TotalTokens = usage.TotalTokens
	if usage.CompletionTokensDetails != nil {
		out.ReasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
	}
	if usage.PromptTokensDetails != nil {
		out.CachedInputTokens = usage.PromptTokensDetails.CachedTokens
	}
	return out
}

// Cycle drives the loop for one client request.
//
// The loop is: ask the model -> the model calls the router's tool -> the
// router runs it -> the result goes back as a tool result -> the model
// continues. Each round is another upstream request, so the steps are
// bounded and every round's usage is counted.
type Cycle struct {
	// runner executes the tools.
	runner Runner
	// body is the current request body (rewritten between steps).
	body []byte
	// messages is the conversation so far.
	messages []map[string]any
	// results maps a tool call id to the result already produced for it, so
	// a replayed step does not run the tool (or charge it) twice.
	results map[string]string
}

// NewCycle prepares a cycle for a request body.
func NewCycle(body []byte, runner Runner) (*Cycle, error) {
	cycle := &Cycle{runner: runner, body: body, results: make(map[string]string)}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("toolloop: parsing request body: %w", err)
	}
	rawMessages, ok := fields["messages"]
	if !ok {
		return cycle, nil
	}
	if err := json.Unmarshal(rawMessages, &cycle.messages); err != nil {
		return nil, fmt.Errorf("toolloop: parsing messages: %w", err)
	}
	return cycle, nil
}

// NextBody builds the body for the next upstream call: the conversation with
// the assistant's tool-call message and the tool results appended.
//
// Only the tools the router owns are run. A client's own function tools are
// left alone: the upstream serves them, and the router does not run a
// client's tools on its behalf. They are not given a result either -- the
// model asked for them, and the answer is the client's to give.
func (cycle *Cycle) NextBody(ctx context.Context, responseBody []byte) ([]byte, error) {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return nil, fmt.Errorf("toolloop: parsing response: %w", err)
	}
	assistant := assistantToolCallMessage(response)
	results := make([]map[string]any, 0, len(response.Choices))
	if len(response.Choices) > 0 {
		for _, call := range OwnedCalls(callsFromResponse(response), cycle.runner) {
			content, ok := cycle.results[call.ID]
			if !ok {
				content = cycle.runner.Run(ctx, call.Name, call.Arguments)
				cycle.results[call.ID] = content
			}
			results = append(results, toolResultMessage(call.ID, call.Name, content))
		}
	}
	if len(results) == 0 {
		// None of the calls is the router's own: there is nothing to run and
		// nothing to add to the conversation. Sending the request again would
		// be sending the very same request -- the provider would answer the
		// same way, and the client would be billed for the request twice.
		return nil, ErrNothingToRun
	}
	cycle.messages = appendToolResults(cycle.messages, assistant, results)
	return cycle.bodyWithMessages()
}

// callsFromResponse reads every tool call out of a Chat Completions response.
func callsFromResponse(response openai.OpenAIResponse) []ToolCall {
	if len(response.Choices) == 0 {
		return nil
	}
	calls := make([]ToolCall, 0, len(response.Choices[0].Message.ToolCalls))
	for _, call := range response.Choices[0].Message.ToolCalls {
		calls = append(calls, ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: call.Function.Arguments})
	}
	return calls
}

// FinalBody builds the body for the loop's last request: the conversation
// with the assistant's tool-call message and the tool results appended, and
// the tools removed.
//
// It is what the loop sends when its step budget is spent: the model is
// asked to answer with what it has found, so the client's answer is an
// answer and not a tool call for a tool the client never declared.
//
// The tools the model asked for are still run: their results are the very
// thing the model needs to answer with.
func (cycle *Cycle) FinalBody(ctx context.Context, responseBody []byte) ([]byte, error) {
	next, err := cycle.NextBody(ctx, responseBody)
	if err != nil {
		return nil, err
	}
	return cycle.withoutTools(next)
}

// FinalBodyFromToolCalls is the streaming counterpart of FinalBody.
func (cycle *Cycle) FinalBodyFromToolCalls(ctx context.Context, calls []ToolCall) ([]byte, error) {
	next, err := cycle.NextBodyFromToolCalls(ctx, calls)
	if err != nil {
		return nil, err
	}
	return cycle.withoutTools(next)
}

// withoutTools removes the tools from a request body: the model must answer
// now, not call a tool again.
func (cycle *Cycle) withoutTools(body []byte) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("toolloop: parsing request body: %w", err)
	}
	delete(fields, "tools")
	delete(fields, "tool_choice")
	final, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("toolloop: encoding request body: %w", err)
	}
	cycle.body = final
	return final, nil
}

// NextBodyFromToolCalls builds the next request body from a streamed tool
// call, the streaming counterpart of NextBody.
func (cycle *Cycle) NextBodyFromToolCalls(ctx context.Context, calls []ToolCall) ([]byte, error) {
	assistant := map[string]any{"role": "assistant"}
	toolCalls := make([]map[string]any, 0, len(calls))
	results := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		if !cycle.runner.Owns(call.Name) {
			continue
		}
		toolCalls = append(toolCalls, map[string]any{
			"id":   call.ID,
			"type": "function",
			"function": map[string]any{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
		content, ok := cycle.results[call.ID]
		if !ok {
			content = cycle.runner.Run(ctx, call.Name, call.Arguments)
			cycle.results[call.ID] = content
		}
		results = append(results, toolResultMessage(call.ID, call.Name, content))
	}
	if len(toolCalls) == 0 {
		// None of the calls is the router's own: there is nothing to run and
		// nothing to add to the conversation. Sending the request again would
		// be sending the very same request -- the provider would answer the
		// same way, and the client would be billed for the request twice.
		return nil, ErrNothingToRun
	}
	assistant["tool_calls"] = toolCalls
	cycle.messages = appendToolResults(cycle.messages, assistant, results)
	return cycle.bodyWithMessages()
}

// bodyWithMessages writes the conversation back into the request body.
func (cycle *Cycle) bodyWithMessages() ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(cycle.body, &fields); err != nil {
		return nil, fmt.Errorf("toolloop: parsing request body: %w", err)
	}
	encodedMessages, err := json.Marshal(cycle.messages)
	if err != nil {
		return nil, fmt.Errorf("toolloop: encoding messages: %w", err)
	}
	fields["messages"] = encodedMessages
	next, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("toolloop: encoding request body: %w", err)
	}
	cycle.body = next
	return next, nil
}

// ToolCall is a tool call the model made.
type ToolCall struct {
	// ID is the call's id: the tool result is matched by it.
	ID string
	// Name is the function's name.
	Name string
	// Arguments is the function's arguments, as a JSON string.
	Arguments string
}

// OwnedCalls keeps the tool calls the router owns.
//
// The router runs only its own tools: a client's own function tool is the
// upstream's to serve, and the router neither runs it nor counts it as a
// reason to loop.
func OwnedCalls(calls []ToolCall, runner Runner) []ToolCall {
	owned := make([]ToolCall, 0, len(calls))
	for _, call := range calls {
		if runner.Owns(call.Name) {
			owned = append(owned, call)
		}
	}
	return owned
}

// CallsFromResponse reads the tool calls the router owns out of a Chat
// Completions response.
//
// A client's own function tools are left out: the upstream serves them, and
// the router does not run a client's tools on its behalf.
func CallsFromResponse(body []byte, runner Runner) []ToolCall {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil
	}
	if len(response.Choices) == 0 {
		return nil
	}
	all := make([]ToolCall, 0, len(response.Choices[0].Message.ToolCalls))
	for _, call := range response.Choices[0].Message.ToolCalls {
		all = append(all, ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: call.Function.Arguments})
	}
	return OwnedCalls(all, runner)
}

// assistantToolCallMessage builds the assistant message that carries the
// model's tool calls.
//
// The message keeps the model's tool calls verbatim (the ids have to match)
// and its reasoning content, so the model's own state survives the round trip.
func assistantToolCallMessage(response openai.OpenAIResponse) map[string]any {
	message := map[string]any{"role": "assistant"}
	if len(response.Choices) == 0 {
		return message
	}
	choice := response.Choices[0]
	if choice.Message.Content != "" {
		message["content"] = choice.Message.Content
	}
	if choice.Message.ReasoningContent != "" {
		message["reasoning_content"] = choice.Message.ReasoningContent
	}
	calls := make([]map[string]any, 0, len(choice.Message.ToolCalls))
	for _, call := range choice.Message.ToolCalls {
		calls = append(calls, map[string]any{
			"id":   call.ID,
			"type": call.Type,
			"function": map[string]any{
				"name":      call.Function.Name,
				"arguments": call.Function.Arguments,
			},
		})
	}
	if len(calls) > 0 {
		message["tool_calls"] = calls
	}
	return message
}

// toolResultMessage builds the tool message carrying one result.
func toolResultMessage(toolCallID, name, content string) map[string]any {
	return map[string]any{
		"role":         "tool",
		"tool_call_id": toolCallID,
		"name":         name,
		"content":      content,
	}
}

// appendToolResults appends the assistant's message and the tool results to
// the conversation.
func appendToolResults(messages []map[string]any, assistant map[string]any, results []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(messages)+1+len(results))
	out = append(out, messages...)
	out = append(out, assistant)
	out = append(out, results...)
	return out
}
