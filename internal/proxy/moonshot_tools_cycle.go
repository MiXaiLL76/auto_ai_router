package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// moonshotToolCycle drives the router-run tool loop for one client request.
//
// The loop is: ask the model -> the model calls the router's search/fetch
// function -> the router runs it -> the result goes back as a tool result ->
// the model continues. Each round is another upstream request, so the steps
// are bounded (server.moonshot_tools.max_steps) and every round's usage is counted.
type moonshotToolCycle struct {
	// runner executes the tools.
	runner *moonshotToolRunner
	// body is the current request body (rewritten between steps).
	body []byte
	// messages is the conversation so far.
	messages []map[string]any
}

// newMoonshotToolCycle prepares a cycle for a request body.
func newMoonshotToolCycle(body []byte, runner *moonshotToolRunner) (*moonshotToolCycle, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("moonshottools: parsing request body: %w", err)
	}
	rawMessages, ok := fields["messages"]
	if !ok {
		return &moonshotToolCycle{runner: runner, body: body}, nil
	}
	var messages []map[string]any
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return nil, fmt.Errorf("moonshottools: parsing messages: %w", err)
	}
	return &moonshotToolCycle{runner: runner, body: body, messages: messages}, nil
}

// nextBody builds the body for the next upstream call: the conversation with
// the assistant's tool-call message and the tool results appended.
func (cycle *moonshotToolCycle) nextBody(responseBody []byte) ([]byte, error) {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return nil, fmt.Errorf("moonshottools: parsing response: %w", err)
	}
	assistant := assistantToolCallMessage(response)
	results := make([]map[string]any, 0, len(response.Choices))
	if len(response.Choices) > 0 {
		for _, call := range response.Choices[0].Message.ToolCalls {
			content, ok := cycle.runner.state.toolResults[call.ID]
			if !ok {
				content = cycle.runTool(call)
				cycle.runner.state.toolResults[call.ID] = content
			}
			results = append(results, toolResultMessage(call.ID, call.Function.Name, content))
		}
	}
	cycle.messages = appendToolResults(cycle.messages, assistant, results)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(cycle.body, &fields); err != nil {
		return nil, fmt.Errorf("moonshottools: parsing request body: %w", err)
	}
	encodedMessages, err := json.Marshal(cycle.messages)
	if err != nil {
		return nil, fmt.Errorf("moonshottools: encoding messages: %w", err)
	}
	fields["messages"] = encodedMessages
	next, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("moonshottools: encoding request body: %w", err)
	}
	cycle.body = next
	return next, nil
}

// runTool executes one tool call and returns the result to send back.
func (cycle *moonshotToolCycle) runTool(call openai.OpenAIToolCall) string {
	return cycle.runToolByName(call.Function.Name, call.Function.Arguments)
}

// runToolByName executes a tool by its name and arguments, and returns the
// result to send back.
//
// Only the router's own tools are executed. A client's own function tools
// are left alone: the upstream serves them, and the router does not run a
// client's tools on its behalf.
func (cycle *moonshotToolCycle) runToolByName(name, arguments string) string {
	switch name {
	case moonshottools.SearchToolName:
		return cycle.runner.runSearch(context.Background(), arguments)
	case moonshottools.SearchProToolName:
		return cycle.runner.runSearchPro(context.Background(), arguments)
	case moonshottools.FetchToolName:
		return cycle.runner.runFetch(context.Background(), arguments)
	default:
		return string(moonshottools.EncodeToolError(fmt.Sprintf("unknown tool %q", name)))
	}
}

// moonshotToolUsage is the usage the cycle accumulated over its steps.
//
// The router's own merge helpers replace a counter with the later step's
// value rather than adding them up -- correct for a stream, where the last
// chunk carries the total, and wrong for a tool loop, where every step is a
// separate upstream request that the client is charged for. So the cycle sums
// the token counts and the tool counters itself.
type moonshotToolUsage struct {
	// promptTokens is the sum of every step's prompt tokens.
	promptTokens int
	// completionTokens is the sum of every step's completion tokens.
	completionTokens int
	// cachedInputTokens is the sum of every step's cached input tokens.
	cachedInputTokens int
	// reasoningTokens is the sum of every step's reasoning tokens.
	reasoningTokens int
	// totalTokens is the sum of every step's total tokens.
	totalTokens int
	// webSearchRequests counts the billable Basic searches.
	webSearchRequests int
	// webSearchProRequests counts the billable Pro searches.
	webSearchProRequests int
	// webFetchRequests counts the billable fetches.
	webFetchRequests int
}

// add folds one step's usage into the accumulator.
func (u *moonshotToolUsage) add(other *moonshotToolUsage) {
	if other == nil {
		return
	}
	u.promptTokens += other.promptTokens
	u.completionTokens += other.completionTokens
	u.cachedInputTokens += other.cachedInputTokens
	u.reasoningTokens += other.reasoningTokens
	u.totalTokens += other.totalTokens
	u.webSearchRequests += other.webSearchRequests
	u.webSearchProRequests += other.webSearchProRequests
	u.webFetchRequests += other.webFetchRequests
}

// usageFromMoonshotResponse reads the token counters out of a Chat
// Completions response body.
func usageFromMoonshotResponse(body []byte) *moonshotToolUsage {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return &moonshotToolUsage{}
	}
	usage := &moonshotToolUsage{}
	if response.Usage != nil {
		usage.promptTokens = response.Usage.PromptTokens
		usage.completionTokens = response.Usage.CompletionTokens
		usage.totalTokens = response.Usage.TotalTokens
		if response.Usage.CompletionTokensDetails != nil {
			usage.reasoningTokens = response.Usage.CompletionTokensDetails.ReasoningTokens
		}
		if response.Usage.PromptTokensDetails != nil {
			usage.cachedInputTokens = response.Usage.PromptTokensDetails.CachedTokens
		}
	}
	return usage
}

// toMoonshotToolUsage converts a step's usage into the loop's own
// accumulator shape.
func (u *moonshotToolUsage) toMoonshotToolUsage() *moonshotToolUsage {
	return &moonshotToolUsage{
		promptTokens:         u.promptTokens,
		completionTokens:     u.completionTokens,
		cachedInputTokens:    u.cachedInputTokens,
		reasoningTokens:      u.reasoningTokens,
		totalTokens:          u.totalTokens,
		webSearchRequests:    u.webSearchRequests,
		webSearchProRequests: u.webSearchProRequests,
		webFetchRequests:     u.webFetchRequests,
	}
}

// toTokenUsage converts the loop's accumulated usage into the
// router's own usage shape, so the regular billing path prices it like any
// other response's.
//
// The tool counters go into the Moonshot fields alone: they are Moonshot's
// own calls, priced by the tool price document, and the built-in tools'
// counters (WebSearchRequests, WebFetchRequests) are the model's own -- the
// provider never reported a built-in search here, so counting the router's
// calls in both would charge the client twice.
func (u *moonshotToolUsage) toTokenUsage() *converter.TokenUsage {
	return &converter.TokenUsage{
		PromptTokens:              u.promptTokens,
		CompletionTokens:          u.completionTokens,
		CachedInputTokens:         u.cachedInputTokens,
		ReasoningTokens:           u.reasoningTokens,
		MoonshotSearchRequests:    u.webSearchRequests,
		MoonshotSearchProRequests: u.webSearchProRequests,
		MoonshotFetchRequests:     u.webFetchRequests,
	}
}

// nextBodyFromToolCalls builds the next request body from a streamed tool
// call, the streaming counterpart of the cycle's nextBody.
func (cycle *moonshotToolCycle) nextBodyFromToolCalls(calls []moonshotStreamToolCall) ([]byte, error) {
	assistant := map[string]any{"role": "assistant"}
	toolCalls := make([]map[string]any, 0, len(calls))
	results := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		switch call.Name {
		case moonshottools.SearchToolName, moonshottools.SearchProToolName, moonshottools.FetchToolName:
		default:
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
		content, ok := cycle.runner.state.toolResults[call.ID]
		if !ok {
			content = cycle.runToolByName(call.Name, call.Arguments)
			cycle.runner.state.toolResults[call.ID] = content
		}
		results = append(results, toolResultMessage(call.ID, call.Name, content))
	}
	if len(toolCalls) == 0 {
		return cycle.body, nil
	}
	assistant["tool_calls"] = toolCalls
	cycle.messages = appendToolResults(cycle.messages, assistant, results)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(cycle.body, &fields); err != nil {
		return nil, err
	}
	encodedMessages, err := json.Marshal(cycle.messages)
	if err != nil {
		return nil, err
	}
	fields["messages"] = encodedMessages
	next, err := json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	cycle.body = next
	return next, nil
}

// errMoonshotToolCycleAborted ends the cycle when the client went away.
var errMoonshotToolCycleAborted = errors.New("moonshottools: client disconnected")
