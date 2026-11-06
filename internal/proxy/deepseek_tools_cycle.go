package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// deepSeekToolCycle drives the router-run tool loop for one client request.
//
// The loop is: ask the model -> the model calls the router's search/fetch
// function -> the router runs it -> the result goes back as a tool result ->
// the model continues. Each round is another upstream request, so the steps
// are bounded (maxDeepSeekToolSteps) and every round's usage is counted.
type deepSeekToolCycle struct {
	// runner executes the tools.
	runner *deepSeekToolRunner
	// body is the current request body (rewritten between steps).
	body []byte
	// messages is the conversation so far.
	messages []map[string]any
	// steps counts the upstream requests made.
	steps int
}

// newDeepSeekToolCycle prepares a cycle for a request body.
func newDeepSeekToolCycle(body []byte, runner *deepSeekToolRunner) (*deepSeekToolCycle, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("deepseektools: parsing request body: %w", err)
	}
	rawMessages, ok := fields["messages"]
	if !ok {
		return &deepSeekToolCycle{runner: runner, body: body}, nil
	}
	var messages []map[string]any
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return nil, fmt.Errorf("deepseektools: parsing messages: %w", err)
	}
	return &deepSeekToolCycle{runner: runner, body: body, messages: messages}, nil
}

// needsAnotherStep reports whether the model asked for one of the adapter's
// tools.
func (cycle *deepSeekToolCycle) needsAnotherStep(responseBody []byte) bool {
	return len(deepSeekToolCallsFromResponse(responseBody)) > 0
}

// nextBody builds the body for the next upstream call: the conversation with
// the assistant's tool-call message and the tool results appended.
func (cycle *deepSeekToolCycle) nextBody(responseBody []byte) ([]byte, error) {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return nil, fmt.Errorf("deepseektools: parsing response: %w", err)
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
		return nil, fmt.Errorf("deepseektools: parsing request body: %w", err)
	}
	encodedMessages, err := json.Marshal(cycle.messages)
	if err != nil {
		return nil, fmt.Errorf("deepseektools: encoding messages: %w", err)
	}
	fields["messages"] = encodedMessages
	next, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("deepseektools: encoding request body: %w", err)
	}
	cycle.body = next
	return next, nil
}

// runTool executes one tool call and returns the result to send back.
func (cycle *deepSeekToolCycle) runTool(call openai.OpenAIToolCall) string {
	return cycle.runToolByName(call.Function.Name, call.Function.Arguments)
}

// runToolByName executes a tool by its name and arguments, and returns the
// result to send back.
func (cycle *deepSeekToolCycle) runToolByName(name, arguments string) string {
	switch name {
	case deepseektools.SearchToolName:
		return cycle.runner.runSearch(context.Background(), arguments)
	case deepseektools.FetchToolName:
		return cycle.runner.runFetch(context.Background(), arguments)
	default:
		return string(deepseektools.EncodeToolError(fmt.Sprintf("unknown tool %q", name)))
	}
}

// deepSeekToolCycleResult is what the cycle produced: the final response body
// and the tool usage the steps accumulated.
type deepSeekToolCycleResult struct {
	// body is the model's final answer (the last step's response).
	body []byte
	// usage is the summed usage of every step.
	usage *deepSeekToolUsage
}

// deepSeekToolUsage is the usage the cycle accumulated over its steps.
//
// The router's own merge helpers replace a counter with the later step's
// value rather than adding them up -- correct for a stream, where the last
// chunk carries the total, and wrong for a tool loop, where every step is a
// separate upstream request that the client is charged for. So the cycle sums
// the token counts and the tool counters itself.
type deepSeekToolUsage struct {
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
	// webSearchRequests counts the billable searches.
	webSearchRequests int
	// webFetchRequests counts the billable fetches.
	webFetchRequests int
	// webSearchContextSize is the search context size the request asked for.
	webSearchContextSize string
}

// add folds one step's usage into the accumulator.
func (u *deepSeekToolUsage) add(other *deepSeekToolUsage) {
	if other == nil {
		return
	}
	u.promptTokens += other.promptTokens
	u.completionTokens += other.completionTokens
	u.cachedInputTokens += other.cachedInputTokens
	u.reasoningTokens += other.reasoningTokens
	u.totalTokens += other.totalTokens
	u.webSearchRequests += other.webSearchRequests
	u.webFetchRequests += other.webFetchRequests
}

// usageFromResponse reads the token counters out of a Chat Completions
// response body.
func usageFromResponse(body []byte) *deepSeekToolUsage {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return &deepSeekToolUsage{}
	}
	usage := &deepSeekToolUsage{}
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

// deepSeekToolCycleConfig is what the cycle needs to run one step.
type deepSeekToolCycleConfig struct {
	// cred is the credential the model call goes to.
	cred *config.CredentialConfig
	// prepared is the orchestrated request, used to rebuild the upstream body.
	prepared *orchestratedRequest
	// modelID and realModelID name the model.
	modelID, realModelID string
	// streaming tells whether the client asked for a stream.
	streaming bool
}

// errDeepSeekToolCycleAborted ends the cycle when the client went away.
var errDeepSeekToolCycleAborted = errors.New("deepseektools: client disconnected")

// runStep sends one upstream request and returns the response body.
//
// It reuses the router's own send path (sendToCredential), so the upstream
// request is built, authenticated and proxied exactly as the first attempt
// of a normal request would be.
func (p *Proxy) runStep(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	cfg deepSeekToolCycleConfig,
	body []byte,
	attempt int,
	start time.Time,
	attemptedCreds map[string]bool,
) ([]byte, error) {
	embedFanOutReplies := &embedContentFanOutReplies{}
	outcome, done, isClientGone, shouldRetry, _, transportErr := p.sendToCredential(
		r, w, logCtx, cfg.cred, cfg.prepared, cfg.modelID, cfg.realModelID, body,
		cfg.streaming, false, false, attempt, start, attemptedCreds, embedFanOutReplies,
	)
	if done {
		// sendToCredential already wrote the client response (a fatal error).
		return nil, errDeepSeekToolCycleAborted
	}
	if isClientGone {
		return nil, errDeepSeekToolCycleAborted
	}
	if shouldRetry {
		return nil, fmt.Errorf("deepseektools: upstream attempt failed: %w", transportErr)
	}
	if outcome.resp == nil {
		return nil, fmt.Errorf("deepseektools: upstream returned no response")
	}
	if outcome.resp.StatusCode < 200 || outcome.resp.StatusCode >= 300 {
		// The upstream refused the request: relay its own body, so the client
		// sees the provider's error, not the router's.
		responseBody, err := readAndCloseResponseBody(outcome.resp)
		if err != nil {
			return nil, fmt.Errorf("deepseektools: reading upstream response: %w", err)
		}
		return nil, &upstreamStatusError{status: outcome.resp.StatusCode, body: responseBody}
	}
	responseBody, err := readAndCloseResponseBody(outcome.resp)
	if err != nil {
		return nil, fmt.Errorf("deepseektools: reading upstream response: %w", err)
	}
	return responseBody, nil
}

// upstreamStatusError is an upstream answer the router relays as it is.
type upstreamStatusError struct {
	status int
	body   []byte
}

func (e *upstreamStatusError) Error() string {
	return fmt.Sprintf("deepseektools: upstream returned HTTP %d", e.status)
}
