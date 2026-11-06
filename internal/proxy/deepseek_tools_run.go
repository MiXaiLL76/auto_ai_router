package proxy

import (
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// deepSeekToolRun is the outcome of the router-run tool loop.
type deepSeekToolRun struct {
	// body is the model's final answer, in the upstream's own Chat shape.
	body []byte
	// usage is the summed usage of every step of the loop.
	usage *converter.TokenUsage
	// searchCalls and fetchCalls are the billable tool counts.
	searchCalls int
	fetchCalls  int
}

// runDeepSeekToolLoop drives the router-run search/fetch loop for a
// non-streaming request.
//
// The upstream answered with a tool call: the router runs the tool, appends
// the result to the conversation and asks the model again, until the model
// answers without asking for a tool (or the step budget runs out).
//
// It returns the final body and the summed usage. A nil result means the
// request needs no loop (the model did not call a tool) and the caller keeps
// what it already has.
func (p *Proxy) runDeepSeekToolLoop(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	firstBody []byte,
	firstResponse []byte,
	start time.Time,
	attemptedCreds map[string]bool,
) (*deepSeekToolRun, error) {
	if !deepSeekToolsEnabledFor(modelID, cred) {
		return nil, nil
	}
	calls := deepSeekToolCallsFromResponse(firstResponse)
	if len(calls) == 0 {
		return nil, nil
	}

	runner := &deepSeekToolRunner{
		client:  p.deepSeekToolsClient(cred),
		quota:   p.deepSeekToolsQuota,
		account: deepSeekAccountKey(cred),
		state:   newDeepSeekToolState(),
	}
	if runner.client == nil {
		// No DeepSeek client could be built for this credential (no key):
		// the loop cannot run, so the model is answered with its tool call
		// untouched rather than with an invented tool result.
		return nil, nil
	}

	cycle, err := newDeepSeekToolCycle(firstBody, runner)
	if err != nil {
		return nil, err
	}
	total := usageFromResponse(firstResponse)
	current := firstResponse

	for step := 1; step < maxDeepSeekToolSteps; step++ {
		next, err := cycle.nextBody(current)
		if err != nil {
			return nil, err
		}
		responseBody, err := p.runDeepSeekToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			return nil, err
		}
		total.add(usageFromResponse(responseBody))
		current = responseBody
		if len(deepSeekToolCallsFromResponse(responseBody)) == 0 {
			break
		}
	}

	total.webSearchRequests = runner.state.billableSearch
	total.webFetchRequests = runner.state.billableFetch
	return &deepSeekToolRun{
		body:        current,
		usage:       total.toTokenUsage(),
		searchCalls: runner.state.billableSearch,
		fetchCalls:  runner.state.billableFetch,
	}, nil
}

// runDeepSeekToolStep sends one upstream request of the loop and returns the
// response body.
//
// It reuses the router's own send path, so the upstream request is built,
// authenticated and proxied exactly as the first attempt of a normal request
// would be -- including the credential's proxy.
func (p *Proxy) runDeepSeekToolStep(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	body []byte,
	start time.Time,
	attemptedCreds map[string]bool,
) ([]byte, error) {
	embedFanOutReplies := &embedContentFanOutReplies{}
	outcome, done, isClientGone, shouldRetry, _, transportErr := p.sendToCredential(
		r, w, logCtx, cred, prepared, modelID, realModelID, body,
		prepared.streaming, false, false, 0, start, attemptedCreds, embedFanOutReplies,
	)
	if done || isClientGone {
		return nil, errDeepSeekToolCycleAborted
	}
	if shouldRetry {
		return nil, errors.New("deepseektools: the upstream call failed: " + transportErr.Error())
	}
	if outcome.resp == nil {
		return nil, errors.New("deepseektools: the upstream returned no response")
	}
	responseBody, err := readAndCloseResponseBody(outcome.resp)
	if err != nil {
		return nil, err
	}
	if outcome.resp.StatusCode < 200 || outcome.resp.StatusCode >= 300 {
		return nil, &upstreamStatusError{status: outcome.resp.StatusCode, body: responseBody}
	}
	return responseBody, nil
}

// deepSeekToolsClient builds the DeepSeek Search/Fetch client for a
// credential, or nil when the credential cannot make those calls.
func (p *Proxy) deepSeekToolsClient(cred *config.CredentialConfig) *deepseektools.Client {
	if cred == nil || cred.APIKey == "" {
		return nil
	}
	return deepseektools.NewClient(deepseektools.ClientConfig{
		BaseURL:    cred.BaseURL,
		APIKey:     cred.APIKey,
		Timeout:    p.deepSeekToolsTimeout,
		ProxyURL:   cred.ProxyURL,
		MaxRetries: p.deepSeekToolsMaxRetries,
	})
}

// toTokenUsage converts the loop's accumulated usage into the router's own
// usage shape, so the regular billing path prices it like any other
// response's.
func (u *deepSeekToolUsage) toTokenUsage() *converter.TokenUsage {
	return &converter.TokenUsage{
		PromptTokens:         u.promptTokens,
		CompletionTokens:     u.completionTokens,
		CachedInputTokens:    u.cachedInputTokens,
		ReasoningTokens:      u.reasoningTokens,
		WebSearchRequests:    u.webSearchRequests,
		WebFetchRequests:     u.webFetchRequests,
		WebSearchContextSize: u.webSearchContextSize,
	}
}

// openAIToolCallNames lists the tool names of a response's tool calls, for a
// log line.
func openAIToolCallNames(responseBody []byte) []string {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return nil
	}
	if len(response.Choices) == 0 {
		return nil
	}
	names := make([]string, 0, len(response.Choices[0].Message.ToolCalls))
	for _, call := range response.Choices[0].Message.ToolCalls {
		names = append(names, call.Function.Name)
	}
	return names
}
