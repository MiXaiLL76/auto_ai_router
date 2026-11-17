package proxy

import (
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// moonshotToolRun is the outcome of the router-run tool loop.
type moonshotToolRun struct {
	// body is the model's final answer, in the upstream's own Chat shape.
	body []byte
	// usage is the summed usage of every step of the loop.
	usage *converter.TokenUsage
	// searchCalls, searchProCalls and fetchCalls are the billable tool
	// counts.
	searchCalls    int
	searchProCalls int
	fetchCalls     int
}

// runMoonshotToolLoop drives the router-run search/fetch loop for a
// non-streaming request.
//
// The upstream answered with a tool call: the router runs the tool, appends
// the result to the conversation and asks the model again, until the model
// answers without asking for a tool (or the step budget runs out).
//
// It returns the final body and the summed usage. A nil result means the
// request needs no loop (the model did not call a tool) and the caller keeps
// what it already has.
func (p *Proxy) runMoonshotToolLoop(
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
) (*moonshotToolRun, error) {
	if !moonshotToolsEnabledFor(modelID, cred) {
		return nil, nil
	}
	calls := moonshotToolCallsFromResponse(firstResponse)
	if len(calls) == 0 {
		return nil, nil
	}

	runner := &moonshotToolRunner{
		client:  p.moonshotToolsClient(cred),
		logger:  p.logger,
		quota:   p.moonshotToolsQuota,
		account: moonshotAccountKey(cred),
		state:   newMoonshotToolState(),
	}
	if runner.client == nil {
		// No Moonshot client could be built for this credential (no key):
		// the loop cannot run, so the model is answered with its tool call
		// untouched rather than with an invented tool result.
		return nil, nil
	}

	cycle, err := newMoonshotToolCycle(firstBody, runner)
	if err != nil {
		return nil, err
	}
	total := usageFromMoonshotResponse(firstResponse)
	current := firstResponse

	for step := 1; step < p.moonshotToolsMaxSteps; step++ {
		next, err := cycle.nextBody(current)
		if err != nil {
			return cycle.runResult(current, total, runner), err
		}
		responseBody, err := p.runMoonshotToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			// A tool that already ran and returned results is billed even
			// though the model that followed it failed, so the result is
			// returned with the error rather than dropped.
			return cycle.runResult(current, total, runner), err
		}
		total.add(usageFromMoonshotResponse(responseBody))
		current = responseBody
		if len(moonshotToolCallsFromResponse(responseBody)) == 0 {
			break
		}
	}

	return cycle.runResult(current, total, runner), nil
}

// runMoonshotToolStep sends one upstream request of the loop and returns
// the response body.
//
// It reuses the router's own send path, so the upstream request is built,
// authenticated and proxied exactly as the first attempt of a normal request
// would be -- including the credential's proxy.
func (p *Proxy) runMoonshotToolStep(
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
		return nil, errMoonshotToolCycleAborted
	}
	if shouldRetry {
		return nil, errors.New("moonshottools: the upstream call failed: " + transportErr.Error())
	}
	if outcome.resp == nil {
		return nil, errors.New("moonshottools: the upstream returned no response")
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

// moonshotToolsClient builds the Moonshot Search/Search Pro/Fetch client for
// a credential, or nil when the credential cannot make those calls.
func (p *Proxy) moonshotToolsClient(cred *config.CredentialConfig) *moonshottools.Client {
	if cred == nil || cred.APIKey == "" {
		return nil
	}
	return moonshottools.NewClient(moonshottools.ClientConfig{
		BaseURL:    cred.BaseURL,
		APIKey:     cred.APIKey,
		Timeout:    p.moonshotToolsTimeout,
		ProxyURL:   cred.ProxyURL,
		MaxRetries: p.moonshotToolsMaxRetries,
		Logger:     p.logger,
	})
}

// runResult is the loop's outcome: the body so far, the usage every step
// accumulated, and the tool counters.
//
// The counters are read from the runner, not from the accumulated usage, so
// a step that fails leaves the loop early and the search that already ran
// and returned results is still billed -- "a successful tool counts even when
// the model that follows it fails".
func (cycle *moonshotToolCycle) runResult(body []byte, usage *moonshotToolUsage, runner *moonshotToolRunner) *moonshotToolRun {
	usage.webSearchRequests = runner.state.billableSearch
	usage.webSearchProRequests = runner.state.billableSearchPro
	usage.webFetchRequests = runner.state.billableFetch
	return &moonshotToolRun{
		body:           body,
		usage:          usage.toTokenUsage(),
		searchCalls:    runner.state.billableSearch,
		searchProCalls: runner.state.billableSearchPro,
		fetchCalls:     runner.state.billableFetch,
	}
}

// moonshotToolCallsFromResponse reads the tool calls out of a Chat
// Completions response, keeping only the ones the adapter owns.
//
// A client's own function tools are left out: the upstream serves them, and
// the router does not run a client's tools on its behalf.
func moonshotToolCallsFromResponse(body []byte) []openai.OpenAIToolCall {
	var response openai.OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil
	}
	if len(response.Choices) == 0 {
		return nil
	}
	calls := make([]openai.OpenAIToolCall, 0, len(response.Choices[0].Message.ToolCalls))
	for _, call := range response.Choices[0].Message.ToolCalls {
		switch call.Function.Name {
		case moonshottools.SearchToolName, moonshottools.SearchProToolName, moonshottools.FetchToolName:
			calls = append(calls, call)
		}
	}
	return calls
}
