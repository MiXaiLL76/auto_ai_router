package proxy

import (
	"errors"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// deepSeekToolRun is the outcome of the router-run tool loop.
type deepSeekToolRun struct {
	// bodyBytes is the model's final answer, in the upstream's own Chat shape.
	bodyBytes []byte
	// usageValue is the summed usage of every step of the loop.
	usageValue *converter.TokenUsage
	// searchCount and fetchCount are the billable tool counts.
	searchCount int
	fetchCount  int
}

// body is the model's final answer.
func (r *deepSeekToolRun) body() []byte { return r.bodyBytes }

// usage is the summed usage of every step of the loop.
func (r *deepSeekToolRun) usage() *converter.TokenUsage { return r.usageValue }

// searchCalls is the billable Basic search count.
func (r *deepSeekToolRun) searchCalls() int { return r.searchCount }

// searchProCalls is the billable Pro search count: DeepSeek has none.
func (r *deepSeekToolRun) searchProCalls() int { return 0 }

// fetchCalls is the billable fetch count.
func (r *deepSeekToolRun) fetchCalls() int { return r.fetchCount }

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
	runner := p.newDeepSeekToolRunner(cred)
	calls := toolloop.CallsFromResponse(firstResponse, runner)
	if len(calls) == 0 {
		return nil, nil
	}
	if runner.client == nil {
		// No DeepSeek client could be built for this credential (no key):
		// the loop cannot run, so the model is answered with its tool call
		// untouched rather than with an invented tool result.
		return nil, nil
	}

	cycle, err := toolloop.NewCycle(firstBody, runner)
	if err != nil {
		return nil, err
	}
	total := toolloop.UsageFromResponse(firstResponse)
	current := firstResponse

	for step := 1; step < p.deepSeekToolsMaxSteps; step++ {
		next, err := cycle.NextBody(r.Context(), current)
		if err != nil {
			return deepSeekRunResult(current, total, runner), err
		}
		responseBody, err := p.runDeepSeekToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			// A tool that already ran and returned results is billed even
			// though the model that followed it failed, so the result is
			// returned with the error rather than dropped.
			return deepSeekRunResult(current, total, runner), err
		}
		total.Add(toolloop.UsageFromResponse(responseBody))
		current = responseBody
		if len(toolloop.CallsFromResponse(responseBody, runner)) == 0 {
			break
		}
	}

	return deepSeekRunResult(current, total, runner), nil
}

// newDeepSeekToolRunner builds the runner for a credential.
func (p *Proxy) newDeepSeekToolRunner(cred *config.CredentialConfig) *deepSeekToolRunner {
	return &deepSeekToolRunner{
		client:  p.deepSeekToolsClient(cred),
		logger:  p.logger,
		quota:   p.deepSeekToolsQuota,
		account: deepSeekAccountKey(cred),
		state:   newDeepSeekToolState(),
	}
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
		Logger:     p.logger,
		// The cached client: the connections to DeepSeek survive between
		// the requests instead of being opened and dropped each time.
		HTTPClient: p.deepSeekToolsClients.Client(cred.BaseURL, cred.ProxyURL, p.deepSeekToolsTimeout),
	})
}

// deepSeekRunResult is the loop's outcome: the body so far, the usage every
// step accumulated, and the tool counters.
//
// The counters are read from the runner, not from the accumulated usage, so
// a step that fails leaves the loop early and the search that already ran
// and returned results is still billed -- "a successful tool counts even when
// the model that follows it fails".
func deepSeekRunResult(body []byte, total *toolloop.Usage, runner *deepSeekToolRunner) *deepSeekToolRun {
	billable := runner.Billable()
	usage := &converter.TokenUsage{
		PromptTokens:      total.PromptTokens,
		CompletionTokens:  total.CompletionTokens,
		CachedInputTokens: total.CachedInputTokens,
		ReasoningTokens:   total.ReasoningTokens,
		WebSearchRequests: billable[deepseektools.SearchToolName],
		WebFetchRequests:  billable[deepseektools.FetchToolName],
	}
	return &deepSeekToolRun{
		bodyBytes:   body,
		usageValue:  usage,
		searchCount: usage.WebSearchRequests,
		fetchCount:  usage.WebFetchRequests,
	}
}
