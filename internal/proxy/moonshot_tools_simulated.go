package proxy

import (
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// moonshotSimulatedLoop runs the router-run tool loop over the simulated
// (force_stream_for_ttft) non-streaming path.
//
// That path is the one a non-streaming client actually takes when the router
// forces stream:true upstream: the provider's SSE is consumed and reassembled
// into the Chat JSON the client asked for. A tool call the model made in that
// stream is therefore only visible after the reassembly -- and the client
// cannot be shown it (it declared a web_search, not the router's function),
// so the router runs the tool and asks the model again.
//
// It returns the reassembled body of the final step, the summed usage of
// every step, and the tool counters.
func (p *Proxy) moonshotSimulatedLoop(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	requestBody []byte,
	assembled []byte,
	start time.Time,
	attemptedCreds map[string]bool,
) ([]byte, *converter.TokenUsage, int, int, int, error) {
	if !moonshotToolsEnabledFor(modelID, cred) {
		return assembled, nil, 0, 0, 0, nil
	}
	calls := moonshotToolCallsFromResponse(assembled)
	if len(calls) == 0 {
		return assembled, nil, 0, 0, 0, nil
	}

	runner := &moonshotToolRunner{
		client:  p.moonshotToolsClient(cred),
		logger:  p.logger,
		quota:   p.moonshotToolsQuota,
		account: moonshotAccountKey(cred),
		state:   newMoonshotToolState(),
	}
	if runner.client == nil {
		// No key: the tool cannot be run, so the client's answer is the
		// tool call it already got.
		return assembled, nil, 0, 0, 0, nil
	}

	cycle, err := newMoonshotToolCycle(requestBody, runner)
	if err != nil {
		return assembled, nil, 0, 0, 0, err
	}
	total := usageFromMoonshotResponse(assembled)
	current := assembled

	for step := 1; step < p.moonshotToolsMaxSteps; step++ {
		next, err := cycle.nextBody(current)
		if err != nil {
			return cycle.result(current, total, runner, err)
		}
		body, usage, err := p.moonshotSimulatedStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			return cycle.result(current, total, runner, err)
		}
		total.add(usage)
		current = body
		if len(moonshotToolCallsFromResponse(body)) == 0 {
			break
		}
	}
	return cycle.result(current, total, runner, nil)
}

// moonshotSimulatedStep sends one upstream request of the simulated loop and
// returns the reassembled body and its usage.
//
// The request goes out the same way the first attempt did -- streamed, so
// that the provider's answer is reassembled the same way -- and the stream is
// read and reassembled here.
func (p *Proxy) moonshotSimulatedStep(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	body []byte,
	start time.Time,
	attemptedCreds map[string]bool,
) ([]byte, *moonshotToolUsage, error) {
	embedFanOutReplies := embedContentFanOutReplies{}
	outcome, done, isClientGone, shouldRetry, _, transportErr := p.sendToCredential(
		r, w, logCtx, cred, prepared, modelID, realModelID, body,
		true, false, false, 0, start, attemptedCreds, &embedFanOutReplies,
	)
	if done || isClientGone {
		return nil, nil, errMoonshotToolCycleAborted
	}
	if shouldRetry {
		return nil, nil, transportErr
	}
	if outcome.resp == nil {
		return nil, nil, errMoonshotToolCycleAborted
	}
	if outcome.resp.StatusCode < 200 || outcome.resp.StatusCode >= 300 {
		responseBody, err := readAndCloseResponseBody(outcome.resp)
		if err != nil {
			return nil, nil, err
		}
		return nil, nil, &upstreamStatusError{status: outcome.resp.StatusCode, body: responseBody}
	}
	raw, _, usage, err := collectMoonshotStreamToolCalls(outcome.resp)
	if err != nil {
		return nil, nil, err
	}
	return reassembleChatCompletion(raw), usageFromMoonshotStreamUsage(usage), nil
}

// result is the loop's outcome: the reassembled body, the summed usage, and
// the tool counters.
func (cycle *moonshotToolCycle) result(
	body []byte,
	total *moonshotToolUsage,
	runner *moonshotToolRunner,
	err error,
) ([]byte, *converter.TokenUsage, int, int, int, error) {
	total.webSearchRequests = runner.state.billableSearch
	total.webSearchProRequests = runner.state.billableSearchPro
	total.webFetchRequests = runner.state.billableFetch
	return body, total.toTokenUsage(),
		runner.state.billableSearch, runner.state.billableSearchPro, runner.state.billableFetch, err
}
