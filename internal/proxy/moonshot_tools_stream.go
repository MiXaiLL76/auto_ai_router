package proxy

import (
	"errors"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
	"github.com/mixaill76/auto_ai_router/internal/sse"
	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// errMoonshotToolCycleAborted ends the cycle when the client went away.
//
// It is the shared sentinel (internal/toolloop.ErrAborted), so the router's
// dispatch checks it once for every provider's loop: a disconnect is not a
// failure of the loop.
var errMoonshotToolCycleAborted = toolloop.ErrAborted

// moonshotStreamCycleResult is what the streaming loop produced.
type moonshotStreamCycleResult struct {
	// body is the final answer's stream, in the upstream's own SSE form.
	body []byte
	// usage is the summed usage of every step.
	usage *converter.TokenUsage
	// totalTokens is the steps' token count, the streaming handler's own
	// accounting figure.
	totalTokens int
	// lastChunk is the last chunk of the final stream, which the streaming
	// log reads its figures from.
	lastChunk []byte
	// partial marks a body that is the fragment of the provider's FIRST
	// answer, cut short by a broken connection: it carries no tool call of
	// the router's, so it can be relayed to the client as the answer it
	// is. A step of the loop that broke is not partial in this sense --
	// it may carry the tool call the client never declared -- and is
	// never relayed.
	partial bool
}

// runMoonshotStreamToolLoop drives the router-run tool loop for a streaming
// request.
//
// A stream that asks for one of the router's tools cannot be relayed to the
// client as it is: the client would see a tool call for a tool it never
// declared. So the router runs the tool, asks the model again, and relays
// the final stream.
//
// The steps' streams are not relayed (the client asked for an answer, not
// for the tool calls), so only the last step's stream reaches the client.
func (p *Proxy) runMoonshotStreamToolLoop(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	body []byte,
	resp *http.Response,
	start time.Time,
	attemptedCreds map[string]bool,
) (*moonshotStreamCycleResult, error) {
	if !moonshotToolsEnabledFor(modelID, cred) {
		return nil, nil
	}
	result, err := p.runMoonshotStreamToolLoopBuffered(r, w, logCtx, prepared, cred, modelID, realModelID, body, resp, start, attemptedCreds)
	if err != nil {
		// The provider's stream broke part-way. The buffered loop hands the
		// bytes that arrived back with the error, and those bytes are the
		// answer so far: the model had already produced them. They are
		// relayed to the client here, as the stream they are -- a stream
		// that ends without a finish_reason is what a broken provider's
		// answer looks like, and the client can use the text it got.
		if result != nil && result.partial {
			p.replayToolStream(w, resp, r, modelID, logCtx, result.body)
		}
		return result, err
	}
	// The final step's stream is what the client gets.
	p.replayToolStream(w, resp, r, modelID, logCtx, result.body)
	return result, nil
}

// runMoonshotStreamToolLoopBuffered runs the loop and returns the final
// step's stream, without writing it to the client.
//
// The split matters for the Responses API: its client expects Responses-shaped
// SSE, and the provider answered in Chat Completions SSE. So the loop's
// result is converted on the way out, and the conversion needs the buffered
// stream rather than one already written.
func (p *Proxy) runMoonshotStreamToolLoopBuffered(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	body []byte,
	resp *http.Response,
	start time.Time,
	attemptedCreds map[string]bool,
) (*moonshotStreamCycleResult, error) {
	runner := p.newMoonshotToolRunner(cred)
	// The collector returns the router's OWN calls only.
	raw, owned, usage, err := toolloop.CollectStreamToolCalls(resp, runner)
	if err != nil {
		// The provider's stream broke part-way. The bytes that arrived
		// are the answer so far -- the model had already produced them,
		// and the client is waiting for exactly that -- so the partial
		// stream is handed over with the error instead of being dropped.
		//
		// Unless it carries a call of the ROUTER'S OWN: the client never
		// declared it, so relaying it would show the client a function
		// it does not know and no answer. That case is the failure the
		// client is told about.
		if len(raw) == 0 || len(owned) > 0 {
			return nil, err
		}
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		return &moonshotStreamCycleResult{
			body:        raw,
			usage:       moonshotUsageOf(streamUsage, nil),
			totalTokens: streamUsage.TotalTokens,
			lastChunk:   sse.LastPayload(raw),
			partial:     true,
		}, err
	}
	if len(owned) == 0 {
		// No tool call: the client gets the provider's stream as it is.
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		return &moonshotStreamCycleResult{
			body:        raw,
			usage:       moonshotUsageOf(streamUsage, nil),
			totalTokens: streamUsage.TotalTokens,
			lastChunk:   sse.LastPayload(raw),
		}, nil
	}
	if runner.client == nil {
		// No key: the tool cannot be run, so the stream is relayed as it is.
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		return &moonshotStreamCycleResult{
			body:        raw,
			usage:       moonshotUsageOf(streamUsage, nil),
			totalTokens: streamUsage.TotalTokens,
			lastChunk:   sse.LastPayload(raw),
		}, nil
	}

	cycle, err := toolloop.NewCycle(body, runner)
	if err != nil {
		return nil, err
	}
	total := toolloop.UsageFromStreamUsage(usage)
	current := raw

	for step := 1; step < p.moonshotToolsMaxSteps; step++ {
		// The last step of the budget asks the model to answer with what it
		// has found, with the tools taken away: a tool call for the
		// router's own function must never reach the client, which never
		// declared it.
		last := step == p.moonshotToolsMaxSteps-1
		var next []byte
		var err error
		if last {
			next, err = cycle.FinalBodyFromToolCalls(r.Context(), owned)
		} else {
			next, err = cycle.NextBodyFromToolCalls(r.Context(), owned)
		}
		if errors.Is(err, toolloop.ErrNothingToRun) {
			// The step's calls are none of the router's own, so there is
			// nothing to run and nothing to add to the conversation: the
			// loop ends here and the client gets the answer as it is.
			//
			// Sending the request again would be sending the very same
			// request -- the provider would answer the same way, and the
			// client would be billed for both sends.
			break
		}
		if err != nil {
			return moonshotStreamResult(current, total, runner), err
		}
		stepBody, stepUsage, err := p.runMoonshotStreamToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			// A tool that already ran and returned results is billed even
			// though the model that followed it failed, so the result is
			// returned with the error rather than dropped.
			return moonshotStreamResult(current, total, runner), err
		}
		total.Add(stepUsage)
		current = stepBody
		owned = toolloop.CallsFromStream(stepBody, runner)
		if len(owned) == 0 {
			break
		}
	}

	return moonshotStreamResult(current, total, runner), nil
}

// runMoonshotStreamToolStep sends one upstream request of the streaming loop
// and returns the response's stream and its usage.
func (p *Proxy) runMoonshotStreamToolStep(
	r *http.Request,
	w http.ResponseWriter,
	logCtx *RequestLogContext,
	prepared *orchestratedRequest,
	cred *config.CredentialConfig,
	modelID, realModelID string,
	body []byte,
	start time.Time,
	attemptedCreds map[string]bool,
) ([]byte, *toolloop.Usage, error) {
	embedFanOutReplies := &embedContentFanOutReplies{}
	outcome, done, isClientGone, shouldRetry, _, transportErr := p.sendToCredential(
		r, w, logCtx, cred, prepared, modelID, realModelID, body,
		true, false, false, 0, start, attemptedCreds, embedFanOutReplies,
	)
	if done || isClientGone {
		return nil, nil, errMoonshotToolCycleAborted
	}
	if shouldRetry {
		return nil, nil, errors.New("moonshottools: the upstream call failed: " + transportErr.Error())
	}
	if outcome.resp == nil {
		return nil, nil, errors.New("moonshottools: the upstream returned no response")
	}
	if outcome.resp.StatusCode < 200 || outcome.resp.StatusCode >= 300 {
		responseBody, err := readAndCloseResponseBody(outcome.resp)
		if err != nil {
			return nil, nil, err
		}
		return nil, nil, &upstreamStatusError{status: outcome.resp.StatusCode, body: responseBody}
	}
	raw, usage, err := toolloop.ReadStream(outcome.resp)
	if err != nil {
		return nil, nil, err
	}
	return raw, toolloop.UsageFromStreamUsage(usage), nil
}

// moonshotUsageOf converts the loop's usage into the router's own usage
// shape, with the billable tool counts.
func moonshotUsageOf(total *toolloop.Usage, billable map[string]int) *converter.TokenUsage {
	usage := &converter.TokenUsage{
		PromptTokens:      total.PromptTokens,
		CompletionTokens:  total.CompletionTokens,
		CachedInputTokens: total.CachedInputTokens,
		ReasoningTokens:   total.ReasoningTokens,
	}
	if billable != nil {
		// The tool counters go into the Moonshot fields alone: they are
		// Moonshot's own calls, priced by the tool price document, and the
		// built-in tools' counters are the model's own -- counting the
		// router's calls in both would charge the client twice.
		usage.MoonshotSearchRequests = billable[moonshottools.SearchToolName]
		usage.MoonshotSearchProRequests = billable[moonshottools.SearchProToolName]
		usage.MoonshotFetchRequests = billable[moonshottools.FetchToolName]
	}
	return usage
}

// moonshotStreamResult is the streaming loop's outcome.
func moonshotStreamResult(body []byte, total *toolloop.Usage, runner *moonshotToolRunner) *moonshotStreamCycleResult {
	return &moonshotStreamCycleResult{
		body:        body,
		usage:       moonshotUsageOf(total, runner.Billable()),
		totalTokens: total.TotalTokens,
		lastChunk:   sse.LastPayload(body),
	}
}
