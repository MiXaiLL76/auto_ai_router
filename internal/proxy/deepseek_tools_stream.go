package proxy

import (
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
	"github.com/mixaill76/auto_ai_router/internal/sse"
	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// deepSeekStreamCycleResult is what the streaming loop produced.
type deepSeekStreamCycleResult struct {
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

// runDeepSeekStreamToolLoop drives the router-run tool loop for a streaming
// request.
//
// A stream that asks for one of the router's tools cannot be relayed to the
// client as it is: the client would see a tool call for a tool it never
// declared. So the router runs the tool, asks the model again, and relays
// the final stream.
//
// The steps' streams are not relayed (the client asked for an answer, not
// for the tool calls), so only the last step's stream reaches the client.
func (p *Proxy) runDeepSeekStreamToolLoop(
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
) (*deepSeekStreamCycleResult, error) {
	if !deepSeekToolsEnabledFor(modelID, cred) {
		return nil, nil
	}
	result, err := p.runDeepSeekStreamToolLoopBuffered(r, w, logCtx, prepared, cred, modelID, realModelID, body, resp, start, attemptedCreds)
	if err != nil {
		// The provider's stream broke part-way. The buffered loop hands the
		// bytes that arrived back with the error, and those bytes are the
		// answer so far: the model had already produced them. They are
		// relayed to the client here, as the stream they are -- a stream
		// that ends without a finish_reason is what a broken provider's
		// answer looks like, and the client can use the text it got.
		if result != nil && result.partial {
			p.replayToolStream(w, r, modelID, logCtx, result.body)
		}
		return result, err
	}
	// The final step's stream is what the client gets.
	p.replayToolStream(w, r, modelID, logCtx, result.body)
	return result, nil
}

// runDeepSeekStreamToolLoopBuffered runs the loop and returns the final
// step's stream, without writing it to the client.
//
// The split matters for the Responses API: its client expects Responses-shaped
// SSE, and the provider answered in Chat Completions SSE. So the loop's
// result is converted on the way out, and the conversion needs the buffered
// stream rather than one already written.
func (p *Proxy) runDeepSeekStreamToolLoopBuffered(
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
) (*deepSeekStreamCycleResult, error) {
	runner := p.newDeepSeekToolRunner(cred)
	// The collector returns the router's OWN calls only: a call of the
	// client's own (a coding agent's read_file, say) is the client's
	// business, and the router relays it as it is. Counting it made the
	// router send the very same request to the provider a second time --
	// the client never noticed (it got the second answer's stream), but the
	// request was billed twice.
	raw, calls, usage, err := toolloop.CollectStreamToolCalls(resp, runner)
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
		//
		// A call whose name never arrived counts as one of ours: the
		// stream broke between the piece with the name and the pieces
		// with its arguments, and an unnamed call cannot be told apart
		// from the router's own.
		if len(raw) == 0 || len(calls) > 0 || toolloop.HasUnnamedToolCall(raw) {
			return nil, err
		}
		// The client went away: the upstream request rode the client's
		// context, so its read fails with the cancellation when the
		// client disconnects. That is not a broken provider -- there is
		// nobody left to answer -- and the loop must stop without
		// writing to the dead connection.
		if isClientDisconnectError(err) {
			return nil, errDeepSeekToolCycleAborted
		}
		// The provider reported no usage (its stream broke before the
		// usage frame), so the loop's own counts are empty. The answer's
		// size is counted locally instead, from the text that was
		// streamed: an empty usage would make the spend row zero, and
		// the client got a real answer.
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		loopUsage := deepSeekUsageOf(streamUsage, nil)
		loopUsage.CompletionTokens = p.countStreamedTextTokens(modelID, raw)
		return &deepSeekStreamCycleResult{
			body:        raw,
			usage:       loopUsage,
			totalTokens: streamUsage.TotalTokens,
			lastChunk:   sse.LastPayload(raw),
			partial:     true,
		}, err
	}
	if len(calls) == 0 {
		// No tool call of ours: the client gets the provider's stream as it is.
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		return &deepSeekStreamCycleResult{
			body:        raw,
			usage:       deepSeekUsageOf(streamUsage, nil),
			totalTokens: streamUsage.TotalTokens,
			lastChunk:   sse.LastPayload(raw),
		}, nil
	}
	if runner.client == nil {
		// No key: the tool cannot be run, so the stream is relayed as it is.
		streamUsage := toolloop.UsageFromStreamUsage(usage)
		return &deepSeekStreamCycleResult{
			body:        raw,
			usage:       deepSeekUsageOf(streamUsage, nil),
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

	for step := 1; step < p.deepSeekToolsMaxSteps; step++ {
		// The last step of the budget asks the model to answer with what it
		// has found, with the tools taken away: a tool call for the
		// router's own function must never reach the client, which never
		// declared it.
		last := step == p.deepSeekToolsMaxSteps-1
		var next []byte
		var err error
		if last {
			next, err = cycle.FinalBodyFromToolCalls(r.Context(), calls)
		} else {
			next, err = cycle.NextBodyFromToolCalls(r.Context(), calls)
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
			return deepSeekStreamResult(current, total, runner), err
		}
		stepBody, stepUsage, err := p.runDeepSeekStreamToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			// A tool that already ran and returned results is billed even
			// though the model that followed it failed, so the result is
			// returned with the error rather than dropped.
			return deepSeekStreamResult(current, total, runner), err
		}
		total.Add(stepUsage)
		current = stepBody
		nextCalls := toolloop.CallsFromStream(stepBody, runner)
		if len(nextCalls) == 0 {
			break
		}
		calls = nextCalls
	}

	return deepSeekStreamResult(current, total, runner), nil
}

// runDeepSeekStreamToolStep sends one upstream request of the streaming loop
// and returns the response's stream and its usage.
func (p *Proxy) runDeepSeekStreamToolStep(
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
		return nil, nil, errDeepSeekToolCycleAborted
	}
	if shouldRetry {
		return nil, nil, errors.New("deepseektools: the upstream call failed: " + transportErr.Error())
	}
	if outcome.resp == nil {
		return nil, nil, errors.New("deepseektools: the upstream returned no response")
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

// replayToolStream writes the buffered final step of the tool loop.
func (p *Proxy) replayToolStream(w http.ResponseWriter, r *http.Request, modelID string, logCtx *RequestLogContext, raw []byte) {
	header := w.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	header.Del("Content-Length")
	if logCtx != nil {
		applyCredentialResponseHeaders(w, logCtx.Credential)
	}
	finishFilter := p.newChatPostFinishFilter(r.Context(), header, endpointFromRequest(r), http.StatusOK, modelID, logCtx)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(finishFilter.filter(raw, true)) //nolint:gosec // G705: proxied SSE
}

// deepSeekUsageOf converts the loop's usage into the router's own usage
// shape, with the billable tool counts.
func deepSeekUsageOf(total *toolloop.Usage, billable map[string]int) *converter.TokenUsage {
	usage := &converter.TokenUsage{
		PromptTokens:      total.PromptTokens,
		CompletionTokens:  total.CompletionTokens,
		CachedInputTokens: total.CachedInputTokens,
		ReasoningTokens:   total.ReasoningTokens,
	}
	if billable != nil {
		usage.WebSearchRequests = billable[deepseektools.SearchToolName]
		usage.WebFetchRequests = billable[deepseektools.FetchToolName]
	}
	return usage
}

// deepSeekStreamResult is the streaming loop's outcome.
func deepSeekStreamResult(body []byte, total *toolloop.Usage, runner *deepSeekToolRunner) *deepSeekStreamCycleResult {
	return &deepSeekStreamCycleResult{
		body:        body,
		usage:       deepSeekUsageOf(total, runner.Billable()),
		totalTokens: total.TotalTokens,
		lastChunk:   sse.LastPayload(body),
	}
}

// readAndCloseResponseBody reads a response body and closes it.
func readAndCloseResponseBody(resp *http.Response) ([]byte, error) {
	defer func() { _ = resp.Body.Close() }()
	return io.ReadAll(resp.Body)
}
