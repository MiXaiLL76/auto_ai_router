package proxy

import (
	"errors"
	"io"
	"net/http"
	"strings"
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
	runner := p.newDeepSeekToolRunner(cred)
	raw, calls, usage, err := toolloop.CollectStreamToolCalls(resp)
	if err != nil {
		return nil, err
	}
	if len(calls) == 0 {
		// No tool call: the client gets the provider's stream as it is.
		replayToolStream(w, resp, raw)
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
		replayToolStream(w, resp, raw)
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

	// The final step's stream is what the client gets.
	replayToolStream(w, resp, current)
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
	raw, _, usage, err := toolloop.CollectStreamToolCalls(outcome.resp)
	if err != nil {
		return nil, nil, err
	}
	return raw, toolloop.UsageFromStreamUsage(usage), nil
}

// replayToolStream writes a buffered stream to the client unchanged.
func replayToolStream(w http.ResponseWriter, resp *http.Response, raw []byte) {
	header := w.Header()
	for name, values := range resp.Header {
		if strings.EqualFold(name, "Content-Length") {
			continue
		}
		for _, value := range values {
			header.Add(name, value)
		}
	}
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	header.Del("Content-Length")
	w.WriteHeader(http.StatusOK)
	// The provider's own SSE, relayed byte for byte: the client asked for a
	// stream and expects exactly what the provider sent. It is not content the
	// router composes, so it is not an XSS sink here.
	_, _ = w.Write(raw) //nolint:gosec // G705: the provider's stream, relayed verbatim
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
