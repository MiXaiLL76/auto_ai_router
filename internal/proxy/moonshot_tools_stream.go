package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// moonshotStreamToolCall is the tool call a streaming model asked for.
type moonshotStreamToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// collectMoonshotStreamToolCalls reads a Chat Completions SSE stream and
// returns the tool calls the model made in it.
//
// The stream is read whole: the router has to know whether the model asked
// for a tool before it can decide whether to relay the stream to the client
// as it is or to run the tool first. A stream that ends without a tool call
// is replayed to the client byte for byte, so the client sees exactly what
// the provider sent.
//
// It returns the stream's bytes, the tool calls and the usage the stream
// carried.
func collectMoonshotStreamToolCalls(resp *http.Response) ([]byte, []moonshotStreamToolCall, *openai.OpenAIUsage, error) {
	if resp == nil || resp.Body == nil {
		return nil, nil, nil, errors.New("moonshottools: no stream to read")
	}
	defer func() { _ = resp.Body.Close() }()

	var buffered bytes.Buffer
	if _, err := io.Copy(&buffered, resp.Body); err != nil {
		return nil, nil, nil, err
	}
	raw := buffered.Bytes()

	calls := map[string]*moonshotStreamToolCall{}
	var order []string
	usage := &openai.OpenAIUsage{}
	for _, payload := range splitSSEPayloads(raw, nil) {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *openai.OpenAIUsage `json:"usage"`
		}
		if err := json.Unmarshal(payload, &chunk); err != nil {
			continue
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			for _, delta := range choice.Delta.ToolCalls {
				// A streamed tool call's pieces arrive under the same index:
				// the id and the name only on the first piece, the arguments
				// split across the rest. The index is therefore the key --
				// the id is not (a later piece may carry it empty, and two
				// calls may share one id in a provider's stream).
				key := strconv.Itoa(delta.Index)
				call, seen := calls[key]
				if !seen {
					call = &moonshotStreamToolCall{ID: delta.ID}
					calls[key] = call
					order = append(order, key)
				}
				if delta.ID != "" {
					call.ID = delta.ID
				}
				if delta.Function.Name != "" {
					call.Name = delta.Function.Name
				}
				call.Arguments += delta.Function.Arguments
			}
		}
	}

	var owned []moonshotStreamToolCall
	for _, key := range order {
		call := calls[key]
		switch call.Name {
		case moonshottools.SearchToolName, moonshottools.SearchProToolName, moonshottools.FetchToolName:
			owned = append(owned, *call)
		}
	}
	return raw, owned, usage, nil
}

// usageFromMoonshotStreamUsage converts a stream's usage into the loop's
// accumulator shape.
func usageFromMoonshotStreamUsage(usage *openai.OpenAIUsage) *moonshotToolUsage {
	out := &moonshotToolUsage{}
	if usage == nil {
		return out
	}
	out.promptTokens = usage.PromptTokens
	out.completionTokens = usage.CompletionTokens
	out.totalTokens = usage.TotalTokens
	if usage.CompletionTokensDetails != nil {
		out.reasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
	}
	if usage.PromptTokensDetails != nil {
		out.cachedInputTokens = usage.PromptTokensDetails.CachedTokens
	}
	return out
}

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
}

// runMoonshotStreamToolLoop drives the router-run tool loop for a streaming
// request.
//
// A stream that asks for one of the router's tools cannot be relayed to
// the client as it is: the client would see a tool call for a tool it never
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
	raw, calls, usage, err := collectMoonshotStreamToolCalls(resp)
	if err != nil {
		return nil, err
	}
	if len(calls) == 0 {
		// No tool call: the client gets the provider's stream as it is.
		replayDeepSeekStream(w, resp, raw)
		streamUsage := usageFromMoonshotStreamUsage(usage)
		return &moonshotStreamCycleResult{
			body:        raw,
			usage:       streamUsage.toTokenUsage(),
			totalTokens: streamUsage.totalTokens,
			lastChunk:   lastChunkOf(raw),
		}, nil
	}

	runner := &moonshotToolRunner{
		client:  p.moonshotToolsClient(cred),
		logger:  p.logger,
		quota:   p.moonshotToolsQuota,
		account: moonshotAccountKey(cred),
		state:   newMoonshotToolState(),
	}
	if runner.client == nil {
		// No key: the tool cannot be run, so the stream is relayed as it is.
		replayDeepSeekStream(w, resp, raw)
		streamUsage := usageFromMoonshotStreamUsage(usage)
		return &moonshotStreamCycleResult{
			body:        raw,
			usage:       streamUsage.toTokenUsage(),
			totalTokens: streamUsage.totalTokens,
			lastChunk:   lastChunkOf(raw),
		}, nil
	}

	cycle, err := newMoonshotToolCycle(body, runner)
	if err != nil {
		return nil, err
	}
	total := usageFromMoonshotStreamUsage(usage)
	current := raw

	for step := 1; step < p.moonshotToolsMaxSteps; step++ {
		next, err := cycle.nextBodyFromToolCalls(calls)
		if err != nil {
			return nil, err
		}
		stepBody, stepUsage, err := p.runMoonshotStreamToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			return nil, err
		}
		total.add(stepUsage.toMoonshotToolUsage())
		current = stepBody
		nextCalls, err := moonshotToolCallsFromStream(stepBody)
		if err != nil {
			return nil, err
		}
		if len(nextCalls) == 0 {
			break
		}
		calls = nextCalls
	}

	// The final step's stream is what the client gets.
	replayDeepSeekStream(w, resp, current)
	total.webSearchRequests = runner.state.billableSearch
	total.webSearchProRequests = runner.state.billableSearchPro
	total.webFetchRequests = runner.state.billableFetch
	return &moonshotStreamCycleResult{
		body:        current,
		usage:       total.toTokenUsage(),
		totalTokens: total.totalTokens,
		lastChunk:   lastChunkOf(current),
	}, nil
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
) ([]byte, *moonshotToolUsage, error) {
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
	raw, _, usage, err := collectMoonshotStreamToolCalls(outcome.resp)
	if err != nil {
		return nil, nil, err
	}
	return raw, usageFromMoonshotStreamUsage(usage), nil
}

// moonshotToolCallsFromStream reads the tool calls out of a buffered stream.
func moonshotToolCallsFromStream(raw []byte) ([]moonshotStreamToolCall, error) {
	// The pieces of one streamed tool call arrive under the same index: the id
	// and the name only on the first piece, the arguments split across the
	// rest.
	calls := map[string]*moonshotStreamToolCall{}
	var order []string
	for _, payload := range splitSSEPayloads(raw, nil) {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(payload, &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			for _, delta := range choice.Delta.ToolCalls {
				key := strconv.Itoa(delta.Index)
				call, seen := calls[key]
				if !seen {
					call = &moonshotStreamToolCall{ID: delta.ID}
					calls[key] = call
					order = append(order, key)
				}
				if delta.ID != "" {
					call.ID = delta.ID
				}
				if delta.Function.Name != "" {
					call.Name = delta.Function.Name
				}
				call.Arguments += delta.Function.Arguments
			}
		}
	}

	var owned []moonshotStreamToolCall
	for _, key := range order {
		call := calls[key]
		switch call.Name {
		case moonshottools.SearchToolName, moonshottools.SearchProToolName, moonshottools.FetchToolName:
			owned = append(owned, *call)
		}
	}
	return owned, nil
}
