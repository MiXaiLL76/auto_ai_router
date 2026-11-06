package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// deepSeekStreamToolCall is the tool call a streaming model asked for.
type deepSeekStreamToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// collectDeepSeekStreamToolCalls reads a Chat Completions SSE stream and
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
func collectDeepSeekStreamToolCalls(resp *http.Response) ([]byte, []deepSeekStreamToolCall, *openai.OpenAIUsage, error) {
	if resp == nil || resp.Body == nil {
		return nil, nil, nil, errors.New("deepseektools: no stream to read")
	}
	defer func() { _ = resp.Body.Close() }()

	var buffered bytes.Buffer
	if _, err := io.Copy(&buffered, resp.Body); err != nil {
		return nil, nil, nil, err
	}
	raw := buffered.Bytes()

	calls := map[string]*deepSeekStreamToolCall{}
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
				key := delta.ID
				if key == "" {
					// A streamed tool call's arguments arrive in pieces under
					// the same index; the id may only be on the first piece.
					key = "index:" + string(rune('0'+delta.Index))
				}
				call, seen := calls[key]
				if !seen {
					call = &deepSeekStreamToolCall{ID: delta.ID}
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

	var owned []deepSeekStreamToolCall
	for _, key := range order {
		call := calls[key]
		if call.Name == deepseektools.SearchToolName || call.Name == deepseektools.FetchToolName {
			owned = append(owned, *call)
		}
	}
	return raw, owned, usage, nil
}

// replayDeepSeekStream writes a buffered stream to the client unchanged.
func replayDeepSeekStream(w http.ResponseWriter, resp *http.Response, raw []byte) {
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
	_, _ = w.Write(raw)
}

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
	raw, calls, usage, err := collectDeepSeekStreamToolCalls(resp)
	if err != nil {
		return nil, err
	}
	if len(calls) == 0 {
		// No tool call: the client gets the provider's stream as it is.
		replayDeepSeekStream(w, resp, raw)
		streamUsage := usageFromStreamUsage(usage)
		return &deepSeekStreamCycleResult{
			body:        raw,
			usage:       streamUsage.toTokenUsage(),
			totalTokens: streamUsage.totalTokens,
			lastChunk:   lastChunkOf(raw),
		}, nil
	}

	runner := &deepSeekToolRunner{
		client:  p.deepSeekToolsClient(cred),
		quota:   p.deepSeekToolsQuota,
		account: deepSeekAccountKey(cred),
		state:   newDeepSeekToolState(),
	}
	if runner.client == nil {
		// No key: the tool cannot be run, so the stream is relayed as it is.
		replayDeepSeekStream(w, resp, raw)
		streamUsage := usageFromStreamUsage(usage)
		return &deepSeekStreamCycleResult{
			body:        raw,
			usage:       streamUsage.toTokenUsage(),
			totalTokens: streamUsage.totalTokens,
			lastChunk:   lastChunkOf(raw),
		}, nil
	}

	cycle, err := newDeepSeekToolCycle(body, runner)
	if err != nil {
		return nil, err
	}
	total := usageFromStreamUsage(usage)
	current := raw

	for step := 1; step < p.deepSeekToolsMaxSteps; step++ {
		next, err := cycle.nextBodyFromToolCalls(current, calls)
		if err != nil {
			return nil, err
		}
		stepBody, stepUsage, err := p.runDeepSeekStreamToolStep(r, w, logCtx, prepared, cred, modelID, realModelID, next, start, attemptedCreds)
		if err != nil {
			return nil, err
		}
		total.add(stepUsage)
		current = stepBody
		nextCalls, err := deepSeekToolCallsFromStream(stepBody)
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
	total.webFetchRequests = runner.state.billableFetch
	return &deepSeekStreamCycleResult{
		body:        current,
		usage:       total.toTokenUsage(),
		totalTokens: total.totalTokens,
		lastChunk:   lastChunkOf(current),
	}, nil
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
) ([]byte, *deepSeekToolUsage, error) {
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
	raw, _, usage, err := collectDeepSeekStreamToolCalls(outcome.resp)
	if err != nil {
		return nil, nil, err
	}
	return raw, usageFromStreamUsage(usage), nil
}

// deepSeekToolCallsFromStream reads the tool calls out of a buffered stream.
func deepSeekToolCallsFromStream(raw []byte) ([]deepSeekStreamToolCall, error) {
	var calls []deepSeekStreamToolCall
	for _, payload := range splitSSEPayloads(raw, nil) {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
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
				calls = append(calls, deepSeekStreamToolCall{
					ID:        delta.ID,
					Name:      delta.Function.Name,
					Arguments: delta.Function.Arguments,
				})
			}
		}
	}
	return calls, nil
}

// usageFromStreamUsage converts a stream's usage into the loop's shape.
func usageFromStreamUsage(usage *openai.OpenAIUsage) *deepSeekToolUsage {
	out := &deepSeekToolUsage{}
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

// nextBodyFromToolCalls builds the next request body from a streamed tool
// call, the streaming counterpart of the cycle's nextBody.
func (cycle *deepSeekToolCycle) nextBodyFromToolCalls(raw []byte, calls []deepSeekStreamToolCall) ([]byte, error) {
	assistant := map[string]any{"role": "assistant"}
	toolCalls := make([]map[string]any, 0, len(calls))
	results := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		if call.Name != deepseektools.SearchToolName && call.Name != deepseektools.FetchToolName {
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

// lastChunkOf returns the last SSE payload of a buffered stream, which the
// streaming log reads its figures from.
func lastChunkOf(raw []byte) []byte {
	payloads := splitSSEPayloads(raw, nil)
	if len(payloads) == 0 {
		return nil
	}
	return payloads[len(payloads)-1]
}
