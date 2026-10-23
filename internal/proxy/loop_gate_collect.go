package proxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/loopgate"
)

// loopGateForcedStream is a non-streamed chat completion the loop gate sends
// upstream as a stream (enforce mode with force_stream): the stream is read
// through the gate, restarted on a loop, and rebuilt into the one
// chat.completion the client asked for.
type loopGateForcedStream struct {
	settings     config.LoopGateSettings
	modelID      string
	respawn      loopGateRespawn
	promptTokens func() int // prompt estimate for a cut attempt without upstream usage
}

// loopGateForceStream returns the body to send upstream instead of body and
// the plan for its response, or (body, nil) when the gate does not apply: a
// streamed request, an upstream answer the gate does not watch as a chat
// completion, or enforce/force_stream off in the attempt's settings.
func (p *Proxy) loopGateForceStream(ctx context.Context, cred *config.CredentialConfig, modelID string, format loopgate.Format, settings config.LoopGateSettings, streaming bool, body []byte) ([]byte, *loopGateForcedStream) {
	if streaming || format != loopgate.FormatChat || settings.Mode != config.LoopGateEnforce || !settings.ForceStreamEnabled() {
		return body, nil
	}
	streamBody, err := loopGateStreamBody(body, cred.Type == config.ProviderTypeVLLM)
	if err != nil {
		p.logger.DebugContext(ctx, "Loop gate: request not streamed upstream", "error", err)
		return body, nil
	}
	return streamBody, &loopGateForcedStream{settings: settings, modelID: modelID}
}

// loopGateStreamBody turns a non-streamed chat completion request into a
// streamed one with usage. stream and stream_options are replaced, never
// added twice (a duplicate key is a 400 for some routers in front of vLLM).
// continuousUsage asks vLLM for the running usage on every chunk, so an
// attempt cut short still reports what the engine generated.
func loopGateStreamBody(body []byte, continuousUsage bool) ([]byte, error) {
	top, err := loopGateJSONObject(body)
	if err != nil {
		return nil, err
	}
	top["stream"] = json.RawMessage("true")
	top["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	if continuousUsage {
		top["stream_options"] = json.RawMessage(`{"include_usage":true,"continuous_usage_stats":true}`)
	}
	return json.Marshal(top)
}

type loopGateAttempt struct {
	agg   *chatStreamAggregator
	obs   *loopGateObserver
	cut   *loopGateCut
	words int
}

// tokens returns the attempt's total tokens from its usage, else an estimate
// of its generated tokens from the word count.
func (a loopGateAttempt) tokens() int {
	var usage struct {
		TotalTokens int `json:"total_tokens"`
	}
	if len(a.agg.usage) > 0 && json.Unmarshal(a.agg.usage, &usage) == nil && usage.TotalTokens > 0 {
		return usage.TotalTokens
	}
	return loopGateWordsToTokens(a.words)
}

// errorBodyReader fails every read with err, so the regular response body
// read reports a transport error or an oversized body as it would for a
// non-streamed reply.
type errorBodyReader struct{ err error }

func (r errorBodyReader) Read([]byte) (int, error) { return 0, r.err }

// failedAggregatedResponse returns resp as a 200 whose body read fails with
// err, so the caller's body read takes its transport-error path.
func failedAggregatedResponse(resp *http.Response, err error) *http.Response {
	failed := aggregatedResponse(resp, http.StatusOK, nil)
	failed.Body = io.NopCloser(errorBodyReader{err: err})
	failed.ContentLength = -1
	return failed
}

// collectLoopGateStream reads the forced stream of resp through the loop gate
// and returns resp rebuilt as the non-streamed chat.completion (or the error
// the stream carried). A looping attempt is aborted and restarted while
// max_attempts and retry_deadline allow; when they run out, the best attempt
// is returned with its looped tail trimmed and finish_reason "length".
func (p *Proxy) collectLoopGateStream(ctx context.Context, logCtx *RequestLogContext, resp *http.Response, plan *loopGateForcedStream) *http.Response {
	if resp.StatusCode != http.StatusOK || !IsStreamingResponse(resp) {
		return resp
	}
	start := time.Now()
	var attempts []loopGateAttempt
	var last *loopGateObserver
	total := 0 // attempts sent upstream, failed ones included
	body := resp.Body
	for n := 1; ; n++ {
		total = n
		o := p.newLoopGateObserverWith(ctx, logCtx, plan.modelID, plan.settings, loopgate.FormatChat)
		o.restarts = n - 1
		if plan.promptTokens != nil {
			o.promptTokens = plan.promptTokens
		}
		if n == 1 {
			p.metrics.RecordLoopGateStream(o.credential, o.model, string(plan.settings.Mode))
		}
		agg := newChatStreamAggregator()
		err := p.consumeLoopGateAttempt(agg, o, body)
		_ = body.Close()
		last = o

		if err != nil || agg.errStatus != 0 {
			o.finishAttempt("")
			if len(attempts) == 0 {
				if err != nil {
					return failedAggregatedResponse(resp, err)
				}
				return aggregatedResponse(resp, agg.errStatus, agg.errBody)
			}
			// A restart failed midway: answer with what the earlier attempts
			// got. Every earlier attempt was cut, so the request ends cut.
			p.logger.WarnContext(ctx, "Loop gate: restarted attempt failed, answering with an earlier one",
				"credential", o.credential, "model", o.model, "attempt", n,
				"error", err, "status", agg.errStatus, "request_id", o.requestID)
			attempts[len(attempts)-1].obs.reportCut(n)
			break
		}

		words := o.detector.Content().Words() + o.detector.Reasoning().Words()
		attempts = append(attempts, loopGateAttempt{agg: agg, obs: o, cut: o.cut, words: words})
		if o.cut == nil {
			o.finishAttempt("")
			if n > 1 {
				o.reportRecovered(n, time.Since(start))
			}
			break
		}

		restart := n < plan.settings.MaxAttempts && time.Since(start) < plan.settings.RetryDeadline && plan.respawn != nil
		if !restart {
			o.finishAttempt("loop_gate_cut")
			o.reportCut(n)
			break
		}
		penalty := plan.settings.PresencePenaltyForRestart(n)
		next, rerr := plan.respawn(n, penalty)
		if rerr != nil {
			o.finishAttempt("loop_gate_cut")
			o.reportCut(n)
			p.logger.WarnContext(ctx, "Loop gate: restart failed, answering with the trimmed attempt",
				"credential", o.credential, "model", o.model, "attempt", n+1,
				"error", rerr, "request_id", o.requestID)
			break
		}
		o.finishAttempt("loop_gate_restart")
		o.reportRestart(o.cut.channel, o.cut.words, o.cut.ratio, time.Since(start), n+1, penalty, false)
		body = next
	}

	chosen := len(attempts) - 1
	if attempts[chosen].cut != nil {
		chosen = loopGateBestAttempt(attempts, plan.settings)
		a := attempts[chosen]
		marker := a.cut.marker(total)
		a.agg.loopGate = &marker
	}
	// The response's usage covers the returned attempt only (for a cut vLLM
	// attempt: the running usage at the cut): every other attempt counts
	// against the credential's TPM, not the client's bill.
	other := 0
	for i, a := range attempts {
		if i != chosen {
			other += a.tokens()
		}
	}
	last.consumeTokens(other)

	out, err := attempts[chosen].agg.completion()
	if err != nil {
		return failedAggregatedResponse(resp, err)
	}
	return aggregatedResponse(resp, http.StatusOK, out)
}

// consumeLoopGateAttempt reads one forced-stream attempt into agg within
// request_timeout, as the regular non-streamed body read is bounded: a hung
// upstream fails the attempt with a read error (retried on another
// credential for the first attempt) instead of holding the request until the
// client leaves.
func (p *Proxy) consumeLoopGateAttempt(agg *chatStreamAggregator, o *loopGateObserver, body io.ReadCloser) error {
	if p.requestTimeout <= 0 {
		return agg.consume(o.wrap(body), p.maxResponseBodySize)
	}
	// A body closed mid-read may read as a clean EOF: the flag, not the read
	// error, tells a timed-out attempt.
	var timedOut atomic.Bool
	timer := time.AfterFunc(p.requestTimeout, func() {
		timedOut.Store(true)
		_ = body.Close()
	})
	err := agg.consume(o.wrap(body), p.maxResponseBodySize)
	timer.Stop()
	if timedOut.Load() {
		return fmt.Errorf("loop gate: upstream stream exceeded request_timeout %s: %w", p.requestTimeout, context.DeadlineExceeded)
	}
	return err
}

// loopGateBestAttempt trims the looped tail off every cut attempt (one round
// of the loop is kept) and returns the index of the one with the longest
// remaining answer, then the longest remaining reasoning. Its finish_reason
// becomes "length".
func loopGateBestAttempt(attempts []loopGateAttempt, s config.LoopGateSettings) int {
	best, bestContent, bestReasoning := len(attempts)-1, -1, -1
	for i, a := range attempts {
		c := a.agg.choice0()
		if c == nil || a.cut == nil {
			continue
		}
		content := loopgate.Trim(c.content.String(), s.NGram, s.MinRepeat)
		reasoning := loopgate.Trim(c.reasoning.String(), s.NGram, s.MinRepeat)
		c.content.Reset()
		c.content.WriteString(content)
		c.reasoning.Reset()
		c.reasoning.WriteString(reasoning)
		length := "length"
		c.finishReason = &length
		if len(content) > bestContent || (len(content) == bestContent && len(reasoning) > bestReasoning) {
			best, bestContent, bestReasoning = i, len(content), len(reasoning)
		}
	}
	return best
}

// observeLoopGateBody runs the detector over a non-streamed answer the gate
// did not stream (observe mode, force_stream off, or a Responses / Messages
// passthrough): it only logs and counts, the response is not changed.
func (p *Proxy) observeLoopGateBody(ctx context.Context, logCtx *RequestLogContext, modelID string, body []byte) {
	format, settings := logCtx.loopGateFormat, logCtx.loopGateSettings
	if format == loopgate.FormatNone {
		return
	}
	content, reasoning, finish, ok := loopGateAnswerText(format, body)
	if !ok {
		return
	}
	o := p.newLoopGateObserverWith(ctx, logCtx, modelID, settings, format)
	p.metrics.RecordLoopGateStream(o.credential, o.model, string(settings.Mode))
	o.record(o.detector.ObserveText(content, reasoning, finish, o.now()))
	o.reportAttempt(o.detector.FinishReason())
}

// loopGateAnswerText extracts the answer and reasoning text of a
// non-streamed response body: choice 0 of a chat completion, the message and
// reasoning items of a Responses API response, the text and thinking blocks
// of a Messages API message.
func loopGateAnswerText(format loopgate.Format, body []byte) (content, reasoning, finish string, ok bool) {
	var b strings.Builder
	var rb strings.Builder
	switch format {
	case loopgate.FormatChat:
		var completion struct {
			Choices []struct {
				Index   int `json:"index"`
				Message struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal(body, &completion) != nil {
			return "", "", "", false
		}
		for _, c := range completion.Choices {
			if c.Index != 0 {
				continue
			}
			r := c.Message.ReasoningContent
			if r == "" {
				r = c.Message.Reasoning
			}
			return c.Message.Content, r, c.FinishReason, true
		}
		return "", "", "", false
	case loopgate.FormatResponses:
		var response struct {
			Status            string `json:"status"`
			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Output []struct {
				Type    string `json:"type"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
				Summary []struct {
					Text string `json:"text"`
				} `json:"summary"`
			} `json:"output"`
		}
		if json.Unmarshal(body, &response) != nil {
			return "", "", "", false
		}
		for _, item := range response.Output {
			switch item.Type {
			case "message":
				for _, part := range item.Content {
					b.WriteString(part.Text)
				}
			case "reasoning":
				for _, part := range item.Content {
					rb.WriteString(part.Text)
				}
				for _, part := range item.Summary {
					rb.WriteString(part.Text)
				}
			}
		}
		finish = response.Status
		if response.IncompleteDetails != nil && response.IncompleteDetails.Reason != "" {
			finish = response.IncompleteDetails.Reason
		}
		return b.String(), rb.String(), finish, true
	case loopgate.FormatMessages:
		var message struct {
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			} `json:"content"`
			StopReason string `json:"stop_reason"`
		}
		if json.Unmarshal(body, &message) != nil {
			return "", "", "", false
		}
		for _, block := range message.Content {
			switch block.Type {
			case "text":
				b.WriteString(block.Text)
			case "thinking":
				rb.WriteString(block.Thinking)
			}
		}
		return b.String(), rb.String(), message.StopReason, true
	}
	return "", "", "", false
}
