package proxy

// What the router's two "send a non-streamed request upstream as a stream"
// mechanisms share: the server's force_stream_for_ttft (stream_simulated.go)
// and the loop gate's force_stream (loop_gate_collect.go).
//
//   - forcedStreamBlocker: the request options a provider refuses, or answers
//     differently, with stream:true. Such a request is sent as the client
//     sent it.
//   - forcedStreamRequestBody: the one rewrite of the request body.
//   - streamReadDeadline: request_timeout on the read of the forced stream,
//     as the body read of a non-streamed reply is bounded.
//   - drainForcedStream: the read loop (size limit, TTFT).
//   - reassembleForcedStream: the rebuild of the non-streamed body, which
//     refuses a stream that ended before its answer was complete.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/sse"
)

// errForcedStreamTruncated is a forced stream that ended cleanly (EOF, no
// transport error) before the answer was complete: no "data: [DONE]" and a
// choice without a finish_reason. A proxy in front of the model, or the
// model server going away, closes a stream like that. The client asked for a
// non-streamed answer and must not get the cut one as a 200: the attempt
// failed, as a body read that ended early fails a non-streamed reply.
var errForcedStreamTruncated = fmt.Errorf("forced stream: upstream stream ended before the answer was complete (no [DONE], no finish_reason): %w", io.ErrUnexpectedEOF)

// errForcedStreamEmpty is a forced stream with nothing to rebuild an answer
// from.
var errForcedStreamEmpty = errors.New("forced stream: no reassemblable content in upstream stream")

var (
	forcedStreamBestOfKey     = []byte(`"best_of"`)
	forcedStreamBeamSearchKey = []byte(`"use_beam_search"`)
)

// forcedStreamBlocker names the option of a non-streamed chat completion or
// completion request that rules out sending it upstream as a stream, "" when
// there is none:
//
//   - best_of above 1: OpenAI's /v1/completions refuses it with stream:true,
//     vLLM refuses it or silently does not stream;
//   - use_beam_search (vLLM): refused with stream:true.
//
// The request worked non-streamed, so it stays non-streamed: it only misses
// the TTFT (force_stream_for_ttft) or is observed instead of gated (the loop
// gate's force_stream).
func forcedStreamBlocker(body []byte) string {
	// Neither key in the bytes: most requests skip the decode.
	if !bytes.Contains(body, forcedStreamBestOfKey) && !bytes.Contains(body, forcedStreamBeamSearchKey) {
		return ""
	}
	var req struct {
		BestOf        json.RawMessage `json:"best_of"`
		UseBeamSearch json.RawMessage `json:"use_beam_search"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	var bestOf float64
	if json.Unmarshal(req.BestOf, &bestOf) == nil && bestOf > 1 {
		return "best_of"
	}
	var beam bool
	if json.Unmarshal(req.UseBeamSearch, &beam) == nil && beam {
		return "use_beam_search"
	}
	return ""
}

// forcedStreamOptions is how a non-streamed request is rewritten into a
// streamed one.
type forcedStreamOptions struct {
	// includeUsage asks for the usage chunk (stream_options.include_usage):
	// without it an OpenAI-wire stream carries no usage, and neither does the
	// rebuilt answer.
	includeUsage bool
	// continuousUsage asks vLLM for the running usage on every chunk.
	continuousUsage bool
	// replaceOptions drops the stream_options the client sent first.
	replaceOptions bool
}

// forcedStreamRequestBody returns body with "stream": true and the
// stream_options opts asks for. stream and stream_options are replaced, never
// added twice. ok is false, and body is returned as is, when body is not a
// JSON object.
func forcedStreamRequestBody(body []byte, opts forcedStreamOptions) (out []byte, ok bool) {
	return editStreamRequest(body, streamRequestEdit{
		forceStream:     true,
		includeUsage:    opts.includeUsage,
		continuousUsage: opts.continuousUsage,
		replaceOptions:  opts.replaceOptions,
	})
}

// streamReadDeadline bounds the read of a forced stream by request_timeout:
// when it passes, the body is closed, which ends the read. The client waits
// for the whole answer, so a hung upstream must fail the attempt (and let the
// next credential try) instead of holding the request until the client
// leaves.
type streamReadDeadline struct {
	timer   *time.Timer
	timeout time.Duration
	fired   atomic.Bool
}

// startStreamReadDeadline arms the deadline on body; timeout <= 0 means none.
func startStreamReadDeadline(body io.Closer, timeout time.Duration) *streamReadDeadline {
	d := &streamReadDeadline{timeout: timeout}
	if timeout <= 0 || body == nil {
		return d
	}
	d.timer = time.AfterFunc(timeout, func() {
		d.fired.Store(true)
		_ = body.Close()
	})
	return d
}

// stop disarms the deadline and returns the read's error: readErr, or the
// deadline's own when it fired. A body closed mid-read may read as a clean
// EOF, so the flag, not the read error, tells a timed-out read.
func (d *streamReadDeadline) stop(readErr error) error {
	if d.timer != nil {
		d.timer.Stop()
	}
	if d.fired.Load() {
		return fmt.Errorf("forced stream: upstream stream exceeded request_timeout %s: %w", d.timeout, context.DeadlineExceeded)
	}
	return readErr
}

// drainForcedStream reads r to its end and hands every chunk to onChunk (the
// chunk is only valid during the call). It stamps the request's TTFT on the
// first real content delta, with the scanner the live streaming path uses.
// overflow is true when the stream is larger than max_response_body_size: the
// chunk that crossed the limit is not handed over and the read stops. err is
// the read error, nil for a stream that ended (EOF).
func (p *Proxy) drainForcedStream(r io.Reader, logCtx *RequestLogContext, onChunk func(chunk []byte)) (overflow bool, err error) {
	var ttftScan ttftScanState
	ttftPending := logCtx != nil && logCtx.CompletionStartTime.IsZero()
	readBuf := make([]byte, 8192)
	var size int64
	for {
		n, readErr := r.Read(readBuf)
		if n > 0 {
			chunk := readBuf[:n]
			size += int64(n)
			if size > p.maxResponseBodySize {
				return true, nil
			}
			if onChunk != nil {
				onChunk(chunk)
			}
			if ttftPending && ttftScan.observe(chunk) {
				logCtx.markFirstToken()
				ttftPending = false
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return false, nil
			}
			return false, readErr
		}
	}
}

// reassembleForcedStream rebuilds the non-streamed body of a forced stream:
// a chat.completion from chat.completion.chunk events, or, for the legacy
// /v1/completions (completions), a text_completion. It returns
// errForcedStreamEmpty when there is nothing to rebuild (or the stream
// carried an error event), and errForcedStreamTruncated when the stream
// ended before the answer was complete.
func reassembleForcedStream(stream []byte, completions bool) ([]byte, error) {
	if completions {
		agg := newCompletionsStreamAggregator()
		agg.consume(stream)
		if len(agg.choices) == 0 {
			return nil, errForcedStreamEmpty
		}
		if !agg.complete() {
			return nil, errForcedStreamTruncated
		}
		if agg.id == "" {
			agg.id = "cmpl-simulated"
		}
		return agg.completion()
	}
	agg := newChatStreamAggregator()
	if err := agg.consume(bytes.NewReader(stream), 0); err != nil || agg.errStatus != 0 || len(agg.choices) == 0 {
		return nil, errForcedStreamEmpty
	}
	if !agg.complete() {
		return nil, errForcedStreamTruncated
	}
	if agg.id == "" {
		agg.id = "chatcmpl-simulated"
	}
	return agg.completion()
}

// reassembleChatCompletion folds an OpenAI Chat-Completions SSE body (as
// produced by the provider converters after conv.StreamTo) into a single
// Chat-Completions non-stream JSON body. Returns nil when nothing
// reassemblable was found, the stream carried an error event, or it ended
// before the answer was complete.
func reassembleChatCompletion(stream []byte) []byte {
	b, err := reassembleForcedStream(stream, false)
	if err != nil {
		return nil
	}
	return b
}

// reassembleCompletions is reassembleChatCompletion for a legacy OpenAI
// Completions SSE body (choices[].text, object "text_completion").
func reassembleCompletions(stream []byte) []byte {
	b, err := reassembleForcedStream(stream, true)
	if err != nil {
		return nil
	}
	return b
}

// completionsStreamAggregator rebuilds the text_completion a non-streamed
// /v1/completions request would have received from the events of a stream
// sent upstream instead. The legacy shape has no message, delta or
// tool_calls: per choice the text is concatenated, the logprobs arrays
// (tokens, token_logprobs, top_logprobs, text_offset) are appended, and
// finish_reason and vLLM's stop_reason come from the chunk that carried them.
// usage comes from the last chunk carrying it.
type completionsStreamAggregator struct {
	id                string
	model             string
	created           int64
	systemFingerprint string
	usage             json.RawMessage
	choices           map[int]*aggCompletionsChoice
	sawDone           bool
}

type aggCompletionsChoice struct {
	index        int
	text         strings.Builder
	finishReason *string
	stopReason   json.RawMessage
	logprobs     aggCompletionsLogprobs
	logprobsSeen bool
}

// aggCompletionsLogprobs is the logprobs object of a legacy completion
// choice. Its elements are kept as the provider sent them.
type aggCompletionsLogprobs struct {
	Tokens        []json.RawMessage `json:"tokens"`
	TokenLogprobs []json.RawMessage `json:"token_logprobs"`
	TopLogprobs   []json.RawMessage `json:"top_logprobs"`
	TextOffset    []json.RawMessage `json:"text_offset"`
}

func newCompletionsStreamAggregator() *completionsStreamAggregator {
	return &completionsStreamAggregator{choices: make(map[int]*aggCompletionsChoice)}
}

// consume aggregates every event of the buffered stream.
func (a *completionsStreamAggregator) consume(stream []byte) {
	for len(stream) > 0 {
		var line []byte
		if i := bytes.IndexByte(stream, '\n'); i >= 0 {
			line, stream = stream[:i], stream[i+1:]
		} else {
			line, stream = stream, nil
		}
		if sse.IsDone(line) {
			a.sawDone = true
			continue
		}
		if payload, ok := sse.LinePayload(line); ok {
			a.add(payload)
		}
	}
}

func (a *completionsStreamAggregator) add(payload []byte) {
	var chunk struct {
		ID                string `json:"id"`
		Model             string `json:"model"`
		Created           int64  `json:"created"`
		SystemFingerprint string `json:"system_fingerprint"`
		Choices           []struct {
			Index        int                     `json:"index"`
			Text         string                  `json:"text"`
			FinishReason *string                 `json:"finish_reason"`
			StopReason   json.RawMessage         `json:"stop_reason"`
			Logprobs     *aggCompletionsLogprobs `json:"logprobs"`
		} `json:"choices"`
		Usage json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return
	}
	if chunk.ID != "" {
		a.id = chunk.ID
	}
	if chunk.Model != "" {
		a.model = chunk.Model
	}
	if chunk.Created != 0 {
		a.created = chunk.Created
	}
	if chunk.SystemFingerprint != "" {
		a.systemFingerprint = chunk.SystemFingerprint
	}
	if !isJSONNull(chunk.Usage) {
		a.usage = append(json.RawMessage(nil), chunk.Usage...)
	}
	for i := range chunk.Choices {
		in := &chunk.Choices[i]
		c := a.choices[in.Index]
		if c == nil {
			c = &aggCompletionsChoice{index: in.Index}
			a.choices[in.Index] = c
		}
		c.text.WriteString(in.Text)
		if in.FinishReason != nil && *in.FinishReason != "" {
			fr := *in.FinishReason
			c.finishReason = &fr
		}
		if !isJSONNull(in.StopReason) {
			c.stopReason = append(json.RawMessage(nil), in.StopReason...)
		}
		if in.Logprobs != nil {
			c.logprobsSeen = true
			c.logprobs.Tokens = append(c.logprobs.Tokens, in.Logprobs.Tokens...)
			c.logprobs.TokenLogprobs = append(c.logprobs.TokenLogprobs, in.Logprobs.TokenLogprobs...)
			c.logprobs.TopLogprobs = append(c.logprobs.TopLogprobs, in.Logprobs.TopLogprobs...)
			c.logprobs.TextOffset = append(c.logprobs.TextOffset, in.Logprobs.TextOffset...)
		}
	}
}

// complete reports whether the stream carried the whole answer: its [DONE],
// or a finish_reason for every choice.
func (a *completionsStreamAggregator) complete() bool {
	if a.sawDone {
		return true
	}
	for _, c := range a.choices {
		if c.finishReason == nil {
			return false
		}
	}
	return len(a.choices) > 0
}

// completion returns the rebuilt text_completion body.
func (a *completionsStreamAggregator) completion() ([]byte, error) {
	type choiceOut struct {
		Index        int                     `json:"index"`
		Text         string                  `json:"text"`
		Logprobs     *aggCompletionsLogprobs `json:"logprobs"`
		FinishReason *string                 `json:"finish_reason,omitempty"`
		StopReason   json.RawMessage         `json:"stop_reason,omitempty"`
	}
	out := struct {
		ID                string          `json:"id"`
		Object            string          `json:"object"`
		Created           int64           `json:"created"`
		Model             string          `json:"model"`
		SystemFingerprint string          `json:"system_fingerprint,omitempty"`
		Choices           []choiceOut     `json:"choices"`
		Usage             json.RawMessage `json:"usage,omitempty"`
	}{
		ID:                a.id,
		Object:            "text_completion",
		Created:           a.created,
		Model:             a.model,
		SystemFingerprint: a.systemFingerprint,
		Usage:             a.usage,
	}
	indexes := make([]int, 0, len(a.choices))
	for i := range a.choices {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	out.Choices = make([]choiceOut, 0, len(indexes))
	for _, i := range indexes {
		c := a.choices[i]
		choice := choiceOut{Index: c.index, Text: c.text.String(), FinishReason: c.finishReason, StopReason: c.stopReason}
		if c.logprobsSeen {
			choice.Logprobs = &c.logprobs
		}
		out.Choices = append(out.Choices, choice)
	}
	return json.Marshal(out)
}
