package proxy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"

	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/loopgate"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

// loopGateUpstreamFormat returns the wire format of an attempt's upstream
// answer when the loop gate can watch it: a chat completion from an
// OpenAI-wire credential (whatever API the client called: Responses and
// Messages requests converted to chat completions included), or a Responses /
// Messages API answer relayed as is (passthrough).
func loopGateUpstreamFormat(cred *config.CredentialConfig, path string, passthroughResponses, passthroughMessages bool) loopgate.Format {
	switch {
	case strings.HasSuffix(path, "/chat/completions") && loopGateOpenAIWire(cred):
		return loopgate.FormatChat
	case passthroughResponses && strings.HasSuffix(path, "/responses"):
		return loopgate.FormatResponses
	case passthroughMessages && strings.HasSuffix(path, "/messages"):
		return loopgate.FormatMessages
	default:
		return loopgate.FormatNone
	}
}

// loopGateOpenAIWire reports whether cred answers chat completions in the
// OpenAI wire format (not converted from a provider-native one).
func loopGateOpenAIWire(cred *config.CredentialConfig) bool {
	switch cred.EffectiveProviderType() {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini, config.ProviderTypeAnthropic,
		config.ProviderTypeCometAPI, config.ProviderTypeProMan, config.ProviderTypeBedrock:
		return false
	default:
		return true
	}
}

// loopGateFormatOf returns the upstream format of logCtx's current attempt.
func loopGateFormatOf(logCtx *RequestLogContext) loopgate.Format {
	if logCtx == nil {
		return loopgate.FormatNone
	}
	return logCtx.loopGateFormat
}

// loopGateMaxLine bounds the partial SSE line kept between upstream reads. A
// longer line (a huge tool call or image event) is skipped by the detector.
const loopGateMaxLine = 1 << 20

// loopGateRespawn re-sends the streaming request of the current attempt with
// the given presence_penalty and a new seed, and returns the new event stream.
type loopGateRespawn func(presencePenalty float64) (io.ReadCloser, error)

// loopGateRespawner returns the restart function of one upstream attempt: a
// copy of req (headers, auth, URL) with body changed only in seed and
// presence_penalty. It refuses once the client is gone, so a drained stream
// never starts a generation nobody reads.
func (p *Proxy) loopGateRespawner(ctx context.Context, client, req *http.Request, body []byte) loopGateRespawn {
	return func(presencePenalty float64) (io.ReadCloser, error) {
		if err := client.Context().Err(); err != nil {
			return nil, err
		}
		newBody, err := loopGateRestartBody(body, presencePenalty, rand.Int64N(1<<31))
		if err != nil {
			return nil, err
		}
		restart := req.Clone(ctx)
		restart.Body = io.NopCloser(bytes.NewReader(newBody))
		restart.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(newBody)), nil }
		restart.ContentLength = int64(len(newBody))
		resp, err := p.client.Do(restart) //nolint:gosec // G704: same credential URL as the attempt being restarted
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK || !IsStreamingResponse(resp) {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("restart got status %d, content type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
		}
		return resp.Body, nil
	}
}

// loopGateRestartBody sets seed and presence_penalty on a chat completion
// request body; every other field is carried through as raw bytes.
func loopGateRestartBody(body []byte, presencePenalty float64, seed int64) ([]byte, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, fmt.Errorf("decode request body: %w", err)
	}
	if top == nil {
		return nil, fmt.Errorf("request body is not a JSON object")
	}
	top["seed"] = json.RawMessage(fmt.Sprint(seed))
	penalty, err := json.Marshal(presencePenalty)
	if err != nil {
		return nil, err
	}
	top["presence_penalty"] = penalty
	return json.Marshal(top)
}

// loopGateObserver runs the loop gate detector over one streamed chat
// completion. It sits on the upstream body as an io.Reader and sees every SSE
// line before anything downstream does. In observe mode the bytes pass
// through untouched. In enforce mode, once a channel loops at the cut ratio:
//
//   - while only reasoning has been streamed, the upstream generation is
//     aborted and restarted with a new seed and a higher presence_penalty; the
//     new attempt's events continue the same client stream (the looped
//     reasoning the client already received stays);
//   - otherwise, or when restarts run out, the stream ends right after the
//     line that tipped it: a final chunk with finish_reason "length" and an
//     air_loop_gate marker, then [DONE]. The upstream body is closed so the
//     engine stops generating.
type loopGateObserver struct {
	p           *Proxy
	ctx         context.Context
	settings    config.LoopGateSettings
	detector    *loopgate.Detector
	credential  string
	modelID     string // routing alias, for rate limits
	model       string // metrics label
	publicModel string // model name the client sees
	requestID   string
	firings     []loopgate.Firing
	now         func() time.Time
	start       time.Time

	format         loopgate.Format // wire format of the upstream stream
	normalizeUsage bool            // the upstream body was wrapped in the usage normalizer; restarts get it too
	respawn        loopGateRespawn // nil = no restarts
	visionInject   bool            // descriptions were injected into attempt 1: never restart
	restarts       int
	abortedWords   int // words of restarted attempts, missing from the final usage

	upstream     io.ReadCloser
	carry        []byte // unfinished SSE line from the previous read
	carryDropped bool   // the current line outgrew loopGateMaxLine
	cut          *loopGateCut
	pending      []byte // bytes to return before reading upstream again
}

type loopGateCut struct {
	channel string
	words   int
	ratio   float64
	elapsed time.Duration
}

// loopGateSettings returns the effective loop gate settings of a request for
// modelID served by cred; ok is false when the gate is off for them.
func (p *Proxy) loopGateSettings(ctx context.Context, cred *config.CredentialConfig, modelID string) (config.LoopGateSettings, bool) {
	if cred == nil {
		return config.LoopGateSettings{}, false
	}
	var modelOverride *config.LoopGateOverride
	if p.modelManager != nil {
		modelOverride = p.modelManager.GetLoopGateOverride(modelID, cred.Name)
	}
	settings, err := p.loopGate.Resolve(cred.Type, cred.LoopGate, modelOverride)
	if err != nil {
		p.logger.WarnContext(ctx, "Loop gate disabled: invalid effective settings",
			"credential", cred.Name, "model", modelID, "error", err)
		return settings, false
	}
	return settings, settings.Enabled()
}

// newLoopGateObserver returns an observer for a successful stream served by
// the request's credential, or nil when the loop gate is disabled for that
// credential and model or the upstream stream is not in format (the format
// of each attempt is decided where its request is built: logCtx.loopGateFormat).
func (p *Proxy) newLoopGateObserver(ctx context.Context, logCtx *RequestLogContext, modelID string, statusCode int, format loopgate.Format) *loopGateObserver {
	if logCtx == nil || format == loopgate.FormatNone || logCtx.loopGateFormat != format ||
		statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return nil
	}
	settings, ok := p.loopGateSettings(ctx, logCtx.Credential, modelID)
	if !ok {
		return nil
	}
	o := p.newLoopGateObserverWith(ctx, logCtx, modelID, settings, format)
	o.respawn = logCtx.loopGateRespawn
	o.visionInject = logCtx.visionInject != nil
	p.metrics.RecordLoopGateStream(o.credential, o.model, string(settings.Mode))
	return o
}

// newLoopGateObserverWith returns an observer for one attempt, without
// restarts of its own.
func (p *Proxy) newLoopGateObserverWith(ctx context.Context, logCtx *RequestLogContext, modelID string, settings config.LoopGateSettings, format loopgate.Format) *loopGateObserver {
	o := &loopGateObserver{
		p:           p,
		ctx:         ctx,
		settings:    settings,
		format:      format,
		credential:  logCtx.Credential.Name,
		modelID:     modelID,
		model:       metricModelID(modelID, logCtx),
		publicModel: clientVisibleResponseModel(logCtx, modelID),
		requestID:   logCtx.RequestID,
		now:         time.Now,
	}
	o.start = o.now()
	o.detector = o.newDetector()
	return o
}

func (o *loopGateObserver) newDetector() *loopgate.Detector {
	return loopgate.NewDetector(loopgate.Config{
		NGram:     o.settings.NGram,
		ArmWords:  o.settings.ArmWords,
		MinRepeat: o.settings.MinRepeat,
		MinRatio:  o.settings.MinRatio,
	}, o.now())
}

// wrap returns the reader the stream handler reads instead of body.
func (o *loopGateObserver) wrap(body io.ReadCloser) io.Reader {
	o.upstream = body
	return o
}

func (o *loopGateObserver) Read(p []byte) (int, error) {
	if len(o.pending) > 0 {
		n := copy(p, o.pending)
		o.pending = o.pending[n:]
		return n, nil
	}
	if o.cut != nil {
		return 0, io.EOF
	}
	n, err := o.upstream.Read(p)
	if n > 0 {
		if end, ch := o.scan(p[:n]); end >= 0 {
			o.onLoop(ch)
			return end, nil
		}
	}
	return n, err
}

// scan feeds the complete SSE lines of b to the detector. In enforce mode it
// returns the offset just past the line that made a channel loop at the cut
// ratio, and that channel; -1 otherwise.
func (o *loopGateObserver) scan(b []byte) (int, string) {
	enforce := o.settings.Mode == config.LoopGateEnforce
	cut := o.settings.CutRatio()
	pos := 0
	for {
		i := bytes.IndexByte(b[pos:], '\n')
		if i < 0 {
			o.keep(b[pos:])
			return -1, ""
		}
		line := b[pos : pos+i]
		if len(o.carry) > 0 || o.carryDropped {
			if !o.carryDropped {
				o.carry = append(o.carry, line...)
				line = o.carry
			} else {
				line = nil
			}
		}
		o.observeLine(line)
		o.carry, o.carryDropped = o.carry[:0], false
		pos += i + 1
		if enforce {
			if ch := o.detector.LoopingChannel(cut); ch != "" {
				return pos, ch
			}
		}
	}
}

func (o *loopGateObserver) keep(rest []byte) {
	if o.carryDropped || len(rest) == 0 {
		return
	}
	if len(o.carry)+len(rest) > loopGateMaxLine {
		o.carry, o.carryDropped = o.carry[:0], true
		return
	}
	o.carry = append(o.carry, rest...)
}

func (o *loopGateObserver) observeLine(line []byte) {
	payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), sseDataPrefix)
	if !ok {
		return
	}
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 || bytes.Equal(payload, sseDoneSentinel) {
		return
	}
	o.record(o.detector.Observe(o.format, payload, o.now()))
}

func (o *loopGateObserver) channel(name string) *loopgate.Channel {
	if name == loopgate.ChannelReasoning {
		return o.detector.Reasoning()
	}
	return o.detector.Content()
}

// onLoop acts on a channel looping at the cut ratio: the upstream generation
// is abandoned either way, then restarted or cut.
func (o *loopGateObserver) onLoop(ch string) {
	_ = o.upstream.Close()
	if o.canRestart(ch) && o.restart(ch) {
		return
	}
	c := o.channel(ch)
	o.cut = &loopGateCut{channel: ch, words: c.Words(), ratio: c.Ratio(), elapsed: o.now().Sub(o.start)}
	o.pending = o.cutTail()
}

// canRestart reports whether a loop on ch can be restarted in the same
// stream. Only a chat stream can take a second attempt's events: a Responses
// or Messages stream would get a second response.created / message_start.
func (o *loopGateObserver) canRestart(ch string) bool {
	return o.format == loopgate.FormatChat &&
		ch == loopgate.ChannelReasoning &&
		!o.detector.AnswerStarted() &&
		o.respawn != nil &&
		!o.visionInject &&
		o.restarts+1 < o.settings.MaxAttempts &&
		o.now().Sub(o.start) < o.settings.RetryDeadline
}

// restart replaces the upstream with a new attempt. false = the restart
// request failed, and the stream is cut instead.
func (o *loopGateObserver) restart(ch string) bool {
	c := o.channel(ch)
	penalty := o.settings.PresencePenaltyForRestart(o.restarts + 1)
	body, err := o.respawn(penalty)
	if err != nil {
		o.p.logger.WarnContext(o.ctx, "Loop gate: restart failed, cutting the stream",
			"credential", o.credential, "model", o.model, "attempt", o.restarts+2,
			"error", err, "request_id", o.requestID)
		return false
	}
	if o.normalizeUsage {
		body, _ = modelutils.NewUsageNormalizingReadCloser(body, o.publicModel)
	}

	o.p.metrics.RecordLoopGateRestart(o.credential, o.model, ch)
	o.p.logger.WarnContext(o.ctx, "Loop gate: generation restarted",
		"credential", o.credential,
		"model", o.model,
		"channel", ch,
		"words", c.Words(),
		"ratio", c.Ratio(),
		"elapsed_s", o.now().Sub(o.start).Seconds(),
		"attempt", o.restarts+2,
		"presence_penalty", penalty,
		"request_id", o.requestID,
	)
	o.reportAttempt("loop_gate_restart")
	o.abortedWords += o.detector.Content().Words() + o.detector.Reasoning().Words()
	o.restarts++
	o.upstream = body
	o.detector = o.newDetector()
	o.firings = nil
	o.carry, o.carryDropped = o.carry[:0], false
	// The line that tipped the loop ended with its own "\n"; the blank line
	// closes its event before the new attempt's events.
	o.pending = []byte("\n")
	return true
}

func (o *loopGateObserver) marker() loopGateMarker {
	return loopGateMarker{
		Action:   "cut",
		Channel:  o.cut.channel,
		Words:    o.cut.words,
		Ratio:    o.cut.ratio,
		Attempts: o.restarts + 1,
	}
}

// cutTail returns the closing events of a cut stream in its wire format. The
// line that tipped the loop ended with its own "\n"; the blank line each tail
// starts with closes its event.
func (o *loopGateObserver) cutTail() []byte {
	switch o.format {
	case loopgate.FormatResponses:
		return o.cutTailResponses()
	case loopgate.FormatMessages:
		return o.cutTailMessages()
	default:
		return o.cutTailChat()
	}
}

// cutTailResponses ends a Responses API stream with response.incomplete
// (reason max_output_tokens), the event a provider sends when it stops at
// the output limit.
func (o *loopGateObserver) cutTailResponses() []byte {
	meta := o.detector.Meta()
	event := map[string]any{
		"type":            "response.incomplete",
		"sequence_number": o.detector.Sequence() + 1,
		"response": map[string]any{
			"id":                 meta.ID,
			"object":             "response",
			"created_at":         meta.Created,
			"model":              meta.Model,
			"status":             "incomplete",
			"incomplete_details": map[string]any{"reason": "max_output_tokens"},
			"output":             []any{},
		},
		"air_loop_gate": o.marker(),
	}
	return loopGateSSEEvents([]loopGateSSEEvent{{name: "response.incomplete", data: event}})
}

// cutTailMessages ends an Anthropic Messages stream the way it ends at
// max_tokens: the open content block is closed, message_delta carries
// stop_reason "max_tokens", then message_stop.
func (o *loopGateObserver) cutTailMessages() []byte {
	var events []loopGateSSEEvent
	if index, open := o.detector.OpenBlock(); open {
		events = append(events, loopGateSSEEvent{name: "content_block_stop", data: map[string]any{"type": "content_block_stop", "index": index}})
	}
	words := o.detector.Content().Words() + o.detector.Reasoning().Words()
	events = append(events,
		loopGateSSEEvent{name: "message_delta", data: map[string]any{
			"type":          "message_delta",
			"delta":         map[string]any{"stop_reason": "max_tokens", "stop_sequence": nil},
			"usage":         map[string]any{"output_tokens": loopGateWordsToTokens(words)},
			"air_loop_gate": o.marker(),
		}},
		loopGateSSEEvent{name: "message_stop", data: map[string]any{"type": "message_stop"}},
	)
	return loopGateSSEEvents(events)
}

type loopGateSSEEvent struct {
	name string
	data any
}

func loopGateSSEEvents(events []loopGateSSEEvent) []byte {
	out := []byte{'\n'}
	for _, e := range events {
		payload, err := json.Marshal(e.data)
		if err != nil {
			continue
		}
		out = append(out, "event: "...)
		out = append(out, e.name...)
		out = append(out, "\ndata: "...)
		out = append(out, payload...)
		out = append(out, "\n\n"...)
	}
	return out
}

// cutTailChat ends a chat completion stream with finish_reason "length".
func (o *loopGateObserver) cutTailChat() []byte {
	meta := o.detector.Meta()
	chunk := loopGateCutChunk{
		ID:       meta.ID,
		Object:   "chat.completion.chunk",
		Created:  meta.Created,
		Model:    meta.Model,
		Choices:  []loopGateCutChoice{{Index: 0, Delta: struct{}{}, FinishReason: "length"}},
		LoopGate: o.marker(),
	}
	payload, err := json.Marshal(chunk)
	if err != nil {
		payload = []byte(`{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`)
	}
	tail := make([]byte, 0, len(payload)+32)
	tail = append(tail, '\n')
	tail = append(tail, "data: "...)
	tail = append(tail, payload...)
	tail = append(tail, "\n\ndata: [DONE]\n\n"...)
	return tail
}

type loopGateCutChunk struct {
	ID       string              `json:"id,omitempty"`
	Object   string              `json:"object"`
	Created  int64               `json:"created,omitempty"`
	Model    string              `json:"model,omitempty"`
	Choices  []loopGateCutChoice `json:"choices"`
	LoopGate loopGateMarker      `json:"air_loop_gate"`
}

type loopGateCutChoice struct {
	Index        int      `json:"index"`
	Delta        struct{} `json:"delta"`
	FinishReason string   `json:"finish_reason"`
}

// loopGateMarker tells the client that finish_reason "length" came from the
// loop gate, not from its max_tokens.
type loopGateMarker struct {
	Action   string  `json:"action"`
	Channel  string  `json:"channel"`
	Words    int     `json:"words"`
	Ratio    float64 `json:"ratio"`
	Attempts int     `json:"attempts"`
}

func (o *loopGateObserver) record(firings []loopgate.Firing) {
	mode := string(o.settings.Mode)
	for _, f := range firings {
		o.firings = append(o.firings, f)
		o.p.metrics.RecordLoopGateFiring(o.credential, o.model, f.Channel, mode)
		o.p.logger.WarnContext(o.ctx, "Loop gate: generation is looping",
			"mode", mode,
			"credential", o.credential,
			"model", o.model,
			"channel", f.Channel,
			"words", f.Words,
			"elapsed_s", f.Elapsed.Seconds(),
			"top_repeat", f.Repeat,
			"ratio", f.Ratio,
			"attempt", o.restarts+1,
			"request_id", o.requestID,
		)
	}
}

// reportAttempt reports, for every fired channel of the current attempt, how
// much it generated after the firing and how the attempt ended.
func (o *loopGateObserver) reportAttempt(outcome string) {
	mode := string(o.settings.Mode)
	for _, f := range o.firings {
		ch := o.channel(f.Channel)
		after := ch.Words() - f.Words
		o.p.metrics.RecordLoopGateWordsAfterFiring(o.credential, o.model, f.Channel, mode, after)
		o.p.logger.InfoContext(o.ctx, "Loop gate: looping stream ended",
			"mode", mode,
			"credential", o.credential,
			"model", o.model,
			"channel", f.Channel,
			"words_total", ch.Words(),
			"words_after_firing", after,
			"final_ratio", ch.Ratio(),
			"final_top_repeat", ch.Top(),
			"finish_reason", outcome,
			"attempt", o.restarts+1,
			"request_id", o.requestID,
		)
	}
}

// finishAttempt closes the current attempt: the last unfinished word is
// counted and the fired channels are reported with outcome, or with the
// upstream finish_reason when outcome is "".
func (o *loopGateObserver) finishAttempt(outcome string) {
	if o.cut == nil {
		o.record(o.detector.Finish(o.now()))
	}
	if outcome == "" {
		outcome = o.detector.FinishReason()
	}
	o.reportAttempt(outcome)
}

// reportCut records the final cut of a request after attempts attempts.
func (o *loopGateObserver) reportCut(attempts int) {
	o.p.metrics.RecordLoopGateCut(o.credential, o.model, o.cut.channel)
	o.p.logger.WarnContext(o.ctx, "Loop gate: stream cut",
		"credential", o.credential,
		"model", o.model,
		"channel", o.cut.channel,
		"words", o.cut.words,
		"ratio", o.cut.ratio,
		"elapsed_s", o.cut.elapsed.Seconds(),
		"cut_ratio", o.settings.CutRatio(),
		"attempts", attempts,
		"request_id", o.requestID,
	)
}

// reportRecovered records a request whose restart finished uncut.
func (o *loopGateObserver) reportRecovered(attempts int, elapsed time.Duration) {
	o.p.metrics.RecordLoopGateRecovered(o.credential, o.model)
	o.p.logger.InfoContext(o.ctx, "Loop gate: restart recovered the generation",
		"credential", o.credential, "model", o.model,
		"attempts", attempts, "finish_reason", o.detector.FinishReason(),
		"elapsed_s", elapsed.Seconds(),
		"request_id", o.requestID)
}

// consumeAbortedTokens counts an estimate of restarted attempts' tokens
// against the credential's TPM: the final usage covers the last attempt only,
// but the restarted ones used the engine too. Not billed to the client.
func (o *loopGateObserver) consumeAbortedTokens(words int) {
	o.consumeTokens(loopGateWordsToTokens(words))
}

// consumeTokens counts tokens against the credential's (and model's) TPM.
func (o *loopGateObserver) consumeTokens(tokens int) {
	if tokens <= 0 || o.p.rateLimiter == nil {
		return
	}
	o.p.rateLimiter.ConsumeTokens(o.credential, tokens)
	if o.modelID != "" {
		o.p.rateLimiter.ConsumeModelTokens(o.credential, o.modelID, tokens)
	}
}

// finish closes the stream when the handler returns.
func (o *loopGateObserver) finish() {
	outcome := ""
	if o.cut != nil {
		outcome = "loop_gate_cut"
		o.reportCut(o.restarts + 1)
	}
	o.finishAttempt(outcome)
	if o.restarts == 0 {
		return
	}
	if o.cut == nil {
		o.reportRecovered(o.restarts+1, o.now().Sub(o.start))
		// A cut stream has no usage: its token count comes from all streamed
		// text, restarted attempts included, so only a recovered one adds them.
		o.consumeAbortedTokens(o.abortedWords)
	}
}

// loopGateWordsToTokens estimates tokens from whitespace-separated words
// (about 4 tokens per 3 English words; more for other languages, so this
// errs low).
func loopGateWordsToTokens(words int) int {
	return words * 4 / 3
}
