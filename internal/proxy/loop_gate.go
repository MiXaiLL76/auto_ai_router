package proxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
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

// loopGateRespawn re-sends the streaming request of the current attempt as
// restart number restart (1 = the second attempt) with at least the given
// presence_penalty, and returns the new event stream.
type loopGateRespawn func(restart int, presencePenalty float64) (io.ReadCloser, error)

// loopGateRespawner returns the restart function of one upstream attempt
// served by cred for modelID: a copy of req (headers, auth, URL) with body
// changed only in seed and presence_penalty (see loopGateRestartBody). The
// clone keeps req's context, and with it the credential's proxy_url
// (httputil.WithProxyURL). It refuses once the client is gone, so a drained
// stream never starts a generation nobody reads.
//
// A restart is a request to the credential like any other: it counts toward
// its RPM (while under the limit: a running stream is never refused or held
// for RPM) and its outcome goes to fail2ban and the attempt error metric.
func (p *Proxy) loopGateRespawner(client, req *http.Request, body []byte, cred, modelID string) loopGateRespawn {
	return func(attempt int, presencePenalty float64) (io.ReadCloser, error) {
		if err := client.Context().Err(); err != nil {
			return nil, err
		}
		if p.rateLimiter != nil {
			p.rateLimiter.Allow(cred)
			if modelID != "" {
				p.rateLimiter.AllowModel(cred, modelID)
			}
		}
		newBody, err := loopGateRestartBody(body, presencePenalty, attempt, rand.Int64N(1<<31))
		if err != nil {
			return nil, err
		}
		restart := req.Clone(req.Context())
		restart.Body = io.NopCloser(bytes.NewReader(newBody))
		restart.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(newBody)), nil }
		restart.ContentLength = int64(len(newBody))
		resp, err := p.client.Do(restart) //nolint:gosec // G704: same credential URL as the attempt being restarted
		if err != nil {
			status := http.StatusBadGateway
			if isTimeoutError(err) {
				status = http.StatusRequestTimeout
			}
			p.balancer.RecordResponse(cred, modelID, status)
			p.metrics.RecordCredentialAttemptError(cred)
			return nil, err
		}
		p.balancer.RecordResponse(cred, modelID, resp.StatusCode)
		if resp.StatusCode != http.StatusOK || !IsStreamingResponse(resp) {
			_ = resp.Body.Close()
			p.metrics.RecordCredentialAttemptError(cred)
			return nil, fmt.Errorf("restart got status %d, content type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
		}
		return resp.Body, nil
	}
}

// loopGateRestartBody sets seed and presence_penalty on a chat completion
// request body for restart number restart; every other field is carried
// through as raw bytes. The client's presence_penalty is kept when it is
// higher than the gate's. A client seed becomes seed+restart, so the attempts
// stay reproducible yet differ; without one, randomSeed is used.
func loopGateRestartBody(body []byte, presencePenalty float64, restart int, randomSeed int64) ([]byte, error) {
	top, err := loopGateJSONObject(body)
	if err != nil {
		return nil, err
	}
	var clientPenalty float64
	if raw, ok := top["presence_penalty"]; ok && json.Unmarshal(raw, &clientPenalty) == nil {
		presencePenalty = max(presencePenalty, clientPenalty)
	}
	seed := randomSeed
	var clientSeed int64
	if raw, ok := top["seed"]; ok && !isJSONNull(raw) && json.Unmarshal(raw, &clientSeed) == nil {
		seed = clientSeed + int64(restart)
	}
	top["seed"] = json.RawMessage(strconv.FormatInt(seed, 10))
	penalty, err := json.Marshal(presencePenalty)
	if err != nil {
		return nil, err
	}
	top["presence_penalty"] = penalty
	return json.Marshal(top)
}

// loopGateJSONObject decodes the top-level keys of a JSON object request
// body, leaving every value as raw bytes.
func loopGateJSONObject(body []byte) (map[string]json.RawMessage, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, fmt.Errorf("decode request body: %w", err)
	}
	if top == nil {
		return nil, errors.New("request body is not a JSON object")
	}
	return top, nil
}

// loopGateObserver runs the loop gate detector over one streamed chat
// completion. It sits on the upstream body as an io.Reader and sees every SSE
// line before anything downstream does. In observe mode the bytes pass
// through untouched. In enforce mode, once a channel loops at the cut ratio:
//
//   - while only reasoning has been streamed, the upstream generation is
//     aborted and restarted with a new seed and a higher presence_penalty; the
//     new attempt's events continue the same client stream (the looped
//     reasoning the client already received stays). While the new attempt is
//     on its way (request, prefill), the client gets an SSE comment every
//     loopGateKeepaliveInterval;
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
	keepalive      io.Writer       // where waiting keepalives go; nil = into the stream read through the gate
	stripUsage     bool            // vLLM sends the running usage on every chunk for the gate only: wrap drops it
	restarts       int
	abortedTokens  int                    // tokens of restarted attempts, missing from the final usage
	restarting     chan loopGateRestarted // the next attempt being sent; nil = none
	restartCut     *loopGateCut           // the loop being restarted, cut with it when the restart fails

	promptTokens func() int               // local prompt token estimate, 0 = unknown
	lastUsage    loopGateChatUsage        // chat: the last usage the current attempt reported
	responsesOut *loopGateResponsesOutput // enforced Responses stream: its output so far

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

// loopGateAttemptFormat returns the format the loop gate watches for one
// upstream attempt of modelID served by cred (FormatNone when the gate is off
// for them, or format is not watchable) and its settings, resolved once per
// attempt for everything downstream.
func (p *Proxy) loopGateAttemptFormat(ctx context.Context, cred *config.CredentialConfig, modelID string, format loopgate.Format) (loopgate.Format, config.LoopGateSettings) {
	if format == loopgate.FormatNone {
		return format, config.LoopGateSettings{}
	}
	settings, ok := p.loopGateSettings(ctx, cred, modelID)
	if !ok {
		return loopgate.FormatNone, config.LoopGateSettings{}
	}
	return format, settings
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
// and settings of each attempt are decided where its request is built:
// logCtx.loopGateFormat, logCtx.loopGateSettings).
func (p *Proxy) newLoopGateObserver(ctx context.Context, logCtx *RequestLogContext, modelID string, statusCode int, format loopgate.Format) *loopGateObserver {
	if logCtx == nil || format == loopgate.FormatNone || logCtx.loopGateFormat != format ||
		statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return nil
	}
	settings := logCtx.loopGateSettings
	o := p.newLoopGateObserverWith(ctx, logCtx, modelID, settings, format)
	o.respawn = logCtx.loopGateRespawn
	o.visionInject = logCtx.visionInject != nil
	o.stripUsage = logCtx.loopGateStripUsage
	p.metrics.RecordLoopGateStream(o.credential, o.model, string(settings.Mode))
	return o
}

// newLoopGateObserverWith returns an observer for one attempt, without
// restarts of its own.
func (p *Proxy) newLoopGateObserverWith(ctx context.Context, logCtx *RequestLogContext, modelID string, settings config.LoopGateSettings, format loopgate.Format) *loopGateObserver {
	o := &loopGateObserver{
		p:            p,
		ctx:          ctx,
		settings:     settings,
		format:       format,
		credential:   logCtx.Credential.Name,
		modelID:      metricModelID(modelID, logCtx),
		model:        metricModelID(modelID, logCtx),
		publicModel:  clientVisibleResponseModel(logCtx, modelID),
		requestID:    logCtx.RequestID,
		now:          time.Now,
		promptTokens: logCtx.promptTokensEstimate,
	}
	if format == loopgate.FormatResponses && settings.Mode == config.LoopGateEnforce {
		o.responsesOut = newLoopGateResponsesOutput()
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
	if o.stripUsage {
		return newLoopGateUsageStripReader(o)
	}
	return o
}

func (o *loopGateObserver) Read(p []byte) (int, error) {
	for len(o.pending) == 0 && o.restarting != nil {
		if err := o.awaitRestart(); err != nil {
			return 0, err
		}
	}
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
	if err == io.EOF && len(o.carry) > 0 {
		// The upstream ended its last line without "\n": nothing follows to
		// cut, the detector only has to see it.
		o.observeLine(o.carry)
		o.carry = o.carry[:0]
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
	if o.settings.Mode != config.LoopGateEnforce {
		return
	}
	switch {
	case o.responsesOut != nil:
		o.responsesOut.add(payload)
	case o.format == loopgate.FormatChat:
		var usage loopGateChatUsage
		if start, end, ok := chatChunkUsageSpan(payload); ok && json.Unmarshal(payload[start:end], &usage) == nil {
			o.lastUsage = usage
		}
	}
}

// loopGateChatUsage is the usage of a chat completion chunk.
type loopGateChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// cutUsage returns the usage of a cut or restarted attempt: what the engine
// last reported (vLLM's continuous_usage_stats, which the gate asks for, is
// exact), else the prompt estimate and the streamed words as tokens.
func (o *loopGateObserver) cutUsage() loopGateChatUsage {
	u := o.lastUsage
	if u.PromptTokens <= 0 && o.promptTokens != nil {
		u.PromptTokens = o.promptTokens()
	}
	if u.CompletionTokens <= 0 {
		u.CompletionTokens = loopGateWordsToTokens(o.detector.Content().Words() + o.detector.Reasoning().Words())
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	return u
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
	c := o.channel(ch)
	cut := &loopGateCut{channel: ch, words: c.Words(), ratio: c.Ratio(), elapsed: o.now().Sub(o.start)}
	if o.canRestart(ch) {
		o.startRestart(cut)
		// The line that tipped the loop ended with its own "\n"; the blank
		// line closes its event before keepalives and the new attempt's events.
		o.pending = []byte("\n")
		return
	}
	o.cut = cut
	o.pending = o.cutTail()
}

// canRestart reports whether a loop on ch can be restarted in the same
// stream. Only a chat stream can take a second attempt's events: a Responses
// or Messages stream would get a second response.created / message_start.
func (o *loopGateObserver) canRestart(ch string) bool {
	return o.format == loopgate.FormatChat &&
		ch == loopgate.ChannelReasoning &&
		!o.detector.AnswerStarted() &&
		!o.visionInject &&
		o.restartAllowed(o.respawn)
}

// restartAllowed reports whether max_attempts and retry_deadline allow an
// attempt after the current one, sent with respawn.
func (o *loopGateObserver) restartAllowed(respawn loopGateRespawn) bool {
	return respawn != nil &&
		o.restarts+1 < o.settings.MaxAttempts &&
		o.now().Sub(o.start) < o.settings.RetryDeadline
}

// nextRestart returns the number of the restart after the current attempt
// and its presence_penalty.
func (o *loopGateObserver) nextRestart() (int, float64) {
	restart := o.restarts + 1
	return restart, o.settings.PresencePenaltyForRestart(restart)
}

func (o *loopGateObserver) warnRestartFailed(err error, fallback string) {
	o.p.logger.WarnContext(o.ctx, "Loop gate: restart failed, "+fallback,
		"credential", o.credential, "model", o.model, "attempt", o.restarts+2,
		"error", err, "request_id", o.requestID)
}

// loopGateKeepaliveInterval is how often a client stream waiting for a
// restarted attempt gets a keepalive: the new attempt's request and prefill
// can outlast the read timeout of a proxy or SDK in front of the client.
var loopGateKeepaliveInterval = 15 * time.Second

// loopGateKeepalive is an SSE comment: clients and the converters skip it.
var loopGateKeepalive = []byte(": air loop gate restart\n\n")

// loopGateRestarted is the outcome of a restart request: the new attempt's
// stream, once its first bytes arrived, or the error.
type loopGateRestarted struct {
	body    io.ReadCloser
	penalty float64
	err     error
}

// startRestart sends the next attempt in the background; Read waits for it
// in awaitRestart, sending keepalives meanwhile.
func (o *loopGateObserver) startRestart(cut *loopGateCut) {
	restart, penalty := o.nextRestart()
	respawn, normalize, model := o.respawn, o.normalizeUsage, o.publicModel
	done := make(chan loopGateRestarted, 1)
	o.restarting, o.restartCut = done, cut
	go func() {
		body, err := respawn(restart, penalty)
		if err == nil {
			if normalize {
				body, _ = modelutils.NewUsageNormalizingReadCloser(body, model)
			}
			body, err = loopGatePrefetch(body)
		}
		done <- loopGateRestarted{body: body, penalty: penalty, err: err}
	}()
}

// awaitRestart waits for the restarted attempt, or for one keepalive
// interval: the keepalive is then queued for the client (or written to
// o.keepalive). An error means the client is gone.
func (o *loopGateObserver) awaitRestart() error {
	timer := time.NewTimer(loopGateKeepaliveInterval)
	defer timer.Stop()
	select {
	case r := <-o.restarting:
		o.restarting = nil
		o.restarted(r)
	case <-timer.C:
		if o.keepalive == nil {
			o.pending = loopGateKeepalive
			return nil
		}
		if _, err := o.keepalive.Write(loopGateKeepalive); err != nil {
			return err
		}
	case <-o.ctx.Done():
		return o.ctx.Err()
	}
	return nil
}

// restarted continues the stream with the new attempt, or cuts it at the
// restarted loop when the restart failed.
func (o *loopGateObserver) restarted(r loopGateRestarted) {
	cut := o.restartCut
	o.restartCut = nil
	if r.err != nil {
		o.warnRestartFailed(r.err, "cutting the stream")
		o.cut = cut
		o.pending = o.cutTail()
		return
	}
	o.reportRestart(cut, o.restarts+2, r.penalty, true)
	o.reportAttempt("loop_gate_restart")
	o.abortedTokens += o.cutUsage().TotalTokens
	o.restarts++
	o.upstream = r.body
	o.detector = o.newDetector()
	o.lastUsage = loopGateChatUsage{}
	o.firings = nil
	o.carry, o.carryDropped = o.carry[:0], false
}

// loopGatePrefetch waits for the first bytes of a restarted attempt's stream:
// until then the engine is still on prefill, and a stream that fails or ends
// before sending anything is a failed restart.
func loopGatePrefetch(body io.ReadCloser) (io.ReadCloser, error) {
	head := make([]byte, 4096)
	n, err := body.Read(head)
	if n == 0 && err != nil {
		_ = body.Close()
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return &loopGatePrefetched{head: head[:n], err: err, body: body}, nil
}

// loopGatePrefetched is a stream whose first read was made ahead.
type loopGatePrefetched struct {
	head []byte
	err  error // the first read's error, returned once head is drained
	body io.ReadCloser
}

func (b *loopGatePrefetched) Read(p []byte) (int, error) {
	if len(b.head) > 0 {
		n := copy(p, b.head)
		b.head = b.head[n:]
		return n, nil
	}
	if b.err != nil {
		return 0, b.err
	}
	return b.body.Read(p)
}

func (b *loopGatePrefetched) Close() error { return b.body.Close() }

// reportRestart records the restart of the loop c: attempt is the number of
// the new attempt, stream whether it continues a client stream (false: a
// forced stream).
func (o *loopGateObserver) reportRestart(c *loopGateCut, attempt int, penalty float64, stream bool) {
	o.p.metrics.RecordLoopGateRestart(o.credential, o.model, c.channel)
	o.p.logger.WarnContext(o.ctx, "Loop gate: generation restarted",
		"credential", o.credential,
		"model", o.model,
		"channel", c.channel,
		"words", c.words,
		"ratio", c.ratio,
		"elapsed_s", c.elapsed.Seconds(),
		"attempt", attempt,
		"presence_penalty", penalty,
		"stream", stream,
		"request_id", o.requestID,
	)
}

func (o *loopGateObserver) marker() loopGateMarker {
	return o.cut.marker(o.restarts + 1)
}

// marker returns the air_loop_gate marker of a request cut after attempts
// attempts.
func (c *loopGateCut) marker(attempts int) loopGateMarker {
	return loopGateMarker{
		Action:   "cut",
		Channel:  c.channel,
		Words:    c.words,
		Ratio:    c.ratio,
		Attempts: attempts,
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

// cutTailResponses ends a Responses API stream the way a provider ends it at
// the output limit: the open output items are closed (their *.done events,
// then output_item.done with status "incomplete"), then response.incomplete
// (reason max_output_tokens) carries the whole output and an estimated usage.
func (o *loopGateObserver) cutTailResponses() []byte {
	seq := o.detector.Sequence()
	nextSeq := func() int { seq++; return seq }
	var events []loopGateSSEEvent
	output := []any{}
	response := map[string]any{}
	inputTokens := 0
	if out := o.responsesOut; out != nil {
		events, output = out.closeOpen(nextSeq)
		for k, v := range out.response {
			response[k] = v
		}
		inputTokens = out.inputTokens
	}
	if inputTokens <= 0 && o.promptTokens != nil {
		inputTokens = o.promptTokens()
	}
	meta := o.detector.Meta()
	for k, v := range map[string]any{"id": meta.ID, "created_at": meta.Created, "model": meta.Model} {
		if _, ok := response[k]; !ok {
			response[k] = v
		}
	}
	response["object"] = "response"
	response["status"] = "incomplete"
	response["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	response["output"] = output
	outputTokens := loopGateWordsToTokens(o.detector.Content().Words() + o.detector.Reasoning().Words())
	response["usage"] = map[string]any{
		"input_tokens":          inputTokens,
		"input_tokens_details":  map[string]any{"cached_tokens": 0},
		"output_tokens":         outputTokens,
		"output_tokens_details": map[string]any{"reasoning_tokens": loopGateWordsToTokens(o.detector.Reasoning().Words())},
		"total_tokens":          inputTokens + outputTokens,
	}
	events = append(events, loopGateSSEEvent{name: "response.incomplete", data: map[string]any{
		"type":            "response.incomplete",
		"sequence_number": nextSeq(),
		"response":        response,
		"air_loop_gate":   o.marker(),
	}})
	return loopGateSSEEvents(events)
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

// cutTailChat ends a chat completion stream with finish_reason "length",
// then the usage chunk (choices: []) a stream with include_usage ends with:
// AIR always asks the upstream for usage, and the response compatibility
// layer drops it for a client that did not ask.
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
	usage, err := json.Marshal(loopGateUsageChunk{
		ID: meta.ID, Object: "chat.completion.chunk", Created: meta.Created, Model: meta.Model,
		Choices: []struct{}{}, Usage: o.cutUsage(),
	})
	if err != nil {
		usage = nil
	}
	tail := make([]byte, 0, len(payload)+len(usage)+48)
	tail = append(tail, '\n')
	tail = append(tail, "data: "...)
	tail = append(tail, payload...)
	if usage != nil {
		tail = append(tail, "\n\ndata: "...)
		tail = append(tail, usage...)
	}
	tail = append(tail, "\n\ndata: [DONE]\n\n"...)
	return tail
}

type loopGateUsageChunk struct {
	ID      string            `json:"id,omitempty"`
	Object  string            `json:"object"`
	Created int64             `json:"created,omitempty"`
	Model   string            `json:"model,omitempty"`
	Choices []struct{}        `json:"choices"`
	Usage   loopGateChatUsage `json:"usage"`
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

// finish closes the stream when the handler returns. The handler closes only
// the original body, so a restarted attempt's body is closed here: a client
// gone mid-stream must not leave the engine generating. A restart still on
// its way is closed when it arrives.
func (o *loopGateObserver) finish() {
	outcome := ""
	if o.restarting != nil {
		// The client went away while the restart was on its way: the
		// abandoned attempt used the engine all the same.
		go func(done <-chan loopGateRestarted) {
			if r := <-done; r.body != nil {
				_ = r.body.Close()
			}
		}(o.restarting)
		o.restarting, o.restartCut = nil, nil
		o.abortedTokens += o.cutUsage().TotalTokens
		outcome = "loop_gate_restart"
	}
	if o.restarts > 0 {
		_ = o.upstream.Close()
	}
	if o.cut != nil {
		outcome = "loop_gate_cut"
		o.reportCut(o.restarts + 1)
	}
	o.finishAttempt(outcome)
	// The final usage, the upstream's or the cut tail's, covers the last
	// attempt only: the restarted ones (prompt and generated tokens) used the
	// engine too. Counted against the credential's TPM, not billed.
	o.consumeTokens(o.abortedTokens)
	if o.restarts > 0 && outcome == "" {
		o.reportRecovered(o.restarts+1, o.now().Sub(o.start))
	}
}

// loopGateWordsToTokens estimates tokens from whitespace-separated words
// (about 4 tokens per 3 English words; more for other languages, so this
// errs low).
func loopGateWordsToTokens(words int) int {
	return words * 4 / 3
}
