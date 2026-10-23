package proxy

import (
	"context"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/loopgate"
)

// loopGateObserver runs the loop gate detector over one streamed chat
// completion. In observe mode it only logs and counts firings: the stream
// reaches the client untouched.
type loopGateObserver struct {
	p          *Proxy
	ctx        context.Context
	settings   config.LoopGateSettings
	detector   *loopgate.Detector
	credential string
	model      string
	requestID  string
	firings    []loopgate.Firing
	now        func() time.Time
}

// newLoopGateObserver returns an observer for a successful stream served by
// cred, or nil when the loop gate is disabled for that credential and model.
func (p *Proxy) newLoopGateObserver(ctx context.Context, logCtx *RequestLogContext, modelID string, statusCode int) *loopGateObserver {
	if logCtx == nil || logCtx.Credential == nil ||
		statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return nil
	}
	cred := logCtx.Credential
	var modelOverride *config.LoopGateOverride
	if p.modelManager != nil {
		modelOverride = p.modelManager.GetLoopGateOverride(modelID, cred.Name)
	}
	settings, err := p.loopGate.Resolve(cred.Type, cred.LoopGate, modelOverride)
	if err != nil {
		p.logger.WarnContext(ctx, "Loop gate disabled: invalid effective settings",
			"credential", cred.Name, "model", modelID, "error", err)
		return nil
	}
	if !settings.Enabled() {
		return nil
	}
	o := &loopGateObserver{
		p:          p,
		ctx:        ctx,
		settings:   settings,
		credential: cred.Name,
		model:      metricModelID(modelID, logCtx),
		requestID:  logCtx.RequestID,
		now:        time.Now,
	}
	o.detector = loopgate.NewDetector(loopgate.Config{
		NGram:     settings.NGram,
		ArmWords:  settings.ArmWords,
		MinRepeat: settings.MinRepeat,
		MinRatio:  settings.MinRatio,
	}, o.now())
	p.metrics.RecordLoopGateStream(o.credential, o.model, string(settings.Mode))
	return o
}

// observe feeds the SSE data payloads of one stream line batch.
func (o *loopGateObserver) observe(payloads [][]byte) {
	for _, payload := range payloads {
		o.record(o.detector.ObserveChatChunk(payload, o.now()))
	}
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
			"request_id", o.requestID,
		)
	}
}

// finish closes the stream: the last unfinished word is counted and every
// fired channel reports how much it generated after the firing and how the
// generation ended.
func (o *loopGateObserver) finish() {
	o.record(o.detector.Finish(o.now()))
	mode := string(o.settings.Mode)
	for _, f := range o.firings {
		ch := o.detector.Content()
		if f.Channel == loopgate.ChannelReasoning {
			ch = o.detector.Reasoning()
		}
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
			"finish_reason", o.detector.FinishReason(),
			"request_id", o.requestID,
		)
	}
}
