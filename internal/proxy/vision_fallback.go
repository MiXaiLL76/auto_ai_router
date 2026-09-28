package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
)

// HeaderVisionFallback tells the client that image inputs were rewritten because the
// requested model cannot see images. A comma-separated list of: "described=N/M" (N of
// the M images of the current turn were described; the rest became placeholders),
// "restored=K" (K history images got the description recorded in an earlier answer)
// and "stripped" (placeholders only).
const HeaderVisionFallback = "X-AIR-Vision-Fallback"

// maxVisionQuestionChars bounds the user text passed to the describe model as context.
const maxVisionQuestionChars = 4000

type visionBodyFormat int

const (
	visionFormatChat      visionBodyFormat = iota // /v1/chat/completions: messages[].content[] image_url
	visionFormatResponses                         // /v1/responses: input[].content[] input_image
	visionFormatMessages                          // /v1/messages: messages[].content[] image
)

// visionImageRef points at one image part inside a decoded request body. Assigning
// container[index] replaces the part in place.
type visionImageRef struct {
	container []any
	index     int
	url       string             // data: or http(s) URL; empty when the image cannot be described (file_id)
	current   bool               // belongs to the turn after the last assistant message
	restored  *visionDescription // history image: its description found in the answer to its turn
}

type visionDescribeCtxKey struct{}

// isVisionDescribeRequest reports whether ctx belongs to an internal describe call.
// Such a request never triggers another describe round, whatever the configuration.
func isVisionDescribeRequest(ctx context.Context) bool {
	return ctx.Value(visionDescribeCtxKey{}) != nil
}

// visionFallbackResult is what applyVisionFallback did to the body.
type visionFallbackResult struct {
	body      []byte
	rewritten bool   // body differs from the input
	rejected  bool   // mode reject and the body carries images
	outcome   string // HeaderVisionFallback value; "" when no image was rewritten
	images    int    // all image parts, history included
	current   int    // image parts of the current turn
	described int
	restored  int
	inject    *visionResponseInjection // descriptions to prepend to the answer
}

// applyVisionFallback rewrites image inputs of a request to a vLLM model configured
// with supports_vision: false, and removes the image-description markers of earlier
// answers from the conversation of any vLLM-only model. Bodies without images or
// markers, models that support images (or never declared supports_vision) and models
// served by any non-vLLM credential are returned unchanged.
func (p *Proxy) applyVisionFallback(w http.ResponseWriter, r *http.Request, body []byte, modelID string, format visionBodyFormat) visionFallbackResult {
	unchanged := visionFallbackResult{body: body}
	if p.modelManager == nil {
		return unchanged
	}
	hasImage := bytes.Contains(body, []byte("image"))
	hasMarker := bytes.Contains(body, []byte(visionMarkerNeedle))
	if !hasImage && !hasMarker {
		return unchanged
	}
	supported, known := p.modelManager.SupportsVision(modelID)
	textOnly := known && !supported
	if !textOnly && !hasMarker {
		return unchanged
	}
	if !p.servedOnlyByVLLM(modelID) {
		if textOnly {
			p.warnVisionFlagIgnored(r.Context(), modelID)
		}
		return unchanged
	}

	root, err := decodeVisionBody(body)
	if err != nil {
		return unchanged
	}
	result := visionFallbackResult{}
	if textOnly {
		var rejected bool
		result, rejected = p.rewriteVisionImages(w, r, root, modelID, format)
		if rejected {
			return visionFallbackResult{body: body, rejected: true, images: result.images}
		}
	}
	// After the images: their descriptions are read from the markers first.
	stripped := hasMarker && stripVisionMarkers(root, format)
	if result.outcome == "" && !stripped {
		return unchanged
	}

	newBody, err := marshalVisionJSON(root)
	if err != nil {
		p.logger.WarnContext(r.Context(), "Vision fallback: failed to encode rewritten body", "error", err)
		return unchanged
	}
	result.body = newBody
	result.rewritten = true
	return result
}

// rewriteVisionImages replaces the image parts of a conversation for a text-only
// model: current-turn images are described (mode describe), history images get the
// description recorded in the answer to their turn, the rest become placeholders.
func (p *Proxy) rewriteVisionImages(w http.ResponseWriter, r *http.Request, root map[string]any, modelID string, format visionBodyFormat) (visionFallbackResult, bool) {
	refs, question := collectVisionImages(root, format)
	result := visionFallbackResult{images: len(refs)}
	if len(refs) == 0 {
		return result, false
	}

	mode := p.visionFallback.Mode
	if mode == config.VisionFallbackDescribe && isVisionDescribeRequest(r.Context()) {
		mode = config.VisionFallbackStrip
	}
	if mode != config.VisionFallbackStrip && mode != config.VisionFallbackDescribe {
		return result, true
	}

	replacements := make([]string, len(refs))
	for i, ref := range refs {
		switch {
		case ref.current:
			result.current++
			replacements[i] = visionPlaceholderText(true)
		case ref.restored != nil:
			result.restored++
			replacements[i] = visionDescribedText(ref.restored.n, ref.restored.model, modelID, ref.restored.text)
		default:
			replacements[i] = visionPlaceholderText(false)
		}
	}
	var descs []visionDescription
	if mode == config.VisionFallbackDescribe && result.current > 0 {
		descs = p.describeVisionImages(w, r, refs, question, modelID, replacements)
		result.described = len(descs)
	}
	for i, ref := range refs {
		ref.container[ref.index] = visionTextPart(format, replacements[i])
	}

	var outcome []string
	if mode == config.VisionFallbackDescribe && result.current > 0 {
		outcome = append(outcome, fmt.Sprintf("described=%d/%d", result.described, result.current))
	}
	if result.restored > 0 {
		outcome = append(outcome, fmt.Sprintf("restored=%d", result.restored))
	}
	if len(outcome) == 0 {
		outcome = append(outcome, "stripped")
	}
	result.outcome = strings.Join(outcome, ", ")

	if len(descs) > 0 && p.visionFallback.InjectEnabled() && visionInjectAllowed(root, format) {
		result.inject = &visionResponseInjection{prefix: buildVisionMarker(descs), choices: visionChoiceCount(root)}
	}
	return result, false
}

// servedOnlyByVLLM reports whether every credential serving modelID is a vLLM
// credential. Vision fallback is a vLLM-only feature: a supports_vision flag on a
// model reachable through any other provider (OpenAI, Anthropic, an AIR peer, ...)
// is ignored, so those requests are never rewritten.
func (p *Proxy) servedOnlyByVLLM(modelID string) bool {
	names := p.modelManager.GetCredentialsForModel(modelID)
	if len(names) == 0 || p.balancer == nil {
		return false
	}
	types := make(map[string]config.ProviderType)
	for _, cred := range p.balancer.GetCredentialsSnapshot() {
		types[cred.Name] = cred.Type
	}
	for _, name := range names {
		if types[name] != config.ProviderTypeVLLM {
			return false
		}
	}
	return true
}

// warnVisionFlagIgnored logs once per model that its supports_vision: false has no
// effect because a non-vLLM credential serves it. It is logged on the first request
// with images rather than at startup: the model-to-credential mapping only settles
// after the database sync and remote model discovery.
func (p *Proxy) warnVisionFlagIgnored(ctx context.Context, modelID string) {
	if _, seen := p.visionFlagIgnoredWarned.LoadOrStore(modelID, struct{}{}); seen {
		return
	}
	p.logger.WarnContext(ctx, "supports_vision: false is ignored: the model is served by a non-vLLM credential, images are forwarded as-is",
		"model", modelID,
		"credentials", p.modelManager.GetCredentialsForModel(modelID))
}

// applyVisionFallbackToBase runs applyVisionFallback on the orchestrator's base
// body and returns the rewritten body, or nil when the body is unchanged. It writes
// the 400 response and returns false when the request is rejected.
func (p *Proxy) applyVisionFallbackToBase(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	baseBody []byte,
	modelID string,
	format visionBodyFormat,
) ([]byte, bool) {
	vision := p.applyVisionFallback(w, r, baseBody, modelID, format)
	if vision.rejected {
		msg := fmt.Sprintf("Model %s does not support image inputs; remove the images or use a vision-capable model", modelID)
		p.logger.WarnContext(r.Context(), "Rejected image input for a model without vision support",
			"error_code", http.StatusBadRequest,
			"model", modelID,
			"images", vision.images,
			"request_id", logCtx.RequestID)
		logCtx.Status = "failure"
		logCtx.HTTPStatus = http.StatusBadRequest
		logCtx.ErrorMsg = msg
		WriteErrorBadRequest(w, msg)
		return nil, false
	}
	if !vision.rewritten {
		return nil, true
	}
	logCtx.visionInject = vision.inject
	if vision.outcome == "" {
		return vision.body, true // only markers of earlier answers were removed
	}
	w.Header().Set(HeaderVisionFallback, vision.outcome)
	p.logger.InfoContext(r.Context(), "Rewrote image input for a model without vision support",
		"model", modelID,
		"outcome", vision.outcome,
		"images", vision.images,
		"current_turn_images", vision.current,
		"described", vision.described,
		"restored", vision.restored,
		"inject_into_response", vision.inject != nil,
		"describe_model", p.visionFallback.DescribeModel,
		"request_id", logCtx.RequestID)
	return vision.body, true
}

// describeVisionImages describes the current-turn images (at most max_images) in
// parallel and stores each description in replacements. It returns the successful
// descriptions in image order; failed ones keep their placeholder.
func (p *Proxy) describeVisionImages(w http.ResponseWriter, r *http.Request, refs []visionImageRef, question, modelID string, replacements []string) []visionDescription {
	cfg := p.visionFallback

	// Nothing is written to the client while images are described, but the server
	// write_timeout keeps running (HTTP/2 even resets the stream when it fires). Lift
	// the deadline for the describe step (each call is bounded by cfg.Timeout) and
	// give the rest of the request a full write_timeout afterwards, as it would have
	// had without the describe step.
	rc := http.NewResponseController(w)
	_ = rc.SetWriteDeadline(time.Time{})
	defer func() {
		if p.serverWriteTimeout > 0 {
			_ = rc.SetWriteDeadline(time.Now().Add(p.serverWriteTimeout))
		}
	}()

	var wg sync.WaitGroup
	results := make([]*visionDescription, len(refs)) // each goroutine writes only its own slot
	number := 0
	for i, ref := range refs {
		if !ref.current || ref.url == "" {
			continue
		}
		if cfg.MaxImages > 0 && number >= cfg.MaxImages {
			replacements[i] = "[image omitted: too many images in one message, only the first ones were described]"
			continue
		}
		number++
		wg.Add(1)
		go func(i, number int, url string) {
			defer wg.Done()
			// The router's panic recovery covers only the handler goroutine; a panic
			// here would otherwise take down the whole process.
			defer func() {
				if rec := recover(); rec != nil {
					p.logger.ErrorContext(r.Context(), "Vision fallback: panic in describe call",
						"model", modelID, "describe_model", cfg.DescribeModel, "image", number,
						"panic", rec, "stack", string(debug.Stack()))
					replacements[i] = "[image omitted: the image could not be described]"
				}
			}()
			start := time.Now()
			text, err := p.describeVisionImage(r, url, question)
			if err != nil {
				p.logger.WarnContext(r.Context(), "Vision fallback: describe call failed",
					"model", modelID, "describe_model", cfg.DescribeModel,
					"image", number, "duration", time.Since(start), "error", err)
				replacements[i] = "[image omitted: the image could not be described]"
				return
			}
			text = sanitizeVisionDescription(text)
			replacements[i] = visionDescribedText(number, cfg.DescribeModel, modelID, text)
			results[i] = &visionDescription{n: number, model: cfg.DescribeModel, text: text}
		}(i, number, ref.url)
	}
	wg.Wait()
	var descs []visionDescription
	for _, d := range results {
		if d != nil {
			descs = append(descs, *d)
		}
	}
	return descs
}

// describeVisionImage runs one Chat Completions request against describe_model through
// this router's own pipeline with the caller's headers, so the call is authenticated,
// rate-limited, balanced and billed to the same key and end user as the original.
func (p *Proxy) describeVisionImage(r *http.Request, imageURL, question string) (string, error) {
	cfg := p.visionFallback
	userContent := []any{}
	if question != "" {
		userContent = append(userContent, map[string]any{"type": "text", "text": "User's question about the image: " + question})
	}
	userContent = append(userContent, map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}})
	messages := []any{map[string]any{"role": "user", "content": userContent}}
	if cfg.DescribePrompt != "" {
		messages = append([]any{map[string]any{"role": "system", "content": cfg.DescribePrompt}}, messages...)
	}
	reqBody := map[string]any{"model": cfg.DescribeModel, "stream": false, "messages": messages}
	if cfg.MaxTokens > 0 {
		reqBody["max_tokens"] = cfg.MaxTokens
	}
	payload, err := marshalVisionJSON(reqBody)
	if err != nil {
		return "", err
	}

	ctx, cancel := visionDescribeContext(r.Context(), cfg.Timeout)
	defer cancel()
	p.logger.DebugContext(r.Context(), "Vision fallback: describe call",
		"request_id", requestid.FromContext(r.Context()),
		"describe_request_id", requestid.FromContext(ctx),
		"describe_model", cfg.DescribeModel)

	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/chat/completions", bytes.NewReader(payload))
	req.Header = r.Header.Clone()
	for _, h := range []string{"Content-Length", "Content-Encoding", "Accept-Encoding", HeaderAIRProxyClient, HeaderLegacyAIRProxyClient} {
		req.Header.Del(h)
	}
	dropRepresentationIntegrityHeaders(req.Header)
	req.Header.Set("Content-Type", "application/json")
	req.ContentLength = int64(len(payload))
	req.RemoteAddr = r.RemoteAddr

	rec := httptest.NewRecorder()
	p.proxyRequest(rec, req)
	if rec.Code != http.StatusOK {
		return "", fmt.Errorf("describe model returned status %d: %s", rec.Code, truncateForLog(rec.Body.String(), 300))
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		return "", fmt.Errorf("decode describe response: %w", err)
	}
	if len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
		return "", fmt.Errorf("describe model returned an empty description")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// visionDescribeContext derives the context of an internal describe call from the
// caller's one. It keeps client cancellation, the trace span (describe calls show up
// in the caller's trace) and in-process identity, and replaces what belongs to one
// request only: a new request_id (it is the spend-log key of the describe call; the
// parent one is logged next to it), no LiteLLM-compat response state and no native
// WebSocket routing. Tried credentials and the credential denylist are reset by
// proxyRequest itself.
func visionDescribeContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	ctx := context.WithValue(parent, visionDescribeCtxKey{}, requestid.FromContext(parent))
	ctx = context.WithValue(ctx, responseCompatContextKey{}, (*responseCompatRequest)(nil))
	ctx = context.WithValue(ctx, nativeWSRoutingKey{}, (*nativeWSRouting)(nil))
	ctx = requestid.WithID(ctx, requestid.New())
	if timeout > 0 {
		return context.WithTimeout(ctx, timeout)
	}
	return context.WithCancel(ctx)
}

func truncateForLog(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func visionPlaceholderText(current bool) string {
	if current {
		return "[image omitted: the model cannot see images]"
	}
	return "[image from an earlier turn omitted]"
}

func visionTextPart(format visionBodyFormat, text string) map[string]any {
	if format == visionFormatResponses {
		return map[string]any{"type": "input_text", "text": text}
	}
	return map[string]any{"type": "text", "text": text}
}

func decodeVisionBody(body []byte) (map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber() // keep numbers (seed, max_tokens, logit_bias...) exactly as sent
	var root map[string]any
	if err := dec.Decode(&root); err != nil {
		return nil, err
	}
	return root, nil
}

// marshalVisionJSON encodes a rewritten body or a describe request without HTML
// escaping. A variable so tests can make it fail.
var marshalVisionJSON = func(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// collectVisionImages finds every image part of the conversation and the user text of
// the current turn (everything after the last model output).
func collectVisionImages(root map[string]any, format visionBodyFormat) ([]visionImageRef, string) {
	items, _ := root[visionItemsKey(format)].([]any)
	if len(items) == 0 {
		return nil, ""
	}

	boundary := -1
	for i, raw := range items {
		if item, ok := raw.(map[string]any); ok && isVisionModelOutput(item, format) {
			boundary = i
		}
	}
	answers := visionTurnAnswers(items, format)

	var refs []visionImageRef
	var question []string
	number := 0 // described images of the turn so far, numbered as on the turn they were described
	for i, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if isVisionModelOutput(item, format) {
			number = 0
		}
		current := i > boundary
		first := len(refs)
		for _, field := range []string{"content", "output"} {
			parts, ok := item[field].([]any)
			if !ok {
				continue
			}
			refs = collectVisionParts(refs, parts, format, current)
		}
		if !current {
			for j := first; j < len(refs); j++ {
				if refs[j].url == "" {
					continue
				}
				number++
				if d, ok := answers[i][number]; ok {
					refs[j].restored = &d
				}
			}
		}
		if current && item["role"] == "user" {
			question = append(question, visionItemText(item["content"])...)
		}
	}

	text := strings.TrimSpace(strings.Join(question, "\n"))
	if runes := []rune(text); len(runes) > maxVisionQuestionChars {
		text = string(runes[len(runes)-maxVisionQuestionChars:]) // the latest text is the most relevant
	}
	return refs, text
}

// visionTurnAnswers maps every conversation item that is not a model output to the
// image descriptions recorded in the model outputs that answered its turn (the run
// of model outputs right after it). Items of the current turn get nil.
func visionTurnAnswers(items []any, format visionBodyFormat) []map[int]visionDescription {
	answers := make([]map[int]visionDescription, len(items))
	var answer map[int]visionDescription
	inRun := false
	for i := len(items) - 1; i >= 0; i-- {
		item, _ := items[i].(map[string]any)
		if item != nil && isVisionModelOutput(item, format) {
			if !inRun {
				answer = make(map[int]visionDescription)
				inRun = true
			}
			parseVisionMarkers(answer, visionItemText(item["content"]))
			continue
		}
		inRun = false
		answers[i] = answer
	}
	return answers
}

func visionItemsKey(format visionBodyFormat) string {
	if format == visionFormatResponses {
		return "input"
	}
	return "messages"
}

// isVisionModelOutput reports whether a conversation item was produced by the model,
// so everything after it belongs to the current turn.
func isVisionModelOutput(item map[string]any, format visionBodyFormat) bool {
	if item["role"] == "assistant" {
		return true
	}
	if format != visionFormatResponses {
		return false
	}
	itemType, _ := item["type"].(string)
	return itemType == "reasoning" || strings.HasSuffix(itemType, "_call")
}

// collectVisionParts appends the image parts of one content array, descending into
// nested tool results (Anthropic tool_result.content).
func collectVisionParts(refs []visionImageRef, parts []any, format visionBodyFormat, current bool) []visionImageRef {
	for idx, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if url, isImage := visionImageURL(part, format); isImage {
			refs = append(refs, visionImageRef{container: parts, index: idx, url: url, current: current})
			continue
		}
		if nested, ok := part["content"].([]any); ok {
			refs = collectVisionParts(refs, nested, format, current)
		}
	}
	return refs
}

// visionImageURL recognizes an image part and returns a URL the describe model can
// fetch (data: or http(s)). isImage is true even when no usable URL exists.
func visionImageURL(part map[string]any, format visionBodyFormat) (url string, isImage bool) {
	partType, _ := part["type"].(string)
	if format == visionFormatMessages {
		if partType != "image" {
			return "", false
		}
		// Same conversion as the Messages-to-Chat converter, so an image the
		// converter forwards is also one that can be described.
		return anthropicconv.SourceToImageURL(part["source"]), true
	}

	// Chat Completions: {"type":"image_url","image_url":{"url":...}}; Responses:
	// {"type":"input_image","image_url":"..."}. Both shapes are accepted for either,
	// as SDKs mix them. An input_image with only file_id has no usable URL.
	wantType := "image_url"
	if format == visionFormatResponses {
		wantType = "input_image"
	}
	if partType != wantType {
		return "", false
	}
	switch v := part["image_url"].(type) {
	case string:
		return v, true
	case map[string]any:
		u, _ := v["url"].(string)
		return u, true
	}
	return "", true
}

// visionItemText returns the text of a message content (string or text parts).
func visionItemText(content any) []string {
	switch v := content.(type) {
	case string:
		return []string{v}
	case []any:
		var out []string
		for _, raw := range v {
			part, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			switch part["type"] {
			case "text", "input_text", "output_text":
				if text, ok := part["text"].(string); ok {
					out = append(out, text)
				}
			}
		}
		return out
	}
	return nil
}
