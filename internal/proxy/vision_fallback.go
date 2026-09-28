package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// HeaderVisionFallback tells the client that image inputs were rewritten because the
// requested model cannot see images. Values: "described", "stripped".
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
	url       string // data: or http(s) URL; empty when the image cannot be described (file_id)
	current   bool   // belongs to the turn after the last assistant message
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
	rejected  bool   // mode reject and the body carries images
	outcome   string // HeaderVisionFallback value; "" when the body was not changed
	images    int
	described int
}

// applyVisionFallback rewrites image inputs of a request to a vLLM model configured
// with supports_vision: false. Bodies without images, models that support images (or
// never declared supports_vision) and models served by any non-vLLM credential are
// returned unchanged.
func (p *Proxy) applyVisionFallback(r *http.Request, body []byte, modelID string, format visionBodyFormat) visionFallbackResult {
	unchanged := visionFallbackResult{body: body}
	if p.modelManager == nil || !bytes.Contains(body, []byte("image")) {
		return unchanged
	}
	if supported, known := p.modelManager.SupportsVision(modelID); !known || supported {
		return unchanged
	}
	if !p.servedOnlyByVLLM(modelID) {
		return unchanged
	}

	root, err := decodeVisionBody(body)
	if err != nil {
		return unchanged
	}
	refs, question := collectVisionImages(root, format)
	if len(refs) == 0 {
		return unchanged
	}

	mode := p.visionFallback.Mode
	if mode == config.VisionFallbackDescribe && isVisionDescribeRequest(r.Context()) {
		mode = config.VisionFallbackStrip
	}
	if mode != config.VisionFallbackStrip && mode != config.VisionFallbackDescribe {
		return visionFallbackResult{body: body, rejected: true, images: len(refs)}
	}

	replacements := make([]string, len(refs))
	for i := range refs {
		replacements[i] = visionPlaceholderText(refs[i].current)
	}
	described := 0
	if mode == config.VisionFallbackDescribe {
		described = p.describeVisionImages(r, refs, question, modelID, replacements)
	}
	for i, ref := range refs {
		ref.container[ref.index] = visionTextPart(format, replacements[i])
	}

	newBody, err := marshalVisionJSON(root)
	if err != nil {
		p.logger.WarnContext(r.Context(), "Vision fallback: failed to encode rewritten body", "error", err)
		return unchanged
	}
	outcome := "stripped"
	if described > 0 {
		outcome = "described"
	}
	return visionFallbackResult{body: newBody, outcome: outcome, images: len(refs), described: described}
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

// applyVisionFallbackToBase runs applyVisionFallback on the orchestrator's base
// bodies. It writes the 400 response and returns false when the request is rejected.
func (p *Proxy) applyVisionFallbackToBase(
	w http.ResponseWriter,
	r *http.Request,
	logCtx *RequestLogContext,
	baseBody, baseProxyBody *[]byte,
	modelID, baseRealModelID string,
	format visionBodyFormat,
) bool {
	vision := p.applyVisionFallback(r, *baseBody, modelID, format)
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
		return false
	}
	if vision.outcome == "" {
		return true
	}
	*baseBody = vision.body
	*baseProxyBody = vision.body
	if modelID != baseRealModelID {
		*baseProxyBody = openai.ReplaceModelInBody(vision.body, baseRealModelID, modelID)
	}
	w.Header().Set(HeaderVisionFallback, vision.outcome)
	p.logger.InfoContext(r.Context(), "Rewrote image input for a model without vision support",
		"model", modelID,
		"outcome", vision.outcome,
		"images", vision.images,
		"described", vision.described,
		"describe_model", p.visionFallback.DescribeModel,
		"request_id", logCtx.RequestID)
	return true
}

// describeVisionImages describes the current-turn images (at most max_images) in
// parallel and stores each description in replacements. It returns how many
// descriptions succeeded; failed ones keep their placeholder.
func (p *Proxy) describeVisionImages(r *http.Request, refs []visionImageRef, question, modelID string, replacements []string) int {
	cfg := p.visionFallback
	var wg sync.WaitGroup
	var mu sync.Mutex
	described, number := 0, 0
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
			start := time.Now()
			text, err := p.describeVisionImage(r, url, question)
			if err != nil {
				p.logger.WarnContext(r.Context(), "Vision fallback: describe call failed",
					"model", modelID, "describe_model", cfg.DescribeModel,
					"image", number, "duration", time.Since(start), "error", err)
				replacements[i] = "[image omitted: the image could not be described]"
				return
			}
			replacements[i] = fmt.Sprintf("[Image %d, described by %s because %s cannot see images]\n%s",
				number, cfg.DescribeModel, modelID, text)
			mu.Lock()
			described++
			mu.Unlock()
		}(i, number, ref.url)
	}
	wg.Wait()
	return described
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
	reqBody := map[string]any{
		"model":      cfg.DescribeModel,
		"stream":     false,
		"max_tokens": cfg.MaxTokens,
		"messages": []any{
			map[string]any{"role": "system", "content": cfg.DescribePrompt},
			map[string]any{"role": "user", "content": userContent},
		},
	}
	payload, err := marshalVisionJSON(reqBody)
	if err != nil {
		return "", err
	}

	// A fresh context: the original one carries per-request routing state (tried
	// credentials, denylists) that must not leak into the describe call. Client
	// cancellation is still propagated.
	ctx, cancel := context.WithTimeout(context.WithValue(context.Background(), visionDescribeCtxKey{}, true), cfg.Timeout)
	defer cancel()
	stop := context.AfterFunc(r.Context(), cancel)
	defer stop()

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
	key := "messages"
	if format == visionFormatResponses {
		key = "input"
	}
	items, _ := root[key].([]any)
	if len(items) == 0 {
		return nil, ""
	}

	boundary := -1
	for i, raw := range items {
		if item, ok := raw.(map[string]any); ok && isVisionModelOutput(item, format) {
			boundary = i
		}
	}

	var refs []visionImageRef
	var question []string
	for i, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		current := i > boundary
		for _, field := range []string{"content", "output"} {
			parts, ok := item[field].([]any)
			if !ok {
				continue
			}
			refs = collectVisionParts(refs, parts, format, current)
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
		source, _ := part["source"].(map[string]any)
		switch source["type"] {
		case "base64":
			mediaType, _ := source["media_type"].(string)
			data, _ := source["data"].(string)
			if mediaType != "" && data != "" {
				return "data:" + mediaType + ";base64," + data, true
			}
		case "url":
			u, _ := source["url"].(string)
			return u, true
		}
		return "", true
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
			case "text", "input_text":
				if text, ok := part["text"].(string); ok {
					out = append(out, text)
				}
			}
		}
		return out
	}
	return nil
}
