package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Image descriptions made by the vision fallback are written into the assistant
// answer as collapsible blocks, so the client keeps them in its history and later
// turns can restore them instead of describing the image again (the router keeps
// no state):
//
//	<details type="air-vision" n="1" model="qwen-vl">
//	<summary>Image 1 described by qwen-vl</summary>
//	...description...
//	</details>
//
// Open WebUI renders <details> as a collapsed block and sends the answer text back
// verbatim; the newlines after the opening tag and after </summary> are required by
// its markdown extension. The blocks are removed from assistant messages before the
// conversation goes upstream again.

// visionMarkerNeedle is a cheap pre-check for bodies that may carry a marker.
const visionMarkerNeedle = "air-vision"

var visionMarkerRe = regexp.MustCompile(`(?s)<details type="air-vision" n="(\d+)" model="([^"]*)">\s*<summary>.*?</summary>\s*(.*?)\s*</details>[ \t]*\n*`)

// visionDescription is one image description of a turn; n numbers the described
// images of that turn in order.
type visionDescription struct {
	n     int
	model string
	text  string
}

// visionResponseInjection is what gets prepended to the answer of a request whose
// current-turn images were described.
type visionResponseInjection struct {
	prefix  string
	choices int // Chat Completions n
}

// sanitizeVisionDescription makes a description safe to embed in a marker: it can
// never close the block early. The same text is used in the upstream request, so
// the restored text of later turns is identical and vLLM's prefix cache still hits.
func sanitizeVisionDescription(text string) string {
	return strings.TrimSpace(strings.ReplaceAll(text, "</details>", "</ details>"))
}

// visionDescribedText is the text an image is replaced with, on the turn it is
// described and on every later turn it is restored.
func visionDescribedText(n int, describeModel, modelID, text string) string {
	return fmt.Sprintf("[Image %d, described by %s because %s cannot see images]\n%s", n, describeModel, modelID, text)
}

var visionMarkerAttrReplacer = strings.NewReplacer(`"`, "", "<", "", ">", "", "\n", " ")

func buildVisionMarker(descs []visionDescription) string {
	var b strings.Builder
	for _, d := range descs {
		model := visionMarkerAttrReplacer.Replace(d.model)
		fmt.Fprintf(&b, "<details type=\"air-vision\" n=\"%d\" model=\"%s\">\n<summary>Image %d described by %s</summary>\n%s\n</details>\n\n",
			d.n, model, d.n, model, d.text)
	}
	return b.String()
}

// parseVisionMarkers returns the descriptions found in texts, keyed by n.
func parseVisionMarkers(descs map[int]visionDescription, texts []string) {
	for _, text := range texts {
		if !strings.Contains(text, visionMarkerNeedle) {
			continue
		}
		for _, m := range visionMarkerRe.FindAllStringSubmatch(text, -1) {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			if _, dup := descs[n]; !dup {
				descs[n] = visionDescription{n: n, model: m[2], text: m[3]}
			}
		}
	}
}

func stripVisionMarkerText(text string) (string, bool) {
	if !strings.Contains(text, visionMarkerNeedle) {
		return text, false
	}
	stripped := visionMarkerRe.ReplaceAllString(text, "")
	return stripped, stripped != text
}

// stripVisionMarkers removes the markers from every model output of the
// conversation. Text parts that become empty are dropped; a Responses message item
// left without content is dropped as well.
func stripVisionMarkers(root map[string]any, format visionBodyFormat) bool {
	key := visionItemsKey(format)
	items, _ := root[key].([]any)
	changed := false
	kept := items[:0:0]
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok || !isVisionModelOutput(item, format) {
			kept = append(kept, raw)
			continue
		}
		itemChanged, empty := stripVisionMarkersFromContent(item)
		changed = changed || itemChanged
		if itemChanged && empty && format == visionFormatResponses && item["type"] == "message" {
			continue
		}
		kept = append(kept, raw)
	}
	if changed {
		root[key] = kept
	}
	return changed
}

// stripVisionMarkersFromContent strips the markers from item["content"] and reports
// whether it changed and whether the content is now empty.
func stripVisionMarkersFromContent(item map[string]any) (changed, empty bool) {
	switch content := item["content"].(type) {
	case string:
		stripped, ok := stripVisionMarkerText(content)
		if ok {
			item["content"] = stripped
		}
		return ok, strings.TrimSpace(stripped) == ""
	case []any:
		parts := content[:0:0]
		for _, raw := range content {
			part, isMap := raw.(map[string]any)
			text, isText := part["text"].(string)
			if !isMap || !isText {
				parts = append(parts, raw)
				continue
			}
			stripped, ok := stripVisionMarkerText(text)
			if !ok {
				parts = append(parts, raw)
				continue
			}
			changed = true
			if strings.TrimSpace(stripped) == "" {
				continue
			}
			part["text"] = stripped
			parts = append(parts, part)
		}
		if !changed {
			return false, false
		}
		if len(parts) == 0 {
			item["content"] = ""
			return true, true
		}
		item["content"] = parts
		return true, false
	}
	return false, false
}

// visionInjectAllowed reports whether a description may be prepended to the answer:
// not when the client asked for machine-readable output (structured output or a
// forced tool call).
func visionInjectAllowed(root map[string]any, format visionBodyFormat) bool {
	switch format {
	case visionFormatChat:
		if rf, ok := root["response_format"].(map[string]any); ok && rf["type"] != nil && rf["type"] != "text" {
			return false
		}
		return !visionToolChoiceForced(root["tool_choice"])
	case visionFormatResponses:
		if text, ok := root["text"].(map[string]any); ok {
			if f, ok := text["format"].(map[string]any); ok && f["type"] != nil && f["type"] != "text" {
				return false
			}
		}
		return !visionToolChoiceForced(root["tool_choice"])
	case visionFormatMessages:
		if root["output_format"] != nil {
			return false
		}
		if oc, ok := root["output_config"].(map[string]any); ok && oc["format"] != nil {
			return false
		}
		if tc, ok := root["tool_choice"].(map[string]any); ok && (tc["type"] == "any" || tc["type"] == "tool") {
			return false
		}
		return true
	}
	return false
}

func visionToolChoiceForced(toolChoice any) bool {
	switch tc := toolChoice.(type) {
	case string:
		return tc == "required"
	case map[string]any:
		if tc["type"] != "allowed_tools" {
			return true // a named function / custom tool
		}
		mode := tc["mode"] // Responses
		if nested, ok := tc["allowed_tools"].(map[string]any); ok {
			mode = nested["mode"] // Chat Completions
		}
		return mode == "required"
	}
	return false
}

func visionChoiceCount(root map[string]any) int {
	if n, ok := root["n"].(json.Number); ok {
		if v, err := n.Int64(); err == nil && v > 1 && v <= 128 {
			return int(v)
		}
	}
	return 1
}

// visionInjectionFor returns the descriptions to write into a successful answer whose
// upstream format is Chat Completions or Responses, nil otherwise. A /v1/messages
// passthrough (upstream answers in Anthropic format) is not written into.
func visionInjectionFor(logCtx *RequestLogContext, prepared *orchestratedRequest, status int) *visionResponseInjection {
	if logCtx.visionInject == nil || prepared.nativeResponses || prepared.passthroughMessages ||
		status < 200 || status >= 300 {
		return nil
	}
	return logCtx.visionInject
}

// injectVisionIntoResponse prepends the descriptions to a non-streaming answer in
// the upstream format (Chat Completions, or Responses for Responses passthrough).
// Conversions to the client format run afterwards and carry the text along.
func injectVisionIntoResponse(body []byte, inj *visionResponseInjection, responsesFormat bool) ([]byte, bool) {
	root, err := decodeVisionBody(body)
	if err != nil {
		return body, false
	}
	var changed bool
	if responsesFormat {
		changed = injectVisionIntoResponsesOutput(root, inj.prefix)
	} else {
		changed = injectVisionIntoChatChoices(root, inj.prefix)
	}
	if !changed {
		return body, false
	}
	out, err := marshalVisionJSON(root)
	if err != nil {
		return body, false
	}
	return out, true
}

func injectVisionIntoChatChoices(root map[string]any, prefix string) bool {
	choices, _ := root["choices"].([]any)
	changed := false
	for _, raw := range choices {
		choice, _ := raw.(map[string]any)
		msg, _ := choice["message"].(map[string]any)
		if msg == nil {
			continue
		}
		switch content := msg["content"].(type) {
		case nil:
			msg["content"] = strings.TrimRight(prefix, "\n")
		case string:
			msg["content"] = prefix + content
		case []any:
			msg["content"] = append([]any{map[string]any{"type": "text", "text": prefix}}, content...)
		default:
			continue
		}
		changed = true
	}
	return changed
}

func injectVisionIntoResponsesOutput(root map[string]any, prefix string) bool {
	output, ok := root["output"].([]any)
	if !ok {
		return false
	}
	if text, ok := root["output_text"].(string); ok {
		root["output_text"] = prefix + text
	}
	for _, raw := range output {
		item, _ := raw.(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		parts, _ := item["content"].([]any)
		for _, rawPart := range parts {
			if part, _ := rawPart.(map[string]any); part != nil && part["type"] == "output_text" {
				text, _ := part["text"].(string)
				part["text"] = prefix + text
				return true
			}
		}
		item["content"] = append([]any{visionOutputTextPart(prefix)}, parts...)
		return true
	}
	// No message item (tool calls only): add one after the leading reasoning.
	at := 0
	for at < len(output) {
		if item, _ := output[at].(map[string]any); item == nil || item["type"] != "reasoning" {
			break
		}
		at++
	}
	msg := map[string]any{
		"type": "message", "id": "msg_air_vision", "role": "assistant", "status": "completed",
		"content": []any{visionOutputTextPart(strings.TrimRight(prefix, "\n"))},
	}
	root["output"] = append(output[:at:at], append([]any{msg}, output[at:]...)...)
	return true
}

func visionOutputTextPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
}

// newVisionStreamInjector wraps an upstream SSE body so the descriptions reach the
// client as the start of the answer text.
func newVisionStreamInjector(body io.ReadCloser, inj *visionResponseInjection, responsesFormat bool) io.ReadCloser {
	r := &visionSSEInjector{source: bufio.NewReader(body), closer: body, inj: inj}
	if responsesFormat {
		r.rewrite = r.rewriteResponsesLine
	} else {
		r.rewrite = r.rewriteChatLine
		r.seen = make(map[int]bool)
		r.injected = make(map[int]bool)
	}
	return r
}

type visionSSEInjector struct {
	source  *bufio.Reader
	closer  io.Closer
	inj     *visionResponseInjection
	rewrite func(line []byte) []byte
	pending []byte
	err     error
	done    bool // nothing left to rewrite: lines pass through untouched

	// Chat Completions: per choice index
	seen     map[int]bool
	injected map[int]bool
	skipped  bool // a chunk passed without being decoded: the role was already sent

	// Responses: an "event:" line waits for its data line, so events can be
	// inserted before the pair.
	heldEvent []byte

	// Responses: the output_text part that carries the prefix
	target struct {
		set          bool
		itemID       string
		contentIndex string
	}
}

func (r *visionSSEInjector) Read(p []byte) (int, error) {
	for len(r.pending) == 0 {
		if r.err != nil {
			return 0, r.err
		}
		line, err := r.source.ReadBytes('\n')
		if len(line) > 0 {
			if r.done {
				r.pending = line
			} else {
				r.pending = r.rewrite(line)
			}
		}
		if err != nil {
			r.pending = append(r.pending, r.heldEvent...)
			r.heldEvent = nil
			r.err = err
		}
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *visionSSEInjector) Close() error { return r.closer.Close() }

// sseDataPayload splits "data: {...}\n" into the prefix, JSON payload and line ending.
func sseDataPayload(line []byte) (head, payload, ending []byte, ok bool) {
	if !bytes.HasPrefix(line, []byte("data:")) {
		return nil, nil, nil, false
	}
	start := len("data:")
	for start < len(line) && (line[start] == ' ' || line[start] == '\t') {
		start++
	}
	payload, ending = splitLineEnding(line[start:])
	if len(payload) == 0 || payload[0] != '{' {
		return nil, nil, nil, false
	}
	return line[:start], payload, ending, true
}

func sseDataLine(head []byte, event any, ending []byte) ([]byte, bool) {
	payload, err := marshalVisionJSON(event)
	if err != nil {
		return nil, false
	}
	return append(append(append([]byte(nil), head...), payload...), ending...), true
}

// rewriteChatLine inserts a synthetic content chunk before the first chunk of each
// choice that carries answer content, a tool call or the finish reason. Reasoning
// chunks pass first, so the description starts the answer text, not the reasoning.
func (r *visionSSEInjector) rewriteChatLine(line []byte) []byte {
	head, payload, ending, ok := sseDataPayload(line)
	if !ok {
		return line
	}
	// Reasoning models stream hundreds of chunks before the answer starts: decode
	// only chunks that may start it.
	if !visionChatChunkMayStartAnswer(payload) {
		r.skipped = true
		return line
	}
	chunk, err := decodeVisionBody(payload)
	if err != nil {
		return line
	}
	choices, _ := chunk["choices"].([]any)
	var out []byte
	for _, raw := range choices {
		choice, _ := raw.(map[string]any)
		if choice == nil {
			continue
		}
		index := 0
		if n, ok := choice["index"].(json.Number); ok {
			if v, err := n.Int64(); err == nil {
				index = int(v)
			}
		}
		if !r.injected[index] && visionChatChunkStartsAnswer(choice) {
			delta := map[string]any{"content": r.inj.prefix}
			if !r.seen[index] && !r.skipped {
				delta["role"] = "assistant"
			}
			synthetic := make(map[string]any, len(chunk))
			for k, v := range chunk {
				if k != "choices" && k != "usage" {
					synthetic[k] = v
				}
			}
			synthetic["choices"] = []any{map[string]any{"index": index, "delta": delta, "finish_reason": nil}}
			if data, ok := sseDataLine(head, synthetic, ending); ok {
				out = append(out, data...)
				if len(ending) > 0 {
					out = append(out, ending...) // blank line ends the SSE event
				}
			}
			r.injected[index] = true
		}
		r.seen[index] = true
	}
	if len(r.injected) >= r.inj.choices {
		r.done = true
	}
	if out == nil {
		return line
	}
	return append(out, line...)
}

// visionChatChunkMayStartAnswer is a byte-level pre-check for
// visionChatChunkStartsAnswer: a tool call, a finish reason or non-empty content.
func visionChatChunkMayStartAnswer(payload []byte) bool {
	if bytes.Contains(payload, []byte(`"tool_calls"`)) {
		return true
	}
	return visionJSONStringFieldNonEmpty(payload, `"finish_reason":`) || visionJSONStringFieldNonEmpty(payload, `"content":`)
}

// visionJSONStringFieldNonEmpty reports whether key (quoted, with the colon) is
// followed by a non-empty string somewhere in payload. "reasoning_content" does not
// match "content": the needle starts with the quote.
func visionJSONStringFieldNonEmpty(payload []byte, key string) bool {
	for rest := payload; ; {
		i := bytes.Index(rest, []byte(key))
		if i < 0 {
			return false
		}
		rest = bytes.TrimLeft(rest[i+len(key):], " \t")
		if len(rest) >= 2 && rest[0] == '"' && rest[1] != '"' {
			return true
		}
	}
}

func visionChatChunkStartsAnswer(choice map[string]any) bool {
	if choice["finish_reason"] != nil {
		return true
	}
	delta, _ := choice["delta"].(map[string]any)
	if content, _ := delta["content"].(string); content != "" {
		return true
	}
	calls, _ := delta["tool_calls"].([]any)
	return len(calls) > 0
}

// rewriteResponsesLine prepends the descriptions to the first output_text of the
// answer and keeps every later event that repeats that text (the .done events and
// the final response) consistent with it. An answer without any output_text (tool
// calls only) gets a message item with the descriptions appended before the final
// response event.
func (r *visionSSEInjector) rewriteResponsesLine(line []byte) []byte {
	if bytes.HasPrefix(line, []byte("event:")) {
		r.heldEvent = append([]byte(nil), line...)
		return nil
	}
	held := r.heldEvent
	r.heldEvent = nil
	before, data := r.rewriteResponsesData(line)
	return slices.Concat(before, held, data)
}

// visionResponsesEventTypes are the events rewriteResponsesData may change, before
// and after the prefixed output_text is known.
var (
	visionResponsesTypesBefore = []string{`"response.output_text.delta"`, `"response.completed"`, `"response.incomplete"`}
	visionResponsesTypesAfter  = []string{`"response.output_text.done"`, `"response.content_part.done"`, `"response.output_item.done"`, `"response.completed"`, `"response.incomplete"`}
)

func (r *visionSSEInjector) rewriteResponsesData(line []byte) (before, data []byte) {
	head, payload, ending, ok := sseDataPayload(line)
	if !ok {
		return nil, line
	}
	types := visionResponsesTypesBefore
	if r.target.set {
		types = visionResponsesTypesAfter
	}
	relevant := false
	for _, t := range types {
		if bytes.Contains(payload, []byte(t)) {
			relevant = true
			break
		}
	}
	if !relevant {
		return nil, line
	}
	event, err := decodeVisionBody(payload)
	if err != nil {
		return nil, line
	}
	prefix := r.inj.prefix
	changed := false
	switch event["type"] {
	case "response.output_text.delta":
		if !r.target.set {
			delta, _ := event["delta"].(string)
			event["delta"] = prefix + delta
			r.target.set = true
			r.target.itemID, _ = event["item_id"].(string)
			r.target.contentIndex = fmt.Sprint(event["content_index"])
			changed = true
		}
	case "response.output_text.done":
		if r.isTarget(event["item_id"], event["content_index"]) {
			text, _ := event["text"].(string)
			event["text"] = prefix + text
			changed = true
		}
	case "response.content_part.done":
		if part, _ := event["part"].(map[string]any); part != nil && r.isTarget(event["item_id"], event["content_index"]) {
			text, _ := part["text"].(string)
			part["text"] = prefix + text
			changed = true
		}
	case "response.output_item.done":
		if item, _ := event["item"].(map[string]any); item != nil {
			changed = r.prefixTargetItem(item)
		}
	case "response.completed", "response.incomplete":
		if resp, _ := event["response"].(map[string]any); resp != nil {
			if r.target.set {
				output, _ := resp["output"].([]any)
				for _, raw := range output {
					if item, _ := raw.(map[string]any); item != nil && r.prefixTargetItem(item) {
						changed = true
					}
				}
				if text, ok := resp["output_text"].(string); ok {
					resp["output_text"] = prefix + text
					changed = true
				}
			} else {
				before = r.appendResponsesMessageItem(event, resp, head, ending)
				changed = before != nil
			}
		}
		r.done = true
	}
	if !changed {
		return nil, line
	}
	if out, ok := sseDataLine(head, event, ending); ok {
		return before, out
	}
	return nil, line
}

// appendResponsesMessageItem adds a message item holding the descriptions to the
// final response and returns the events that stream it (added, text, done), numbered
// before the final event.
func (r *visionSSEInjector) appendResponsesMessageItem(final, resp map[string]any, head, ending []byte) []byte {
	output, _ := resp["output"].([]any)
	outputIndex := len(output)
	text := strings.TrimRight(r.inj.prefix, "\n")
	const itemID = "msg_air_vision"
	item := map[string]any{
		"type": "message", "id": itemID, "role": "assistant", "status": "completed",
		"content": []any{visionOutputTextPart(text)},
	}
	part := map[string]any{"item_id": itemID, "output_index": outputIndex, "content_index": 0}
	withPart := func(fields map[string]any) map[string]any {
		for k, v := range part {
			fields[k] = v
		}
		return fields
	}
	events := []map[string]any{
		{"type": "response.output_item.added", "output_index": outputIndex, "item": map[string]any{
			"type": "message", "id": itemID, "role": "assistant", "status": "in_progress", "content": []any{},
		}},
		withPart(map[string]any{"type": "response.content_part.added", "part": visionOutputTextPart("")}),
		withPart(map[string]any{"type": "response.output_text.delta", "delta": text}),
		withPart(map[string]any{"type": "response.output_text.done", "text": text}),
		withPart(map[string]any{"type": "response.content_part.done", "part": visionOutputTextPart(text)}),
		{"type": "response.output_item.done", "output_index": outputIndex, "item": item},
	}
	var seq int64 = -1
	if n, ok := final["sequence_number"].(json.Number); ok {
		if v, err := n.Int64(); err == nil {
			seq = v
		}
	}
	var out []byte
	for i, event := range events {
		if seq >= 0 {
			event["sequence_number"] = seq + int64(i)
		}
		data, ok := sseDataLine(head, event, ending)
		if !ok {
			return nil
		}
		out = append(out, "event: "+event["type"].(string)+string(ending)...)
		out = append(append(out, data...), ending...)
	}
	if seq >= 0 {
		final["sequence_number"] = seq + int64(len(events))
	}
	resp["output"] = append(output, item)
	if existing, ok := resp["output_text"].(string); ok {
		resp["output_text"] = r.inj.prefix + existing
	}
	return out
}

func (r *visionSSEInjector) isTarget(itemID, contentIndex any) bool {
	id, _ := itemID.(string)
	return r.target.set && id == r.target.itemID && fmt.Sprint(contentIndex) == r.target.contentIndex
}

func (r *visionSSEInjector) prefixTargetItem(item map[string]any) bool {
	if !r.target.set || item["id"] != r.target.itemID {
		return false
	}
	parts, _ := item["content"].([]any)
	index, err := strconv.Atoi(r.target.contentIndex)
	if err != nil || index < 0 || index >= len(parts) {
		return false
	}
	part, _ := parts[index].(map[string]any)
	if part == nil {
		return false
	}
	text, _ := part["text"].(string)
	part["text"] = r.inj.prefix + text
	return true
}
