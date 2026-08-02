package litellm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
)

type streamReader struct {
	ctx             Context
	source          *bufio.Reader
	pending         bytes.Buffer
	id              string
	created         any
	choices         map[int]*choiceStreamState
	sentChatPrelude bool
	sentDone        bool
}

type choiceStreamState struct {
	sentRole      bool
	sentFinish    bool
	sawTools      bool
	pendingFinish string
}

func newStreamReader(ctx Context, source io.Reader) io.Reader {
	return &streamReader{
		ctx:     ctx,
		source:  bufio.NewReader(source),
		choices: make(map[int]*choiceStreamState),
	}
}

func (r *streamReader) Read(target []byte) (int, error) {
	for r.pending.Len() == 0 && !r.sentDone {
		if err := r.readFrame(); err != nil {
			if err != io.EOF {
				return 0, err
			}
			r.finish()
		}
	}
	if r.pending.Len() > 0 {
		return r.pending.Read(target)
	}
	return 0, io.EOF
}

func (r *streamReader) readFrame() error {
	var lines []string
	for {
		line, err := r.source.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				break
			}
			lines = append(lines, trimmed)
		}
		if err != nil {
			if len(lines) > 0 {
				break
			}
			return err
		}
	}
	if len(lines) == 0 {
		return nil
	}

	dataIndex := -1
	for index, line := range lines {
		if strings.HasPrefix(line, "data:") {
			dataIndex = index
			break
		}
	}
	if dataIndex < 0 {
		r.writeFrame(strings.Join(lines, "\n"))
		return nil
	}

	payload := strings.TrimSpace(strings.TrimPrefix(lines[dataIndex], "data:"))
	if payload == "[DONE]" {
		r.finish()
		return nil
	}

	var body map[string]any
	if json.Unmarshal([]byte(payload), &body) != nil {
		r.writeFrame(strings.Join(lines, "\n"))
		return nil
	}

	switch r.ctx.Endpoint {
	case "/v1/chat/completions":
		if !r.normalizeChatChunk(body) {
			return nil
		}
		if !r.sentChatPrelude {
			r.writeChatPrelude()
		}
	case "/v1/completions":
		if !r.normalizeTextChunk(body) {
			return nil
		}
	default:
		r.normalizeResponsesChunk(body)
	}

	encoded, err := json.Marshal(stripNulls(body, false))
	if err != nil {
		return err
	}
	lines[dataIndex] = "data: " + string(encoded)
	r.writeFrame(strings.Join(lines, "\n"))
	r.writePendingFinishes()
	return nil
}

func (r *streamReader) normalizeTextChunk(body map[string]any) bool {
	body["object"] = "text_completion"
	if r.ctx.RequestedModel != "" {
		body["model"] = r.ctx.RequestedModel
	}
	if r.id == "" {
		r.id, _ = body["id"].(string)
		if r.id == "" {
			r.id = "cmpl-" + uuid.NewString()
		}
	}
	body["id"] = r.id
	if r.created == nil {
		r.created = body["created"]
		if r.created == nil {
			r.created = time.Now().Unix()
		}
	}
	body["created"] = r.created
	if usage, ok := body["usage"].(map[string]any); ok {
		normalizeUsage(usage)
		if !r.ctx.IncludeUsage {
			delete(body, "usage")
		}
	}
	choices, _ := body["choices"].([]any)
	for index, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		if choice["index"] == nil {
			choice["index"] = index
		}
		if reason, _ := choice["finish_reason"].(string); reason != "" {
			choice["finish_reason"] = mapFinishReason(reason)
		}
	}
	_, hasUsage := body["usage"]
	return len(choices) > 0 || hasUsage
}

func (r *streamReader) normalizeChatChunk(body map[string]any) bool {
	usage, hasUsage := body["usage"].(map[string]any)
	hasUsage = hasUsage && usage != nil
	if !hasUsage {
		delete(body, "usage")
	}
	choices, _ := body["choices"].([]any)
	if len(choices) == 0 && !hasUsage {
		return false
	}

	body["object"] = "chat.completion.chunk"
	if r.ctx.RequestedModel != "" {
		body["model"] = r.ctx.RequestedModel
	}
	if r.id == "" {
		r.id, _ = body["id"].(string)
		if r.id == "" {
			r.id = "chatcmpl-" + uuid.NewString()
		}
	}
	body["id"] = r.id
	if r.created == nil {
		r.created = body["created"]
		if r.created == nil {
			r.created = time.Now().Unix()
		}
	}
	body["created"] = r.created

	if hasUsage {
		if !r.ctx.IncludeUsage {
			delete(body, "usage")
			hasUsage = false
		} else {
			body["usage"] = liteLLMStreamUsage()
			for _, field := range []string{"latency_checkpoint", "obfuscation", "service_tier", "system_fingerprint"} {
				delete(body, field)
			}
		}
	}

	for index, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		hadContentFilter := choice["content_filter_results"] != nil || choice["content_filter_offsets"] != nil
		delete(choice, "content_filter_results")
		delete(choice, "content_filter_offsets")
		if choice["index"] == nil {
			choice["index"] = index
		}
		choiceIndex := streamChoiceIndex(choice, index)
		state := r.choiceState(choiceIndex)
		delta, ok := choice["delta"].(map[string]any)
		if !ok {
			delta = map[string]any{}
			choice["delta"] = delta
		}
		if reasoning, ok := delta["reasoning"]; ok {
			if _, exists := delta["reasoning_content"]; !exists {
				delta["reasoning_content"] = reasoning
			}
			delete(delta, "reasoning")
		}
		hasDelta := hasStreamDelta(delta)
		if hadContentFilter && !hasDelta && choice["finish_reason"] == nil {
			return false
		}
		if toolCalls, ok := delta["tool_calls"].([]any); ok && len(toolCalls) > 0 {
			state.sawTools = true
		}
		if !state.sentRole {
			delta["role"] = "assistant"
			state.sentRole = true
		} else {
			delete(delta, "role")
		}

		if reason, _ := choice["finish_reason"].(string); reason != "" {
			if reason == "stop" && state.sawTools {
				reason = "tool_calls"
			}
			reason = mapFinishReason(reason)
			if hasDelta {
				choice["finish_reason"] = nil
				state.pendingFinish = reason
			} else {
				choice["finish_reason"] = reason
				state.sentFinish = true
			}
		}
	}
	if hasUsage {
		body["choices"] = r.usageChoices()
	}
	return len(choices) > 0 || hasUsage
}

func (r *streamReader) writeChatPrelude() {
	r.writeFrame("data: " + string(mustMarshal(map[string]any{
		"id":      "",
		"object":  "chat.completion.chunk",
		"created": r.created,
		"model":   r.ctx.RequestedModel,
		"choices": []any{},
	})))
	r.sentChatPrelude = true
}

func (r *streamReader) usageChoices() []any {
	indexes := r.choiceIndexes(func(state *choiceStreamState) bool { return state.sentRole })
	if len(indexes) == 0 {
		indexes = []int{0}
	}
	choices := make([]any, 0, len(indexes))
	for _, index := range indexes {
		choices = append(choices, map[string]any{"index": index, "delta": map[string]any{}})
	}
	return choices
}

func liteLLMStreamUsage() map[string]any {
	return map[string]any{
		"completion_tokens":         0,
		"completion_tokens_details": map[string]any{"reasoning_tokens": 0},
		"prompt_tokens":             0,
		"total_tokens":              0,
	}
}

func mustMarshal(value any) []byte {
	encoded, _ := json.Marshal(value)
	return encoded
}

func hasStreamDelta(delta map[string]any) bool {
	for _, field := range []string{"content", "tool_calls", "function_call", "reasoning_content", "audio"} {
		if value, ok := delta[field]; ok && value != nil {
			return true
		}
	}
	return false
}

func streamChoiceIndex(choice map[string]any, fallback int) int {
	value, ok := choice["index"].(float64)
	if !ok || value < 0 || value != float64(int(value)) {
		return fallback
	}
	return int(value)
}

func (r *streamReader) choiceState(index int) *choiceStreamState {
	state := r.choices[index]
	if state == nil {
		state = &choiceStreamState{}
		r.choices[index] = state
	}
	return state
}

func (r *streamReader) writePendingFinishes() {
	indexes := r.choiceIndexes(func(state *choiceStreamState) bool {
		return state.pendingFinish != ""
	})
	if len(indexes) == 0 {
		return
	}

	choices := make([]any, 0, len(indexes))
	for _, index := range indexes {
		state := r.choices[index]
		choices = append(choices, terminalChoice(index, state.pendingFinish))
		state.pendingFinish = ""
		state.sentFinish = true
	}

	body := r.terminalChunk(choices)
	encoded, _ := json.Marshal(stripNulls(body, false))
	r.writeFrame("data: " + string(encoded))
}

func (r *streamReader) normalizeResponsesChunk(body map[string]any) {
	if response, ok := body["response"].(map[string]any); ok && r.ctx.RequestedModel != "" {
		response["model"] = r.ctx.RequestedModel
	}
	if errorBody, ok := body["error"].(map[string]any); ok && errorBody["code"] == nil {
		errorBody["code"] = "unknown_error"
	}
}

func (r *streamReader) finish() {
	if r.sentDone {
		return
	}
	if r.ctx.Endpoint == "/v1/chat/completions" {
		indexes := r.choiceIndexes(func(state *choiceStreamState) bool {
			return state.sentRole && !state.sentFinish
		})
		if len(indexes) > 0 {
			choices := make([]any, 0, len(indexes))
			for _, index := range indexes {
				choices = append(choices, terminalChoice(index, "stop"))
				r.choices[index].sentFinish = true
			}
			encoded, _ := json.Marshal(stripNulls(r.terminalChunk(choices), false))
			r.writeFrame("data: " + string(encoded))
		}
	}
	r.writeFrame("data: [DONE]")
	r.sentDone = true
}

func (r *streamReader) choiceIndexes(include func(*choiceStreamState) bool) []int {
	indexes := make([]int, 0, len(r.choices))
	for index, state := range r.choices {
		if include(state) {
			indexes = append(indexes, index)
		}
	}
	sort.Ints(indexes)
	return indexes
}

func (r *streamReader) terminalChunk(choices []any) map[string]any {
	return map[string]any{
		"id":      r.id,
		"object":  "chat.completion.chunk",
		"created": r.created,
		"model":   r.ctx.RequestedModel,
		"choices": choices,
	}
}

func terminalChoice(index int, finishReason string) map[string]any {
	return map[string]any{
		"index":         index,
		"delta":         map[string]any{"content": nil},
		"finish_reason": finishReason,
	}
}

func (r *streamReader) writeFrame(frame string) {
	r.pending.WriteString(frame)
	r.pending.WriteString("\n\n")
}
