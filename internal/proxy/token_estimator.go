package proxy

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode"
	"unicode/utf8"

	// goccyjson is used only for appendChatCompletionDeltaText/appendResponsesDeltaText
	// (small, per-SSE-chunk typed unmarshals in the streaming hot path) — benchmarked
	// separately from this file's other encoding/json uses, which stay on stdlib.
	goccyjson "github.com/goccy/go-json"
	"github.com/tiktoken-go/tokenizer"
)

// Fixed OpenAI Chat Completions framing tokens around messages and the assistant reply.
const (
	openAIChatTokensPerMessage = 3
	openAIChatTokensPerName    = 1
	openAIChatReplyPrimer      = 3
)

const (
	tokenizerMaxExactWordBytes     = 4096
	tokenizerLongWordSampleBytes   = 256
	tokenizerLongWordFallbackRatio = 4
)

var openAIInjectedPromptFields = []string{"tools", "functions", "response_format"}

// estimatePromptTokens estimates the number of prompt tokens from the request body.
func estimatePromptTokens(body []byte) int {
	return estimatePromptTokensForModel(body, "")
}

func estimatePromptTokensForModel(body []byte, model string) int {
	if len(body) == 0 {
		return 0
	}

	raw, ok := decodeRequestBody(body)
	if !ok {
		return 0
	}
	if model == "" {
		model, _ = raw["model"].(string)
	}

	enc := tokenizerForModel(model)
	if tokens := countOpenAIChatPromptTokensFromRaw(enc, raw); tokens > 0 {
		return tokens
	}
	if tokens := countPromptTextTokens(enc, raw); tokens > 0 {
		return tokens
	}
	return 1
}

func decodeRequestBody(body []byte) (map[string]interface{}, bool) {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, false
	}
	return raw, true
}

func countOpenAIChatPromptTokensFromRaw(enc tokenizer.Codec, raw map[string]interface{}) int {
	messages, ok := raw["messages"].([]interface{})
	if !ok {
		return 0
	}

	total := openAIChatReplyPrimer
	for _, item := range messages {
		msg, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		total += countOpenAIMessageTokens(enc, msg)
	}

	for _, key := range openAIInjectedPromptFields {
		if value, exists := raw[key]; exists {
			total += countCompactJSONTokens(enc, value)
		}
	}

	return total
}

func countPromptTextTokens(enc tokenizer.Codec, raw map[string]interface{}) int {
	total := 0
	for _, key := range []string{"input", "instructions", "prompt", "system"} {
		if value, exists := raw[key]; exists {
			total += countPromptValueTokens(enc, value)
		}
	}
	for _, key := range openAIInjectedPromptFields {
		if value, exists := raw[key]; exists {
			total += countCompactJSONTokens(enc, value)
		}
	}
	return total
}

func countPromptValueTokens(enc tokenizer.Codec, value interface{}) int {
	switch v := value.(type) {
	case string:
		return countTextTokens(enc, v)
	case []interface{}:
		total := 0
		for _, item := range v {
			total += countPromptValueTokens(enc, item)
		}
		return total
	case map[string]interface{}:
		total := 0
		if content, exists := v["content"]; exists {
			total += countPromptValueTokens(enc, content)
		}
		if input, exists := v["input"]; exists {
			total += countPromptValueTokens(enc, input)
		}
		if text, ok := v["text"].(string); ok {
			total += countTextTokens(enc, text)
		}
		return total
	default:
		return 0
	}
}

func countOpenAIMessageTokens(enc tokenizer.Codec, msg map[string]interface{}) int {
	total := openAIChatTokensPerMessage
	for key, value := range msg {
		switch key {
		case "name":
			if text, ok := value.(string); ok {
				total += countTextTokens(enc, text) + openAIChatTokensPerName
			}
		case "role":
			if text, ok := value.(string); ok {
				total += countTextTokens(enc, text)
			}
		case "content":
			total += countMessageContentTokens(enc, value)
		case "tool_calls", "function_call":
			total += countCompactJSONTokens(enc, value)
		case "tool_call_id", "refusal", "reasoning_content":
			if text, ok := value.(string); ok {
				total += countTextTokens(enc, text)
			}
		}
	}
	return total
}

func countMessageContentTokens(enc tokenizer.Codec, content interface{}) int {
	switch v := content.(type) {
	case string:
		return countTextTokens(enc, v)
	case []interface{}:
		total := 0
		for _, part := range v {
			block, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch block["type"] {
			case "text", "input_text", "output_text":
				if text, ok := block["text"].(string); ok {
					total += countTextTokens(enc, text)
				}
			case "tool_result":
				total += countMessageContentTokens(enc, block["content"])
			default:
				if text, ok := block["text"].(string); ok {
					total += countTextTokens(enc, text)
				}
			}
		}
		return total
	default:
		return 0
	}
}

func countCompactJSONTokens(enc tokenizer.Codec, value interface{}) int {
	data, err := json.Marshal(value)
	if err != nil {
		return 0
	}
	return countTextTokens(enc, string(data))
}

func countTextTokensForModel(model, text string) int {
	return countTextTokens(tokenizerForModel(model), text)
}

func countTextTokens(enc tokenizer.Codec, text string) int {
	if enc == nil || text == "" {
		return 0
	}
	if hasLongWord(text) {
		return countTextTokensWithEstimatedLongWords(enc, text)
	}
	return countTextTokensDirect(enc, text)
}

func countTextTokensDirect(enc tokenizer.Codec, text string) int {
	count, err := enc.Count(text)
	if err != nil {
		return 0
	}
	return count
}

func countTextTokensWithEstimatedLongWords(enc tokenizer.Codec, text string) int {
	total := 0
	segmentStart := 0
	wordStart := -1

	for offset, r := range text {
		if unicode.IsSpace(r) {
			if isLongWord(text, wordStart, offset) {
				total += countTextTokensDirect(enc, text[segmentStart:wordStart])
				total += estimateLongWordTokens(enc, text[wordStart:offset])
				segmentStart = offset
			}
			wordStart = -1
			continue
		}
		if wordStart < 0 {
			wordStart = offset
		}
	}

	if isLongWord(text, wordStart, len(text)) {
		total += countTextTokensDirect(enc, text[segmentStart:wordStart])
		total += estimateLongWordTokens(enc, text[wordStart:])
		return total
	}

	return total + countTextTokensDirect(enc, text[segmentStart:])
}

func estimateLongWordTokens(enc tokenizer.Codec, text string) int {
	samples := longWordSamples(text)
	sampleBytes := 0
	sampleTokens := 0
	for _, sample := range samples {
		sampleBytes += len(sample)
		sampleTokens += countTextTokensDirect(enc, sample)
	}
	if sampleBytes == 0 || sampleTokens == 0 {
		return estimateLongWordTokensByBytes(len(text))
	}
	return max(1, (len(text)*sampleTokens+sampleBytes-1)/sampleBytes)
}

func longWordSamples(text string) []string {
	if len(text) <= tokenizerLongWordSampleBytes {
		return []string{text}
	}

	return []string{
		textSample(text, 0),
		textSample(text, len(text)/2-tokenizerLongWordSampleBytes/2),
		textSample(text, len(text)-tokenizerLongWordSampleBytes),
	}
}

func textSample(text string, start int) string {
	if start < 0 {
		start = 0
	}
	if maxStart := len(text) - tokenizerLongWordSampleBytes; start > maxStart {
		start = maxStart
	}

	start = previousRuneBoundary(text, start)
	end := nextRuneBoundary(text, start+tokenizerLongWordSampleBytes)
	return text[start:end]
}

func estimateLongWordTokensByBytes(bytes int) int {
	return max(1, (bytes+tokenizerLongWordFallbackRatio-1)/tokenizerLongWordFallbackRatio)
}

func hasLongWord(text string) bool {
	wordStart := -1
	for offset, r := range text {
		if unicode.IsSpace(r) {
			wordStart = -1
			continue
		}
		if wordStart < 0 {
			wordStart = offset
		}
		if isLongWord(text, wordStart, offset) {
			return true
		}
	}
	return isLongWord(text, wordStart, len(text))
}

func isLongWord(text string, start, end int) bool {
	return start >= 0 && end-start > tokenizerMaxExactWordBytes
}

func previousRuneBoundary(text string, offset int) int {
	for offset > 0 && !utf8.RuneStart(text[offset]) {
		offset--
	}
	return offset
}

func nextRuneBoundary(text string, offset int) int {
	if offset >= len(text) {
		return len(text)
	}
	for offset < len(text) && !utf8.RuneStart(text[offset]) {
		offset++
	}
	return offset
}

func tokenizerForModel(model string) tokenizer.Codec {
	normalized := normalizeOpenAIModelName(model)
	if normalized != "" {
		if enc, err := tokenizer.ForModel(tokenizer.Model(normalized)); err == nil {
			return enc
		}
	}

	switch {
	case strings.HasPrefix(normalized, "gpt-5"),
		strings.HasPrefix(normalized, "gpt-4.1"),
		strings.HasPrefix(normalized, "gpt-4.5"),
		strings.HasPrefix(normalized, "gpt-4o"),
		strings.HasPrefix(normalized, "chatgpt-4o"),
		strings.HasPrefix(normalized, "o1"),
		strings.HasPrefix(normalized, "o3"),
		strings.HasPrefix(normalized, "o4"):
		return tokenizerForEncoding(tokenizer.O200kBase)
	case strings.HasPrefix(normalized, "gpt-4"),
		strings.HasPrefix(normalized, "gpt-3.5"),
		strings.HasPrefix(normalized, "gpt-35"),
		strings.HasPrefix(normalized, "ft:gpt-4"),
		strings.HasPrefix(normalized, "ft:gpt-3.5"):
		return tokenizerForEncoding(tokenizer.Cl100kBase)
	default:
		return tokenizerForEncoding(tokenizer.O200kBase)
	}
}

func tokenizerForEncoding(encoding tokenizer.Encoding) tokenizer.Codec {
	enc, err := tokenizer.Get(encoding)
	if err != nil {
		return nil
	}
	return enc
}

func normalizeOpenAIModelName(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	for _, prefix := range []string{"openai/", "azure/"} {
		model = strings.TrimPrefix(model, prefix)
	}
	return model
}

type completionTokenAccumulator struct {
	model string
	text  strings.Builder
	// payloadBuf is reused across AddChunk calls (dst[:0] pattern) so a
	// stream's per-chunk delta-text extraction doesn't allocate a fresh
	// [][]byte on every chunk. Safe because each accumulator is created fresh
	// per stream (see newCompletionTokenAccumulator) — never shared across
	// concurrent streams.
	payloadBuf [][]byte
}

func newCompletionTokenAccumulator(model string) *completionTokenAccumulator {
	return &completionTokenAccumulator{model: model}
}

// newCompletionTokenAccumulator returns nil when tiktoken_enabled=false, so the per-chunk
// AddChunk/TokenCount calls in the streaming hot path become no-ops (both are already
// nil-safe) instead of doing a wasted per-chunk JSON decode whose result is then discarded.
func (p *Proxy) newCompletionTokenAccumulator(model string) *completionTokenAccumulator {
	if !p.tiktokenEnabled {
		return nil
	}
	return newCompletionTokenAccumulator(model)
}

// setPromptTokensEstimate arms logCtx's lazy local-tokenizer fallback estimate for the
// given request body/model. The estimator itself (full body JSON decode + tiktoken BPE)
// only runs if a consumer later calls logCtx.promptTokensEstimate() — i.e. only when the
// provider genuinely didn't report usage. body must not be mutated by the caller after
// this point (verified true at every current call site: the retry loops that could
// reassign body/proxyBody have already broken out by the time streaming starts).
// No-op (leaves the estimator unarmed, so promptTokensEstimate() returns 0) when
// tiktoken_enabled=false.
func (p *Proxy) setPromptTokensEstimate(logCtx *RequestLogContext, body []byte, model string) {
	if logCtx == nil || !p.tiktokenEnabled {
		return
	}
	logCtx.promptTokensEstimateFn = func() int {
		return estimatePromptTokensForModel(body, model)
	}
}

func (a *completionTokenAccumulator) AddChunk(chunk []byte) {
	if a == nil {
		return
	}
	a.payloadBuf = splitSSEPayloads(chunk, a.payloadBuf)
	a.appendPayloadsText(a.payloadBuf)
}

// AddPayloads adds delta text from payloads a caller already split out of the
// current chunk (e.g. tokenCapturingWriter.Write or an onChunk closure that
// splits once and shares the sub-slices with every consumer) — avoids a
// second, redundant split of the same chunk that AddChunk would otherwise do.
func (a *completionTokenAccumulator) AddPayloads(payloads [][]byte) {
	if a == nil {
		return
	}
	a.appendPayloadsText(payloads)
}

func (a *completionTokenAccumulator) appendPayloadsText(payloads [][]byte) {
	for _, payload := range payloads {
		appendDeltaTextForPayload(&a.text, payload)
	}
}

func (a *completionTokenAccumulator) TokenCount() int {
	if a == nil || a.text.Len() == 0 {
		return 0
	}
	return countTextTokensForModel(a.model, a.text.String())
}

func extractCompletionDeltaText(chunk []byte) string {
	payloads := splitSSEPayloads(chunk, nil)
	if len(payloads) == 0 {
		return ""
	}

	var b strings.Builder
	for _, payload := range payloads {
		appendDeltaTextForPayload(&b, payload)
	}
	return b.String()
}

// appendDeltaTextForPayload picks which shape-specific decoder(s) to run for a
// single SSE payload. A stream is always either Chat-Completions-shaped or
// Responses-shaped, never both, so unconditionally running both
// appendChatCompletionDeltaText and appendResponsesDeltaText — as this used
// to do — means one of the two always does a full, wasted json.Unmarshal of
// the same bytes. A raw substring check is a cheap, allocation-free way to
// rule out the shape that can't match before paying for its unmarshal; a
// false positive (the substring appearing inside delta text itself, e.g. a
// model streaming a code sample containing `"type"`) only costs one harmless
// extra unmarshal that finds no matching top-level field, same as today —
// never a wrong result. (A map[string]json.RawMessage pre-decode was tried
// first and measured slower here: the extra map/RawMessage allocations
// outweigh the unmarshal it saves at this payload size.)
func appendDeltaTextForPayload(b *strings.Builder, payload []byte) {
	if bytes.Contains(payload, []byte(`"choices"`)) {
		appendChatCompletionDeltaText(b, payload)
	}
	if bytes.Contains(payload, []byte(`"content_block_delta"`)) {
		appendMessagesDeltaText(b, payload)
	}
	if bytes.Contains(payload, []byte(`"type"`)) {
		appendResponsesDeltaText(b, payload)
	}
}

func appendChatCompletionDeltaText(b *strings.Builder, payload []byte) {
	var data struct {
		Choices []struct {
			Delta struct {
				Content      interface{} `json:"content"`
				Refusal      string      `json:"refusal"`
				FunctionCall *struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function_call,omitempty"`
				ToolCalls []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := goccyjson.Unmarshal(payload, &data); err != nil {
		return
	}
	for _, choice := range data.Choices {
		appendDeltaValueText(b, choice.Delta.Content)
		b.WriteString(choice.Delta.Refusal)
		if choice.Delta.FunctionCall != nil {
			b.WriteString(choice.Delta.FunctionCall.Name)
			b.WriteString(choice.Delta.FunctionCall.Arguments)
		}
		for _, call := range choice.Delta.ToolCalls {
			b.WriteString(call.Function.Name)
			b.WriteString(call.Function.Arguments)
		}
	}
}

func appendResponsesDeltaText(b *strings.Builder, payload []byte) {
	var event struct {
		Type  string      `json:"type"`
		Delta interface{} `json:"delta"`
	}
	if err := goccyjson.Unmarshal(payload, &event); err != nil || event.Type == "" {
		return
	}

	switch event.Type {
	case "response.output_text.delta",
		"response.refusal.delta",
		"response.reasoning_text.delta",
		"response.reasoning_summary_text.delta",
		"response.function_call_arguments.delta",
		"response.mcp_call_arguments.delta",
		"response.custom_tool_call_input.delta",
		"response.code_interpreter_call_code.delta":
		appendDeltaValueText(b, event.Delta)
	}
}

func appendMessagesDeltaText(b *strings.Builder, payload []byte) {
	var event struct {
		Type  string `json:"type"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			PartialJSON string `json:"partial_json"`
		} `json:"delta"`
	}
	if err := goccyjson.Unmarshal(payload, &event); err != nil || event.Type != "content_block_delta" {
		return
	}
	switch event.Delta.Type {
	case "text_delta":
		b.WriteString(event.Delta.Text)
	case "thinking_delta":
		b.WriteString(event.Delta.Thinking)
	case "input_json_delta":
		b.WriteString(event.Delta.PartialJSON)
	}
}

func appendDeltaValueText(b *strings.Builder, value interface{}) {
	switch v := value.(type) {
	case string:
		b.WriteString(v)
	case []interface{}:
		for _, item := range v {
			block, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := block["text"].(string); ok {
				b.WriteString(text)
			}
		}
	}
}
