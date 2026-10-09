// Package modelutils normalizes model usage and token accounting data.
package modelutils

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

const maxSSEUsageLineBytes = 1 << 20

func NormalizeCompletionUsage(body []byte, modelID string) ([]byte, bool) {
	var response map[string]json.RawMessage
	if json.Unmarshal(body, &response) != nil {
		return body, false
	}
	if !isQwenModel(modelID) {
		var responseModel string
		if json.Unmarshal(response["model"], &responseModel) != nil || !isQwenModel(responseModel) {
			return body, false
		}
	}

	usageRaw, ok := response["usage"]
	if !ok {
		return body, false
	}

	var usage map[string]json.RawMessage
	if json.Unmarshal(usageRaw, &usage) != nil {
		return body, false
	}

	completionTokens, ok := nonNegativeJSONInteger(usage["completion_tokens"])
	if !ok {
		return body, false
	}

	detailsRaw, ok := usage["completion_tokens_details"]
	if !ok {
		return body, false
	}

	var details map[string]json.RawMessage
	if json.Unmarshal(detailsRaw, &details) != nil {
		return body, false
	}

	reasoningTokens, reasoningOK := nonNegativeJSONInteger(details["reasoning_tokens"])
	if !reasoningOK || reasoningTokens > completionTokens {
		return body, false
	}

	expectedTextTokens := completionTokens - reasoningTokens
	if textRaw, exists := details["text_tokens"]; exists && string(textRaw) != "null" {
		textTokens, textOK := nonNegativeJSONInteger(textRaw)
		if !textOK || (textTokens > 0 && textTokens <= expectedTextTokens) {
			return body, false
		}
	}

	details["text_tokens"] = json.RawMessage(strconv.AppendInt(nil, expectedTextTokens, 10))
	normalizedDetails, err := json.Marshal(details)
	if err != nil {
		return body, false
	}
	usage["completion_tokens_details"] = normalizedDetails

	normalizedUsage, err := json.Marshal(usage)
	if err != nil {
		return body, false
	}
	response["usage"] = normalizedUsage

	normalizedResponse, err := json.Marshal(response)
	if err != nil {
		return body, false
	}
	return normalizedResponse, true
}

// NormalizeCacheCreationUsage ensures an OpenAI chat-completions response
// carries the standard cache-write naming in usage.prompt_tokens_details
// whenever the cache write is known only under Requesty's spelling
// (caching_tokens / caching_token_details). Downstream usage-metering matches
// the standard cache_creation_tokens / cache_creation_token_details in its
// pricing jsonpaths, but has no path for the Requesty names, so an
// Requesty-only spelling drops the cache write from billing. The Requesty
// fields are preserved; the standard ones are added.
//
// This is deliberately Requesty-only. Other cache-write spellings are already
// matched directly by the pricing jsonpaths (cache_write_tokens, Alibaba's
// cache_creation.ephemeral_*), so synthesizing the standard name for them
// would risk double-counting. Returns the (possibly unchanged) body and
// whether it changed.
func NormalizeCacheCreationUsage(body []byte) ([]byte, bool) {
	// Cheap gate: the only Requesty fields this reads (caching_tokens,
	// caching_token_details, caching_5m/1h_tokens) all contain the substring
	// "caching", so a body without it can never need a rewrite. Skips the JSON
	// parse on the hot non-streaming path for the vast majority of responses.
	if !bytes.Contains(body, []byte("caching")) {
		return body, false
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return body, false
	}
	usageRaw, ok := top["usage"]
	if !ok {
		return body, false
	}
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usage); err != nil {
		return body, false
	}
	detailsRaw, ok := usage["prompt_tokens_details"]
	if !ok {
		return body, false
	}
	// Already carries a positive standard count — leave it untouched.
	if existing, ok := nonNegativeJSONInteger(detailsRawKey(detailsRaw, "cache_creation_tokens")); ok && existing > 0 {
		return body, false
	}
	// Read ONLY Requesty's naming (caching_tokens / caching_token_details).
	// CachingWrite() is the Requesty-extension reader; it ignores the standard,
	// cache_write and Alibaba spellings, so those responses are left alone.
	var td openai.TokenDetails
	if err := json.Unmarshal(detailsRaw, &td); err != nil {
		return body, false
	}
	total, fiveMinutes, oneHour := td.CachingWrite()
	if total <= 0 && fiveMinutes <= 0 && oneHour <= 0 {
		return body, false
	}

	var details map[string]json.RawMessage
	if err := json.Unmarshal(detailsRaw, &details); err != nil {
		return body, false
	}
	if total > 0 {
		details["cache_creation_tokens"] = json.RawMessage(strconv.FormatInt(int64(total), 10))
	}
	if fiveMinutes > 0 || oneHour > 0 {
		ctd := make(map[string]json.RawMessage, 2)
		if fiveMinutes > 0 {
			ctd["ephemeral_5m_input_tokens"] = json.RawMessage(strconv.FormatInt(int64(fiveMinutes), 10))
		}
		if oneHour > 0 {
			ctd["ephemeral_1h_input_tokens"] = json.RawMessage(strconv.FormatInt(int64(oneHour), 10))
		}
		encCTD, err := json.Marshal(ctd)
		if err != nil {
			return body, false
		}
		details["cache_creation_token_details"] = encCTD
	}
	encDetails, err := json.Marshal(details)
	if err != nil {
		return body, false
	}
	usage["prompt_tokens_details"] = encDetails
	encUsage, err := json.Marshal(usage)
	if err != nil {
		return body, false
	}
	top["usage"] = encUsage
	encTop, err := json.Marshal(top)
	if err != nil {
		return body, false
	}
	return encTop, true
}

// detailsRawKey digs a named key out of a raw prompt_tokens_details object.
func detailsRawKey(detailsRaw json.RawMessage, key string) json.RawMessage {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(detailsRaw, &m); err != nil {
		return nil
	}
	return m[key]
}

func nonNegativeJSONInteger(raw json.RawMessage) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	encoded := strings.Trim(string(raw), `"`)
	if value, err := strconv.ParseInt(encoded, 10, 64); err == nil {
		return value, value >= 0
	}
	value, err := strconv.ParseFloat(encoded, 64)
	if err != nil || value < 0 || value > math.MaxInt64 || math.Trunc(value) != value {
		return 0, false
	}
	return int64(value), true
}

func NewUsageNormalizingReadCloser(source io.ReadCloser, modelID string) (io.ReadCloser, bool) {
	if source == nil {
		return source, false
	}
	return &usageNormalizingReadCloser{
		source:  source,
		reader:  bufio.NewReader(source),
		modelID: modelID,
	}, true
}

func isQwenModel(modelID string) bool {
	modelName := strings.ToLower(strings.TrimSpace(modelID))
	if slash := strings.LastIndexByte(modelName, '/'); slash >= 0 {
		modelName = modelName[slash+1:]
	}
	return strings.HasPrefix(modelName, "qwen")
}

type usageNormalizingReadCloser struct {
	source      io.ReadCloser
	reader      *bufio.Reader
	modelID     string
	pending     []byte
	line        []byte
	passthrough bool
	terminalErr error
}

func (r *usageNormalizingReadCloser) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		if r.terminalErr != nil {
			err := r.terminalErr
			if err != io.EOF {
				r.terminalErr = io.EOF
			}
			return 0, err
		}
		r.readNextFragment()
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *usageNormalizingReadCloser) readNextFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.reader.ReadSlice('\n')
		if r.passthrough {
			r.pending = fragment
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
			}
		} else {
			r.consumeFragment(fragment, err)
		}
		if err != nil && !errors.Is(err, bufio.ErrBufferFull) {
			r.terminalErr = err
		}
	}
}

func (r *usageNormalizingReadCloser) consumeFragment(fragment []byte, readErr error) {
	if errors.Is(readErr, bufio.ErrBufferFull) {
		if len(r.line)+len(fragment) <= maxSSEUsageLineBytes {
			r.line = append(r.line, fragment...)
			return
		}
		r.pending = slices.Concat(r.line, fragment)
		r.line = nil
		r.passthrough = true
		return
	}

	line := slices.Concat(r.line, fragment)
	r.line = nil
	if len(line) == 0 {
		return
	}
	if (readErr == nil || errors.Is(readErr, io.EOF)) && len(line) <= maxSSEUsageLineBytes {
		r.pending = normalizeSSEUsageLine(line, r.modelID)
		return
	}
	r.pending = line
}

func (r *usageNormalizingReadCloser) Close() error {
	return r.source.Close()
}

func normalizeSSEUsageLine(line []byte, modelID string) []byte {
	content, ending := splitSSELineEnding(line)
	if !bytes.HasPrefix(content, []byte("data:")) {
		return line
	}

	rawPayload := content[len("data:"):]
	payload := bytes.TrimSpace(rawPayload)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || payload[0] != '{' {
		return line
	}

	// Cheap substring gates before the normalizers' full JSON parse: a streamed
	// content delta carries neither key, so the common case costs a byte scan
	// instead of an unmarshal. Each gate matches the key its normalizer can
	// actually rewrite, so nothing eligible is skipped.
	normalizedPayload := payload
	changed := false
	if bytes.Contains(payload, []byte("completion_tokens_details")) {
		if p, didChange := NormalizeCompletionUsage(payload, modelID); didChange {
			normalizedPayload = p
			changed = true
		}
	}
	// NormalizeCacheCreationUsage self-gates on the "caching" substring, so a
	// content-delta line costs only that scan before it returns unchanged.
	if p, didChange := NormalizeCacheCreationUsage(normalizedPayload); didChange {
		normalizedPayload = p
		changed = true
	}
	if !changed {
		return line
	}

	payloadStart := bytes.Index(rawPayload, payload)
	payloadEnd := payloadStart + len(payload)
	normalized := make([]byte, 0, len(line)-len(payload)+len(normalizedPayload))
	normalized = append(normalized, content[:len("data:")]...)
	normalized = append(normalized, rawPayload[:payloadStart]...)
	normalized = append(normalized, normalizedPayload...)
	normalized = append(normalized, rawPayload[payloadEnd:]...)
	normalized = append(normalized, ending...)
	return normalized
}

func splitSSELineEnding(line []byte) (content, ending []byte) {
	if len(line) == 0 || line[len(line)-1] != '\n' {
		return line, nil
	}
	if len(line) >= 2 && line[len(line)-2] == '\r' {
		return line[:len(line)-2], line[len(line)-2:]
	}
	return line[:len(line)-1], line[len(line)-1:]
}
