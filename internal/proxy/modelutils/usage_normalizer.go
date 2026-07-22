package modelutils

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"strings"
)

const (
	qwenReasoningUsageModel = "qwen3.6-35b-a3b"
	maxQwenSSELineBytes     = 1 << 20
)

// NormalizeCompletionUsage applies model-specific response usage fixes. It
// intentionally fails open: malformed or ambiguous payloads are returned
// byte-for-byte unchanged.
func NormalizeCompletionUsage(body []byte, modelID string) ([]byte, bool) {
	return normalizeQwenCompletionUsage(body, modelID)
}

// NewUsageNormalizingReadCloser wraps streaming responses when a model-specific
// SSE usage normalizer exists.
func NewUsageNormalizingReadCloser(source io.ReadCloser, modelID string) (io.ReadCloser, bool) {
	if source == nil || !isQwenReasoningUsageModel(modelID) {
		return source, false
	}
	return newQwenUsageNormalizingReadCloser(source, modelID), true
}

// normalizeQwenCompletionUsage fixes the overlapping text/reasoning token
// breakdown observed for qwen3.6-35b-a3b.
func normalizeQwenCompletionUsage(body []byte, modelID string) ([]byte, bool) {
	if !isQwenReasoningUsageModel(modelID) {
		return body, false
	}

	var response map[string]json.RawMessage
	if err := json.Unmarshal(body, &response); err != nil {
		return body, false
	}

	usageRaw, ok := response["usage"]
	if !ok {
		return body, false
	}

	var usage map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usage); err != nil {
		return body, false
	}

	completionTokens, ok := positiveJSONInteger(usage["completion_tokens"])
	if !ok {
		return body, false
	}

	detailsRaw, ok := usage["completion_tokens_details"]
	if !ok {
		return body, false
	}

	var details map[string]json.RawMessage
	if err := json.Unmarshal(detailsRaw, &details); err != nil {
		return body, false
	}

	textRaw, textPresent := details["text_tokens"]
	if !textPresent {
		return body, false
	}
	textTokens, ok := positiveJSONInteger(textRaw)
	if !ok {
		return body, false
	}

	reasoningTokens, ok := positiveJSONInteger(details["reasoning_tokens"])
	if !ok || reasoningTokens > completionTokens {
		return body, false
	}

	expectedTextTokens := completionTokens - reasoningTokens
	if textTokens <= expectedTextTokens {
		return body, false
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

func positiveJSONInteger(raw json.RawMessage) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}

	value, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil || value <= 0 {
		return 0, false
	}
	return value, true
}

func isQwenReasoningUsageModel(modelID string) bool {
	modelName := strings.ToLower(strings.TrimSpace(modelID))
	if slash := strings.LastIndexByte(modelName, '/'); slash >= 0 {
		modelName = modelName[slash+1:]
	}
	return modelName == qwenReasoningUsageModel || strings.HasPrefix(modelName, qwenReasoningUsageModel+"-")
}

// qwenUsageNormalizingReadCloser processes complete SSE lines. Using a
// synchronous reader is important here: callers can continue draining this
// exact reader after a downstream client disconnects without losing buffered
// upstream bytes in a separate goroutine.
type qwenUsageNormalizingReadCloser struct {
	source      io.ReadCloser
	reader      *bufio.Reader
	modelID     string
	pending     []byte
	line        []byte
	passthrough bool
	terminalErr error
}

func newQwenUsageNormalizingReadCloser(source io.ReadCloser, modelID string) io.ReadCloser {
	if source == nil || !isQwenReasoningUsageModel(modelID) {
		return source
	}
	return &qwenUsageNormalizingReadCloser{
		source:  source,
		reader:  bufio.NewReader(source),
		modelID: modelID,
	}
}

func (r *qwenUsageNormalizingReadCloser) Read(p []byte) (int, error) {
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

		r.readNextSSEFragment()
	}

	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *qwenUsageNormalizingReadCloser) readNextSSEFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.reader.ReadSlice('\n')

		if r.passthrough {
			if len(fragment) > 0 {
				r.pending = fragment
			}
			if !errors.Is(err, bufio.ErrBufferFull) {
				r.passthrough = false
			}
		} else {
			r.consumeBufferedSSEFragment(fragment, err)
		}

		if err != nil && !errors.Is(err, bufio.ErrBufferFull) {
			r.terminalErr = err
		}
	}
}

func (r *qwenUsageNormalizingReadCloser) consumeBufferedSSEFragment(fragment []byte, readErr error) {
	if errors.Is(readErr, bufio.ErrBufferFull) {
		if len(r.line)+len(fragment) <= maxQwenSSELineBytes {
			r.line = append(r.line, fragment...)
			return
		}

		r.pending = append(r.line, fragment...)
		r.line = nil
		r.passthrough = true
		return
	}

	line := append(r.line, fragment...)
	r.line = nil
	if len(line) == 0 {
		return
	}

	if readErr == nil || errors.Is(readErr, io.EOF) {
		if len(line) <= maxQwenSSELineBytes {
			r.pending = normalizeQwenSSELine(line, r.modelID)
			return
		}
	}

	// A line above the limit or interrupted by a transport error is ambiguous.
	// Preserve it exactly and resume normalization at the next complete line.
	r.pending = line
}

func (r *qwenUsageNormalizingReadCloser) Close() error {
	return r.source.Close()
}

func normalizeQwenSSELine(line []byte, modelID string) []byte {
	content, ending := splitSSELineEnding(line)
	if !bytes.HasPrefix(content, []byte("data:")) {
		return line
	}

	rawPayload := content[len("data:"):]
	payload := bytes.TrimSpace(rawPayload)
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || payload[0] != '{' {
		return line
	}

	normalizedPayload, changed := normalizeQwenCompletionUsage(payload, modelID)
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
