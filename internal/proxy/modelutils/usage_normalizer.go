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

const maxSSEUsageLineBytes = 1 << 20

func NormalizeCompletionUsage(body []byte, modelID string) ([]byte, bool) {
	if !isQwenModel(modelID) {
		return body, false
	}
	var response map[string]json.RawMessage
	if json.Unmarshal(body, &response) != nil {
		return body, false
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

	textTokens, textOK := nonNegativeJSONInteger(details["text_tokens"])
	reasoningTokens, reasoningOK := nonNegativeJSONInteger(details["reasoning_tokens"])
	if !textOK || !reasoningOK || reasoningTokens > completionTokens {
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

func nonNegativeJSONInteger(raw json.RawMessage) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	value, err := strconv.ParseInt(string(raw), 10, 64)
	return value, err == nil && value >= 0
}

func NewUsageNormalizingReadCloser(source io.ReadCloser, modelID string) (io.ReadCloser, bool) {
	if source == nil || !isQwenModel(modelID) {
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

	normalizedPayload, changed := NormalizeCompletionUsage(payload, modelID)
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
