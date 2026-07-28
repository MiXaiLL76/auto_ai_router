package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

const maxModelAliasSSELineBytes = 1 << 20

func rewriteResponseModelAlias(body []byte, realModelID, displayModelID string) []byte {
	if displayModelID == "" {
		return body
	}
	if rewritten, ok := rewriteTopLevelResponseModel(body, displayModelID); ok {
		return rewritten
	}
	if realModelID == "" || realModelID == displayModelID {
		return body
	}
	return openai.ReplaceModelInBody(body, realModelID, displayModelID)
}

func newOpenAIModelAliasReader(source io.Reader, realModelID, displayModelID string) io.Reader {
	if source == nil || displayModelID == "" {
		return source
	}
	return &modelAliasSSEReader{
		source:         bufio.NewReader(source),
		realModelID:    realModelID,
		displayModelID: displayModelID,
	}
}

type modelAliasSSEReader struct {
	source         *bufio.Reader
	realModelID    string
	displayModelID string
	pending        []byte
	line           []byte
	passthrough    bool
	terminalErr    error
}

func (r *modelAliasSSEReader) Read(p []byte) (int, error) {
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

func (r *modelAliasSSEReader) readNextFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.source.ReadSlice('\n')
		if r.passthrough {
			if len(fragment) > 0 {
				r.pending = fragment
			}
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

func (r *modelAliasSSEReader) consumeFragment(fragment []byte, readErr error) {
	if errors.Is(readErr, bufio.ErrBufferFull) {
		if len(r.line)+len(fragment) <= maxModelAliasSSELineBytes {
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
		if len(line) <= maxModelAliasSSELineBytes {
			r.pending = rewriteResponseModelAlias(line, r.realModelID, r.displayModelID)
			return
		}
	}
	r.pending = line
}

func rewriteTopLevelResponseModel(body []byte, displayModelID string) ([]byte, bool) {
	if rewritten, ok := rewriteJSONResponseModel(body, displayModelID); ok {
		return rewritten, true
	}
	return rewriteSSEDataResponseModel(body, displayModelID)
}

func rewriteSSEDataResponseModel(line []byte, displayModelID string) ([]byte, bool) {
	if !bytes.HasPrefix(line, []byte("data:")) {
		return nil, false
	}

	payloadStart := len("data:")
	for payloadStart < len(line) && (line[payloadStart] == ' ' || line[payloadStart] == '\t') {
		payloadStart++
	}

	payload := line[payloadStart:]
	payload, ending := splitLineEnding(payload)
	if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
		return nil, false
	}

	rewrittenPayload, ok := rewriteJSONResponseModel(payload, displayModelID)
	if !ok {
		return nil, false
	}

	out := make([]byte, 0, payloadStart+len(rewrittenPayload)+len(ending))
	out = append(out, line[:payloadStart]...)
	out = append(out, rewrittenPayload...)
	out = append(out, ending...)
	return out, true
}

func splitLineEnding(line []byte) ([]byte, []byte) {
	switch {
	case bytes.HasSuffix(line, []byte("\r\n")):
		return line[:len(line)-2], line[len(line)-2:]
	case bytes.HasSuffix(line, []byte("\n")):
		return line[:len(line)-1], line[len(line)-1:]
	default:
		return line, nil
	}
}

func rewriteJSONResponseModel(body []byte, displayModelID string) ([]byte, bool) {
	if len(bytes.TrimSpace(body)) == 0 {
		return nil, false
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, false
	}

	rawModel, ok := obj["model"]
	if !ok {
		return nil, false
	}

	var currentModel string
	if err := json.Unmarshal(rawModel, &currentModel); err != nil {
		return nil, false
	}
	if currentModel == displayModelID {
		return body, true
	}

	model, err := json.Marshal(displayModelID)
	if err != nil {
		return nil, false
	}
	obj["model"] = model

	rewritten, err := json.Marshal(obj)
	if err != nil {
		return nil, false
	}
	return rewritten, true
}
