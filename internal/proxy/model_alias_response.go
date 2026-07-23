package proxy

import (
	"bufio"
	"errors"
	"io"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

const maxModelAliasSSELineBytes = 1 << 20

func rewriteResponseModelAlias(body []byte, realModelID, displayModelID string) []byte {
	if realModelID == "" || displayModelID == "" || realModelID == displayModelID {
		return body
	}
	return openai.ReplaceModelInBody(body, realModelID, displayModelID)
}

func newOpenAIModelAliasReader(source io.Reader, realModelID, displayModelID string) io.Reader {
	if source == nil || realModelID == "" || displayModelID == "" || realModelID == displayModelID {
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
