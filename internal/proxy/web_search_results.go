package proxy

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"slices"
	"strings"

	json "github.com/goccy/go-json"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

// A Z.AI web_search tool only returns its results array when the request asks
// for it, and that array is the only evidence a (billable) search ran. The
// request side (openai.ForceWebSearchResults) therefore always asks for it;
// the helpers below remove the array again from responses to clients that did
// not, after it has been counted for billing.

var webSearchResultsNeedle = []byte(`"web_search"`)

const maxWebSearchResultsSSELineBytes = 16 << 20

// webSearchResultsHiddenFromClient reports whether the client request enables
// a Z.AI web_search tool without asking for its results. Only Chat Completions
// carry that tool shape.
func webSearchResultsHiddenFromClient(path string, body []byte) bool {
	return isChatCompletionsPath(path) && openai.WebSearchResultsHidden(body)
}

func isChatCompletionsPath(path string) bool {
	return strings.HasSuffix(strings.TrimSuffix(path, "/"), "/chat/completions")
}

// forceWebSearchResultsForPath asks for the results on a request forwarded
// as-is to a remote router, so that router returns them for billing here
// instead of stripping them as if this router were the end client.
func forceWebSearchResultsForPath(path string, body []byte) []byte {
	if !isChatCompletionsPath(path) {
		return body
	}
	return openai.ForceWebSearchResults(body)
}

// stripWebSearchResults removes the top-level "web_search" key from a JSON
// response object. It reports false and leaves body untouched when there is
// nothing to remove.
func stripWebSearchResults(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, webSearchResultsNeedle) {
		return body, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil || obj == nil {
		return body, false
	}
	if _, ok := obj["web_search"]; !ok {
		return body, false
	}
	delete(obj, "web_search")
	stripped, err := json.Marshal(obj)
	if err != nil {
		return body, false
	}
	return stripped, true
}

// stripSSEDataWebSearchResults applies stripWebSearchResults to the JSON
// payload of one SSE data line, keeping the prefix and line ending. It also
// returns the original payload so the caller can bill what was removed.
func stripSSEDataWebSearchResults(line []byte) (out []byte, removedFrom []byte) {
	if !bytes.HasPrefix(line, []byte("data:")) || !bytes.Contains(line, webSearchResultsNeedle) {
		return line, nil
	}
	payloadStart := len("data:")
	for payloadStart < len(line) && (line[payloadStart] == ' ' || line[payloadStart] == '\t') {
		payloadStart++
	}
	payload, ending := splitLineEnding(line[payloadStart:])
	stripped, ok := stripWebSearchResults(payload)
	if !ok {
		return line, nil
	}
	out = make([]byte, 0, payloadStart+len(stripped)+len(ending))
	out = append(out, line[:payloadStart]...)
	out = append(out, stripped...)
	out = append(out, ending...)
	return out, payload
}

// newWebSearchResultsStripReader strips the "web_search" results from every
// SSE data line of source. onStripped gets the original JSON payload of each
// line it rewrote, before the results are gone, so they can still be billed.
func newWebSearchResultsStripReader(source io.Reader, onStripped func(payload []byte)) io.Reader {
	if source == nil {
		return source
	}
	return &webSearchResultsStripReader{source: bufio.NewReader(source), onStripped: onStripped}
}

type webSearchResultsStripReader struct {
	source      *bufio.Reader
	onStripped  func(payload []byte)
	pending     []byte
	line        []byte
	passthrough bool
	terminalErr error
}

func (r *webSearchResultsStripReader) Read(p []byte) (int, error) {
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

func (r *webSearchResultsStripReader) readNextFragment() {
	for len(r.pending) == 0 && r.terminalErr == nil {
		fragment, err := r.source.ReadSlice('\n')
		bufferFull := errors.Is(err, bufio.ErrBufferFull)
		switch {
		case r.passthrough:
			// Tail of a line too long to buffer: forward it unchanged.
			r.pending = slices.Clone(fragment)
			r.passthrough = bufferFull
		case bufferFull && len(r.line)+len(fragment) <= maxWebSearchResultsSSELineBytes:
			r.line = append(r.line, fragment...)
		case bufferFull:
			r.pending = slices.Concat(r.line, fragment)
			r.line = nil
			r.passthrough = true
		default:
			line := slices.Concat(r.line, fragment)
			r.line = nil
			if len(line) > 0 {
				out, removedFrom := stripSSEDataWebSearchResults(line)
				if removedFrom != nil && r.onStripped != nil {
					r.onStripped(removedFrom)
				}
				r.pending = out
			}
		}
		if err != nil && !bufferFull {
			r.terminalErr = err
		}
	}
}
