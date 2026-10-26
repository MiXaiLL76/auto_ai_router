package proxy

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"slices"
)

// loopGateUsageStripReader drops the running usage that vLLM's
// continuous_usage_stats puts on every chunk of a stream the loop gate asked
// it for: the usage becomes null (as OpenAI sends it on every chunk but the
// last) on each chunk with choices, and the usage-only chunk (choices: [])
// the stream ends with keeps it. The client, and AIR's own usage accounting
// (which adds up the usage it sees in a chat stream), get the stream they
// would have got without the option. The gate below has already read the
// running usage.
type loopGateUsageStripReader struct {
	reader      *bufio.Reader
	line        []byte // the current line, while it is longer than reader's buffer
	pending     []byte
	passthrough bool // the current line outgrew loopGateMaxLine: relayed as is
	err         error
}

func newLoopGateUsageStripReader(source io.Reader) *loopGateUsageStripReader {
	return &loopGateUsageStripReader{reader: bufio.NewReaderSize(source, 64*1024)}
}

func (r *loopGateUsageStripReader) Read(p []byte) (int, error) {
	for len(r.pending) == 0 {
		if r.err != nil {
			return 0, r.err
		}
		fragment, err := r.reader.ReadSlice('\n')
		full := errors.Is(err, bufio.ErrBufferFull)
		if err != nil && !full {
			r.err = err
		}
		switch {
		case r.passthrough:
			r.pending = fragment
			r.passthrough = full
		case full && len(r.line)+len(fragment) > loopGateMaxLine:
			r.pending = slices.Concat(r.line, fragment)
			r.line, r.passthrough = nil, true
		case full:
			r.line = append(r.line, fragment...)
		case len(r.line) > 0:
			r.pending = stripRunningUsage(slices.Concat(r.line, fragment))
			r.line = nil
		default:
			r.pending = stripRunningUsage(fragment)
		}
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

var loopGateNullUsage = []byte("null")

// stripRunningUsage returns line with the usage of a chat chunk with choices
// replaced by null; any other line is returned as is.
func stripRunningUsage(line []byte) []byte {
	payload, ok := sseLinePayload(line)
	if !ok {
		return line
	}
	start, end, ok := chatChunkUsageSpan(payload)
	if !ok || !chatChunkHasChoices(payload) {
		return line
	}
	offset := cap(line) - cap(payload) // payload is a sub-slice of line
	return slices.Concat(line[:offset+start], loopGateNullUsage, line[offset+end:])
}

// chatChunkUsageSpan returns the byte range of the usage object of a chat
// completion chunk payload; ok is false when it has none (or usage is null).
// A key inside a string value has its quotes escaped, so the last `"usage"`
// in the payload is the chunk's own.
func chatChunkUsageSpan(payload []byte) (start, end int, ok bool) {
	i := bytes.LastIndex(payload, sseUsageNeedle)
	if i < 0 {
		return 0, 0, false
	}
	i = skipJSONSpace(payload, i+len(sseUsageNeedle))
	if i >= len(payload) || payload[i] != ':' {
		return 0, 0, false
	}
	i = skipJSONSpace(payload, i+1)
	if i >= len(payload) || payload[i] != '{' {
		return 0, 0, false
	}
	end = jsonObjectEnd(payload, i)
	if end < 0 {
		return 0, 0, false
	}
	return i, end, true
}

// chatChunkHasChoices reports whether a chat completion chunk payload has a
// non-empty choices array.
func chatChunkHasChoices(payload []byte) bool {
	i := bytes.Index(payload, sseChoicesNeedle)
	if i < 0 {
		return false
	}
	i = skipJSONSpace(payload, i+len(sseChoicesNeedle))
	if i >= len(payload) || payload[i] != ':' {
		return false
	}
	i = skipJSONSpace(payload, i+1)
	if i >= len(payload) || payload[i] != '[' {
		return false
	}
	i = skipJSONSpace(payload, i+1)
	return i < len(payload) && payload[i] != ']'
}

var sseChoicesNeedle = []byte(`"choices"`)

func skipJSONSpace(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return i
}

// jsonObjectEnd returns the offset just past the object that starts at
// b[start] ('{'), or -1 when it does not end within b.
func jsonObjectEnd(b []byte, start int) int {
	depth, inString := 0, false
	for i := start; i < len(b); i++ {
		c := b[i]
		switch {
		case inString && c == '\\':
			i++
		case c == '"':
			inString = !inString
		case inString:
		case c == '{' || c == '[':
			depth++
		case c == '}' || c == ']':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}
