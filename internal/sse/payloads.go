// Package sse splits a Server-Sent Events stream into its data payloads.
//
// It is the router's own reader of the providers' streams: the streaming
// handler, the tool loop's collector and the simulated path all need the
// payloads of the same stream, and they must agree on what a payload is.
package sse

import "bytes"

// dataPrefix and doneSentinel are shared byte-literal needles for the
// repeated comparison: a package-level []byte avoids the allocation a
// string conversion would make on every line.
var (
	dataPrefix   = []byte("data:")
	doneSentinel = []byte("[DONE]")
)

// SplitPayloads splits a chunk of an SSE stream into its data payloads.
//
// dst is the slice the payloads are appended to; it is reused, so a caller in
// a hot path can hand the same slice over every time.
//
// nil for one-off, non-hot-path calls.
//
// ⚠️ Every returned payload is a sub-slice of chunk itself — zero copies.
// The result is only valid until chunk's backing array is next overwritten
// (e.g. the next Read() into a buffer pulled from a pool). Every current
// caller is synchronous within the call that produced chunk and finishes
// before returning, so this is safe today — but don't stash a returned payload
// past that call. The caller that must survive past the current chunk's
// lifetime copies explicitly.
func SplitPayloads(chunk []byte, dst [][]byte) [][]byte {
	dst = dst[:0]
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return dst
	}

	// Fast path: non-SSE plain JSON.
	if !bytes.Contains(trimmed, dataPrefix) {
		return append(dst, trimmed)
	}

	rest := trimmed
	for {
		var line []byte
		if idx := bytes.IndexByte(rest, '\n'); idx >= 0 {
			line = rest[:idx]
			rest = rest[idx+1:]
		} else {
			line = rest
			rest = nil
		}
		if payload, ok := LinePayload(line); ok {
			dst = append(dst, payload)
		}
		if rest == nil {
			break
		}
	}

	return dst
}

// LinePayload returns the data payload of one SSE line: the text after
// "data:" with surrounding whitespace trimmed, a sub-slice of line. ok is
// false for any other line, an empty payload and the [DONE] sentinel.
func LinePayload(line []byte) ([]byte, bool) {
	payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), dataPrefix)
	if !ok {
		return nil, false
	}
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 || bytes.Equal(payload, doneSentinel) {
		return nil, false
	}
	return payload, true
}

// IsDone reports whether line is the "data: [DONE]" sentinel that ends an
// OpenAI-style stream.
func IsDone(line []byte) bool {
	payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), dataPrefix)
	return ok && bytes.Equal(bytes.TrimSpace(payload), doneSentinel)
}

// LastPayload returns the last data payload of a buffered stream, which the
// streaming log reads its figures from.
func LastPayload(raw []byte) []byte {
	payloads := SplitPayloads(raw, nil)
	if len(payloads) == 0 {
		return nil
	}
	return payloads[len(payloads)-1]
}
