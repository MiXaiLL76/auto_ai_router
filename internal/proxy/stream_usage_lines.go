package proxy

import "bytes"

// streamUsageLines reassembles SSE data lines before usage extraction. Reads
// from an HTTP body may split JSON anywhere, including inside an image-bearing
// event that also carries usage. The bound matches image stream model rewriting.
type streamUsageLines struct {
	pending []byte
	discard bool
}

func (s *streamUsageLines) Observe(chunk []byte, consume func([]byte)) {
	for len(chunk) > 0 {
		end := bytes.IndexByte(chunk, '\n')
		length := len(chunk)
		if end >= 0 {
			length = end + 1
		}
		part := chunk[:length]
		chunk = chunk[length:]
		if !s.discard {
			if len(s.pending)+len(part) > maxSSEModelRewriteLineBytes {
				s.pending = nil
				s.discard = true
			} else if len(s.pending) == 0 && end >= 0 {
				consume(part)
			} else {
				s.pending = append(s.pending, part...)
				if end >= 0 {
					consume(s.pending)
					s.pending = nil
				}
			}
		}
		if end >= 0 {
			s.discard = false
		}
	}
}

func (s *streamUsageLines) Finalize(consume func([]byte)) {
	if !s.discard && len(s.pending) > 0 {
		consume(s.pending)
	}
	s.pending = nil
	s.discard = false
}
