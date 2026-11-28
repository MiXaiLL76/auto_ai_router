package jsonscan

// A JSON syntax checker that only finds where values end.
//
// ScanTop and ScanArray have to walk an entire request body to learn where a
// few top-level values start and stop, and nearly all of that body is string
// content nobody asked about. A general-purpose tokenizer validates UTF-8 and
// prepares each token for decoding as it goes; this one does neither, which
// makes the walk several times cheaper.
//
// It accepts exactly the documents encoding/json accepts: the JSON grammar of
// RFC 8259, with invalid UTF-8 inside strings allowed, duplicate object names
// allowed, and nesting limited to maxDepth. The tests hold it to that by
// comparing it with encoding/json on every input the fuzzers produce.

// maxDepth is encoding/json's own nesting limit.
const maxDepth = 10000

// plainStringByte marks the bytes that can appear inside a string without
// ending it or starting an escape: everything from 0x20 up except '"' and
// '\'. Bytes above 0x7f are plain too -- UTF-8 is not validated.
var plainStringByte = func() (t [256]bool) {
	for c := 0x20; c < 256; c++ {
		t[c] = c != '"' && c != '\\'
	}
	return t
}()

type syntaxScanner struct {
	b []byte
	i int
	// escapedASCII is set when a string contained an escape of a printable
	// ASCII character (see HasEscapedASCII), i.e. when some key or value in
	// the document may be spelled differently from its text.
	escapedASCII bool
	// float64Numbers makes a number that does not fit float64 a syntax
	// error, as it is an error for a decode into interface{}.
	float64Numbers bool
}

func (s *syntaxScanner) skipSpace() {
	for s.i < len(s.b) {
		switch s.b[s.i] {
		case ' ', '\t', '\n', '\r':
			s.i++
		default:
			return
		}
	}
}

// skipString consumes a string starting at the opening quote. escaped
// reports whether it contained a backslash.
func (s *syntaxScanner) skipString() (escaped, ok bool) {
	b, i := s.b, s.i+1
	for {
		for i < len(b) && plainStringByte[b[i]] {
			i++
		}
		if i >= len(b) {
			return escaped, false
		}
		switch b[i] {
		case '"':
			s.i = i + 1
			return escaped, true
		case '\\':
			escaped = true
			if i+1 >= len(b) {
				return escaped, false
			}
			switch b[i+1] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				i += 2
			case 'u':
				if i+6 > len(b) || !isHex(b[i+2]) || !isHex(b[i+3]) || !isHex(b[i+4]) || !isHex(b[i+5]) {
					return escaped, false
				}
				if b[i+2] == '0' && b[i+3] == '0' && b[i+4] >= '2' && b[i+4] <= '7' {
					s.escapedASCII = true
				}
				i += 6
			default:
				return escaped, false
			}
		default: // a raw control character
			return escaped, false
		}
	}
}

func isHex(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// skipNumber consumes a number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
func (s *syntaxScanner) skipNumber() bool {
	b, i := s.b, s.i
	if i < len(b) && b[i] == '-' {
		i++
	}
	switch {
	case i < len(b) && b[i] == '0':
		i++
	case i < len(b) && b[i] >= '1' && b[i] <= '9':
		for i < len(b) && isDigit(b[i]) {
			i++
		}
	default:
		return false
	}
	if i < len(b) && b[i] == '.' {
		i++
		if i >= len(b) || !isDigit(b[i]) {
			return false
		}
		for i < len(b) && isDigit(b[i]) {
			i++
		}
	}
	if i < len(b) && (b[i] == 'e' || b[i] == 'E') {
		i++
		if i < len(b) && (b[i] == '+' || b[i] == '-') {
			i++
		}
		if i >= len(b) || !isDigit(b[i]) {
			return false
		}
		for i < len(b) && isDigit(b[i]) {
			i++
		}
	}
	s.i = i
	return true
}

func (s *syntaxScanner) skipLiteral(lit string) bool {
	if len(s.b)-s.i < len(lit) || string(s.b[s.i:s.i+len(lit)]) != lit {
		return false
	}
	s.i += len(lit)
	return true
}

// skipValue consumes one value starting at s.i (no leading whitespace).
// depth is the number of arrays and objects already open around it.
func (s *syntaxScanner) skipValue(depth int) bool {
	if s.i >= len(s.b) {
		return false
	}
	switch c := s.b[s.i]; {
	case c == '"':
		_, ok := s.skipString()
		return ok
	case c == '{':
		if depth >= maxDepth {
			return false
		}
		s.i++
		s.skipSpace()
		if s.i < len(s.b) && s.b[s.i] == '}' {
			s.i++
			return true
		}
		for {
			if s.i >= len(s.b) || s.b[s.i] != '"' {
				return false
			}
			if _, ok := s.skipString(); !ok {
				return false
			}
			s.skipSpace()
			if s.i >= len(s.b) || s.b[s.i] != ':' {
				return false
			}
			s.i++
			s.skipSpace()
			if !s.skipValue(depth + 1) {
				return false
			}
			s.skipSpace()
			if s.i >= len(s.b) {
				return false
			}
			switch s.b[s.i] {
			case ',':
				s.i++
				s.skipSpace()
			case '}':
				s.i++
				return true
			default:
				return false
			}
		}
	case c == '[':
		if depth >= maxDepth {
			return false
		}
		s.i++
		s.skipSpace()
		if s.i < len(s.b) && s.b[s.i] == ']' {
			s.i++
			return true
		}
		for {
			if !s.skipValue(depth + 1) {
				return false
			}
			s.skipSpace()
			if s.i >= len(s.b) {
				return false
			}
			switch s.b[s.i] {
			case ',':
				s.i++
				s.skipSpace()
			case ']':
				s.i++
				return true
			default:
				return false
			}
		}
	case c == '-' || isDigit(c):
		start := s.i
		return s.skipNumber() && (!s.float64Numbers || FitsFloat64(s.b[start:s.i]))
	case c == 't':
		return s.skipLiteral("true")
	case c == 'f':
		return s.skipLiteral("false")
	case c == 'n':
		return s.skipLiteral("null")
	}
	return false
}

// atEnd reports whether only whitespace is left.
func (s *syntaxScanner) atEnd() bool {
	s.skipSpace()
	return s.i == len(s.b)
}
