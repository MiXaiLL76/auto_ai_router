package jsonscan

import (
	"bytes"
	"strconv"
	"unicode/utf8"
)

// maxWalkObjectKeys bounds the duplicate-name check FindKey makes on every
// object it walks (each new name is compared with the names before it). An
// object with more members than this is not walked; FindKey reports that it
// cannot answer and the caller decodes instead.
const maxWalkObjectKeys = 128

// FindKey walks a JSON value and returns the raw value of every object
// member named key, at any depth, in document order. It does not descend
// into the values it returns.
//
// It exists to replace "decode the value into interface{} and search the
// result" where the search is for one key: nothing is allocated for the
// parts of the document that do not match.
//
// ok is false whenever the walk cannot promise the answer a decode-and-search
// would give, and the caller should then do exactly that instead:
//   - raw is not one valid JSON value;
//   - some object has two members with the same name (a decode keeps only
//     the last, so anything under the earlier one would be found here but
//     not there), or has more than maxWalkObjectKeys members;
//   - some object name is written with an escape or is not valid UTF-8 (it
//     may then equal key, or another name, without being the same bytes);
//   - some number does not fit float64 (a decode into interface{} fails).
func FindKey(raw []byte, key string) (values [][]byte, ok bool) {
	w := keyWalker{syntaxScanner: syntaxScanner{b: raw, float64Numbers: true}, key: key}
	w.skipSpace()
	if !w.walk(0) || !w.atEnd() {
		return nil, false
	}
	return w.found, true
}

type keyWalker struct {
	syntaxScanner
	key   string
	found [][]byte
	// names holds the raw names (without quotes) of the members seen so far
	// in every object currently open, innermost last; an object's names start
	// at the index its frame recorded.
	names [][]byte
}

func (w *keyWalker) walk(depth int) bool {
	if w.i >= len(w.b) {
		return false
	}
	switch w.b[w.i] {
	case '{':
		if depth >= maxDepth {
			return false
		}
		w.i++
		w.skipSpace()
		if w.i < len(w.b) && w.b[w.i] == '}' {
			w.i++
			return true
		}
		frame := len(w.names)
		for {
			if w.i >= len(w.b) || w.b[w.i] != '"' {
				return false
			}
			nameStart := w.i + 1
			escaped, ok := w.skipString()
			if !ok || escaped {
				return false
			}
			name := w.b[nameStart : w.i-1]
			if !utf8.Valid(name) || len(w.names)-frame >= maxWalkObjectKeys {
				return false
			}
			for _, earlier := range w.names[frame:] {
				if bytes.Equal(earlier, name) {
					return false
				}
			}
			w.names = append(w.names, name)

			w.skipSpace()
			if w.i >= len(w.b) || w.b[w.i] != ':' {
				return false
			}
			w.i++
			w.skipSpace()
			if string(name) == w.key {
				start := w.i
				if !w.skipValue(depth + 1) {
					return false
				}
				w.found = append(w.found, w.b[start:w.i:w.i])
			} else if !w.walk(depth + 1) {
				return false
			}
			w.skipSpace()
			if w.i >= len(w.b) {
				return false
			}
			switch w.b[w.i] {
			case ',':
				w.i++
				w.skipSpace()
			case '}':
				w.i++
				w.names = w.names[:frame]
				return true
			default:
				return false
			}
		}
	case '[':
		if depth >= maxDepth {
			return false
		}
		w.i++
		w.skipSpace()
		if w.i < len(w.b) && w.b[w.i] == ']' {
			w.i++
			return true
		}
		for {
			if !w.walk(depth + 1) {
				return false
			}
			w.skipSpace()
			if w.i >= len(w.b) {
				return false
			}
			switch w.b[w.i] {
			case ',':
				w.i++
				w.skipSpace()
			case ']':
				w.i++
				return true
			default:
				return false
			}
		}
	default:
		return w.skipValue(depth)
	}
}

// FitsFloat64 reports whether a JSON number literal decodes into float64,
// which is what a decode into interface{} requires of every number. Only a
// literal with an exponent or hundreds of digits can fail, so nothing is
// parsed for an ordinary number.
func FitsFloat64(literal []byte) bool {
	if len(literal) < 300 && bytes.IndexByte(literal, 'e') < 0 && bytes.IndexByte(literal, 'E') < 0 {
		return true
	}
	_, err := strconv.ParseFloat(string(literal), 64)
	return err == nil
}

// ScanArrayMembers is ScanArray that also reports, for every element that is
// an object, the raw value of its top-level member named member (the last
// one, as a decode would keep): values[i] is nil for an element that is not
// an object or has no such member.
//
// It answers "what is the type of each item" for a whole array in the same
// single pass that finds the items, so a caller that acts on a few item types
// never has to look into the others. member must be valid UTF-8 (it is a key
// name from the caller's own code).
func ScanArrayMembers(raw []byte, member string) (elements, values [][]byte, ok bool) {
	s := syntaxScanner{b: raw}
	s.skipSpace()
	if s.i >= len(raw) || raw[s.i] != '[' {
		return nil, nil, false
	}
	s.i++
	s.skipSpace()
	elements = make([][]byte, 0, 64)
	values = make([][]byte, 0, 64)
	if s.i < len(raw) && raw[s.i] == ']' {
		s.i++
		return elements, values, s.atEnd()
	}
	for {
		start := s.i
		var value []byte
		if s.i < len(raw) && raw[s.i] == '{' {
			var elementOK bool
			if value, elementOK = s.skipObjectFinding(member); !elementOK {
				return nil, nil, false
			}
		} else if !s.skipValue(1) {
			return nil, nil, false
		}
		elements = append(elements, raw[start:s.i:s.i])
		values = append(values, value)
		s.skipSpace()
		if s.i >= len(raw) {
			return nil, nil, false
		}
		switch raw[s.i] {
		case ',':
			s.i++
			s.skipSpace()
		case ']':
			s.i++
			if !s.atEnd() {
				return nil, nil, false
			}
			return elements, values, true
		default:
			return nil, nil, false
		}
	}
}

// skipObjectFinding consumes an object that is an element of the array being
// scanned (so one level deep) and returns the raw value of its last member
// named member, nil if it has none.
func (s *syntaxScanner) skipObjectFinding(member string) (value []byte, ok bool) {
	s.i++ // '{'
	s.skipSpace()
	if s.i < len(s.b) && s.b[s.i] == '}' {
		s.i++
		return nil, true
	}
	for {
		if s.i >= len(s.b) || s.b[s.i] != '"' {
			return nil, false
		}
		nameStart := s.i
		escaped, strOK := s.skipString()
		if !strOK {
			return nil, false
		}
		quoted := s.b[nameStart:s.i]
		matches := false
		if !escaped && len(quoted) == len(member)+2 && string(quoted[1:len(quoted)-1]) == member {
			matches = true
		} else if escaped {
			// An escaped spelling is decoded; rare, and exact.
			name, nameOK := decodeName(quoted, true)
			if !nameOK {
				return nil, false
			}
			matches = name == member
		}
		s.skipSpace()
		if s.i >= len(s.b) || s.b[s.i] != ':' {
			return nil, false
		}
		s.i++
		s.skipSpace()
		start := s.i
		if !s.skipValue(2) {
			return nil, false
		}
		if matches {
			value = s.b[start:s.i:s.i]
		}
		s.skipSpace()
		if s.i >= len(s.b) {
			return nil, false
		}
		switch s.b[s.i] {
		case ',':
			s.i++
			s.skipSpace()
		case '}':
			s.i++
			return value, true
		default:
			return nil, false
		}
	}
}
