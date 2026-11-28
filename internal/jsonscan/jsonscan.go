// Package jsonscan answers questions about a JSON request body from one pass
// over its top-level keys, without decoding the values nobody asked for.
//
// A request body is mostly one large value ("messages" or "input"); the
// router's routing and bookkeeping decisions depend on a handful of small
// top-level fields next to it. Decoding the whole document into
// map[string]interface{} for each such decision costs a full pass and ~2.5x
// the body in allocations every time. ScanTop walks the document once,
// remembers where each top-level value starts and ends, and lets callers
// decode only the values they read.
package jsonscan

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	"slices"
	"strings"
	"unicode/utf8"
)

type entry struct {
	name       string
	start, end int64 // value bytes are body[start:end]
}

// Top is the top-level layout of a JSON object document. The zero value and
// a nil *Top are valid and describe "no keys", which is what every accessor
// reports for a body that is not a JSON object.
type Top struct {
	body    []byte
	entries []entry
	// escapedASCII: see EscapedASCII.
	escapedASCII bool
	// invalidNames: see HasInvalidNames.
	invalidNames bool
}

// ScanTop indexes the top-level keys of body. It returns nil, false unless
// body is exactly one syntactically valid JSON object (surrounding whitespace
// allowed). The returned Top keeps referring to body: do not modify body while
// the Top is in use.
//
// Like encoding/json, it accepts duplicate names (the last one wins in every
// accessor) and invalid UTF-8 inside strings. Unlike a decode into
// map[string]interface{}, it does not convert numbers, so a number literal
// outside the float64 range (1e999) does not make the document invalid.
func ScanTop(body []byte) (*Top, bool) {
	s := syntaxScanner{b: body}
	s.skipSpace()
	if s.i >= len(body) || body[s.i] != '{' {
		return nil, false
	}
	s.i++
	s.skipSpace()
	top := &Top{body: body, entries: make([]entry, 0, 16)}
	if s.i < len(body) && body[s.i] == '}' {
		s.i++
		if !s.atEnd() {
			return nil, false
		}
		return top, true
	}
	for {
		if s.i >= len(body) || body[s.i] != '"' {
			return nil, false
		}
		keyStart := s.i
		escaped, ok := s.skipString()
		if !ok {
			return nil, false
		}
		name, ok := decodeName(body[keyStart:s.i], escaped)
		if !ok {
			return nil, false
		}
		if !utf8.Valid(body[keyStart:s.i]) {
			top.invalidNames = true
		}
		s.skipSpace()
		if s.i >= len(body) || body[s.i] != ':' {
			return nil, false
		}
		s.i++
		s.skipSpace()
		start := s.i
		if !s.skipValue(1) {
			return nil, false
		}
		top.entries = append(top.entries, entry{name: name, start: int64(start), end: int64(s.i)})
		s.skipSpace()
		if s.i >= len(body) {
			return nil, false
		}
		switch body[s.i] {
		case ',':
			s.i++
			s.skipSpace()
		case '}':
			s.i++
			if !s.atEnd() {
				return nil, false
			}
			top.escapedASCII = s.escapedASCII
			return top, true
		default:
			return nil, false
		}
	}
}

// EscapedASCII reports whether any string in the document -- at any depth,
// key or value -- contains an escape of a printable ASCII character. It is
// HasEscapedASCII's answer for the whole body, learned during the scan
// instead of by another pass over the bytes.
//
// When it is false, every key and every word in the document is spelled as
// its own text, so "this key does not occur in these bytes" can be decided
// by a substring search with no false negatives.
func (t *Top) EscapedASCII() bool {
	return t != nil && t.escapedASCII
}

// HasInvalidNames reports whether a top-level name is not valid UTF-8 as
// written. encoding/json decodes such a name with replacement characters and
// writes that repaired name when it encodes the object again, so a document
// with one is not left byte-for-byte alone by a decode/encode round trip even
// when nothing in it was changed.
func (t *Top) HasInvalidNames() bool {
	return t != nil && t.invalidNames
}

// decodeName returns the name a quoted object key decodes to. A key of plain
// ASCII with no escapes is its own text; anything else is decoded by
// encoding/json, so that escapes and invalid UTF-8 come out exactly as they
// do in a map decoded by it.
func decodeName(quoted []byte, escaped bool) (string, bool) {
	inner := quoted[1 : len(quoted)-1]
	if !escaped {
		ascii := true
		for _, c := range inner {
			if c >= 0x80 {
				ascii = false
				break
			}
		}
		if ascii {
			return string(inner), true
		}
	}
	var name string
	if err := json.Unmarshal(quoted, &name); err != nil {
		return "", false
	}
	return name, true
}

// Has reports whether the document has a top-level key name (even when its
// value is null).
func (t *Top) Has(name string) bool {
	_, ok := t.Raw(name)
	return ok
}

// Raw returns the bytes of the value of top-level key name (a sub-slice of
// the scanned body, not a copy). With duplicate names the last one wins,
// matching encoding/json.
func (t *Top) Raw(name string) ([]byte, bool) {
	if t == nil {
		return nil, false
	}
	for i := len(t.entries) - 1; i >= 0; i-- {
		if e := t.entries[i]; e.name == name {
			return t.body[e.start:e.end], true
		}
	}
	return nil, false
}

// RawFoldAll returns, in document order, the value of every top-level key
// that encoding/json would store into a struct field named name: keys equal
// to name under Unicode case folding. Decoding each returned value in order
// into the same destination reproduces what decoding the whole body into a
// struct with that one field does, duplicates included.
func (t *Top) RawFoldAll(name string) [][]byte {
	if t == nil {
		return nil
	}
	var values [][]byte
	for _, e := range t.entries {
		if strings.EqualFold(e.name, name) {
			values = append(values, t.body[e.start:e.end])
		}
	}
	return values
}

// IsNull reports whether raw is the JSON literal null.
func IsNull(raw []byte) bool {
	return string(raw) == "null"
}

// Project decodes only the named top-level keys into a
// map[string]interface{}, exactly as json.Unmarshal of the whole body into a
// map[string]interface{} would have decoded them. Keys that are absent from
// the body are absent from the result. It returns false when one of the
// requested values cannot be decoded (a number outside the float64 range).
//
// Code that reads a fixed set of top-level keys from the fully decoded map
// gets the same answers from the projection, for the price of the values it
// actually reads.
func (t *Top) Project(names ...string) (map[string]interface{}, bool) {
	fields := make(map[string]interface{}, len(names))
	for _, name := range names {
		raw, ok := t.Raw(name)
		if !ok {
			continue
		}
		var value interface{}
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, false
		}
		fields[name] = value
	}
	return fields, true
}

// Range calls fn for every top-level member in document order, duplicates
// included (a later duplicate is the one encoding/json keeps).
func (t *Top) Range(fn func(name string, raw []byte)) {
	if t == nil {
		return
	}
	for _, e := range t.entries {
		fn(e.name, t.body[e.start:e.end])
	}
}

// HasDuplicates reports whether a top-level name occurs more than once.
func (t *Top) HasDuplicates() bool {
	if t == nil {
		return false
	}
	for i := range t.entries {
		for j := i + 1; j < len(t.entries); j++ {
			if t.entries[i].name == t.entries[j].name {
				return true
			}
		}
	}
	return false
}

// Edit changes one top-level member in Rewrite: Raw replaces the value of
// Name (or adds the member when the document has none), a nil Raw removes it.
type Edit struct {
	Name string
	Raw  []byte
}

// Rewrite returns a new document with edits applied. Members that are not
// edited are copied byte for byte, in their original order; this is what
// lets a request body be changed in a small field without decoding and
// re-encoding the conversation next to it. Like a decode into a map, the
// result has each name once: of duplicated members only the last survives.
// Added members follow the existing ones in the order given.
//
// The result is equivalent JSON to "decode into a map, apply the edits,
// encode", not the same bytes: that round trip also sorts keys and rewrites
// string escapes and whitespace, which Rewrite leaves as the client sent them.
func (t *Top) Rewrite(edits ...Edit) []byte {
	if t == nil {
		return nil
	}
	return t.RewriteTop(edits...).body
}

// RewriteTop is Rewrite returning the result already scanned: the layout of
// the new document is known while it is being written, so the next reader
// does not have to scan it again. Bytes returns the document.
func (t *Top) RewriteTop(edits ...Edit) *Top {
	if t == nil {
		return nil
	}
	size := len(t.body) + 2
	for _, e := range edits {
		size += len(e.Name) + len(e.Raw) + 8
	}
	out := make([]byte, 0, size)
	out = append(out, '{')
	entries := make([]entry, 0, len(t.entries)+len(edits))
	applied := make([]bool, len(edits))
	for i, entry := range t.entries {
		last := true
		for _, later := range t.entries[i+1:] {
			if later.name == entry.name {
				last = false
				break
			}
		}
		if !last {
			continue
		}
		raw := t.body[entry.start:entry.end]
		removed := false
		// The last edit of a name is the one that counts.
		for j := len(edits) - 1; j >= 0; j-- {
			if edits[j].Name != entry.name {
				continue
			}
			for k := 0; k <= j; k++ {
				if edits[k].Name == entry.name {
					applied[k] = true
				}
			}
			raw, removed = edits[j].Raw, edits[j].Raw == nil
			break
		}
		if removed {
			continue
		}
		out, entries = appendMember(out, entries, entry.name, raw)
	}
	for j, e := range edits {
		if applied[j] || e.Raw == nil {
			continue
		}
		// A name added more than once: the last value wins, at the first position.
		raw := e.Raw
		for k := j + 1; k < len(edits); k++ {
			if edits[k].Name == e.Name {
				applied[k] = true
				raw = edits[k].Raw
			}
		}
		if raw == nil {
			continue
		}
		out, entries = appendMember(out, entries, e.Name, raw)
	}
	// The edits' own values are not inspected: keeping the flag set when the
	// source had an escape, and setting it when an edit brings one in, errs
	// on the side of "may be escaped".
	escaped := t.escapedASCII
	for _, e := range edits {
		escaped = escaped || HasEscapedASCII(e.Raw)
	}
	// Names are written from their decoded form, so they are valid now.
	return &Top{body: append(out, '}'), entries: entries, escapedASCII: escaped}
}

func appendMember(out []byte, entries []entry, name string, raw []byte) ([]byte, []entry) {
	if len(out) > 1 {
		out = append(out, ',')
	}
	out, _ = jsontext.AppendQuote(out, name)
	out = append(out, ':')
	start := len(out)
	out = append(out, raw...)
	return out, append(entries, entry{name: name, start: int64(start), end: int64(len(out))})
}

// Bytes returns the document the Top describes.
func (t *Top) Bytes() []byte {
	if t == nil {
		return nil
	}
	return t.body
}

// Describes reports whether t is a scan of exactly this slice (the same
// memory, not merely equal contents), so that it can stand in for scanning
// body again.
func (t *Top) Describes(body []byte) bool {
	return t != nil && len(t.body) == len(body) && (len(body) == 0 || &t.body[0] == &body[0])
}

// ScanArray returns the elements of a JSON array as sub-slices of raw, in
// order, without decoding them. It returns false unless raw is exactly one
// syntactically valid JSON array.
func ScanArray(raw []byte) ([][]byte, bool) {
	s := syntaxScanner{b: raw}
	s.skipSpace()
	if s.i >= len(raw) || raw[s.i] != '[' {
		return nil, false
	}
	s.i++
	s.skipSpace()
	elements := make([][]byte, 0, 64)
	if s.i < len(raw) && raw[s.i] == ']' {
		s.i++
		return elements, s.atEnd()
	}
	for {
		start := s.i
		if !s.skipValue(1) {
			return nil, false
		}
		elements = append(elements, raw[start:s.i:s.i])
		s.skipSpace()
		if s.i >= len(raw) {
			return nil, false
		}
		switch raw[s.i] {
		case ',':
			s.i++
			s.skipSpace()
		case ']':
			s.i++
			if !s.atEnd() {
				return nil, false
			}
			return elements, true
		default:
			return nil, false
		}
	}
}

// JoinArray is the inverse of ScanArray: a JSON array of the given raw
// elements.
func JoinArray(elements [][]byte) []byte {
	size := 2
	for _, e := range elements {
		size += len(e) + 1
	}
	out := make([]byte, 0, size)
	out = append(out, '[')
	for i, e := range elements {
		if i > 0 {
			out = append(out, ',')
		}
		out = append(out, e...)
	}
	return append(out, ']')
}

// Object is a request body opened for editing its top-level members. It
// stands in for the pattern
//
//	var fields map[string]json.RawMessage
//	json.Unmarshal(body, &fields)
//	... read and change fields ...
//	body, _ = json.Marshal(fields)
//
// which copies every member on the way in and re-encodes every member on the
// way out. Fields holds the same names and the same raw values, but as
// sub-slices of the body, and Encode writes back only what was changed.
type Object struct {
	// Fields are the top-level members, last duplicate winning, exactly as
	// json.Unmarshal into map[string]json.RawMessage yields them. Callers add,
	// replace and delete entries freely; the values must not be modified in
	// place.
	Fields map[string]json.RawMessage

	top      *Top
	original map[string]json.RawMessage
}

// ParseObject opens body. It returns false for anything json.Unmarshal would
// not decode into a non-nil map[string]json.RawMessage.
func ParseObject(body []byte) (*Object, bool) {
	top, ok := ScanTop(body)
	if !ok {
		return nil, false
	}
	fields := make(map[string]json.RawMessage, len(top.entries)+2)
	for _, e := range top.entries {
		fields[e.name] = body[e.start:e.end:e.end]
	}
	original := make(map[string]json.RawMessage, len(fields))
	for name, raw := range fields {
		original[name] = raw
	}
	return &Object{Fields: fields, top: top, original: original}, true
}

// Top returns the scan the object was opened from.
func (o *Object) Top() *Top { return o.top }

// Changed reports whether Fields differs from the body the object was opened
// from.
func (o *Object) Changed() bool { return len(o.edits()) > 0 }

func (o *Object) edits() []Edit {
	var edits []Edit
	for _, e := range o.top.entries {
		raw, kept := o.Fields[e.name]
		switch {
		case !kept:
			edits = append(edits, Edit{Name: e.name})
		case !sameBytes(raw, o.original[e.name]):
			edits = append(edits, Edit{Name: e.name, Raw: raw})
		}
	}
	var added []string
	for name := range o.Fields {
		if _, existed := o.original[name]; !existed {
			added = append(added, name)
		}
	}
	slices.Sort(added)
	for _, name := range added {
		edits = append(edits, Edit{Name: name, Raw: o.Fields[name]})
	}
	return edits
}

func sameBytes(a, b []byte) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0] || bytes.Equal(a, b))
}

// Encode returns the body with the changes made to Fields. Members that were
// not changed keep their bytes and their order; added members follow, sorted
// by name. With nothing changed -- and nothing a decode/encode round trip
// would have normalised on its own: no duplicate names to resolve, no name
// that is invalid UTF-8 to repair -- the result is the original body itself,
// not a copy.
//
// The result is the same JSON value json.Marshal(Fields) encodes, in a
// different layout: json.Marshal also sorts the members, compacts them and
// re-escapes their strings.
func (o *Object) Encode() []byte {
	edits := o.edits()
	if len(edits) == 0 && !o.top.HasDuplicates() && !o.top.HasInvalidNames() {
		return o.top.body
	}
	return o.top.Rewrite(edits...)
}

var escapePrefix = []byte(`\u00`)

// HasEscapedASCII reports whether data contains a \u00XX escape of a
// printable ASCII character (  to \u007f).
//
// It makes substring pre-checks exact. A JSON string can spell a letter,
// digit or underscore without writing it only through such an escape, so a
// body that contains neither the text of a key (or of a value like a type
// name) nor one of these escapes cannot contain that key or value at all.
// Escapes of control characters (\u0000 to \u001f, common in captured
// terminal output) and of non-ASCII characters cannot spell one and are not
// reported.
func HasEscapedASCII(data []byte) bool {
	for {
		i := bytes.Index(data, escapePrefix)
		if i < 0 {
			return false
		}
		data = data[i+len(escapePrefix):]
		if len(data) > 0 && data[0] >= '2' && data[0] <= '7' {
			return true
		}
	}
}
