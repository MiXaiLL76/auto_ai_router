package jsonscan

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	"io"
	"strings"
	"testing"
)

// scanTopTokenizer is ScanTop written on encoding/json/jsontext, the
// implementation the hand-written syntax scanner replaced. It is kept as a
// second, independent reference next to encoding/json itself.
func scanTopTokenizer(body []byte) (map[string]string, bool) {
	if len(body) == 0 {
		return nil, false
	}
	dec := jsontext.NewDecoder(bytes.NewBuffer(body),
		jsontext.AllowDuplicateNames(true), jsontext.AllowInvalidUTF8(true))
	if tok, err := dec.ReadToken(); err != nil || tok.Kind() != '{' {
		return nil, false
	}
	fields := map[string]string{}
	for dec.PeekKind() != '}' {
		key, err := dec.ReadToken()
		if err != nil || key.Kind() != '"' {
			return nil, false
		}
		name := key.String()
		start := dec.InputOffset()
		if err := dec.SkipValue(); err != nil {
			return nil, false
		}
		raw := bytes.TrimLeft(body[start:dec.InputOffset()], ": \t\r\n")
		fields[name] = string(raw)
	}
	if _, err := dec.ReadToken(); err != nil {
		return nil, false
	}
	if _, err := dec.ReadToken(); err != io.EOF {
		return nil, false
	}
	return fields, true
}

var syntaxCases = []string{
	// numbers
	`{"n":0}`, `{"n":-0}`, `{"n":10}`, `{"n":-1.5}`, `{"n":1e5}`, `{"n":1E+5}`, `{"n":1.25e-3}`, `{"n":0.0}`, `{"n":1e999}`,
	`{"n":01}`, `{"n":-}`, `{"n":1.}`, `{"n":.5}`, `{"n":1e}`, `{"n":1e+}`, `{"n":+1}`, `{"n":0x10}`, `{"n":1.2.3}`, `{"n":--1}`, `{"n":1 2}`, `{"n":Infinity}`, `{"n":NaN}`,
	// literals
	`{"a":true,"b":false,"c":null}`, `{"a":tru}`, `{"a":True}`, `{"a":nul}`, `{"a":nulll}`, `{"a":falsey}`, `{"a":truefalse}`,
	// strings
	`{"s":""}`, `{"s":"a\"b\\c\/d\b\f\n\r\t"}`, `{"s":"\a"}`, `{"s":"\x41"}`, `{"s":"\"}`, `{"s":"\`, `{"s":"tab	inside"}`, "{\"s\":\"nl\ninside\"}", "{\"s\":\"\x00\"}", "{\"s\":\"\x1f\"}", "{\"s\":\"\x7f\"}",
	"{\"s\":\"\xff\xfe\xc0\"}", "{\"\xff\":1}", `{"s":"unterminated`, `{"s":'single'}`,
	// structure
	`{}`, ` {} `, "\t\n\r {\"a\" \t:\n [ ] \r} \n", `{"a":{}}`, `{"a":[]}`, `{"a":[[],{},[{}]]}`, `{"a":[1,2,]}`, `{"a":[,1]}`, `{"a":[1 2]}`, `{"a":{"b":1,}}`, `{"a":{,}}`,
	`{"a":1,}`, `{,"a":1}`, `{"a" 1}`, `{"a"::1}`, `{"a":}`, `{:1}`, `{1:1}`, `{"a":1 "b":2}`, `{"a":1}}`, `{{"a":1}`, `{"a":[1}`, `{"a":{]}`, `{"a":1}x`, `{"a":1}{}`, `x{"a":1}`,
	`{"":1}`, `{"a":1,"a":2,"":3,"":4}`, "\ufeff{}", `{"a":/*c*/1}`, `{"a":1 // c` + "\n}",
	// not objects
	``, ` `, `[]`, `[{}]`, `null`, `true`, `1`, `"s"`,
}

func TestSyntaxScannerAgreesWithBothReferences(t *testing.T) {
	cases := append(append([]string{}, syntaxCases...), realEscapes(
		`{"s":"%u0041%u00e9%u4e2d%uD83D%uDE00"}`, `{"s":"%ud83d"}`, `{"s":"%udc00%ud83d"}`, `{"s":"%u12"}`, `{"s":"%u12G4"}`, `{"s":"%U0041"}`, `{"s":"%u"}`,
		`{"k%u0065y":1}`, `{"%u0000":1}`, `{"%ud800":1,"%udfff":2}`,
	)...)
	cases = append(cases, validDocs...)
	cases = append(cases, invalidDocs...)
	for _, doc := range cases {
		assertScanTopMatchesReferences(t, []byte(doc))
	}
}

func assertScanTopMatchesReferences(t testing.TB, body []byte) {
	t.Helper()
	top, ok := ScanTop(body)

	var std map[string]json.RawMessage
	stdOK := json.Unmarshal(body, &std) == nil && std != nil
	tok, tokOK := scanTopTokenizer(body)
	if ok != stdOK || ok != tokOK {
		t.Fatalf("ScanTop ok=%v, encoding/json=%v, tokenizer=%v for %q", ok, stdOK, tokOK, body)
	}
	if !ok {
		return
	}
	seen := map[string]string{}
	top.Range(func(name string, raw []byte) { seen[name] = string(raw) })
	if len(seen) != len(std) || len(seen) != len(tok) {
		t.Fatalf("%q: %d names, encoding/json %d, tokenizer %d", body, len(seen), len(std), len(tok))
	}
	for name, raw := range std {
		if seen[name] != string(raw) || tok[name] != string(raw) {
			t.Fatalf("%q: %q = %q, encoding/json %q, tokenizer %q", body, name, seen[name], raw, tok[name])
		}
	}
}

// The nesting limit must sit exactly where encoding/json puts it.
func TestSyntaxScannerNestingLimit(t *testing.T) {
	for _, depth := range []int{1, 2, 9998, 9999, 10000, 10001, 10002, 20000} {
		for _, shape := range []struct{ open, close string }{{"[", "]"}, {`{"a":`, "}"}} {
			inner := strings.Repeat(shape.open, depth)
			if shape.open != "[" {
				inner += "0"
			}
			body := []byte(`{"v":` + inner + strings.Repeat(shape.close, depth) + `}`)
			var std map[string]json.RawMessage
			stdOK := json.Unmarshal(body, &std) == nil
			if _, ok := ScanTop(body); ok != stdOK {
				t.Fatalf("depth %d of %q: ScanTop ok=%v, encoding/json ok=%v", depth, shape.open, ok, stdOK)
			}
			arr := []byte(`[` + inner + strings.Repeat(shape.close, depth) + `]`)
			var stdArr []json.RawMessage
			stdArrOK := json.Unmarshal(arr, &stdArr) == nil
			if _, ok := ScanArray(arr); ok != stdArrOK {
				t.Fatalf("array depth %d of %q: ScanArray ok=%v, encoding/json ok=%v", depth, shape.open, ok, stdArrOK)
			}
		}
	}
}

func FuzzSyntaxScanner(f *testing.F) {
	for _, doc := range syntaxCases {
		f.Add([]byte(doc))
	}
	for _, doc := range append(append([]string{}, validDocs...), invalidDocs...) {
		f.Add([]byte(doc))
	}
	for _, doc := range realEscapes(`{"s":"%u0041%u00e9%uD83D%uDE00","k%u0065y":[1,{"%u0061":null}]}`, `{"s":"%ud83d","t":"%u12G4"}`) {
		f.Add([]byte(doc))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		assertScanTopMatchesReferences(t, body)
		// The same bytes as an array element and as a nested value.
		wrapped := append(append([]byte(`[`), body...), ']')
		var std []json.RawMessage
		stdOK := json.Unmarshal(wrapped, &std) == nil
		got, ok := ScanArray(wrapped)
		if ok != stdOK {
			t.Fatalf("ScanArray ok=%v, encoding/json=%v for %q", ok, stdOK, wrapped)
		}
		if ok {
			if len(got) != len(std) {
				t.Fatalf("%q: %d elements, encoding/json %d", wrapped, len(got), len(std))
			}
			for i := range std {
				if string(got[i]) != string(std[i]) {
					t.Fatalf("%q[%d] = %q, encoding/json %q", wrapped, i, got[i], std[i])
				}
			}
		}
	})
}

func BenchmarkScanTop(b *testing.B) {
	msg := `{"role":"user","content":"` + strings.Repeat(`line of code with \"quotes\" and a newline\n and кириллица, `, 60) + `"},`
	body := []byte(`{"model":"m","messages":[` + strings.Repeat(msg, 300) + `{}],"stream":true}`)
	b.Run("syntax_scanner", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, ok := ScanTop(body); !ok {
				b.Fatal("refused")
			}
		}
	})
	b.Run("jsontext_tokenizer", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, ok := scanTopTokenizer(body); !ok {
				b.Fatal("refused")
			}
		}
	})
	b.Run("json_valid", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		for i := 0; i < b.N; i++ {
			_ = json.Valid(body)
		}
	})
}
