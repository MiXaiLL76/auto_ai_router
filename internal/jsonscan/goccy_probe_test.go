package jsonscan

import (
	"bytes"
	"testing"

	goccyjson "github.com/goccy/go-json"
)

// The request sanitizer used to read bodies with goccy/go-json; it now reads
// the ones ScanTop accepts through ScanTop. This pins the assumption that the
// two agree on what an object document is and on every top-level value.
func FuzzGoccyVsScanTop(f *testing.F) {
	for _, doc := range append(append([]string{}, validDocs...), invalidDocs...) {
		f.Add([]byte(doc))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		var g map[string]goccyjson.RawMessage
		gerr := goccyjson.Unmarshal(body, &g)
		top, ok := ScanTop(body)
		if (gerr == nil && g != nil) != ok {
			t.Fatalf("goccy err=%v nilmap=%v, scan ok=%v for %q", gerr, g == nil, ok, body)
		}
		if gerr != nil {
			return
		}
		for name, raw := range g {
			got, found := top.Raw(name)
			if !found || !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(raw)) {
				t.Fatalf("key %q: scan %q (%v) goccy %q in %q", name, got, found, raw, body)
			}
		}
	})
}
