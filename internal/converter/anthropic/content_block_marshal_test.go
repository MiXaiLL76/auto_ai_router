package anthropic

import (
	"bytes"
	"encoding/json"
	"testing"
)

// ContentBlock has two marshalers: MarshalJSON, and MarshalJSONTo which
// encoding/json prefers because it writes into the request being encoded
// instead of into a buffer of its own. They must produce the same bytes, so
// that which one an encoder picks is not observable.
func TestContentBlockMarshalersAgree(t *testing.T) {
	blocks := []ContentBlock{
		{},
		{Type: "text"},
		{Type: "text", Text: "plain"},
		{Type: "text", Text: "<b>html & \"quotes\"</b>   line\nbreak\t\x00 \xff invalid utf8 кириллица 😀"},
		{Type: "thinking"},
		{Type: "thinking", Thinking: "step <1> & step 2", Signature: "sig=="},
		{Type: "thinking", Thinking: "", Signature: "only-signature"},
		{Type: "redacted_thinking", Data: "opaque"},
		{Type: "tool_use", ID: "toolu_1", Name: "read", Input: map[string]interface{}{"path": "/a<b>", "n": 1.5, "nested": []interface{}{nil, true, map[string]interface{}{"z": 1, "a": 2}}}},
		{Type: "tool_use", ID: "toolu_2", Name: "noargs", Input: map[string]interface{}{}},
		{Type: "tool_result", ToolUseID: "toolu_1", Content: "result text"},
		{Type: "tool_result", ToolUseID: "toolu_1", Content: []ContentBlock{{Type: "text", Text: "nested"}, {Type: "thinking", Thinking: "inner"}}},
		{Type: "tool_result", ToolUseID: "toolu_1", Content: []interface{}{map[string]interface{}{"type": "text", "text": "raw"}}},
		{Type: "image", Source: &MediaSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
		{Type: "document", Source: &MediaSource{Type: "url", URL: "https://example.com/a?b=1&c=<2>"}},
		{Type: "text", Text: "cached", CacheControl: map[string]interface{}{"type": "ephemeral", "ttl": "1h"}},
	}
	for _, block := range blocks {
		// MarshalJSON called directly is the reference; json.Marshal on the
		// value, on a pointer to it and inside a container goes through
		// MarshalJSONTo.
		want, err := block.MarshalJSON()
		if err != nil {
			t.Fatalf("%+v: %v", block, err)
		}
		got, err := json.Marshal(block)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("json.Marshal(block)\n got %s (%v)\nwant %s", got, err, want)
		}
		if got, err := json.Marshal(&block); err != nil || !bytes.Equal(got, want) {
			t.Fatalf("json.Marshal(&block)\n got %s (%v)\nwant %s", got, err, want)
		}
		wrapped, err := json.Marshal(AnthropicMessage{Role: "user", Content: []ContentBlock{block, block}})
		if err != nil {
			t.Fatal(err)
		}
		wantWrapped := `{"role":"user","content":[` + string(want) + `,` + string(want) + `]}`
		if string(wrapped) != wantWrapped {
			t.Fatalf("inside a message\n got %s\nwant %s", wrapped, wantWrapped)
		}
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(block); err != nil {
			t.Fatal(err)
		}
		var viaEncoder, viaMarshal interface{}
		if json.Unmarshal(buf.Bytes(), &viaEncoder) != nil || json.Unmarshal(want, &viaMarshal) != nil {
			t.Fatalf("not JSON: %s / %s", buf.Bytes(), want)
		}
	}
}
