package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertDocumentParts(t *testing.T) {
	tests := []struct {
		name string
		body string
		// wantPart is the converted first content part of the first message;
		// nil means the body is returned unchanged.
		wantPart map[string]any
	}{
		{
			name: "base64 document becomes a file without cache_control",
			body: `{"model":"gpt","messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="},"cache_control":{"type":"ephemeral"}}]}]}`,
			wantPart: map[string]any{
				"type": "file",
				"file": map[string]any{"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "document.pdf"},
			},
		},
		{
			name: "title becomes the file name",
			body: `{"messages":[{"role":"user","content":[{"type":"document","title":"report.pdf","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`,
			wantPart: map[string]any{
				"type": "file",
				"file": map[string]any{"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "report.pdf"},
			},
		},
		{
			name: "type spelled with escapes is not looked for",
			body: `{"messages":[{"role":"user","content":[{"type":"documen\u0074","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`,
		},
		{
			name: "only the exact type key counts",
			body: `{"messages":[{"role":"user","content":[{"type":"text","TYPE":"document","text":"keep me","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`,
		},
		{
			name: "url document has no Chat form",
			body: `{"messages":[{"role":"user","content":[{"type":"document","source":{"type":"url","url":"https://example.com/a.pdf"}}]}]}`,
		},
		{
			name: "tool message is left alone",
			body: `{"messages":[{"role":"tool","tool_call_id":"call_1","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`,
		},
		{
			name: "no document",
			body: `{"messages":[{"role":"user","content":"read the document"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertDocumentParts([]byte(tt.body))
			if tt.wantPart == nil {
				assert.Equal(t, tt.body, string(got))
				return
			}
			var parsed struct {
				Messages []struct {
					Content []map[string]any `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(got, &parsed))
			assert.Equal(t, tt.wantPart, parsed.Messages[0].Content[0])
		})
	}
}

func TestConvertDocumentParts_KeepsOtherFields(t *testing.T) {
	body := []byte(`{"model":"gpt","seed":12345678901234567890,"messages":[{"role":"system","content":"<b>a & b</b>"},` +
		`{"role":"user","content":[{"type":"text","text":"<keep>"},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`)

	got := string(ConvertDocumentParts(body))

	assert.Contains(t, got, `"seed":12345678901234567890`)
	assert.Contains(t, got, `"content":"<b>a & b</b>"`)
	assert.Contains(t, got, `"text":"<keep>"`)
	assert.Contains(t, got, `"type":"file"`)
}
