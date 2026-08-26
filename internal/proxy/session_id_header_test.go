package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractSessionIDFromHeaders(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "no session header",
			headers: map[string]string{"Content-Type": "application/json"},
			want:    "",
		},
		{
			name:    "bare Session-Id",
			headers: map[string]string{"Session-Id": "sess-abc"},
			want:    "sess-abc",
		},
		{
			name:    "bare Session_Id with underscore",
			headers: map[string]string{"Session_Id": "sess-underscore"},
			want:    "sess-underscore",
		},
		{
			name:    "prefixed X-Codex-Session-Id",
			headers: map[string]string{"X-Codex-Session-Id": "sess-codex"},
			want:    "sess-codex",
		},
		{
			name:    "plain X-Session-Id without middle segment does not match",
			headers: map[string]string{"X-Session-Id": "sess-plain"},
			want:    "",
		},
		{
			name:    "unrelated header does not match",
			headers: map[string]string{"X-Session-Token": "not-a-match"},
			want:    "",
		},
		{
			name: "bare Session-Id preferred over prefixed variant",
			headers: map[string]string{
				"X-Codex-Session-Id": "sess-codex",
				"Session-Id":         "sess-bare",
			},
			want: "sess-bare",
		},
		{
			name:    "empty header value is skipped",
			headers: map[string]string{"Session-Id": ""},
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range tt.headers {
				h.Set(k, v)
			}
			assert.Equal(t, tt.want, extractSessionIDFromHeaders(h))
		})
	}
}
