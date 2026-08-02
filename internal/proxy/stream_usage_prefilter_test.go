package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestChunkMayCarryTokenUsage covers the three shapes
// converter.ExtractTokenUsageWithOptions can derive a non-nil, billable
// TokenUsage from — a "usage" key, a completed output[]/response.output[]
// item of type "web_search_call", or a choices[].message.annotations[] entry
// — plus the plain-content case that should be filtered out.
func TestChunkMayCarryTokenUsage(t *testing.T) {
	tests := []struct {
		name  string
		chunk string
		want  bool
	}{
		{
			name:  "usage-only",
			chunk: `data: {"usage":{"prompt_tokens":10,"completion_tokens":5}}` + "\n\n",
			want:  true,
		},
		{
			name:  "web_search_call-only, no usage key",
			chunk: `data: {"output":[{"type":"web_search_call","status":"completed"}]}` + "\n\n",
			want:  true,
		},
		{
			name:  "annotations-only, no usage key",
			chunk: `data: {"choices":[{"message":{"annotations":[{"type":"url_citation"}]}}]}` + "\n\n",
			want:  true,
		},
		{
			name:  "plain content chunk",
			chunk: `data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n",
			want:  false,
		},
		{
			name:  "empty chunk",
			chunk: "",
			want:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, chunkMayCarryTokenUsage([]byte(tt.chunk)))
		})
	}
}
