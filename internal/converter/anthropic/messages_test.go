package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIToAnthropic_AdaptiveThinkingDisplay(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "reasoning effort requests summary",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"reasoning_effort":"high"}`,
		},
		{
			name: "native display is preserved",
			body: `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"thinking":{"type":"adaptive","effort":"high","display":"summarized"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := OpenAIToAnthropic([]byte(tt.body), "claude-opus-4-8")
			require.NoError(t, err)

			var request map[string]interface{}
			require.NoError(t, json.Unmarshal(result, &request))
			thinking := request["thinking"].(map[string]interface{})
			assert.Equal(t, "adaptive", thinking["type"])
			assert.Equal(t, "summarized", thinking["display"])
			assert.Equal(t, "high", request["output_config"].(map[string]interface{})["effort"])
		})
	}
}

func TestExtractSystemBlocks(t *testing.T) {
	ephemeral := map[string]interface{}{"type": "ephemeral"}

	tests := []struct {
		name    string
		content interface{}
		want    []ContentBlock
	}{
		{
			name:    "string system prompt",
			content: "You are a helpful assistant.",
			want:    []ContentBlock{{Type: "text", Text: "You are a helpful assistant."}},
		},
		{
			name: "array with text blocks",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "First instruction."},
				map[string]interface{}{"type": "text", "text": "Second instruction."},
			},
			want: []ContentBlock{
				{Type: "text", Text: "First instruction."},
				{Type: "text", Text: "Second instruction."},
			},
		},
		{
			name: "cache_control preserved",
			content: []interface{}{
				map[string]interface{}{
					"type":          "text",
					"text":          "Cached instruction.",
					"cache_control": ephemeral,
				},
			},
			want: []ContentBlock{
				{Type: "text", Text: "Cached instruction.", CacheControl: ephemeral},
			},
		},
		{
			name: "mixed: one cached one plain",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "Plain."},
				map[string]interface{}{"type": "text", "text": "Cached.", "cache_control": ephemeral},
			},
			want: []ContentBlock{
				{Type: "text", Text: "Plain."},
				{Type: "text", Text: "Cached.", CacheControl: ephemeral},
			},
		},
		{
			name: "array with non-text blocks ignored",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": "Keep this."},
				map[string]interface{}{"type": "image_url", "url": "https://example.com/img.png"},
			},
			want: []ContentBlock{{Type: "text", Text: "Keep this."}},
		},
		{
			name: "array with empty text included as empty block",
			content: []interface{}{
				map[string]interface{}{"type": "text", "text": ""},
			},
			want: []ContentBlock{{Type: "text", Text: ""}},
		},
		{
			name:    "nil returns nil",
			content: nil,
			want:    nil,
		},
		{
			name:    "empty string returns nil",
			content: "",
			want:    nil,
		},
		{
			name:    "non-string non-slice returns nil",
			content: 12345,
			want:    nil,
		},
		{
			name: "array with non-map elements ignored",
			content: []interface{}{
				"not a map",
				map[string]interface{}{"type": "text", "text": "Valid block."},
			},
			want: []ContentBlock{{Type: "text", Text: "Valid block."}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractSystemBlocks(tt.content)
			assert.Equal(t, tt.want, got)
		})
	}
}
