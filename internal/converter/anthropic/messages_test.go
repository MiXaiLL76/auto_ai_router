package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractSystemText(t *testing.T) {
	tests := []struct {
		name    string
		content interface{}
		want    []string
	}{
		{
			name:    "string system prompt",
			content: "You are a helpful assistant.",
			want:    []string{"You are a helpful assistant."},
		},
		{
			name: "array with text blocks",
			content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "First instruction.",
				},
				map[string]interface{}{
					"type": "text",
					"text": "Second instruction.",
				},
			},
			want: []string{"First instruction.", "Second instruction."},
		},
		{
			name: "array with non-text blocks ignored",
			content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Keep this.",
				},
				map[string]interface{}{
					"type": "image_url",
					"url":  "https://example.com/img.png",
				},
			},
			want: []string{"Keep this."},
		},
		{
			name: "array with empty text skipped",
			content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "",
				},
			},
			want: nil,
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
				map[string]interface{}{
					"type": "text",
					"text": "Valid block.",
				},
			},
			want: []string{"Valid block."},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractSystemText(tt.content)
			assert.Equal(t, tt.want, got)
		})
	}
}
