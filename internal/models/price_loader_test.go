package models

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeModelName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple lowercase",
			input:    "gpt-4",
			expected: "gpt-4",
		},
		{
			name:     "uppercase to lowercase",
			input:    "GPT-4-Turbo",
			expected: "gpt-4-turbo",
		},
		{
			name:     "with provider prefix slash",
			input:    "openai/gpt-4-turbo",
			expected: "gpt-4-turbo",
		},
		{
			name:     "with nested provider prefix",
			input:    "vertex/gemini-1.5-pro",
			expected: "gemini-1.5-pro",
		},
		{
			name:     "dot-separated namespace without slash",
			input:    "anthropic.claude",
			expected: "claude",
		},
		{
			name:     "with whitespace",
			input:    "  gpt-4  ",
			expected: "gpt-4",
		},
		{
			name:     "mixed case with provider",
			input:    "OpenAI/GPT-4o",
			expected: "gpt-4o",
		},
		{
			name:     "deep path prefix",
			input:    "a/b/c/model-name",
			expected: "model-name",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "dot splits without slash takes last segment",
			input:    "claude-3.5-sonnet@20241022",
			expected: "5-sonnet@20241022",
		},
		{
			name:     "slash preserves dots in model name",
			input:    "anthropic/claude-3.5-sonnet@20241022",
			expected: "claude-3.5-sonnet@20241022",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeModelName(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestHasPathTraversal(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "simple traversal",
			path:     "../etc/passwd",
			expected: true,
		},
		{
			name:     "normal path",
			path:     "normal/path/file.json",
			expected: false,
		},
		{
			name:     "nested traversal",
			path:     "path/../../etc/passwd",
			expected: true,
		},
		{
			name:     "clean absolute path",
			path:     "/home/user/data/prices.json",
			expected: false,
		},
		{
			name:     "dot only segment",
			path:     "./current/dir",
			expected: false,
		},
		{
			name:     "traversal at end",
			path:     "some/path/..",
			expected: true,
		},
		{
			name:     "empty path",
			path:     "",
			expected: false,
		},
		{
			name:     "just double dot",
			path:     "..",
			expected: true,
		},
		{
			name:     "triple dot is not traversal",
			path:     ".../safe",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasPathTraversal(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}
