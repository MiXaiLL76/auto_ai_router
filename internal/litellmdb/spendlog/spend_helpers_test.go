package spendlog

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDerefString(t *testing.T) {
	tests := []struct {
		name     string
		input    *string
		expected string
	}{
		{
			name:     "nil pointer returns empty string",
			input:    nil,
			expected: "",
		},
		{
			name:     "pointer to hello returns hello",
			input:    strPtr("hello"),
			expected: "hello",
		},
		{
			name:     "pointer to empty string returns empty string",
			input:    strPtr(""),
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := derefString(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSafeAPIKeyPrefix(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "standard key truncated to 8 chars",
			input:    "sk-1234567890abcdef",
			expected: "sk-12345...",
		},
		{
			name:     "short key gets ellipsis appended",
			input:    "abc",
			expected: "abc...",
		},
		{
			name:     "empty key returns <empty>",
			input:    "",
			expected: "<empty>",
		},
		{
			name:     "exactly 8 chars gets ellipsis appended",
			input:    "12345678",
			expected: "12345678...",
		},
		{
			name:     "9 chars truncated to 8",
			input:    "123456789",
			expected: "12345678...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := safeAPIKeyPrefix(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// strPtr is a helper to create a string pointer
func strPtr(s string) *string {
	return &s
}
