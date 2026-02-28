package modelupdate

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSplitCredentialModel(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "standard credential:model",
			input:    "cred:model",
			expected: []string{"cred", "model"},
		},
		{
			name:     "no colon returns single element",
			input:    "no-colon",
			expected: []string{"no-colon"},
		},
		{
			name:     "multiple colons splits on first only",
			input:    "a:b:c",
			expected: []string{"a", "b:c"},
		},
		{
			name:     "empty string",
			input:    "",
			expected: []string{""},
		},
		{
			name:     "colon at start",
			input:    ":model-name",
			expected: []string{"", "model-name"},
		},
		{
			name:     "colon at end",
			input:    "credential:",
			expected: []string{"credential", ""},
		},
		{
			name:     "only colon",
			input:    ":",
			expected: []string{"", ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SplitCredentialModel(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}
