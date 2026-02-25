package main

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/modelupdate"
	"github.com/stretchr/testify/assert"
)

func TestSplitCredentialModel(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "standard format",
			input:    "openai_main:gpt-4o",
			expected: []string{"openai_main", "gpt-4o"},
		},
		{
			name:     "with multiple colons in model name",
			input:    "openai_main:gpt-4o:turbo",
			expected: []string{"openai_main", "gpt-4o:turbo"},
		},
		{
			name:     "simple names",
			input:    "cred1:model1",
			expected: []string{"cred1", "model1"},
		},
		{
			name:     "with dashes and underscores",
			input:    "openai_backup:gpt-3.5-turbo",
			expected: []string{"openai_backup", "gpt-3.5-turbo"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := modelupdate.SplitCredentialModel(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}
