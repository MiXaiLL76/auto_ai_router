package cryptoutils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDecryptValueHelper(t *testing.T) {
	tests := []struct {
		value      string
		key        string
		signingKey string
		expected   string
	}{
		{
			value:      "OtnbFhpXQKK8xU_FfeRgH40n3c3aqb3OZ5N05pkOJDzmcucyZAJGpA1mw96bUlO4KrLyZOD2MVpGnVE=",
			key:        "api_base",
			signingKey: "sk-1234", // todo
			expected:   "https://example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			result, err := DecryptValueHelper(tt.value, tt.key, tt.signingKey)
			assert.Nil(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}
