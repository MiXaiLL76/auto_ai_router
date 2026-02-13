package litellmdb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHashToken(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "sk- prefixed token",
			input:    "sk-iq0apw_l6s9IJRu2PBVu-g",
			expected: "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
		},
		{
			name:     "already hashed token",
			input:    "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
			expected: "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3",
		},
		{
			name:     "non sk- token",
			input:    "some-other-token",
			expected: "some-other-token",
		},
		{
			name:     "empty token",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HashToken(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func BenchmarkHashToken_WithSKPrefix(b *testing.B) {
	token := "sk-iq0apw_l6s9IJRu2PBVu-g"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HashToken(token)
	}
}

func BenchmarkHashToken_WithoutSKPrefix(b *testing.B) {
	token := "f3d29bbcc0d020bb5875a9097827edea6b6f0944e415a26ded616dcbcaca42f3"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HashToken(token)
	}
}
