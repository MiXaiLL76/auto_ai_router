package cryptoutils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncryptValueHelper(t *testing.T) {
	tests := []struct {
		value      string
		key        string
		signingKey string
	}{
		{
			value:      "https://example.com",
			key:        "api_base",
			signingKey: "sk-1234", // todo remove
		},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			enc_result, err := EncryptValueHelper(tt.value, tt.signingKey)
			assert.Nil(t, err)
			dec_result, err := DecryptValueHelper(enc_result, tt.key, tt.signingKey)
			assert.Nil(t, err)
			assert.Equal(t, tt.value, dec_result)
		})
	}
}
