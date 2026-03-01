package converterutil

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncodeBase64(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
		want  string
	}{
		{
			name:  "encode known bytes",
			input: []byte("hello world"),
			want:  base64.StdEncoding.EncodeToString([]byte("hello world")),
		},
		{
			name:  "encode binary data",
			input: []byte{0x00, 0xFF, 0x7F, 0x80},
			want:  base64.StdEncoding.EncodeToString([]byte{0x00, 0xFF, 0x7F, 0x80}),
		},
		{
			name:  "empty input returns empty string",
			input: []byte{},
			want:  "",
		},
		{
			name:  "nil input returns empty string",
			input: nil,
			want:  "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EncodeBase64(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDecodeBase64(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []byte
	}{
		{
			name:  "decode valid base64",
			input: base64.StdEncoding.EncodeToString([]byte("hello world")),
			want:  []byte("hello world"),
		},
		{
			name:  "decode binary data",
			input: base64.StdEncoding.EncodeToString([]byte{0x00, 0xFF, 0x7F, 0x80}),
			want:  []byte{0x00, 0xFF, 0x7F, 0x80},
		},
		{
			name:  "invalid base64 returns nil",
			input: "!!!not-valid-base64!!!",
			want:  nil,
		},
		{
			name:  "empty string returns nil",
			input: "",
			want:  nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DecodeBase64(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}
