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

func TestNormalizeCachedAudioBreakdown(t *testing.T) {
	tests := []struct {
		name              string
		cachedTokens      int
		cachedAudioTokens int
		wantCached        int
		wantCachedAudio   int
	}{
		{name: "valid", cachedTokens: 80, cachedAudioTokens: 40, wantCached: 80, wantCachedAudio: 40},
		{name: "negative cached tokens", cachedTokens: -80, cachedAudioTokens: 40, wantCached: 0, wantCachedAudio: 0},
		{name: "negative cached audio", cachedTokens: 80, cachedAudioTokens: -40, wantCached: 80, wantCachedAudio: 0},
		{name: "cached audio capped", cachedTokens: 10, cachedAudioTokens: 40, wantCached: 10, wantCachedAudio: 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCached, gotCachedAudio := NormalizeCachedAudioBreakdown(tt.cachedTokens, tt.cachedAudioTokens)

			assert.Equal(t, tt.wantCached, gotCached)
			assert.Equal(t, tt.wantCachedAudio, gotCachedAudio)
		})
	}
}

func TestNormalizeAudioInputTokens(t *testing.T) {
	assert.Equal(t, 60, NormalizeAudioInputTokens(100, 80, 40, true))
	assert.Equal(t, 100, NormalizeAudioInputTokens(100, 80, 40, false))
	assert.Equal(t, 100, NormalizeAudioInputTokens(100, -80, 40, true))
	assert.Equal(t, 0, NormalizeAudioInputTokens(-100, 80, 40, true))
	assert.Equal(t, 0, NormalizeAudioInputTokens(20, 80, 40, true))
}

func TestNonNegativeTokenCount(t *testing.T) {
	assert.Equal(t, 0, NonNegativeTokenCount(-1))
	assert.Equal(t, 0, NonNegativeTokenCount(0))
	assert.Equal(t, 42, NonNegativeTokenCount(42))
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

func TestBuildVersionedURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		apiPath string
		want    string
	}{
		{
			name:    "base without version",
			baseURL: "https://api.example.com",
			apiPath: "/v1/messages",
			want:    "https://api.example.com/v1/messages",
		},
		{
			name:    "base with same version",
			baseURL: "https://api.example.com/v1",
			apiPath: "/v1/messages",
			want:    "https://api.example.com/v1/messages",
		},
		{
			name:    "base with trailing slash and same version",
			baseURL: "https://api.example.com/v1/",
			apiPath: "/v1/messages",
			want:    "https://api.example.com/v1/messages",
		},
		{
			name:    "path without leading slash",
			baseURL: "https://api.example.com/v1",
			apiPath: "v1/messages",
			want:    "https://api.example.com/v1/messages",
		},
		{
			name:    "different versions are preserved",
			baseURL: "https://api.example.com/v2",
			apiPath: "/v1/messages",
			want:    "https://api.example.com/v2/v1/messages",
		},
		{
			name:    "non-version suffix is preserved",
			baseURL: "https://api.example.com/api",
			apiPath: "/v1/messages",
			want:    "https://api.example.com/api/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildVersionedURL(tt.baseURL, tt.apiPath)
			assert.Equal(t, tt.want, got)
		})
	}
}
