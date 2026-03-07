package vertex

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractMimeType(t *testing.T) {
	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"jpeg_base64", "data:image/jpeg;base64", "image/jpeg"},
		{"png_base64", "data:image/png;base64", "image/png"},
		{"pdf_base64", "data:application/pdf;base64", "application/pdf"},
		{"no_data_prefix", "image/jpeg;base64", ""},
		{"no_semicolon", "data:image/jpeg", "image/jpeg"},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractMimeType(tt.header))
		})
	}
}

func TestGetMimeTypeFromURL(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		{"jpg", "https://example.com/photo.jpg", "image/jpeg"},
		{"jpeg", "https://example.com/photo.jpeg", "image/jpeg"},
		{"png", "https://example.com/image.png", "image/png"},
		{"gif", "https://example.com/anim.gif", "image/gif"},
		{"mp4", "https://example.com/video.mp4", "video/mp4"},
		{"pdf", "https://example.com/doc.pdf", "application/pdf"},
		{"with_query", "https://example.com/image.png?w=100", "image/png"},
		{"unknown_ext", "https://example.com/file.xyz", ""},
		{"no_extension", "https://example.com/file", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, getMimeTypeFromURL(tt.url))
		})
	}
}

func TestGetAudioMimeType(t *testing.T) {
	tests := []struct {
		format string
		want   string
	}{
		{"wav", "audio/wav"},
		{"mp3", "audio/mpeg"},
		{"ogg", "audio/ogg"},
		{"opus", "audio/opus"},
		{"aac", "audio/aac"},
		{"flac", "audio/flac"},
		{"WAV", "audio/wav"},
		{"unknown", "audio/wav"}, // default
	}
	for _, tt := range tests {
		t.Run(tt.format, func(t *testing.T) {
			assert.Equal(t, tt.want, getAudioMimeType(tt.format))
		})
	}
}

func TestParseDataURLToPart(t *testing.T) {
	t.Run("valid_data_url", func(t *testing.T) {
		rawData := []byte("hello world")
		encoded := base64.StdEncoding.EncodeToString(rawData)
		dataURL := "data:image/jpeg;base64," + encoded

		part := parseDataURLToPart(dataURL)
		require.NotNil(t, part)
		require.NotNil(t, part.InlineData)
		assert.Equal(t, "image/jpeg", part.InlineData.MIMEType)
		assert.Equal(t, rawData, part.InlineData.Data)
	})

	t.Run("not_data_url", func(t *testing.T) {
		result := parseDataURLToPart("https://example.com/image.jpg")
		assert.Nil(t, result)
	})

	t.Run("invalid_base64", func(t *testing.T) {
		result := parseDataURLToPart("data:image/jpeg;base64,!!!invalid!!!")
		assert.Nil(t, result)
	})

	t.Run("no_comma", func(t *testing.T) {
		result := parseDataURLToPart("data:image/jpeg;base64")
		assert.Nil(t, result)
	})

	t.Run("multiple_commas", func(t *testing.T) {
		// With Split (not SplitN), "data:image/jpeg;base64,abc,def" splits to 3 parts
		result := parseDataURLToPart("data:image/jpeg;base64,abc,def")
		assert.Nil(t, result)
	})

	t.Run("empty_mime_type_returns_semicolon_prefix", func(t *testing.T) {
		// "data:;base64" → extractMimeType returns ";base64" (end==0 falls through)
		// so parseDataURLToPart still produces a part with that as MIMEType
		result := parseDataURLToPart("data:;base64," + base64.StdEncoding.EncodeToString([]byte("test")))
		require.NotNil(t, result)
		assert.Equal(t, ";base64", result.InlineData.MIMEType)
	})
}

func TestParseURLToPart(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		fileObj map[string]interface{}
		wantNil bool
		mime    string
		uri     string
	}{
		{
			name:    "http URL with explicit format",
			url:     "http://example.com/file",
			fileObj: map[string]interface{}{"format": "image/png"},
			wantNil: false,
			mime:    "image/png",
			uri:     "http://example.com/file",
		},
		{
			name:    "https URL with extension-based mime",
			url:     "https://example.com/photo.jpg",
			fileObj: map[string]interface{}{},
			wantNil: false,
			mime:    "image/jpeg",
			uri:     "https://example.com/photo.jpg",
		},
		{
			name:    "gs:// URL",
			url:     "gs://bucket/object.pdf",
			fileObj: map[string]interface{}{},
			wantNil: false,
			mime:    "application/pdf",
			uri:     "gs://bucket/object.pdf",
		},
		{
			name:    "empty URL returns nil",
			url:     "",
			fileObj: map[string]interface{}{},
			wantNil: true,
		},
		{
			name:    "unsupported scheme returns nil",
			url:     "ftp://example.com/file.txt",
			fileObj: map[string]interface{}{},
			wantNil: true,
		},
		{
			name:    "URL without extension falls back to octet-stream",
			url:     "https://example.com/noext",
			fileObj: map[string]interface{}{},
			wantNil: false,
			mime:    "application/octet-stream",
			uri:     "https://example.com/noext",
		},
		{
			name:    "file:// URL supported",
			url:     "file:///tmp/data.txt",
			fileObj: map[string]interface{}{},
			wantNil: false,
			mime:    "text/plain",
			uri:     "file:///tmp/data.txt",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseURLToPart(tt.url, tt.fileObj)
			if tt.wantNil {
				assert.Nil(t, result)
				return
			}
			require.NotNil(t, result)
			require.NotNil(t, result.FileData)
			assert.Equal(t, tt.mime, result.FileData.MIMEType)
			assert.Equal(t, tt.uri, result.FileData.FileURI)
		})
	}
}
