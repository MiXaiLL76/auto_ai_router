package vertex

import (
	"encoding/base64"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
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

		part, err := parseDataURLToPart(dataURL)
		require.NoError(t, err)
		require.NotNil(t, part)
		require.NotNil(t, part.InlineData)
		assert.Equal(t, "image/jpeg", part.InlineData.MIMEType)
		assert.Equal(t, rawData, part.InlineData.Data)
	})

	t.Run("not_data_url", func(t *testing.T) {
		result, err := parseDataURLToPart("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid_base64", func(t *testing.T) {
		result, err := parseDataURLToPart("data:image/jpeg;base64,!!!invalid!!!")
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("no_comma", func(t *testing.T) {
		result, err := parseDataURLToPart("data:image/jpeg;base64")
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("multiple_commas", func(t *testing.T) {
		// With Split (not SplitN), "data:image/jpeg;base64,abc,def" splits to 3 parts
		result, err := parseDataURLToPart("data:image/jpeg;base64,abc,def")
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("empty_mime_type_returns_semicolon_prefix", func(t *testing.T) {
		// "data:;base64" → extractMimeType returns ";base64" (end==0 falls through)
		// so parseDataURLToPart still produces a part with that as MIMEType
		result, err := parseDataURLToPart("data:;base64," + base64.StdEncoding.EncodeToString([]byte("test")))
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, ";base64", result.InlineData.MIMEType)
	})

	t.Run("oversized_payload_returns_413_error", func(t *testing.T) {
		// One byte over the limit is enough; no need to allocate a real 100MB+ string twice over.
		oversized := strings.Repeat("A", maxBase64Size+1)
		result, err := parseDataURLToPart("data:image/jpeg;base64," + oversized)
		require.Nil(t, result)
		require.Error(t, err)
		var validationErr *converterutil.RequestValidationError
		require.ErrorAs(t, err, &validationErr)
		assert.Equal(t, http.StatusRequestEntityTooLarge, validationErr.StatusCode)
	})

	t.Run("payload_at_limit_is_accepted", func(t *testing.T) {
		atLimit := strings.Repeat("A", maxBase64Size)
		_, err := parseDataURLToPart("data:image/jpeg;base64," + atLimit)
		// "A" repeated isn't valid base64 padding at this length necessarily; we only
		// care that the size gate itself doesn't reject it — a decode error (if any)
		// surfaces as (nil, nil), never as the 413 error.
		var validationErr *converterutil.RequestValidationError
		assert.False(t, errors.As(err, &validationErr), "size check must not trigger exactly at the limit")
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
			name:    "file:// URL blocked (SSRF)",
			url:     "file:///tmp/data.txt",
			fileObj: map[string]interface{}{},
			wantNil: true,
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

func TestParseURLToPart_PublicIP(t *testing.T) {
	// Use IP-based URLs to avoid DNS resolution issues in sandboxed environments
	t.Run("https URL with public IP", func(t *testing.T) {
		result := parseURLToPart("https://93.184.216.34/photo.jpg", map[string]interface{}{})
		require.NotNil(t, result)
		require.NotNil(t, result.FileData)
		assert.Equal(t, "image/jpeg", result.FileData.MIMEType)
		assert.Equal(t, "https://93.184.216.34/photo.jpg", result.FileData.FileURI)
	})

	t.Run("http URL with public IP and explicit format", func(t *testing.T) {
		result := parseURLToPart("http://93.184.216.34/file", map[string]interface{}{"format": "image/png"})
		require.NotNil(t, result)
		require.NotNil(t, result.FileData)
		assert.Equal(t, "image/png", result.FileData.MIMEType)
	})

	t.Run("URL with public IP without extension", func(t *testing.T) {
		result := parseURLToPart("https://93.184.216.34/noext", map[string]interface{}{})
		require.NotNil(t, result)
		assert.Equal(t, "application/octet-stream", result.FileData.MIMEType)
	})
}

func TestIsPrivateURL(t *testing.T) {
	assert.True(t, IsPrivateURL("http://127.0.0.1/"))
	assert.True(t, IsPrivateURL("http://10.0.0.1/"))
	assert.True(t, IsPrivateURL("http://192.168.1.1/"))
	assert.True(t, IsPrivateURL("http://172.16.0.1/"))
	assert.True(t, IsPrivateURL("http://169.254.169.254/"))
	assert.True(t, IsPrivateURL("http://localhost/"))
	assert.True(t, IsPrivateURL("http://metadata.google.internal/"))
	assert.False(t, IsPrivateURL("http://93.184.216.34/"))
	assert.False(t, IsPrivateURL("https://8.8.8.8/"))
	// Note: url.Parse is very lenient — "not-a-valid-url" parses with empty host,
	// which resolves to false (not private). This is acceptable since parseURLToPart
	// rejects non-http/https/gs URLs before calling isPrivateURL.
}
