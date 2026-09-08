package vertexresponses

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectMIMEFromURL(t *testing.T) {
	tests := []struct {
		url  string
		want string
	}{
		{"https://example.com/photo.jpg", "image/jpeg"},
		{"https://example.com/photo.png", "image/png"},
		{"https://example.com/doc.pdf", "application/pdf"},
		{"https://example.com/clip.mp4", "video/mp4"},
		// Nothing to go on: previously "application/octet-stream", which Gemini
		// rejects outright. Now empty, so callers can fail the request themselves.
		{"https://example.com/noext", ""},
		{"https://example.com/download?id=42", ""},
	}
	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			assert.Equal(t, tt.want, detectMIMEFromURL(tt.url))
		})
	}
}

func TestConvertInputImageParts_UndetectableMIMEIsRejected(t *testing.T) {
	// A URL whose type cannot be determined is refused here rather than forwarded
	// with a placeholder MIME that the upstream is guaranteed to reject.
	parts, err := convertInputImageParts(map[string]interface{}{
		"image_url": "https://example.com/noext",
	})
	require.Error(t, err)
	assert.Nil(t, parts)
}

func TestConvertInputImageParts_KnownExtensionStillWorks(t *testing.T) {
	parts, err := convertInputImageParts(map[string]interface{}{
		"image_url": "https://example.com/photo.jpg",
	})
	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].FileData)
	assert.Equal(t, "image/jpeg", parts[0].FileData.MIMEType)
	assert.Equal(t, "https://example.com/photo.jpg", parts[0].FileData.FileURI)
}

func TestConvertInputFilePart_UndetectableMIMEIsRejected(t *testing.T) {
	parts, err := convertInputFilePart(map[string]interface{}{
		"file_url": "https://example.com/attachment",
	})
	require.Error(t, err)
	assert.Nil(t, parts)
}

func TestConvertInputFilePart_KnownExtensionStillWorks(t *testing.T) {
	parts, err := convertInputFilePart(map[string]interface{}{
		"file_url": "https://example.com/report.pdf",
	})
	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].FileData)
	assert.Equal(t, "application/pdf", parts[0].FileData.MIMEType)
}

func TestParseDataURLToPart_EmptyPayloadIsNotUsable(t *testing.T) {
	// Base64 of "" decodes without error; the resulting blob has no bytes and is
	// rejected by Gemini with "required oneof field 'data' must have one
	// initialized field". Treated as "not a usable data URL" instead.
	part, err := parseDataURLToPart("data:image/png;base64,")
	require.NoError(t, err)
	assert.Nil(t, part)

	part, err = parseDataURLToPart("data:image/png;base64,====")
	require.NoError(t, err)
	assert.Nil(t, part)
}
