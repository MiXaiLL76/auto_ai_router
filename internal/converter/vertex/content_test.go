package vertex

import (
	"encoding/base64"
	"net/http"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertContentToParts_InputAudioOversizedReturns413(t *testing.T) {
	withMaxBase64Size(t, 16)
	oversized := strings.Repeat("A", 17)
	content := []interface{}{
		map[string]interface{}{
			"type": "input_audio",
			"input_audio": map[string]interface{}{
				"data":   oversized,
				"format": "wav",
			},
		},
	}

	parts, err := convertContentToParts(content)

	require.Nil(t, parts)
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.ErrorAs(t, err, &validationErr)
	assert.Equal(t, http.StatusRequestEntityTooLarge, validationErr.StatusCode)
}

func TestConvertContentToParts_InputAudioValidDecodes(t *testing.T) {
	raw := []byte("fake wav bytes")
	encoded := base64.StdEncoding.EncodeToString(raw)
	content := []interface{}{
		map[string]interface{}{
			"type": "input_audio",
			"input_audio": map[string]interface{}{
				"data":   encoded,
				"format": "wav",
			},
		},
	}

	parts, err := convertContentToParts(content)

	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].InlineData)
	assert.Equal(t, raw, parts[0].InlineData.Data)
	assert.Equal(t, "audio/wav", parts[0].InlineData.MIMEType)
}

func TestConvertContentToParts_FileData(t *testing.T) {
	pdf := []byte("%PDF-1.4")
	encoded := base64.StdEncoding.EncodeToString(pdf)
	tests := []struct {
		name string
		file map[string]interface{}
	}{
		{"data URL", map[string]interface{}{"file_data": "data:application/pdf;base64," + encoded, "filename": "a.pdf"}},
		{"bare base64 with a file name", map[string]interface{}{"file_data": encoded, "filename": "a.pdf"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts, err := convertContentToParts([]interface{}{
				map[string]interface{}{"type": "file", "file": tt.file},
			})

			require.NoError(t, err)
			require.Len(t, parts, 1)
			require.NotNil(t, parts[0].InlineData)
			assert.Equal(t, "application/pdf", parts[0].InlineData.MIMEType)
			assert.Equal(t, pdf, parts[0].InlineData.Data)
		})
	}
}

func TestConvertContentToParts_FileDataWithoutMIMEIsRejected(t *testing.T) {
	for _, file := range []map[string]interface{}{
		{"file_data": "JVBERi0="},
		{"file_data": "data:;base64,JVBERi0="},
		{"file_data": "data:application/;base64,JVBERi0="},
		{"file_data": "JVBERi0=", "mime_type": "/pdf"},
	} {
		parts, err := convertContentToParts([]interface{}{
			map[string]interface{}{"type": "file", "file": file},
		})

		require.Nil(t, parts, file)
		var validationErr *converterutil.RequestValidationError
		require.ErrorAs(t, err, &validationErr, file)
		assert.Equal(t, "file.file_data", validationErr.Param)
	}
}

func TestConvertContentToParts_DocumentBase64(t *testing.T) {
	parts, err := convertContentToParts([]interface{}{
		map[string]interface{}{"type": "document", "source": map[string]interface{}{
			"type": "base64", "media_type": "application/pdf", "data": "JVBERi0=",
		}},
	})

	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].InlineData)
	assert.Equal(t, "application/pdf", parts[0].InlineData.MIMEType)
	assert.Equal(t, []byte("%PDF-"), parts[0].InlineData.Data)
}

func TestConvertContentToParts_DocumentURLIsAPDF(t *testing.T) {
	parts, err := convertContentToParts([]interface{}{
		map[string]interface{}{"type": "document", "source": map[string]interface{}{
			"type": "url", "url": "https://example.com/download?id=42",
		}},
	})

	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].FileData)
	assert.Equal(t, "application/pdf", parts[0].FileData.MIMEType)
	assert.Equal(t, "https://example.com/download?id=42", parts[0].FileData.FileURI)
}

func TestConvertContentToParts_DocumentRejected(t *testing.T) {
	tests := []struct {
		name   string
		source map[string]interface{}
	}{
		{"base64 without media_type", map[string]interface{}{"type": "base64", "data": "JVBERi0="}},
		{"data: URL as a url", map[string]interface{}{"type": "url", "url": "data:application/pdf;base64,JVBERi0="}},
		{"private URL", map[string]interface{}{"type": "url", "url": "http://127.0.0.1/a.pdf"}},
		{"file id", map[string]interface{}{"type": "file", "file_id": "file_1"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts, err := convertContentToParts([]interface{}{
				map[string]interface{}{"type": "document", "source": tt.source},
			})

			require.Nil(t, parts)
			require.Error(t, err)
		})
	}
}
