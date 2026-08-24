package anthropic

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractMediaType(t *testing.T) {
	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"jpeg_base64", "data:image/jpeg;base64", "image/jpeg"},
		{"png_base64", "data:image/png;base64", "image/png"},
		{"pdf_base64", "data:application/pdf;base64", "application/pdf"},
		{"no_colon", "image/jpeg;base64", ""},
		{"no_semicolon", "data:image/jpeg", "image/jpeg"},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractMediaType(tt.header))
		})
	}
}

func TestConvertImageURLToAnthropic(t *testing.T) {
	t.Run("data_url", func(t *testing.T) {
		url := "data:image/jpeg;base64,/9j/4AAQ"
		result, err := convertImageURLToAnthropic(url)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, "image", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "image/jpeg", result.Source.MediaType)
		assert.Equal(t, "/9j/4AAQ", result.Source.Data)
	})

	t.Run("data_url_no_media_type_rejected", func(t *testing.T) {
		url := "data:;base64,abc123"
		result, err := convertImageURLToAnthropic(url)
		require.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("http_url", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Contains(t, r.Header.Get("User-Agent"), "auto-ai-router")
			w.Header().Set("Content-Type", "image/jpeg")
			_, _ = w.Write([]byte{0xff, 0xd8, 0xff, 0xd9})
		}))
		defer server.Close()

		result, err := convertImageURLToAnthropic(server.URL + "/image.jpg")
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, "image", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "image/jpeg", result.Source.MediaType)
	})

	t.Run("invalid_data_url_no_comma", func(t *testing.T) {
		result, err := convertImageURLToAnthropic("data:image/jpeg;base64")
		require.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid_scheme", func(t *testing.T) {
		result, err := convertImageURLToAnthropic("ftp://example.com/image.jpg")
		require.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("pdf_data_url_rejected", func(t *testing.T) {
		result, err := convertImageURLToAnthropic("data:application/pdf;base64,JVBERi0=")
		require.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestConvertDataURLToDocument(t *testing.T) {
	t.Run("pdf_data_url", func(t *testing.T) {
		dataURL := "data:application/pdf;base64,JVBERi0="
		result, err := ConvertDataURLToDocument(dataURL, "file_data")
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, "document", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "application/pdf", result.Source.MediaType)
		assert.Equal(t, "JVBERi0=", result.Source.Data)
	})

	t.Run("text_data_url_rejected", func(t *testing.T) {
		dataURL := "data:text/plain;base64,SGVsbG8="
		result, err := ConvertDataURLToDocument(dataURL, "file_data")
		require.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("image_rejected", func(t *testing.T) {
		dataURL := "data:image/jpeg;base64,/9j/4AAQ"
		result, err := ConvertDataURLToDocument(dataURL, "file_data")
		require.Error(t, err)
		assert.Nil(t, result, "image/* should be rejected for document blocks")
	})

	t.Run("no_comma", func(t *testing.T) {
		result, err := ConvertDataURLToDocument("data:application/pdf;base64", "file_data")
		require.Error(t, err)
		assert.Nil(t, result)
	})
}
