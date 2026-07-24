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
		result := convertImageURLToAnthropic(url)
		require.NotNil(t, result)
		assert.Equal(t, "image", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "image/jpeg", result.Source.MediaType)
		assert.Equal(t, "/9j/4AAQ", result.Source.Data)
	})

	t.Run("data_url_no_media_type", func(t *testing.T) {
		url := "data:;base64,abc123"
		result := convertImageURLToAnthropic(url)
		require.NotNil(t, result)
		assert.Equal(t, "image/jpeg", result.Source.MediaType) // falls back to jpeg
	})

	t.Run("http_url", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Contains(t, r.Header.Get("User-Agent"), "auto-ai-router")
			w.Header().Set("Content-Type", "image/jpeg")
			_, _ = w.Write([]byte{0xff, 0xd8, 0xff, 0xd9})
		}))
		defer server.Close()

		result := convertImageURLToAnthropic(server.URL + "/image.jpg")
		require.NotNil(t, result)
		assert.Equal(t, "image", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "image/jpeg", result.Source.MediaType)
	})

	t.Run("invalid_data_url_no_comma", func(t *testing.T) {
		result := convertImageURLToAnthropic("data:image/jpeg;base64")
		assert.Nil(t, result)
	})

	t.Run("invalid_scheme", func(t *testing.T) {
		result := convertImageURLToAnthropic("ftp://example.com/image.jpg")
		assert.Nil(t, result)
	})
}

func TestConvertDataURLToDocument(t *testing.T) {
	t.Run("pdf_data_url", func(t *testing.T) {
		dataURL := "data:application/pdf;base64,JVBERi0="
		result := convertDataURLToDocument(dataURL)
		require.NotNil(t, result)
		assert.Equal(t, "document", result.Type)
		assert.Equal(t, "base64", result.Source.Type)
		assert.Equal(t, "application/pdf", result.Source.MediaType)
		assert.Equal(t, "JVBERi0=", result.Source.Data)
	})

	t.Run("text_data_url", func(t *testing.T) {
		dataURL := "data:text/plain;base64,SGVsbG8="
		result := convertDataURLToDocument(dataURL)
		require.NotNil(t, result)
		assert.Equal(t, "document", result.Type)
		assert.Equal(t, "text/plain", result.Source.MediaType)
	})

	t.Run("image_rejected", func(t *testing.T) {
		dataURL := "data:image/jpeg;base64,/9j/4AAQ"
		result := convertDataURLToDocument(dataURL)
		assert.Nil(t, result, "image/* should be rejected for document blocks")
	})

	t.Run("no_comma", func(t *testing.T) {
		result := convertDataURLToDocument("data:application/pdf;base64")
		assert.Nil(t, result)
	})
}
