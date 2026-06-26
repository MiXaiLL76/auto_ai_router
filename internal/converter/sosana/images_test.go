package sosana

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImageGenerationRequest(t *testing.T) {
	tests := []struct {
		name        string
		size        string
		wantAspect  string
		wantErrPart string
	}{
		{name: "square", size: "1024x1024", wantAspect: "1:1"},
		{name: "wide", size: "1792x1024", wantAspect: "16:9"},
		{name: "portrait", size: "1024x1792", wantAspect: "9:16"},
		{name: "unknown", size: "333x777", wantAspect: "auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"nano-banana","prompt":"draw a cat","size":"` + tt.size + `","n":1}`)
			got, err := ImageGenerationRequest(body, "fallback-model")
			require.NoError(t, err)

			var req BananaCreateRequest
			require.NoError(t, json.Unmarshal(got, &req))
			assert.Equal(t, "draw a cat", req.Prompt)
			assert.Equal(t, "nano-banana", req.Model)
			assert.Equal(t, tt.wantAspect, req.AspectRatio)
			assert.Empty(t, req.ImageURLs)
		})
	}
}

func TestImageGenerationRequestRejectsMultipleImages(t *testing.T) {
	_, err := ImageGenerationRequest([]byte(`{"model":"nano-banana","prompt":"draw","n":2}`), "nano-banana")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "n=1")
}

func TestImageEditRequest(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "nano-banana",
		"prompt": "make it blue",
		"size":   "1024x1024",
		"n":      "1",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	got, err := ImageEditRequest(body, contentType, "fallback-model")
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "make it blue", req.Prompt)
	assert.Equal(t, "nano-banana", req.Model)
	assert.Equal(t, "1:1", req.AspectRatio)
	require.Len(t, req.ImageURLs, 1)
	assert.True(t, strings.HasPrefix(req.ImageURLs[0], "data:image/png;base64,"))
	assert.Contains(t, req.ImageURLs[0], base64.StdEncoding.EncodeToString(pngBytes()))
}

func TestImageEditRequestRejectsMask(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "nano-banana",
		"prompt": "make it blue",
	}, map[string][]byte{
		"image": pngBytes(),
		"mask":  pngBytes(),
	})

	_, err := ImageEditRequest(body, contentType, "fallback-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mask")
}

func TestImageEditRequestRejectsMultipleImagesCount(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "nano-banana",
		"prompt": "make it blue",
		"n":      "2",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	_, err := ImageEditRequest(body, contentType, "fallback-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "n=1")
}

func TestOpenAIImageResponse(t *testing.T) {
	url := "https://cdn.sosana.art/result.png"
	body, err := OpenAIImageResponse(BananaTaskResponse{
		Status:        StatusCompleted,
		ResultFileURL: &url,
	})
	require.NoError(t, err)

	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	require.Len(t, resp.Data, 1)
	assert.Equal(t, url, resp.Data[0].URL)
	assert.Empty(t, resp.Data[0].B64JSON)
	assert.NotZero(t, resp.Created)
}

func multipartImageEditBody(t *testing.T, fields map[string]string, files map[string][]byte) ([]byte, string) {
	t.Helper()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	for key, value := range fields {
		require.NoError(t, writer.WriteField(key, value))
	}
	for key, data := range files {
		part, err := writer.CreateFormFile(key, key+".png")
		require.NoError(t, err)
		_, err = part.Write(data)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Close())
	return buf.Bytes(), writer.FormDataContentType()
}

func pngBytes() []byte {
	return []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}
}
