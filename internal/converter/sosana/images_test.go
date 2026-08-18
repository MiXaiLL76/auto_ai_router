package sosana

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImageGenerationRequest(t *testing.T) {
	tests := []struct {
		name        string
		size        string
		wantAspect  string
		wantSize    string
		wantErrPart string
	}{
		{name: "one k square", size: "1024x1024", wantAspect: "1:1", wantSize: "1K"},
		{name: "one k wide", size: "1376x768", wantAspect: "16:9", wantSize: "1K"},
		{name: "one k portrait", size: "768x1376", wantAspect: "9:16", wantSize: "1K"},
		{name: "one k tall", size: "512x2048", wantAspect: "1:4", wantSize: "1K"},
		{name: "two k square", size: "2048x2048", wantAspect: "1:1", wantSize: "2K"},
		{name: "two k wide", size: "2752x1536", wantAspect: "16:9", wantSize: "2K"},
		{name: "four k square", size: "4096x4096", wantAspect: "1:1", wantSize: "4K"},
		{name: "four k ultra wide", size: "6336x2688", wantAspect: "21:9", wantSize: "4K"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"banana-2-1k-compliant","prompt":"draw a cat","size":"` + tt.size + `","n":1}`)
			got, concreteModel, err := ImageGenerationRequest(body, "banana-2-{image_size}-compliant")
			require.NoError(t, err)

			var req BananaCreateRequest
			require.NoError(t, json.Unmarshal(got, &req))
			assert.Equal(t, "draw a cat", req.Prompt)
			assert.Equal(t, "banana-2-"+strings.ToLower(tt.wantSize)+"-compliant", req.Model)
			assert.Equal(t, req.Model, concreteModel)
			assert.Equal(t, tt.wantAspect, req.AspectRatio)
			assert.Equal(t, tt.wantSize, req.ImageSize)
			assert.False(t, req.PromptOptimization)
			assert.Empty(t, req.ImageURLs)
		})
	}
}

func TestImageGenerationRequestPrefersProviderModel(t *testing.T) {
	got, concreteModel, err := ImageGenerationRequest([]byte(`{"model":"public-image","prompt":"draw","n":1}`), "banana-2-1k-compliant")
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "banana-2-1k-compliant", req.Model)
	assert.Equal(t, req.Model, concreteModel)
}

func TestImageGenerationRequestMapsPublicGeminiModelToSosanaTier(t *testing.T) {
	got, concreteModel, err := ImageGenerationRequest(
		[]byte(`{"model":"google/gemini-3.1-flash-image-preview","prompt":"draw","image_size":"2K","n":1}`),
		"google/gemini-3.1-flash-image-preview",
	)
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "banana-2-2k-compliant", req.Model)
	assert.Equal(t, "banana-2-2k-compliant", concreteModel)
	assert.Equal(t, "2K", req.ImageSize)
}

func TestImageGenerationRequestUsesExplicitAspectRatio(t *testing.T) {
	got, _, err := ImageGenerationRequest([]byte(`{"model":"banana-2-1k-compliant","prompt":"draw","size":"1024x1024","aspect_ratio":"16:9"}`), "banana-2-1k-compliant")
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "16:9", req.AspectRatio)
}

func TestImageGenerationRequestRejectsMultipleImages(t *testing.T) {
	_, _, err := ImageGenerationRequest([]byte(`{"model":"banana-2-1k-compliant","prompt":"draw","n":2}`), "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "n=1")
}

func TestImageGenerationRequestRejectsURLResponseFormat(t *testing.T) {
	_, _, err := ImageGenerationRequest([]byte(`{"model":"banana-2-1k-compliant","prompt":"draw","response_format":"url"}`), "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "response_format=url")
}

func TestImageGenerationRequestRejectsUnsupportedControls(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "tools", body: `{"model":"banana-2-1k-compliant","prompt":"draw","tools":[{"type":"google_search"}]}`, want: "tools"},
		{name: "thinking", body: `{"model":"banana-2-1k-compliant","prompt":"draw","thinking_level":"high"}`, want: "thinking_level"},
		{name: "output format", body: `{"model":"banana-2-1k-compliant","prompt":"draw","output_format":"jpeg"}`, want: "output_format"},
		{name: "output compression", body: `{"model":"banana-2-1k-compliant","prompt":"draw","output_compression":0}`, want: "output_compression"},
		{name: "quality auto", body: `{"model":"banana-2-1k-compliant","prompt":"draw","quality":"auto"}`, want: "quality"},
		{name: "messages", body: `{"model":"banana-2-1k-compliant","prompt":"draw","messages":[{"role":"user","content":"draw"}]}`, want: "messages"},
		{name: "image size 0.5k", body: `{"model":"banana-2-1k-compliant","prompt":"draw","image_size":"0.5K"}`, want: "image_size"},
		{name: "exact size 0.5k", body: `{"model":"banana-2-1k-compliant","prompt":"draw","size":"512x512"}`, want: "size"},
		{name: "legacy openai size", body: `{"model":"banana-2-1k-compliant","prompt":"draw","size":"1792x1024"}`, want: "size"},
		{name: "unknown size", body: `{"model":"banana-2-1k-compliant","prompt":"draw","size":"333x777"}`, want: "size"},
		{name: "reference images", body: `{"model":"banana-2-1k-compliant","prompt":"draw","image_urls":["https://example.com/a.png"]}`, want: "image_urls"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := ImageGenerationRequest([]byte(tt.body), "banana-2-1k-compliant")
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func TestImageGenerationRequestAllowsPNGOutputFormat(t *testing.T) {
	_, _, err := ImageGenerationRequest([]byte(`{"model":"banana-2-1k-compliant","prompt":"draw","output_format":"png","response_format":"b64_json"}`), "banana-2-1k-compliant")
	require.NoError(t, err)
}

func TestImageEditRequest(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "banana-2-1k-compliant",
		"prompt": "make it blue",
		"size":   "1024x1024",
		"n":      "1",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	got, concreteModel, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "make it blue", req.Prompt)
	assert.Equal(t, "banana-2-1k-compliant", req.Model)
	assert.Equal(t, req.Model, concreteModel)
	assert.Equal(t, "1:1", req.AspectRatio)
	assert.Equal(t, "1K", req.ImageSize)
	assert.False(t, req.PromptOptimization)
	require.Len(t, req.ImageURLs, 1)
	assert.True(t, strings.HasPrefix(req.ImageURLs[0], "data:image/png;base64,"))
	assert.Contains(t, req.ImageURLs[0], base64.StdEncoding.EncodeToString(pngBytes()))
}

func TestImageEditRequestPrefersProviderModel(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "public-image",
		"prompt": "make it blue",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	got, concreteModel, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.NoError(t, err)

	var req BananaCreateRequest
	require.NoError(t, json.Unmarshal(got, &req))
	assert.Equal(t, "banana-2-1k-compliant", req.Model)
	assert.Equal(t, req.Model, concreteModel)
}

func TestImageEditRequestRejectsMask(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "banana-2-1k-compliant",
		"prompt": "make it blue",
	}, map[string][]byte{
		"image": pngBytes(),
		"mask":  pngBytes(),
	})

	_, _, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mask")
}

func TestImageEditRequestRejectsMultipleImagesCount(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "banana-2-1k-compliant",
		"prompt": "make it blue",
		"n":      "2",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	_, _, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "n=1")
}

func TestImageEditRequestRejectsURLResponseFormat(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":           "banana-2-1k-compliant",
		"prompt":          "make it blue",
		"response_format": "url",
	}, map[string][]byte{
		"image": pngBytes(),
	})

	_, _, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "response_format=url")
}

func TestImageEditRequestRejectsJPEGInput(t *testing.T) {
	body, contentType := multipartImageEditBody(t, map[string]string{
		"model":  "banana-2-1k-compliant",
		"prompt": "make it blue",
	}, map[string][]byte{
		"image": jpegBytes(),
	})

	_, _, err := ImageEditRequest(body, contentType, "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "PNG")
}

func TestImageEditRequestRejectsTooManyImages(t *testing.T) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	require.NoError(t, writer.WriteField("model", "banana-2-1k-compliant"))
	require.NoError(t, writer.WriteField("prompt", "make it blue"))
	for i := 0; i < maxInputImages+1; i++ {
		part, err := writer.CreateFormFile("image", fmt.Sprintf("image-%02d.png", i))
		require.NoError(t, err)
		_, err = part.Write(pngBytes())
		require.NoError(t, err)
	}
	require.NoError(t, writer.Close())

	_, _, err := ImageEditRequest(buf.Bytes(), writer.FormDataContentType(), "banana-2-1k-compliant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too many")
}

func TestOpenAIImageResponse(t *testing.T) {
	createdAt := "2026-01-01T00:00:00Z"
	body, err := OpenAIImageResponse(BananaTaskResponse{
		Status:          StatusCompleted,
		CreatedAt:       createdAt,
		OptimizedPrompt: "A detailed result prompt",
	}, pngBytes())
	require.NoError(t, err)

	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	require.Len(t, resp.Data, 1)
	assert.Empty(t, resp.Data[0].URL)
	assert.Equal(t, "A detailed result prompt", resp.Data[0].RevisedPrompt)
	assert.Equal(t, base64.StdEncoding.EncodeToString(pngBytes()), resp.Data[0].B64JSON)
	ts, err := time.Parse(time.RFC3339, createdAt)
	require.NoError(t, err)
	assert.Equal(t, ts.Unix(), resp.Created)
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

func jpegBytes() []byte {
	return []byte{0xff, 0xd8, 0xff, 0xdb, 0, 0x43, 0, 1, 2, 3}
}
