package vertex

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/textproto"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestMapGeminiImageSize(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		size        string
		aspectRatio string
		imageSize   string
	}{
		{name: "2.5 fixed 1K", model: "gemini-2.5-flash-image", size: "1792x2400", aspectRatio: "3:4"},
		{name: "2.5 native dimensions", model: "gemini-2.5-flash-image", size: "896x1152", aspectRatio: "4:5"},
		{name: "3 Pro 1K", model: "gemini-3-pro-image-preview", size: "896x1152", aspectRatio: "4:5", imageSize: "1K"},
		{name: "3 Pro 2K", model: "gemini-3-pro-image", size: "1792x2400", aspectRatio: "3:4", imageSize: "2K"},
		{name: "3 Pro 4K", model: "google/gemini-3-pro-image", size: "3584x4800", aspectRatio: "3:4", imageSize: "4K"},
		{name: "3.1 Flash 512", model: "gemini-3.1-flash-image", size: "512x512", aspectRatio: "1:1", imageSize: "512"},
		{name: "3.1 Flash 1K", model: "gemini-3.1-flash-image-preview", size: "768x1024", aspectRatio: "3:4", imageSize: "1K"},
		{name: "3.1 Flash 2K", model: "gemini-3.1-flash-image", size: "1792x2400", aspectRatio: "3:4", imageSize: "2K"},
		{name: "3.1 Flash nearest 4:5", model: "gemini-3.1-flash-image", size: "896x1152", aspectRatio: "4:5", imageSize: "1K"},
		{name: "3.1 Flash nearest 2:3", model: "gemini-3.1-flash-image", size: "640x1024", aspectRatio: "2:3", imageSize: "1K"},
		{name: "3.1 Flash extended ratio", model: "gemini-3.1-flash-image", size: "1:8", aspectRatio: "1:8", imageSize: "1K"},
		{name: "3.1 Flash ratio with x", model: "gemini-3.1-flash-image", size: "16x9", aspectRatio: "16:9", imageSize: "1K"},
		{name: "3.1 Flash Lite fixed 1K", model: "gemini-3.1-flash-lite-image", size: "1792x2400", aspectRatio: "3:4", imageSize: "1K"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := mapGeminiImageSize(tt.model, tt.size)
			require.NoError(t, err)
			require.NotNil(t, config)
			assert.Equal(t, tt.aspectRatio, config.aspectRatio)
			assert.Equal(t, tt.imageSize, config.imageSize)
		})
	}
}

func TestMapGeminiImageSizeOfficialGrid(t *testing.T) {
	tests := []struct {
		aspectRatio string
		imageSize   string
		width       int
		height      int
	}{
		{aspectRatio: "1:1", imageSize: "512", width: 512, height: 512},
		{aspectRatio: "1:1", imageSize: "1K", width: 1024, height: 1024},
		{aspectRatio: "1:1", imageSize: "2K", width: 2048, height: 2048},
		{aspectRatio: "1:1", imageSize: "4K", width: 4096, height: 4096},
		{aspectRatio: "1:4", imageSize: "512", width: 256, height: 1024},
		{aspectRatio: "1:4", imageSize: "1K", width: 512, height: 2048},
		{aspectRatio: "1:4", imageSize: "2K", width: 1024, height: 4096},
		{aspectRatio: "1:4", imageSize: "4K", width: 2048, height: 8192},
		{aspectRatio: "1:8", imageSize: "512", width: 192, height: 1536},
		{aspectRatio: "1:8", imageSize: "1K", width: 384, height: 3072},
		{aspectRatio: "1:8", imageSize: "2K", width: 768, height: 6144},
		{aspectRatio: "1:8", imageSize: "4K", width: 1536, height: 12288},
		{aspectRatio: "2:3", imageSize: "512", width: 424, height: 632},
		{aspectRatio: "2:3", imageSize: "1K", width: 848, height: 1264},
		{aspectRatio: "2:3", imageSize: "2K", width: 1696, height: 2528},
		{aspectRatio: "2:3", imageSize: "4K", width: 3392, height: 5056},
		{aspectRatio: "3:2", imageSize: "512", width: 632, height: 424},
		{aspectRatio: "3:2", imageSize: "1K", width: 1264, height: 848},
		{aspectRatio: "3:2", imageSize: "2K", width: 2528, height: 1696},
		{aspectRatio: "3:2", imageSize: "4K", width: 5056, height: 3392},
		{aspectRatio: "3:4", imageSize: "512", width: 448, height: 600},
		{aspectRatio: "3:4", imageSize: "1K", width: 896, height: 1200},
		{aspectRatio: "3:4", imageSize: "2K", width: 1792, height: 2400},
		{aspectRatio: "3:4", imageSize: "4K", width: 3584, height: 4800},
		{aspectRatio: "4:1", imageSize: "512", width: 1024, height: 256},
		{aspectRatio: "4:1", imageSize: "1K", width: 2048, height: 512},
		{aspectRatio: "4:1", imageSize: "2K", width: 4096, height: 1024},
		{aspectRatio: "4:1", imageSize: "4K", width: 8192, height: 2048},
		{aspectRatio: "4:3", imageSize: "512", width: 600, height: 448},
		{aspectRatio: "4:3", imageSize: "1K", width: 1200, height: 896},
		{aspectRatio: "4:3", imageSize: "2K", width: 2400, height: 1792},
		{aspectRatio: "4:3", imageSize: "4K", width: 4800, height: 3584},
		{aspectRatio: "4:5", imageSize: "512", width: 464, height: 576},
		{aspectRatio: "4:5", imageSize: "1K", width: 928, height: 1152},
		{aspectRatio: "4:5", imageSize: "2K", width: 1856, height: 2304},
		{aspectRatio: "4:5", imageSize: "4K", width: 3712, height: 4608},
		{aspectRatio: "5:4", imageSize: "512", width: 576, height: 464},
		{aspectRatio: "5:4", imageSize: "1K", width: 1152, height: 928},
		{aspectRatio: "5:4", imageSize: "2K", width: 2304, height: 1856},
		{aspectRatio: "5:4", imageSize: "4K", width: 4608, height: 3712},
		{aspectRatio: "8:1", imageSize: "512", width: 1536, height: 192},
		{aspectRatio: "8:1", imageSize: "1K", width: 3072, height: 384},
		{aspectRatio: "8:1", imageSize: "2K", width: 6144, height: 768},
		{aspectRatio: "8:1", imageSize: "4K", width: 12288, height: 1536},
		{aspectRatio: "9:16", imageSize: "512", width: 384, height: 688},
		{aspectRatio: "9:16", imageSize: "1K", width: 768, height: 1376},
		{aspectRatio: "9:16", imageSize: "2K", width: 1536, height: 2752},
		{aspectRatio: "9:16", imageSize: "4K", width: 3072, height: 5504},
		{aspectRatio: "16:9", imageSize: "512", width: 688, height: 384},
		{aspectRatio: "16:9", imageSize: "1K", width: 1376, height: 768},
		{aspectRatio: "16:9", imageSize: "2K", width: 2752, height: 1536},
		{aspectRatio: "16:9", imageSize: "4K", width: 5504, height: 3072},
		{aspectRatio: "21:9", imageSize: "512", width: 792, height: 168},
		{aspectRatio: "21:9", imageSize: "1K", width: 1584, height: 672},
		{aspectRatio: "21:9", imageSize: "2K", width: 3168, height: 1344},
		{aspectRatio: "21:9", imageSize: "4K", width: 6336, height: 2688},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s/%s", tt.aspectRatio, tt.imageSize)
		t.Run(name, func(t *testing.T) {
			config, err := mapGeminiImageSize(
				"gemini-3.1-flash-image-preview",
				fmt.Sprintf("%dx%d", tt.width, tt.height),
			)
			require.NoError(t, err)
			require.NotNil(t, config)
			assert.Equal(t, tt.aspectRatio, config.aspectRatio)
			assert.Equal(t, tt.imageSize, config.imageSize)
		})
	}
}

func TestMapGeminiImageSizeAuto(t *testing.T) {
	config, err := mapGeminiImageSize("gemini-3.1-flash-image", "auto")
	require.NoError(t, err)
	assert.Nil(t, config)
}

func TestMapGeminiImageSizeRejectsInvalidValue(t *testing.T) {
	config, err := mapGeminiImageSize("gemini-3.1-flash-image", "not-a-size")
	assert.Error(t, err)
	assert.Nil(t, config)
	assert.Contains(t, err.Error(), "expected WxH or W:H")
}

func TestImageRequestToOpenAIChatRequest(t *testing.T) {
	t.Run("basic request with prompt and model", func(t *testing.T) {
		input := `{"model": "gemini-2.0-flash", "prompt": "A cat sitting on a mat"}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		err = json.Unmarshal(result, &chatReq)
		require.NoError(t, err)

		assert.Equal(t, "gemini-2.0-flash", chatReq.Model)
		require.Len(t, chatReq.Messages, 1)
		assert.Equal(t, "user", chatReq.Messages[0].Role)
		assert.Equal(t, "A cat sitting on a mat", chatReq.Messages[0].Content)
		assert.Nil(t, chatReq.Seed)
		assert.Nil(t, chatReq.Temperature)
		assert.Nil(t, chatReq.TopP)

		// Check extra_body has generation_config with IMAGE modality
		require.NotNil(t, chatReq.ExtraBody)
		genConfig, ok := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		require.True(t, ok)
		modalities, ok := genConfig["response_modalities"].([]interface{})
		require.True(t, ok)
		assert.Contains(t, modalities, "IMAGE")
	})

	t.Run("request forwards sampling parameters", func(t *testing.T) {
		input := `{"model":"gemini-3.1-flash-image-preview","prompt":"A landscape","seed":42,"temperature":0.7,"top_p":0.9}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		require.NotNil(t, chatReq.Seed)
		assert.Equal(t, int64(42), *chatReq.Seed)
		require.NotNil(t, chatReq.Temperature)
		assert.Equal(t, 0.7, *chatReq.Temperature)
		require.NotNil(t, chatReq.TopP)
		assert.Equal(t, 0.9, *chatReq.TopP)
	})

	t.Run("zero sampling parameters remain set", func(t *testing.T) {
		input := `{"model":"gemini-3.1-flash-image-preview","prompt":"A landscape","seed":0,"temperature":0,"top_p":0}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		require.NotNil(t, chatReq.Seed)
		assert.Equal(t, int64(0), *chatReq.Seed)
		require.NotNil(t, chatReq.Temperature)
		assert.Equal(t, float64(0), *chatReq.Temperature)
		require.NotNil(t, chatReq.TopP)
		assert.Equal(t, float64(0), *chatReq.TopP)
	})

	for _, seed := range []string{"2147483648", "-2147483649"} {
		t.Run("out of range seed returns error "+seed, func(t *testing.T) {
			input := fmt.Sprintf(`{"model":"gemini-3.1-flash-image-preview","prompt":"A landscape","seed":%s}`, seed)
			result, err := ImageRequestToOpenAIChatRequest([]byte(input))
			assert.Error(t, err)
			assert.Nil(t, result)
			assert.Contains(t, err.Error(), "invalid image generation seed")
		})
	}

	t.Run("request with size includes aspect ratio and image size", func(t *testing.T) {
		input := `{"model": "gemini-3-pro-image", "prompt": "A landscape", "size": "1792x1024"}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		err = json.Unmarshal(result, &chatReq)
		require.NoError(t, err)

		genConfig, ok := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		require.True(t, ok)
		imageConfig, ok := genConfig["image_config"].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "16:9", imageConfig["aspectRatio"])
		assert.Equal(t, "1K", imageConfig["imageSize"])
	})

	t.Run("2.5 image request omits unsupported image size", func(t *testing.T) {
		input := `{"model": "gemini-2.5-flash-image", "prompt": "A landscape", "size": "1792x1024"}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		genConfig := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		imageConfig := genConfig["image_config"].(map[string]interface{})
		assert.Equal(t, "16:9", imageConfig["aspectRatio"])
		assert.NotContains(t, imageConfig, "imageSize")
	})

	t.Run("request clamps n to 10", func(t *testing.T) {
		input := `{"model":"gemini-2.0-flash","prompt":"A landscape","n":99}`
		result, err := ImageRequestToOpenAIChatRequest([]byte(input))
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		require.NotNil(t, chatReq.N)
		assert.Equal(t, 10, *chatReq.N)
	})

	t.Run("invalid JSON input returns error", func(t *testing.T) {
		result, err := ImageRequestToOpenAIChatRequest([]byte("not json"))
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "failed to parse OpenAI image request")
	})
}

func TestImageEditRequestToOpenAIChatRequest(t *testing.T) {
	buildMultipart := func(t *testing.T, model, size string, fields map[string]string) ([]byte, string) {
		t.Helper()

		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.WriteField("model", model))
		require.NoError(t, writer.WriteField("prompt", "Make the object blue"))
		require.NoError(t, writer.WriteField("n", "2"))
		require.NoError(t, writer.WriteField("size", size))
		for name, value := range fields {
			require.NoError(t, writer.WriteField(name, value))
		}

		part, err := writer.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": []string{`form-data; name="image"; filename="input.png"`},
			"Content-Type":        []string{"image/png"},
		})
		require.NoError(t, err)
		_, err = part.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
		require.NoError(t, err)

		require.NoError(t, writer.Close())
		return buf.Bytes(), writer.FormDataContentType()
	}

	t.Run("multipart edit request converts to multimodal chat request", func(t *testing.T) {
		body, contentType := buildMultipart(t, "gemini-2.5-flash-image-preview", "1792x1024", map[string]string{
			"seed":        "0",
			"temperature": "0",
			"top_p":       "0",
		})
		result, err := ImageEditRequestToOpenAIChatRequest(body, contentType)
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		assert.Equal(t, "gemini-2.5-flash-image-preview", chatReq.Model)
		require.NotNil(t, chatReq.N)
		assert.Equal(t, 2, *chatReq.N)
		require.NotNil(t, chatReq.Seed)
		assert.Equal(t, int64(0), *chatReq.Seed)
		require.NotNil(t, chatReq.Temperature)
		assert.Equal(t, float64(0), *chatReq.Temperature)
		require.NotNil(t, chatReq.TopP)
		assert.Equal(t, float64(0), *chatReq.TopP)
		require.Len(t, chatReq.Messages, 1)

		blocks, ok := chatReq.Messages[0].Content.([]interface{})
		require.True(t, ok)
		require.Len(t, blocks, 2)

		textBlock, ok := blocks[0].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "text", textBlock["type"])
		assert.Equal(t, "Make the object blue", textBlock["text"])

		imageBlock, ok := blocks[1].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "image_url", imageBlock["type"])

		genConfig, ok := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		require.True(t, ok)
		imageConfig, ok := genConfig["image_config"].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "16:9", imageConfig["aspectRatio"])
		assert.NotContains(t, imageConfig, "imageSize")
	})

	t.Run("multipart edit uses model size profile", func(t *testing.T) {
		body, contentType := buildMultipart(t, "gemini-3.1-flash-image-preview", "1792x2400", map[string]string{
			"seed":        "42",
			"temperature": "2",
			"top_p":       "1",
		})
		result, err := ImageEditRequestToOpenAIChatRequest(body, contentType)
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		genConfig := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		imageConfig := genConfig["image_config"].(map[string]interface{})
		assert.Equal(t, "3:4", imageConfig["aspectRatio"])
		assert.Equal(t, "2K", imageConfig["imageSize"])
		require.NotNil(t, chatReq.Seed)
		assert.Equal(t, int64(42), *chatReq.Seed)
		require.NotNil(t, chatReq.Temperature)
		assert.Equal(t, float64(2), *chatReq.Temperature)
		require.NotNil(t, chatReq.TopP)
		assert.Equal(t, float64(1), *chatReq.TopP)
	})

	t.Run("omitted sampling parameters remain unset", func(t *testing.T) {
		body, contentType := buildMultipart(t, "gemini-3.1-flash-image-preview", "1024x1024", nil)
		result, err := ImageEditRequestToOpenAIChatRequest(body, contentType)
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		assert.Nil(t, chatReq.Temperature)
		assert.Nil(t, chatReq.TopP)
	})

	t.Run("invalid seed returns error", func(t *testing.T) {
		body, contentType := buildMultipart(t, "gemini-3.1-flash-image-preview", "1024x1024", map[string]string{
			"seed": "invalid",
		})
		result, err := ImageEditRequestToOpenAIChatRequest(body, contentType)
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "invalid image edit seed")
	})

	for _, test := range []struct {
		name  string
		field string
		value string
	}{
		{name: "invalid temperature", field: "temperature", value: "invalid"},
		{name: "NaN temperature", field: "temperature", value: "NaN"},
		{name: "infinite temperature", field: "temperature", value: "+Inf"},
		{name: "invalid top_p", field: "top_p", value: "invalid"},
		{name: "NaN top_p", field: "top_p", value: "NaN"},
		{name: "infinite top_p", field: "top_p", value: "+Inf"},
	} {
		t.Run(test.name+" returns error", func(t *testing.T) {
			body, contentType := buildMultipart(t, "gemini-3.1-flash-image-preview", "1024x1024", map[string]string{
				test.field: test.value,
			})
			result, err := ImageEditRequestToOpenAIChatRequest(body, contentType)
			assert.Error(t, err)
			assert.Nil(t, result)
			assert.Contains(t, err.Error(), "invalid image edit "+test.field)
		})
	}

	t.Run("missing model returns error", func(t *testing.T) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.WriteField("prompt", "Edit this"))
		require.NoError(t, writer.Close())

		result, err := ImageEditRequestToOpenAIChatRequest(buf.Bytes(), writer.FormDataContentType())
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "missing model")
	})

	t.Run("mask image is included in content blocks", func(t *testing.T) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.WriteField("model", "gemini-2.5-flash-image-preview"))
		require.NoError(t, writer.WriteField("prompt", "Edit with mask"))

		imagePart, err := writer.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": []string{`form-data; name="image"; filename="input.png"`},
			"Content-Type":        []string{"image/png"},
		})
		require.NoError(t, err)
		_, err = imagePart.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
		require.NoError(t, err)

		maskPart, err := writer.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": []string{`form-data; name="mask"; filename="mask.png"`},
			"Content-Type":        []string{"image/png"},
		})
		require.NoError(t, err)
		_, err = maskPart.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
		require.NoError(t, err)

		require.NoError(t, writer.Close())

		result, err := ImageEditRequestToOpenAIChatRequest(buf.Bytes(), writer.FormDataContentType())
		require.NoError(t, err)

		var chatReq openai.OpenAIRequest
		require.NoError(t, json.Unmarshal(result, &chatReq))
		blocks, ok := chatReq.Messages[0].Content.([]interface{})
		require.True(t, ok)
		require.Len(t, blocks, 4)

		maskPrompt, ok := blocks[2].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "text", maskPrompt["type"])
		assert.Contains(t, maskPrompt["text"], "mask")

		maskBlock, ok := blocks[3].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "image_url", maskBlock["type"])
	})

	t.Run("non-image mime type returns error", func(t *testing.T) {
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)
		require.NoError(t, writer.WriteField("model", "gemini-2.5-flash-image-preview"))
		require.NoError(t, writer.WriteField("prompt", "Edit this"))

		part, err := writer.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": []string{`form-data; name="image"; filename="input.txt"`},
			"Content-Type":        []string{"text/plain"},
		})
		require.NoError(t, err)
		_, err = part.Write([]byte("not an image"))
		require.NoError(t, err)
		require.NoError(t, writer.Close())

		result, err := ImageEditRequestToOpenAIChatRequest(buf.Bytes(), writer.FormDataContentType())
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "unsupported MIME type")
	})
}

// ---------------------------------------------------------------------------
// convertVertexUsageToImageUsage
// ---------------------------------------------------------------------------

func TestConvertVertexUsageToImageUsage(t *testing.T) {
	t.Run("basic prompt and candidate counts, no modality details", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     268,
			CandidatesTokenCount: 1024,
		}
		got := convertVertexUsageToImageUsage(meta)
		require.NotNil(t, got)
		assert.Equal(t, 268, got.InputTokens)
		assert.Equal(t, 268, got.InputTokensDetails.TextTokens, "all input counted as text when no modality details")
		assert.Equal(t, 0, got.InputTokensDetails.ImageTokens)
		assert.Equal(t, 1024, got.OutputTokens)
		assert.Equal(t, 268+1024, got.TotalTokens)
	})

	t.Run("prompt modality details split into text and image tokens", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     300,
			CandidatesTokenCount: 512,
			PromptTokensDetails: []*genai.ModalityTokenCount{
				{Modality: genai.MediaModality(genai.MediaModalityText), TokenCount: 100},
				{Modality: genai.MediaModality(genai.MediaModalityImage), TokenCount: 200},
			},
		}
		got := convertVertexUsageToImageUsage(meta)
		require.NotNil(t, got)
		assert.Equal(t, 300, got.InputTokens)
		assert.Equal(t, 100, got.InputTokensDetails.TextTokens)
		assert.Equal(t, 200, got.InputTokensDetails.ImageTokens)
		assert.Equal(t, 512, got.OutputTokens)
		assert.Equal(t, 812, got.TotalTokens)
	})

	t.Run("video prompt tokens counted as image tokens", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     150,
			CandidatesTokenCount: 256,
			PromptTokensDetails: []*genai.ModalityTokenCount{
				{Modality: genai.MediaModality(genai.MediaModalityText), TokenCount: 50},
				{Modality: genai.MediaModality(genai.MediaModalityVideo), TokenCount: 100},
			},
		}
		got := convertVertexUsageToImageUsage(meta)
		require.NotNil(t, got)
		assert.Equal(t, 50, got.InputTokensDetails.TextTokens)
		assert.Equal(t, 100, got.InputTokensDetails.ImageTokens)
	})

	t.Run("nil modality detail entries are skipped", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 200,
			PromptTokensDetails: []*genai.ModalityTokenCount{
				nil,
				{Modality: genai.MediaModality(genai.MediaModalityText), TokenCount: 100},
			},
		}
		got := convertVertexUsageToImageUsage(meta)
		require.NotNil(t, got)
		assert.Equal(t, 100, got.InputTokensDetails.TextTokens)
		assert.Equal(t, 0, got.InputTokensDetails.ImageTokens)
	})

	t.Run("zero token counts produce zero usage", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{}
		got := convertVertexUsageToImageUsage(meta)
		require.NotNil(t, got)
		assert.Equal(t, 0, got.InputTokens)
		assert.Equal(t, 0, got.InputTokensDetails.TextTokens)
		assert.Equal(t, 0, got.InputTokensDetails.ImageTokens)
		assert.Equal(t, 0, got.OutputTokens)
		assert.Equal(t, 0, got.TotalTokens)
	})

	t.Run("total equals input plus output", func(t *testing.T) {
		meta := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     400,
			CandidatesTokenCount: 600,
		}
		got := convertVertexUsageToImageUsage(meta)
		assert.Equal(t, got.InputTokens+got.OutputTokens, got.TotalTokens)
	})
}

func TestVertexChatResponseToOpenAIImage_UsageFormat(t *testing.T) {
	t.Run("response without usage metadata omits usage field", func(t *testing.T) {
		body := `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]}}]}`
		result, err := VertexChatResponseToOpenAIImage([]byte(body))
		require.NoError(t, err)

		var resp openai.OpenAIImageResponse
		require.NoError(t, json.Unmarshal(result, &resp))
		assert.Nil(t, resp.Usage)
	})

	t.Run("response with usage metadata returns image usage format", func(t *testing.T) {
		body := `{
			"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]}}],
			"usageMetadata":{"promptTokenCount":268,"candidatesTokenCount":1024,"totalTokenCount":1292}
		}`
		result, err := VertexChatResponseToOpenAIImage([]byte(body))
		require.NoError(t, err)

		var resp openai.OpenAIImageResponse
		require.NoError(t, json.Unmarshal(result, &resp))
		require.NotNil(t, resp.Usage)
		assert.Equal(t, 268, resp.Usage.InputTokens)
		assert.Equal(t, 1024, resp.Usage.OutputTokens)
		assert.Equal(t, 1292, resp.Usage.TotalTokens)
	})

	t.Run("usage JSON has input_tokens not prompt_tokens", func(t *testing.T) {
		body := `{
			"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw=="}}]}}],
			"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":200}
		}`
		result, err := VertexChatResponseToOpenAIImage([]byte(body))
		require.NoError(t, err)

		// Validate raw JSON field names — LiteLLM validates these directly
		var raw map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &raw))
		usageRaw, ok := raw["usage"].(map[string]interface{})
		require.True(t, ok, "usage field must be present")
		assert.Contains(t, usageRaw, "input_tokens", "must have input_tokens (not prompt_tokens)")
		assert.Contains(t, usageRaw, "input_tokens_details", "must have input_tokens_details")
		assert.Contains(t, usageRaw, "output_tokens", "must have output_tokens (not completion_tokens)")
		assert.NotContains(t, usageRaw, "prompt_tokens", "must not have prompt_tokens (chat format)")
		assert.NotContains(t, usageRaw, "completion_tokens", "must not have completion_tokens (chat format)")
	})

	t.Run("no images in candidates returns empty data", func(t *testing.T) {
		body := `{"candidates":[{"content":{"parts":[{"text":"no image here"}]}}]}`
		result, err := VertexChatResponseToOpenAIImage([]byte(body))
		require.NoError(t, err)

		var resp openai.OpenAIImageResponse
		require.NoError(t, json.Unmarshal(result, &resp))
		assert.Empty(t, resp.Data)
	})
}
