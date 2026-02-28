package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSizeToAspectRatio(t *testing.T) {
	tests := []struct {
		size string
		want string
	}{
		{"1024x1024", "1:1"},
		{"512x512", "1:1"},
		{"256x256", "1:1"},
		{"1792x1024", "16:9"},
		{"1024x1792", "9:16"},
		{"1536x1024", "3:2"},
		{"1024x1536", "2:3"},
		{"768x1024", "3:4"},
		{"1024x768", "4:3"},
		{"819x1024", "4:5"},
		{"1024x819", "5:4"},
		{"576x1024", "9:16"},
		{"2016x1008", "21:9"},
		{"unknown", "1:1"}, // default
		{"", "1:1"},        // default
	}
	for _, tt := range tests {
		t.Run(tt.size, func(t *testing.T) {
			assert.Equal(t, tt.want, sizeToAspectRatio(tt.size))
		})
	}
}

func TestSizeToImageSize(t *testing.T) {
	tests := []struct {
		size string
		want string
	}{
		// 1K sizes
		{"1024x1024", "1K"},
		{"512x512", "1K"},
		{"256x256", "1K"},
		{"1792x1024", "1K"},
		{"1024x1792", "1K"},
		{"576x1024", "1K"},
		// 2K sizes
		{"2048x2048", "2K"},
		{"3584x2048", "2K"},
		{"2048x3584", "2K"},
		// 4K sizes
		{"4096x4096", "4K"},
		{"7168x4096", "4K"},
		{"4096x7168", "4K"},
		// Unknown → 1K default
		{"unknown", "1K"},
		{"", "1K"},
	}
	for _, tt := range tests {
		t.Run(tt.size, func(t *testing.T) {
			assert.Equal(t, tt.want, sizeToImageSize(tt.size))
		})
	}
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

		// Check extra_body has generation_config with IMAGE modality
		require.NotNil(t, chatReq.ExtraBody)
		genConfig, ok := chatReq.ExtraBody["generation_config"].(map[string]interface{})
		require.True(t, ok)
		modalities, ok := genConfig["response_modalities"].([]interface{})
		require.True(t, ok)
		assert.Contains(t, modalities, "IMAGE")
	})

	t.Run("request with size includes aspect ratio and image size", func(t *testing.T) {
		input := `{"model": "gemini-2.0-flash", "prompt": "A landscape", "size": "1792x1024"}`
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

	t.Run("invalid JSON input returns error", func(t *testing.T) {
		result, err := ImageRequestToOpenAIChatRequest([]byte("not json"))
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "failed to parse OpenAI image request")
	})
}
